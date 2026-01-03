from __future__ import annotations

from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime
import json
import os

import ollama

from process_query import generate_and_execute

app = Flask(__name__)

CSV_PATH = os.environ.get("EMISSIONS_CSV_PATH", "cleaned_emission_data2.csv")
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")


def _format_number(x):
    try:
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, (int, float)):
            if float(x).is_integer():
                return f"{int(x):,}"
            return f"{x:,}"
        return str(x)
    except Exception:
        return str(x)


def _dict_to_table_rows(d: dict, skip_keys: set[str]) -> list[tuple[str, str]]:
    rows = []
    for k, v in d.items():
        if k in skip_keys:
            continue
        if isinstance(v, (dict, list, tuple)):
            continue
        rows.append((str(k), _format_number(v)))
    return rows


def _render_markdown_table(headers: tuple[str, str], rows: list[tuple[str, str]]) -> str:
    if not rows:
        return f"| {headers[0]} | {headers[1]} |\n|---|---|\n| No data | - |"
    lines = [
        f"| {headers[0]} | {headers[1]} |",
        "|---|---|",
    ]
    for a, b in rows:
        lines.append(f"| {a} | {b} |")
    return "\n".join(lines)


def _infer_metric_from_result(result: dict) -> str:
    """Infer the metric type from the result structure since we no longer have predefined metrics."""
    if not isinstance(result, dict):
        return "unknown"

    # Check for explicit metric field
    if "metric" in result:
        return result["metric"]

    # Infer from structure
    if "earliest_date" in result and "latest_date" in result:
        return "dataset_date_coverage"

    if "date" in result and len(result) > 2:
        return "date_lookup"

    if "industry" in result and "value" in result and "top_n" not in result:
        return "highest_industry_total"

    if "top_n" in result and "results" in result:
        return "top_emitters_total"

    return "custom_analysis"


def _normalize_result_for_prompt(exec_payload: dict) -> dict:
    out: dict = {
        "query": exec_payload.get("query"),
        "success": bool(exec_payload.get("success")),
        "needs_code": bool(exec_payload.get("needs_code")),
    }

    # Handle out-of-domain queries (no code needed)
    if exec_payload.get("needs_code") is False:
        out["response_text"] = exec_payload.get("response")
        return out

    # Get the result from the agent
    result = exec_payload.get("result") or {}
    if not isinstance(result, dict):
        out["error"] = "Invalid result payload (expected dict)."
        return out

    # Handle errors in the result
    if "error" in result:
        out["error"] = result.get("error")
        return out

    # Infer or get the metric type
    metric = _infer_metric_from_result(result)
    out["metric"] = metric

    # Handle dataset date coverage
    if metric == "dataset_date_coverage":
        out["earliest_date"] = result.get("earliest_date")
        out["latest_date"] = result.get("latest_date")
        return out

    # Handle date lookup (emissions on a specific date)
    if metric == "date_lookup":
        date = result.get("date")
        out["date"] = date
        rows = _dict_to_table_rows(result, skip_keys={"metric", "date"})
        out["table_markdown"] = _render_markdown_table(
            ("Industry", "Emission Value (Metric Tons)"),
            rows
        )
        # Find top emitter for that date
        numeric_rows = []
        for k, v in result.items():
            if k in {"metric", "date"}:
                continue
            if isinstance(v, (int, float)):
                numeric_rows.append((k, float(v)))
        if numeric_rows:
            top = max(numeric_rows, key=lambda kv: kv[1])
            out["top_emitter"] = {"industry": top[0], "value": top[1]}
        return out

    # Handle single industry ranking
    if metric == "highest_industry_total":
        out["industry"] = result.get("industry")
        out["value"] = result.get("value")
        return out

    # Handle top N emitters
    if metric == "top_emitters_total":
        out["top_n"] = result.get("top_n")
        out["results"] = result.get("results", [])
        rows = []
        for r in (result.get("results") or []):
            if isinstance(r, dict) and "industry" in r and ("value" in r or "emissions" in r):
                rows.append((str(r["industry"]), _format_number(r.get("value", r.get("emissions")))))
        out["table_markdown"] = _render_markdown_table(
            ("Industry", "Total Emissions (Metric Tons)"),
            rows
        )
        return out

    # Handle custom analysis results - be flexible with structure
    if metric == "custom_analysis":
        # Try to build a table from any list results
        if "results" in result and isinstance(result["results"], list):
            rows = []
            for r in result["results"]:
                if isinstance(r, dict):
                    # Flexible: take first two meaningful fields
                    items = [(k, v) for k, v in r.items() if not isinstance(v, (dict, list))]
                    if len(items) >= 2:
                        rows.append((str(items[0][1]), _format_number(items[1][1])))
            if rows:
                out["table_markdown"] = _render_markdown_table(
                    ("Category", "Value"),
                    rows
                )

        # Include any top-level numeric or string values
        summary = {}
        for k, v in result.items():
            if k not in {"metric", "results"} and not isinstance(v, (dict, list)):
                summary[k] = v
        if summary:
            out["summary"] = summary

    # Fallback: include raw result
    out["raw_result"] = result
    return out


def _build_llm_prompt(exec_payload: dict, normalized: dict) -> str:
    upstream_text = exec_payload.get("response") or ""

    return f"""
You are an AI assistant for an emissions dashboard.

You MUST base your answer ONLY on the structured data provided below (authoritative).
You may use the upstream reasoning text for context, but if there is any conflict,
the structured data wins.

Hard constraints:
- Do NOT fabricate numbers, industries, dates, or rows.
- Do NOT change, round, or "fix" numeric values.
- Do NOT introduce new industries beyond what's shown.
- If the result is an error, explain it and suggest a precise next query.

Style:
- Sound like ChatGPT: direct, analytical, readable.
- Do not mention internal reasoning steps.
- Use headings and bullets.

Output requirements:
1) Title
2) Table if provided (keep it exactly)
3) Key insights (bullets)
4) Conclusion (1–3 sentences)
5) Suggested next question(s)

Data provenance:
- Local dataset file: {CSV_PATH}

User query:
{json.dumps(exec_payload.get("query", ""), ensure_ascii=False)}

Structured result (authoritative):
{json.dumps(normalized, ensure_ascii=False, indent=2)}

Upstream reasoning (non-authoritative):
{upstream_text}
""".strip()


def _sse_send(text: str) -> str:
    """
    SSE requires each message to be prefixed by 'data: ' and end with blank line.
    We also split by lines to keep SSE format valid.
    """
    if text is None:
        return ""
    lines = text.splitlines() or [""]
    return "".join([f"data: {line}\n\n" for line in lines])


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# Non-stream fallback (JSON)
@app.post("/api/message")
def api_message():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    # Generate and execute code with the agentic system
    exec_payload = generate_and_execute(
        query=user_text,
        csv_path=CSV_PATH,
        max_debug_attempts=5,
        max_refine_cycles=1
    )

    normalized = _normalize_result_for_prompt(exec_payload)

    # Handle out-of-domain queries
    if exec_payload.get("needs_code") is False and normalized.get("response_text"):
        return jsonify({
            "reply": str(normalized["response_text"]),
            "timestamp": datetime.now().isoformat()
        })

    # Handle execution failures
    if not exec_payload.get("success", False):
        err = exec_payload.get("error") or "Unknown error."
        return jsonify({
            "reply": f"Error processing query: {err}",
            "timestamp": datetime.now().isoformat()
        }), 200

    # Build prompt and get LLM interpretation
    llm_prompt = _build_llm_prompt(exec_payload, normalized)

    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": llm_prompt}],
        stream=False
    )

    assistant_text = resp.get("message", {}).get("content", "")
    return jsonify({
        "reply": assistant_text,
        "timestamp": datetime.now().isoformat()
    })


# Streaming (SSE)
@app.post("/api/message_stream")
def api_message_stream():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    def event_stream():
        yield _sse_send("[THINKING] Thinking....")

        # Generate and execute code with the agentic system
        exec_payload = generate_and_execute(
            query=user_text,
            csv_path=CSV_PATH,
            max_debug_attempts=2,
            max_refine_cycles=1
        )

        yield _sse_send("[THINKING] Code executed, analyzing results...")

        normalized = _normalize_result_for_prompt(exec_payload)

        # Out-of-domain: stream plain text
        if exec_payload.get("needs_code") is False and normalized.get("response_text"):
            yield _sse_send(str(normalized["response_text"]))
            yield _sse_send("[DONE]")
            return

        # In-domain but failed: stream error text
        if not exec_payload.get("success", False):
            err = exec_payload.get("error") or "Unknown error."
            yield _sse_send(f"Error processing query: {err}")
            yield _sse_send("[DONE]")
            return

        # Build prompt and stream LLM interpretation
        llm_prompt = _build_llm_prompt(exec_payload, normalized)

        yield _sse_send("[THINKING] Preparing response...")

        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": llm_prompt}],
            stream=True
        )

        for chunk in stream:
            # Extract token from chunk
            token = ""
            if isinstance(chunk, dict):
                msg = chunk.get("message") or {}
                token = msg.get("content") or ""
            else:
                msg = getattr(chunk, "message", None)
                if msg is not None:
                    token = getattr(msg, "content", "") or ""

            # Stream only non-empty tokens
            if token:
                yield _sse_send(token)

        yield _sse_send("[DONE]")

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)