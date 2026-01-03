import re
import sys
import traceback
from io import StringIO
from typing import Any, Dict, Optional, Tuple
import multiprocessing as mp

import pandas as pd
import numpy as np
from datetime import datetime

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TOPN_RE = re.compile(r"\btop\s*(\d+)\b", re.IGNORECASE)

_TOTAL_NAME_CANON = re.compile(r"^\s*daily[_\s-]*industry\s*emissions\s*$", re.IGNORECASE)
_TOTAL_HINT_RE = re.compile(r"\b(total|overall|aggregate|all\s+industries|grand\s+total)\b", re.IGNORECASE)


def is_in_domain_query(query: str) -> bool:
    q = (query or "").lower()
    emissions_keywords = [
        "emission", "emissions", "emit", "co2", "carbon",
        "industry", "sector",
        "highest", "lowest", "top", "max", "min",
        "average", "mean", "sum", "total",
        "compare", "vs",
        "worst", "peak", "largest", "biggest", "offender", "damage",
    ]
    coverage_keywords = [
        "date", "day", "month", "year",
        "latest", "most recent", "recent",
        "last recorded", "last available", "last date",
        "earliest", "first recorded", "first date",
        "start date", "end date",
        "range", "coverage", "span",
        "recorded", "available",
    ]
    return (
            any(k in q for k in emissions_keywords)
            or any(k in q for k in coverage_keywords)
            or bool(_DATE_RE.search(q))
    )


def extract_topn_from_query(query: str, default: int = 1) -> int:
    m = _TOPN_RE.search(query or "")
    if not m:
        m2 = re.search(r"\btop[-\s]?(\d+)\b", query or "", re.IGNORECASE)
        if m2:
            try:
                return max(1, int(m2.group(1)))
            except Exception:
                return default
        return default
    try:
        return max(1, int(m.group(1)))
    except Exception:
        return default


def extract_code_from_response(llm_response: str) -> Optional[str]:
    if not llm_response:
        return None
    m = _CODE_BLOCK_RE.search(llm_response)
    return m.group(1).strip() if m else None


def code_uses_only_csv_path(code: str) -> bool:
    calls = re.findall(r"pd\s*\.\s*read_csv\s*\((.*?)\)", code, flags=re.DOTALL)
    if not calls:
        return False
    for arg_str in calls:
        first = arg_str.split(",", 1)[0].strip()
        if first != "csv_path":
            return False
    return True


def _safe_builtins() -> Dict[str, Any]:
    return {
        "None": None, "True": True, "False": False,
        "dict": dict, "list": list, "set": set, "tuple": tuple,
        "str": str, "int": int, "float": float, "bool": bool,
        "len": len, "min": min, "max": max, "sum": sum, "sorted": sorted,
        "range": range, "enumerate": enumerate, "zip": zip,
        "abs": abs, "round": round,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    }


def _worker_exec(code: str, query: str, csv_path: str, q: mp.Queue) -> None:
    exec_globals = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "re": re,
        "datetime": datetime,
        "csv_path": csv_path,
    }

    captured = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    try:
        _real_read_csv = pd.read_csv

        def _safe_read_csv(_path, *args, **kwargs):
            return _real_read_csv(csv_path, *args, **kwargs)

        exec_globals["pd"].read_csv = _safe_read_csv

        exec(code, exec_globals)

        fn = exec_globals.get("process_query")
        if fn is None:
            q.put({"success": False, "result": None, "error": "Missing process_query(query: str)",
                   "output": captured.getvalue()})
            return

        res = fn(query)
        if not isinstance(res, dict):
            q.put({"success": False, "result": None, "error": f"process_query must return dict, got {type(res)}",
                   "output": captured.getvalue()})
            return

        q.put({"success": True, "result": res, "error": None, "output": captured.getvalue()})

    except Exception as e:
        q.put({"success": False, "result": None, "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
               "output": captured.getvalue()})
    finally:
        sys.stdout = old_stdout


def execute_code(code: str, query: str, csv_path: str, timeout: int = 15) -> Dict[str, Any]:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker_exec, args=(code, query, csv_path, q), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"success": False, "result": None, "error": f"TimeoutError: exceeded {timeout}s", "output": ""}

    if not q.empty():
        return q.get()

    return {"success": False, "result": None, "error": "ExecutionError: no result returned", "output": ""}


def validate_result(result: Dict[str, Any], query: str) -> Tuple[bool, Optional[str]]:
    """Validate that the result meets basic quality criteria."""

    if not isinstance(result, dict):
        return False, f"Result must be a dict, got {type(result)}"

    if "error" in result:
        return False, f"Result contains error: {result['error']}"

    # Check for empty or meaningless results
    if len(result) == 0:
        return False, "Result is empty"

    # Check for NaN or None values in critical fields
    for key, value in result.items():
        if key in ["value", "emissions", "total"]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return False, f"Critical field '{key}' has invalid value: {value}"

    # Check for list results
    if "results" in result:
        if not isinstance(result["results"], list) or len(result["results"]) == 0:
            return False, "Results field is empty or not a list"

    return True, None


def call_llm(messages: list, model: str = "deepseek-coder:6.7b") -> str:
    """Call the LLM and return the response text."""
    import ollama

    resp = ollama.chat(
        model=model,
        messages=messages,
        stream=False
    )
    return resp["message"]["content"]


def debug_phase(query: str, csv_path: str, schema_info: str,
                messages: list, max_debug_attempts: int = 5) -> Tuple[bool, Optional[str], Optional[Dict], list]:
    """
    Debug phase: Generate code and iterate until it executes successfully.
    Returns: (success, code, result, updated_messages)
    """

    system_prompt = f"""You are a Python coding assistant specialized in data analysis.

Dataset preview (read-only):
{schema_info}

OUTPUT RULES (STRICT):
- Output ONLY a single Python code block: ```python ... ```
- NO text outside the code block
- NO imports (available: pd, np, re, datetime, csv_path)
- Define: def process_query(query: str) -> dict
- Load data with: df = pd.read_csv(csv_path)
- Return JSON-serializable dict (convert numpy types to Python types)
- Use vectorized pandas operations (no iterrows)
- For ranking queries: sum each industry column across ALL dates
- Exclude date columns and total columns from rankings

The user query is: {query}
"""

    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    last_error = None
    last_code = None

    for attempt in range(1, max_debug_attempts + 1):
        print(f"  [DEBUG] Attempt {attempt}/{max_debug_attempts}")

        if attempt == 1:
            user_msg = f"Generate code to answer: {query}"
        else:
            user_msg = f"""The previous code failed with this error:
{last_error}

Previous code:
```python
{last_code}
```

Fix the code to handle this error. Remember:
- Only use csv_path with pd.read_csv()
- Return a dict with proper structure
- Convert all numpy types to Python types
- Handle edge cases and missing data
"""

        messages.append({"role": "user", "content": user_msg})

        llm_response = call_llm(messages)
        messages.append({"role": "assistant", "content": llm_response})

        code = extract_code_from_response(llm_response)

        if code is None:
            last_error = "No Python code block found in response. Must output ```python ... ```"
            continue

        if not code_uses_only_csv_path(code):
            last_code = code
            last_error = "Safety violation: pd.read_csv must use csv_path only"
            continue

        last_code = code
        exec_result = execute_code(code, query, csv_path, timeout=15)

        if not exec_result["success"]:
            last_error = exec_result["error"]
            continue

        # Success! Code executed
        print(f"  [DEBUG] ✓ Code executed successfully after {attempt} attempts")
        return True, code, exec_result["result"], messages

    print(f"  [DEBUG] ✗ Failed to generate working code after {max_debug_attempts} attempts")
    return False, last_code, None, messages


def refinement_phase(query: str, csv_path: str, schema_info: str,
                     working_code: str, working_result: Dict,
                     messages: list, max_refine_attempts: int = 3) -> Tuple[str, Dict, list]:
    """
    Refinement phase: Improve the working code through iterative refinement.
    If refinement breaks the code, enter debug phase to fix it.
    Returns: (best_code, best_result, updated_messages)
    """

    best_code = working_code
    best_result = working_result

    refinement_prompts = [
        "Improve the code to be more robust and handle edge cases better.",
        "Optimize the code for better performance and readability.",
        "Add better error handling and validation to the code.",
    ]

    for refinement_round in range(1, max_refine_attempts + 1):
        print(f"\n[REFINE] Round {refinement_round}/{max_refine_attempts}")

        refinement_prompt = refinement_prompts[min(refinement_round - 1, len(refinement_prompts) - 1)]

        user_msg = f"""Current working code:
```python
{best_code}
```

Current output: {best_result}

{refinement_prompt}
Maintain the same functionality but improve the implementation.
Output ONLY the improved code in a ```python ...``` block.
"""

        messages.append({"role": "user", "content": user_msg})

        llm_response = call_llm(messages)
        messages.append({"role": "assistant", "content": llm_response})

        refined_code = extract_code_from_response(llm_response)

        if refined_code is None:
            print(f"  [REFINE] No code found in response, keeping current version")
            continue

        if not code_uses_only_csv_path(refined_code):
            print(f"  [REFINE] Safety violation, keeping current version")
            continue

        # Try executing refined code
        exec_result = execute_code(refined_code, query, csv_path, timeout=15)

        if not exec_result["success"]:
            print(f"  [REFINE] Refined code failed, entering debug phase...")
            # Enter debug phase to fix the broken refinement
            debug_success, fixed_code, fixed_result, messages = debug_phase(
                query, csv_path, schema_info, messages, max_debug_attempts=5
            )

            if debug_success:
                best_code = fixed_code
                best_result = fixed_result
                print(f"  [REFINE] ✓ Debug phase fixed the code")
            else:
                print(f"  [REFINE] ✗ Debug phase failed, keeping previous version")
            continue

        # Validate the refined result
        is_valid, validation_error = validate_result(exec_result["result"], query)

        if not is_valid:
            print(f"  [REFINE] Validation failed: {validation_error}, entering debug phase...")
            # Enter debug phase to fix validation issues
            debug_success, fixed_code, fixed_result, messages = debug_phase(
                query, csv_path, schema_info, messages, max_debug_attempts=5
            )

            if debug_success:
                best_code = fixed_code
                best_result = fixed_result
                print(f"  [REFINE] ✓ Debug phase fixed validation issues")
            else:
                print(f"  [REFINE] ✗ Debug phase failed, keeping previous version")
            continue

        # Refinement successful!
        best_code = refined_code
        best_result = exec_result["result"]
        print(f"  [REFINE] ✓ Refinement successful")

    return best_code, best_result, messages


def generate_and_execute(query: str, csv_path: str,
                         max_debug_attempts: int = 5,
                         max_refine_cycles: int = 3) -> Dict[str, Any]:
    """
    Main agent function with debug and refinement cycles.

    Process:
    1. Debug phase: Generate code until it executes successfully
    2. Refinement phase: Improve code 3 times, re-entering debug if it breaks
    3. Return best result
    """

    print(f"\n{'=' * 60}")
    print(f"AGENT PROCESSING QUERY: {query}")
    print(f"{'=' * 60}")

    # Check if this needs code generation
    if not is_in_domain_query(query):
        print("[AGENT] Query is out of domain, generating text response...")
        messages = [{"role": "user", "content": query}]
        response = call_llm(messages)
        return {
            "success": True,
            "needs_code": False,
            "query": query,
            "response": response,
            "result": None,
            "code": None,
        }

    # Load dataset info
    try:
        df_full = pd.read_csv(csv_path)
        df_sample = df_full.head(5)
        schema_info = f"Columns: {', '.join(df_sample.columns.tolist())}\nSample:\n{df_sample.to_string(index=False)}"
    except Exception as e:
        schema_info = f"(Schema unavailable: {e})"
        return {"success": False, "error": f"Failed to load CSV: {e}"}

    messages = []

    # PHASE 1: DEBUG - Generate working code
    print("\n[PHASE 1: DEBUG]")
    debug_success, working_code, working_result, messages = debug_phase(
        query, csv_path, schema_info, messages, max_debug_attempts
    )

    if not debug_success:
        return {
            "success": False,
            "needs_code": True,
            "query": query,
            "error": "Failed to generate working code in debug phase",
            "code": working_code,
        }

    # Validate initial result
    is_valid, validation_error = validate_result(working_result, query)
    if not is_valid:
        print(f"[VALIDATION] Initial result invalid: {validation_error}")
        # Try one more debug cycle to fix validation
        debug_success, working_code, working_result, messages = debug_phase(
            query, csv_path, schema_info, messages, max_debug_attempts=3
        )
        if not debug_success:
            return {
                "success": False,
                "needs_code": True,
                "query": query,
                "error": f"Validation failed: {validation_error}",
                "code": working_code,
                "result": working_result,
            }

    # PHASE 2: REFINEMENT - Improve the code
    print("\n[PHASE 2: REFINEMENT]")
    final_code, final_result, messages = refinement_phase(
        query, csv_path, schema_info, working_code, working_result,
        messages, max_refine_cycles
    )

    print(f"\n{'=' * 60}")
    print(f"[AGENT] ✓ COMPLETED SUCCESSFULLY")
    print(f"{'=' * 60}\n")

    return {
        "success": True,
        "needs_code": True,
        "query": query,
        "code": final_code,
        "result": final_result,
        "message": f"Generated and refined through {max_refine_cycles} cycles",
    }


