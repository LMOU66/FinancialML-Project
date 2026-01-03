# FinancialML-Project Emissions Analysis Agent

**Financial Machine Learning Course Project**  
**Students :** Mouheb Ben Nasr & Jamila Ben Cheikh

---

An intelligent, autonomous agent system that automates the discovery, analysis, and visualization of carbon emissions datasets. The agent dynamically generates, executes, debugs, and refines Python code to answer complex data analysis queries without manual intervention.

## Objective

This project aims to create an **agentic AI system** that:

1. **Automates Dataset Discovery**: Automatically identifies and works with carbon emissions datasets
2. **Autonomous Code Generation**: Writes custom Python code to answer user queries
3. **Self-Debugging**: Iteratively fixes errors and validates outputs without human intervention
4. **Code Refinement**: Continuously improves generated code through multiple refinement cycles
5. **Data Analysis Automation**: Handles mundane data analysis tasks (aggregations, rankings, comparisons, time-series analysis)
6. **Interactive Dashboard**: Provides a ChatGPT-like interface for natural language data exploration

## How It Works

### Agent Architecture

The system operates through a **two-phase agentic workflow**:

#### Phase 1: Debug Phase (Code Generation)
1. User asks a question in natural language
2. Agent analyzes the query and dataset schema
3. LLM generates Python code to answer the query
4. Code is executed in a sandboxed environment
5. If execution fails, agent provides error feedback to LLM
6. Process repeats (up to 5 attempts) until code executes successfully

#### Phase 2: Refinement Phase (Code Optimization)
1. Working code undergoes 3 refinement cycles
2. Each cycle attempts to improve:
   - Robustness and edge case handling
   - Performance and readability
   - Error handling and validation
3. If refinement breaks the code, automatically re-enters debug phase
4. Best working version is kept and returned

### Key Features

- **Autonomous Error Recovery**: Automatically debugs and fixes broken code
- **Zero Manual Intervention**: Generates solutions from query to output
- **Safe Execution**: Sandboxed code execution with timeout protection
- **Smart Result Validation**: Verifies outputs meet quality criteria
- **Natural Language Interface**: Ask questions in plain English
- **Streaming Responses**: Real-time feedback with thinking indicators
- **Beautiful UI**: Modern ChatGPT-style interface

## Project Structure

```
emissions-agent/
│
├── app.py                          # Flask web application
├── process_query.py                # Core agentic system
├── cleaned_emission_data2.csv      # Carbon emissions dataset
│
├── templates/
│   └── index.html                  # Frontend UI
│
├── static/
│   └── style.css                   # UI styling
│
└── README.md                       # This file
```

## Getting Started

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install flask pandas numpy ollama
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd emissions-agent
```

2. **Install Ollama and required models**
```bash
# Install Ollama from https://ollama.ai
ollama pull deepseek-coder:6.7b  # For code generation
ollama pull gemma3:4b             # For response formatting
```

3. **Prepare your dataset**
```bash
# Place your emissions CSV in the root directory
# Default: cleaned_emission_data2.csv
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
```
http://localhost:8000
```

## Usage Examples

### Example Queries

**Ranking & Comparison:**
- "Which industry has the highest total emissions?"
- "Show me the top 5 industries by emissions"
- "Compare emissions between Manufacturing and Transportation"

**Time-Series Analysis:**
- "What are the emissions for 2024-03-15?"
- "Show me the trend of Energy Production over time"
- "What was the highest emission day this year?"

**Dataset Information:**
- "What is the date range of this dataset?"
- "How many industries are tracked?"
- "What's the average daily emission for each sector?"

**Complex Analysis:**
- "Which month had the highest average emissions?"
- "Show me year-over-year comparison for 2023 vs 2024"
- "Calculate the percentage contribution of each industry"

### How the Agent Works (Example)

**User Query:** *"Which industry has the highest total emissions?"*

**Agent Process:**
```
1. [DEBUG Phase]
   ├─ Attempt 1: Generate code
   ├─ Execute code
   ├─ Validate output
   └─ ✓ Success!

2. [REFINEMENT Phase]
   ├─ Round 1: Add edge case handling
   ├─ Round 2: Optimize performance
   └─ Round 3: Improve error handling

3. [OUTPUT]
   └─ Formatted response with table, insights, and suggestions
```

**Generated Code (simplified):**
```python
def process_query(query: str) -> dict:
    df = pd.read_csv(csv_path)
    
    # Get industry columns (exclude date and total columns)
    industry_cols = [col for col in df.columns 
                     if col != 'Emission Date' 
                     and 'total' not in col.lower()]
    
    # Sum emissions across all dates
    totals = df[industry_cols].sum()
    
    # Find highest
    max_industry = totals.idxmax()
    max_value = totals.max()
    
    return {
        "metric": "highest_industry_total",
        "industry": max_industry,
        "value": float(max_value)
    }
```

## Configuration

### Environment Variables

```bash
# CSV file path
export EMISSIONS_CSV_PATH="cleaned_emission_data2.csv"

# LLM model for response formatting
export OLLAMA_MODEL="gemma3:4b"
```

### Customization

**Adjust agent parameters in `app.py`:**
```python
exec_payload = generate_and_execute(
    query=user_text,
    csv_path=CSV_PATH,
    max_debug_attempts=5,    # Max attempts to generate working code
    max_refine_cycles=3      # Number of refinement iterations
)
```

**Modify code generation model in `process_query.py`:**
```python
def call_llm(messages: list, model: str = "deepseek-coder:6.7b") -> str:
    # Change model here
    pass
```

## Security Features

- **Sandboxed Execution**: Code runs in isolated process with timeout
- **Limited Imports**: Only safe libraries (pandas, numpy, re, datetime)
- **File Access Control**: Can only read from specified CSV path
- **No External Network**: Code cannot make HTTP requests
- **Safe Builtins**: Restricted Python built-in functions

## Dataset Format

The agent expects CSV files with the following structure:

```csv
Emission Date,Industry 1,Industry 2,Industry 3,...
2024-01-01,1234,5678,9012,...
2024-01-02,1345,5789,9123,...
```

**Requirements:**
- Date column named `"Emission Date"`
- Numeric columns for each industry/sector
- Daily or periodic time series data


## Under the Hood

### Core Technologies

- **Backend**: Flask (Python web framework)
- **LLM**: Ollama (local LLM inference)
- **Data**: Pandas + NumPy
- **Code Execution**: Multiprocessing with sandboxing
- **Frontend**: Vanilla JavaScript + Server-Sent Events (SSE)

### Agent Components

1. **Query Classifier**: Determines if query needs code generation
2. **Code Generator**: LLM-powered Python code synthesis
3. **Sandbox Executor**: Safe, isolated code execution
4. **Result Validator**: Ensures output quality and structure
5. **Debug Loop**: Iterative error correction
6. **Refinement Loop**: Code optimization and improvement
7. **Response Formatter**: Natural language result presentation

## Limitations

- Requires local Ollama installation
- Computational resources
- Model size and quality 

## Future Enhancements

- [ ] Multi-dataset support
- [ ] Automatic dataset discovery from URLs
- [ ] Visualization generation (charts, graphs)
- [ ] Export results to PDF/Excel
- [ ] Chat history persistence
- [ ] User authentication
- [ ] Advanced statistical analysis
- [ ] Real-time data ingestion
- [ ] API endpoint for programmatic access

