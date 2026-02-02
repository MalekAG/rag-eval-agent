# RAG Evaluation Agent - Product Requirements Document

## Overview

A standalone, reusable RAG evaluation system that tests any RAG implementation for quality, accuracy, and hallucination prevention.

## Problem Statement

RAG systems can silently degrade or hallucinate without obvious symptoms. There's no standard way to:
- Measure if answers are grounded in retrieved context (faithfulness)
- Track retrieval quality across system changes
- Compare different RAG implementations objectively
- Catch regressions before they reach production

## Goals

Create a **standalone, reusable RAG evaluation project** that can test any RAG system for:
- **Faithfulness** (no hallucinations) - Is the answer grounded in the retrieved context?
- **Answer Relevance** - Does the answer actually address the question?
- **Context Relevance** - Are the retrieved documents useful for answering?
- **Retrieval Quality** - Precision, recall of the retrieval step

---

## Project Location

```
C:\Users\Malek\Desktop\Claude_Projects\rag-eval-agent\
```

## Architecture

```
rag-eval-agent/
├── directives/
│   └── evaluate_rag.md          # SOP: when/how to run RAG evaluation
├── execution/
│   ├── evaluator.py             # Core evaluation orchestrator
│   ├── metrics.py               # Metric calculations (RAGAS-based)
│   ├── report.py                # Report generation (MD + JSON)
│   ├── config.py                # Config loading and validation
│   └── adapters/                # Adapters for different RAG systems
│       ├── __init__.py
│       ├── base.py              # Abstract base adapter
│       ├── langchain_adapter.py # LangChain RAG
│       ├── llamaindex_adapter.py# LlamaIndex RAG
│       └── http_adapter.py      # Generic HTTP API endpoint
├── datasets/                    # Sample test datasets
│   └── sample_qa.json
├── results/                     # Evaluation output directory
│   └── results.jsonl            # Append-only history log
├── tests/
│   ├── conftest.py              # Shared fixtures
│   └── test_evaluator.py        # Unit tests
├── eval-config.yaml             # Optional config file
├── .env                         # API keys (ANTHROPIC_API_KEY)
├── requirements.txt
├── DEVLOG.md
├── FORMALEK.md
└── README.md
```

---

## Technical Specification

### 1. LLM-as-Judge System

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Judge Model** | Claude Sonnet 3.5 | Good balance of cost/quality, hardcoded (no override) |
| **Judge Passes** | 3-pass majority vote | Reduces variance from LLM inconsistency |
| **Cost Tracking** | Estimate before run, report actual after | User awareness of token spend |

### 2. Answer Comparison

| Method | Description |
|--------|-------------|
| **Semantic Similarity** | Local sentence-transformers embeddings (all-MiniLM-L6-v2, 80MB) |
| **LLM Comparison** | Ask judge: "Do these answers convey the same information?" |
| **Thresholds** | User-configurable similarity threshold (default: 0.85) |

Both methods run in parallel; user configures which matters via config file.

### 3. Context Handling

| Scenario | Behavior |
|----------|----------|
| **Empty context returned** | Auto-fail faithfulness (score = 0) |
| **Null/missing context metadata** | Warning logged, skip context metrics, still evaluate answer |
| **Context exceeds judge window** | Truncate to fit with warning in report |

### 4. Adapter Contract

**Strict interface** - all adapters must conform:

```python
class RAGAdapter(ABC):
    @abstractmethod
    def query(self, question: str) -> tuple[str, list[str]]:
        """
        Returns (answer, retrieved_contexts)

        - answer: The generated response string
        - retrieved_contexts: List of context strings used to generate answer

        If contexts unavailable, return (answer, None) - context metrics will be skipped.
        """
        pass
```

### 5. HTTP Adapter Authentication

Supports three auth methods (in order of precedence):

1. **Config file** (`eval-config.yaml`):
   ```yaml
   http:
     headers:
       Authorization: "Bearer ${RAG_API_TOKEN}"
       X-Custom-Header: "value"
   ```

2. **CLI flags**: `--header "Authorization: Bearer xxx"` (multiple allowed)

3. **Environment variables**: `RAG_AUTH_HEADER` in `.env`

---

## Metrics System

### Default Weights (Faithfulness-Heavy)

| Metric | Weight | Description |
|--------|--------|-------------|
| Faithfulness | 40% | Is the answer grounded in retrieved context? |
| Answer Relevance | 20% | Does the answer address the question? |
| Context Precision | 20% | Are retrieved docs relevant to the query? |
| Context Recall | 20% | Did we retrieve all needed information? |

Weights are configurable via `eval-config.yaml` using normalized percentages:

```yaml
weights:
  faithfulness: 40
  answer_relevance: 20
  context_precision: 20
  context_recall: 20
# Tool normalizes to 1.0 automatically
```

### Composite Score

- Weighted average of all metrics
- Supports `--fail-under 0.8` for CI integration
- Supports per-metric thresholds: `--fail-under-faithfulness 0.9`

---

## Test Dataset Specification

### Schema

```json
{
  "metadata": {
    "name": "Customer Support QA",
    "created": "2026-01-15",
    "version": "1.0"
  },
  "test_cases": [
    {
      "id": "tc_001",
      "question": "What is the refund policy?",
      "ground_truth": "Full refund within 30 days of purchase.",
      "expected_contexts": ["policy.md"],
      "critical": true,
      "tags": ["billing", "policy"]
    },
    {
      "id": "tc_002",
      "question": "How do I reset my password?",
      "ground_truth": "Click 'Forgot Password' on the login page.",
      "critical": false
    }
  ]
}
```

### Field Definitions

| Field | Required | Description |
|-------|----------|-------------|
| `id` | No | Unique identifier (auto-generated if missing) |
| `question` | **Yes** | The query to send to RAG |
| `ground_truth` | **Yes** | Expected correct answer |
| `expected_contexts` | No | Document names/IDs that should be retrieved |
| `critical` | No | If `true`, failure triggers exit code 2 |
| `tags` | No | For filtering/grouping in reports |

### Validation

- **Upfront validation**: All test cases checked before run starts
- **Warnings on issues**: Missing optional fields logged as warnings, run continues
- **Staleness warning**: If dataset file >30 days old, warn user

---

## Execution Behavior

### Concurrency

```bash
--concurrency 5  # Run 5 test cases in parallel (default: 1)
```

Worker pool manages parallel execution against RAG endpoint.

### Retry Logic

| Setting | Value |
|---------|-------|
| Max retries | 3 |
| Backoff strategy | Exponential (1s, 2s, 4s) |
| On exhausted retries | Skip test case, mark as "error" in report |

### Timeouts

```bash
--timeout 30         # Query timeout in seconds (default: 30)
--slow-threshold 5   # Flag queries slower than this (default: 5s)
```

### Test Ordering

1. **Critical tests run first** - fail fast on core functionality
2. Remaining tests run in dataset order

### Progress Reporting

- **Progress bar**: tqdm-style `[=====>    ] 45/100`
- **Verbosity levels**:
  - `--quiet`: Errors only
  - (default): Progress bar + final summary
  - `--verbose`: Per-test status as it completes

---

## Output Specification

### Dual Output Format

Every run produces **both**:

1. **Markdown report** (`results/eval_report.md`) - Human-readable
2. **JSON report** (`results/eval_report.json`) - Machine-readable

### JSON Append Log

Each run appends to `results/results.jsonl`:

```json
{"timestamp": "2026-02-02T14:30:00Z", "composite_score": 0.87, "test_count": 50, "failures": 3, ...}
```

No automatic rotation - user manages log files.

### Full Trace Diagnostics

For failed/low-scoring test cases, report includes:

```markdown
### FAILED: tc_001 - What is the refund policy?

**Question:** What is the refund policy?

**Retrieved Contexts:**
1. "Returns are accepted within 14 days..." (returns.md)
2. "Contact support for refund requests..." (support.md)

**Generated Answer:** "You can get a refund within 14 days by contacting support."

**Ground Truth:** "Full refund within 30 days of purchase."

**Scores:**
- Faithfulness: 0.67
- Answer Relevance: 0.80

**Judge Reasoning:**
The answer claims "14 days" but the ground truth states "30 days".
The answer is partially grounded in context (returns.md mentions 14 days)
but this appears to be the wrong policy document.

**Unsupported Claims:**
- "14 days" - contradicts ground truth
- "contacting support" - not mentioned in ground truth
```

---

## CLI Interface

### Basic Usage

```bash
# Evaluate via HTTP endpoint
python execution/evaluator.py \
  --adapter http \
  --endpoint "http://localhost:8000/query" \
  --dataset datasets/my_tests.json

# Evaluate LangChain RAG
python execution/evaluator.py \
  --adapter langchain \
  --chain-path "path/to/rag.py:chain" \
  --dataset datasets/tests.json

# Evaluate LlamaIndex RAG
python execution/evaluator.py \
  --adapter llamaindex \
  --index-path "path/to/index" \
  --dataset datasets/tests.json
```

### Full Options

```bash
python execution/evaluator.py \
  --adapter http \
  --endpoint "http://localhost:8000/query" \
  --dataset datasets/tests.json \
  --output results/report.md \
  --config eval-config.yaml \
  --concurrency 5 \
  --timeout 30 \
  --slow-threshold 5 \
  --fail-under 0.8 \
  --fail-under-faithfulness 0.9 \
  --header "Authorization: Bearer xxx" \
  --verbose
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success, all thresholds passed |
| 1 | Threshold failure (composite or per-metric) |
| 2 | Critical test case failed |
| 3 | Fatal error (config invalid, RAG unreachable, etc.) |

---

## Configuration File

### `eval-config.yaml` (Optional)

```yaml
# Adapter settings
adapter: http
endpoint: "http://localhost:8000/query"

# HTTP authentication
http:
  headers:
    Authorization: "Bearer ${RAG_API_TOKEN}"
  timeout: 30
  slow_threshold: 5

# Execution settings
concurrency: 5
retry:
  max_attempts: 3
  backoff: exponential  # or "fixed"

# Metric weights (normalized percentages)
weights:
  faithfulness: 40
  answer_relevance: 20
  context_precision: 20
  context_recall: 20

# Thresholds for CI
thresholds:
  composite: 0.8
  faithfulness: 0.85
  answer_relevance: 0.7

# Comparison settings
comparison:
  semantic_similarity_threshold: 0.85
  embedding_model: "all-MiniLM-L6-v2"

# Output settings
output:
  directory: "results"
  formats: ["markdown", "json"]
```

CLI flags override config file values.

---

## Dependencies

```
# Core
anthropic>=0.18.0         # Claude API for judge
sentence-transformers     # Local embeddings
pandas                    # Data handling
requests                  # HTTP adapter
pyyaml                    # Config parsing
python-dotenv             # Environment management
tqdm                      # Progress bar

# Adapters (optional, install as needed)
langchain>=0.1.0          # LangChain adapter
llama-index>=0.10.0       # LlamaIndex adapter

# Development
pytest                    # Testing
pytest-asyncio            # Async test support
```

---

## Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `execution/evaluator.py` | Main orchestrator | P0 |
| `execution/metrics.py` | Metric calculations | P0 |
| `execution/adapters/base.py` | Abstract adapter | P0 |
| `execution/adapters/http_adapter.py` | HTTP adapter | P0 |
| `execution/report.py` | Report generation | P0 |
| `execution/config.py` | Config loading | P0 |
| `execution/adapters/langchain_adapter.py` | LangChain | P1 |
| `execution/adapters/llamaindex_adapter.py` | LlamaIndex | P1 |
| `directives/evaluate_rag.md` | SOP | P1 |
| `datasets/sample_qa.json` | Example dataset | P1 |
| `tests/test_evaluator.py` | Unit tests | P1 |
| `tests/conftest.py` | Test fixtures | P1 |

---

## Verification Criteria

1. **Unit tests pass**: `pytest tests/`
2. **Dry run works**: `python execution/evaluator.py --dry-run --dataset datasets/sample_qa.json`
3. **HTTP adapter works**: Test against a mock server
4. **Report generates**: Both MD and JSON outputs valid
5. **CI integration**: Exit codes work correctly with `--fail-under`
6. **Critical tests**: Exit code 2 when critical test fails
7. **Cost estimate**: Shows estimated tokens before run

---

## Out of Scope (v1)

- Multi-turn conversation evaluation
- Watch mode / continuous monitoring
- Web dashboard
- Automatic log rotation
- Custom metric plugins
- A/B comparison mode

---

## Example Output Report

```markdown
# RAG Evaluation Report

**Date:** 2026-02-02 14:30:00
**Adapter:** http
**Endpoint:** http://localhost:8000/query
**Test Cases:** 50 (3 critical)
**Duration:** 2m 34s
**Estimated Cost:** $0.42 | **Actual Cost:** $0.38

## Summary

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Composite** | 0.84 | 0.80 | PASS |
| Faithfulness | 0.91 | 0.85 | PASS |
| Answer Relevance | 0.82 | 0.70 | PASS |
| Context Precision | 0.79 | - | - |
| Context Recall | 0.76 | - | - |

**Latency:** avg 234ms | p50 198ms | p95 512ms | slow (>5s): 2

## Critical Tests: 3/3 PASSED

## Issues Found: 5

### FAILED: tc_012 - What is the refund policy?
[Full trace as shown above]

...

## Test Results by Tag

| Tag | Count | Avg Score |
|-----|-------|-----------|
| billing | 12 | 0.89 |
| technical | 28 | 0.82 |
| policy | 10 | 0.78 |
```
