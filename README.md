# RAG Evaluation Agent

A standalone, reusable RAG evaluation system that tests any RAG implementation for quality, accuracy, and hallucination prevention.

## Features

- **Faithfulness Detection**: Catches hallucinations by verifying answers are grounded in context
- **Multi-Metric Evaluation**: Measures faithfulness, answer relevance, context precision, and recall
- **LLM-as-Judge**: Uses Claude Sonnet with 3-pass majority voting for consistent scoring
- **Flexible Adapters**: Works with HTTP APIs, LangChain, and LlamaIndex RAG systems
- **CI/CD Ready**: Exit codes and thresholds for automated quality gates
- **Dual Reporting**: Human-readable Markdown + machine-readable JSON

## Quick Start

### Installation

```bash
cd rag-eval-agent
pip install -r requirements.txt
```

### Set API Key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Run Evaluation

```bash
python execution/evaluator.py \
  --adapter http \
  --endpoint "http://localhost:8000/query" \
  --dataset datasets/sample_qa.json
```

## Configuration

### Command Line

```bash
python execution/evaluator.py \
  --adapter http \
  --endpoint "http://localhost:8000/query" \
  --dataset datasets/tests.json \
  --fail-under 0.8 \
  --fail-under-faithfulness 0.85 \
  --concurrency 5 \
  --verbose
```

### Config File

```yaml
# eval-config.yaml
adapter: http
endpoint: "http://localhost:8000/query"

weights:
  faithfulness: 40
  answer_relevance: 20
  context_precision: 20
  context_recall: 20

thresholds:
  composite: 0.8
  faithfulness: 0.85
```

```bash
python execution/evaluator.py --config eval-config.yaml --dataset datasets/tests.json
```

## Test Dataset Format

```json
{
  "metadata": {
    "name": "My QA Tests",
    "created": "2026-02-02"
  },
  "test_cases": [
    {
      "id": "tc_001",
      "question": "What is the refund policy?",
      "ground_truth": "Full refund within 30 days.",
      "critical": true,
      "tags": ["billing"]
    }
  ]
}
```

## Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| Faithfulness | 40% | Is the answer grounded in retrieved context? |
| Answer Relevance | 20% | Does the answer address the question? |
| Context Precision | 20% | Are retrieved docs relevant? |
| Context Recall | 20% | Did we retrieve all needed info? |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success, all thresholds passed |
| 1 | Threshold failure |
| 2 | Critical test case failed |
| 3 | Fatal error |

## Output

Reports are generated in `results/`:
- `eval_report.md` - Human-readable summary
- `eval_report.json` - Machine-readable data
- `results.jsonl` - Append-only history log

## Adapters

### HTTP (Default)
Works with any RAG system exposed via HTTP API.

```bash
--adapter http --endpoint "http://localhost:8000/query"
```

Expected API response:
```json
{
  "answer": "The response text",
  "contexts": ["Retrieved context 1", "Retrieved context 2"]
}
```

### LangChain (Coming Soon)
```bash
--adapter langchain --chain-path "path/to/rag.py:chain"
```

### LlamaIndex (Coming Soon)
```bash
--adapter llamaindex --index-path "path/to/index"
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Project Structure

```
rag-eval-agent/
├── execution/
│   ├── evaluator.py      # Main CLI
│   ├── metrics.py        # Metric calculations
│   ├── report.py         # Report generation
│   ├── config.py         # Configuration
│   └── adapters/
│       ├── base.py       # Abstract adapter
│       └── http_adapter.py
├── datasets/
│   └── sample_qa.json
├── results/              # Output directory
├── tests/
├── eval-config.yaml
└── requirements.txt
```

## License

MIT
