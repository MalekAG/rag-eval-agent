# Directive: Evaluate RAG System

## Goal
Run quality evaluation on a RAG (Retrieval-Augmented Generation) system to measure faithfulness, relevance, and retrieval quality.

## When to Use
- Before deploying RAG system changes to production
- After modifying retrieval or generation components
- For regression testing in CI/CD pipelines
- When comparing different RAG implementations

## Inputs
| Input | Required | Description |
|-------|----------|-------------|
| `dataset_path` | Yes | Path to JSON file with test cases |
| `endpoint` | Yes (HTTP) | RAG API endpoint URL |
| `config_path` | No | Optional YAML config file |
| `thresholds` | No | Score thresholds for pass/fail |

## Execution Steps

### 1. Validate Prerequisites
```bash
# Check if dependencies are installed
pip list | grep -E "anthropic|sentence-transformers"

# Verify API key is set
echo $ANTHROPIC_API_KEY
```

### 2. Validate Dataset
```bash
# Dry run to validate config and dataset
python execution/evaluator.py \
  --adapter http \
  --endpoint "YOUR_ENDPOINT" \
  --dataset datasets/your_tests.json \
  --dry-run
```

### 3. Run Evaluation
```bash
# Basic evaluation
python execution/evaluator.py \
  --adapter http \
  --endpoint "http://localhost:8000/query" \
  --dataset datasets/sample_qa.json

# With thresholds for CI
python execution/evaluator.py \
  --adapter http \
  --endpoint "http://localhost:8000/query" \
  --dataset datasets/sample_qa.json \
  --fail-under 0.8 \
  --fail-under-faithfulness 0.85

# With custom config
python execution/evaluator.py \
  --config eval-config.yaml \
  --dataset datasets/sample_qa.json
```

### 4. Review Results
- Check `results/eval_report.md` for human-readable report
- Check `results/eval_report.json` for machine-readable data
- Review `results/results.jsonl` for historical trends

## Exit Codes
| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success | Proceed with deployment |
| 1 | Threshold not met | Investigate low-scoring tests |
| 2 | Critical test failed | Block deployment, fix immediately |
| 3 | Fatal error | Check config and connectivity |

## Outputs
| Output | Location | Purpose |
|--------|----------|---------|
| Markdown report | `results/eval_report.md` | Human review |
| JSON report | `results/eval_report.json` | Automation/dashboards |
| History log | `results/results.jsonl` | Trend analysis |

## Metrics Explained

### Faithfulness (Weight: 40%)
- Is the answer grounded in retrieved context?
- Score of 0 = hallucination
- Auto-fails if no context retrieved

### Answer Relevance (Weight: 20%)
- Does the answer address the question?
- High score even if answer is wrong, as long as it's on-topic

### Context Precision (Weight: 20%)
- Are retrieved documents relevant to the query?
- Measures retrieval quality

### Context Recall (Weight: 20%)
- Did we retrieve all needed information?
- Compares context to ground truth requirements

## Cost Estimation
- Evaluation uses Claude Sonnet as judge
- Each test case runs ~12 API calls (4 metrics × 3 passes)
- Estimated cost: ~$0.04 per test case
- Always shown before run starts

## Edge Cases

### No Context Retrieved
- Faithfulness auto-fails (score = 0)
- Other metrics still evaluated
- Warning logged in report

### API Timeout
- Retry 3 times with exponential backoff
- Mark as "error" after retries exhausted
- Test case skipped, not failed

### Dataset Staleness
- Warning if dataset >30 days old
- Consider refreshing test cases

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run RAG Evaluation
  run: |
    python execution/evaluator.py \
      --adapter http \
      --endpoint ${{ secrets.RAG_ENDPOINT }} \
      --dataset datasets/tests.json \
      --fail-under 0.8 \
      --fail-under-faithfulness 0.85 \
      --quiet
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Troubleshooting

### "No module named 'anthropic'"
```bash
pip install anthropic
```

### "Health check failed"
- Verify RAG endpoint is running
- Check network connectivity
- Verify authentication headers

### Low Faithfulness Scores
- Check if retriever is returning relevant documents
- Verify context is being passed to LLM
- Review judge reasoning in report

## Related Files
- `execution/evaluator.py` - Main orchestrator
- `execution/metrics.py` - Metric calculations
- `execution/report.py` - Report generation
- `eval-config.yaml` - Configuration template
