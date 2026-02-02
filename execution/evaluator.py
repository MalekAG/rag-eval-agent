#!/usr/bin/env python3
"""
RAG Evaluation System - Main Orchestrator

Evaluates RAG systems for faithfulness, relevance, and retrieval quality.
"""

import argparse
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import EvalConfig, load_config, validate_config
from metrics import MetricsCalculator, calculate_composite_score
from report import (
    EvaluationReport,
    EvaluationSummary,
    ReportGenerator,
    TestResult,
)
from adapters.base import RAGAdapter
from adapters.http_adapter import HTTPAdapter


# Exit codes per spec
EXIT_SUCCESS = 0
EXIT_THRESHOLD_FAIL = 1
EXIT_CRITICAL_FAIL = 2
EXIT_FATAL_ERROR = 3


def load_dataset(dataset_path: str) -> dict:
    """
    Load and validate test dataset.

    Returns:
        Dataset dict with metadata and test_cases
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Validate required fields
    if "test_cases" not in dataset:
        raise ValueError("Dataset must contain 'test_cases' array")

    for i, tc in enumerate(dataset["test_cases"]):
        if "question" not in tc:
            raise ValueError(f"Test case {i} missing required field 'question'")
        if "ground_truth" not in tc:
            raise ValueError(f"Test case {i} missing required field 'ground_truth'")

        # Generate ID if missing
        if "id" not in tc:
            tc["id"] = f"tc_{i+1:03d}"

        # Default optional fields
        tc.setdefault("critical", False)
        tc.setdefault("tags", [])
        tc.setdefault("expected_contexts", None)

    # Check for staleness
    if "metadata" in dataset and "created" in dataset["metadata"]:
        try:
            created = datetime.fromisoformat(dataset["metadata"]["created"])
            age_days = (datetime.now() - created).days
            if age_days > 30:
                print(f"Warning: Dataset is {age_days} days old. Consider updating.", file=sys.stderr)
        except ValueError:
            pass

    return dataset


def create_adapter(config: EvalConfig) -> RAGAdapter:
    """Create RAG adapter based on configuration."""
    if config.adapter == "http":
        if not config.endpoint:
            raise ValueError("HTTP adapter requires --endpoint")

        return HTTPAdapter(
            endpoint=config.endpoint,
            headers=config.http.headers,
            timeout=config.http.timeout,
        )

    elif config.adapter == "langchain":
        # Lazy import to avoid requiring langchain if not used
        try:
            from adapters.langchain_adapter import LangChainAdapter
            return LangChainAdapter()
        except ImportError:
            raise ImportError("LangChain adapter requires langchain package. Install with: pip install langchain")

    elif config.adapter == "llamaindex":
        try:
            from adapters.llamaindex_adapter import LlamaIndexAdapter
            return LlamaIndexAdapter()
        except ImportError:
            raise ImportError("LlamaIndex adapter requires llama-index package. Install with: pip install llama-index")

    else:
        raise ValueError(f"Unknown adapter: {config.adapter}")


def evaluate_single_case(
    adapter: RAGAdapter,
    calculator: MetricsCalculator,
    test_case: dict,
    weights: dict[str, float],
    retry_config: dict,
) -> TestResult:
    """
    Evaluate a single test case with retry logic.

    Returns:
        TestResult with scores and metadata
    """
    tc_id = test_case["id"]
    question = test_case["question"]
    ground_truth = test_case["ground_truth"]
    critical = test_case.get("critical", False)
    tags = test_case.get("tags", [])
    expected_contexts = test_case.get("expected_contexts")

    max_retries = retry_config.get("max_attempts", 3)
    backoff = retry_config.get("backoff", "exponential")
    base_delay = retry_config.get("base_delay", 1.0)

    last_error = None
    latency_ms = 0

    for attempt in range(max_retries):
        try:
            # Query the RAG system
            start_time = time.time()
            answer, contexts = adapter.query(question)
            latency_ms = (time.time() - start_time) * 1000

            # Calculate metrics
            scores = calculator.evaluate(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                contexts=contexts,
                expected_contexts=expected_contexts,
            )

            # Calculate composite score
            composite = calculate_composite_score(scores, weights)

            # Determine pass/fail (based on semantic similarity for now)
            passed = scores.semantic_similarity >= 0.7 and scores.faithfulness >= 0.5

            return TestResult(
                id=tc_id,
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                contexts=contexts,
                faithfulness=scores.faithfulness,
                answer_relevance=scores.answer_relevance,
                context_precision=scores.context_precision,
                context_recall=scores.context_recall,
                semantic_similarity=scores.semantic_similarity,
                composite_score=composite,
                latency_ms=latency_ms,
                passed=passed,
                critical=critical,
                tags=tags,
                judge_reasoning=scores.judge_reasoning,
                unsupported_claims=scores.unsupported_claims,
                error=None,
            )

        except (TimeoutError, ConnectionError, ValueError) as e:
            last_error = str(e)
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Test case {tc_id}: Attempt {attempt + 1}/{max_retries} failed: {e}"
            )

            if attempt < max_retries - 1:
                # Calculate delay
                if backoff == "exponential":
                    delay = base_delay * (2 ** attempt)
                else:
                    delay = base_delay

                logger.info(f"Test case {tc_id}: Retrying in {delay:.1f}s...")
                time.sleep(delay)

    # All retries exhausted
    return TestResult(
        id=tc_id,
        question=question,
        ground_truth=ground_truth,
        answer="",
        contexts=None,
        faithfulness=0,
        answer_relevance=0,
        context_precision=None,
        context_recall=None,
        semantic_similarity=0,
        composite_score=0,
        latency_ms=latency_ms,
        passed=False,
        critical=critical,
        tags=tags,
        judge_reasoning="",
        unsupported_claims=[],
        error=last_error,
    )


def run_evaluation(
    config: EvalConfig,
    dataset: dict,
    adapter: RAGAdapter,
) -> tuple[list[TestResult], float, float]:
    """
    Run evaluation on all test cases.

    Returns:
        Tuple of (results, estimated_cost, actual_cost)
    """
    test_cases = dataset["test_cases"]

    # Sort: critical tests first
    test_cases_sorted = sorted(test_cases, key=lambda x: not x.get("critical", False))

    # Initialize calculator
    calculator = MetricsCalculator(
        embedding_model=config.comparison.embedding_model,
        similarity_threshold=config.comparison.semantic_similarity_threshold,
    )

    # Estimate cost
    estimated_cost = calculator.estimate_cost(len(test_cases))
    if not config.quiet:
        print(f"Estimated cost: ${estimated_cost:.2f}")

    weights = config.weights.normalized()
    retry_config = {
        "max_attempts": config.retry.max_attempts,
        "backoff": config.retry.backoff,
        "base_delay": config.retry.base_delay,
    }

    results = []
    critical_failed = False

    if config.concurrency == 1:
        # Sequential execution
        pbar = tqdm(
            test_cases_sorted,
            desc="Evaluating",
            disable=config.quiet,
            ncols=80,
        )

        for tc in pbar:
            result = evaluate_single_case(
                adapter, calculator, tc, weights, retry_config
            )
            results.append(result)

            # Check for critical failure
            if result.critical and not result.passed:
                critical_failed = True
                if not config.verbose:
                    break

            if config.verbose:
                status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
                pbar.write(f"  {result.id}: {status} (composite: {result.composite_score:.2f})")

    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {
                executor.submit(
                    evaluate_single_case, adapter, calculator, tc, weights, retry_config
                ): tc
                for tc in test_cases_sorted
            }

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating",
                disable=config.quiet,
                ncols=80,
            )

            for future in pbar:
                result = future.result()
                results.append(result)

                if result.critical and not result.passed:
                    critical_failed = True

                if config.verbose:
                    status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
                    pbar.write(f"  {result.id}: {status}")

    actual_cost = calculator.total_usage.estimated_cost

    return results, estimated_cost, actual_cost


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG systems for quality and faithfulness"
    )

    # Adapter options
    parser.add_argument(
        "--adapter",
        choices=["http", "langchain", "llamaindex"],
        help="RAG adapter type",
    )
    parser.add_argument("--endpoint", help="HTTP endpoint URL")
    parser.add_argument(
        "--header",
        action="append",
        dest="headers",
        help="HTTP header (can be specified multiple times)",
    )

    # Dataset
    parser.add_argument("--dataset", required=True, help="Path to test dataset JSON")

    # Config
    parser.add_argument("--config", help="Path to config YAML file")

    # Execution
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of parallel test cases",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Query timeout in seconds",
    )
    parser.add_argument(
        "--slow-threshold",
        type=int,
        help="Flag queries slower than this (seconds)",
    )

    # Thresholds
    parser.add_argument(
        "--fail-under",
        type=float,
        help="Fail if composite score below this",
    )
    parser.add_argument(
        "--fail-under-faithfulness",
        type=float,
        help="Fail if faithfulness below this",
    )
    parser.add_argument(
        "--fail-under-answer-relevance",
        type=float,
        help="Fail if answer relevance below this",
    )

    # Output
    parser.add_argument(
        "--output",
        help="Output report path (default: results/eval_report.md)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-test status",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Errors only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset without running",
    )

    args = parser.parse_args()

    try:
        # Load config
        config = load_config(args.config)

        # Apply CLI overrides
        if args.adapter:
            config.adapter = args.adapter
        if args.endpoint:
            config.endpoint = args.endpoint
        if args.concurrency:
            config.concurrency = args.concurrency
        if args.timeout:
            config.http.timeout = args.timeout
        if args.slow_threshold:
            config.http.slow_threshold = args.slow_threshold
        if args.fail_under:
            config.thresholds.composite = args.fail_under
        if args.fail_under_faithfulness:
            config.thresholds.faithfulness = args.fail_under_faithfulness
        if args.fail_under_answer_relevance:
            config.thresholds.answer_relevance = args.fail_under_answer_relevance
        if args.headers:
            for h in args.headers:
                if ":" in h:
                    key, value = h.split(":", 1)
                    config.http.headers[key.strip()] = value.strip()

        config.verbose = args.verbose
        config.quiet = args.quiet
        config.dry_run = args.dry_run

        # Validate config
        warnings = validate_config(config)
        for w in warnings:
            print(f"Warning: {w}", file=sys.stderr)

        # Load dataset
        dataset = load_dataset(args.dataset)
        test_count = len(dataset["test_cases"])

        if not config.quiet:
            print(f"Loaded {test_count} test cases from {args.dataset}")

        # Dry run - just validate
        if config.dry_run:
            print("Dry run complete. Config and dataset are valid.")
            return EXIT_SUCCESS

        # Create adapter
        adapter = create_adapter(config)

        # Health check
        if not adapter.health_check():
            print(f"Warning: RAG endpoint health check failed", file=sys.stderr)

        # Run evaluation
        start_time = time.time()
        results, estimated_cost, actual_cost = run_evaluation(config, dataset, adapter)
        duration = time.time() - start_time

        # Generate reports
        report_gen = ReportGenerator(config.output.directory)

        thresholds = {
            "composite": config.thresholds.composite,
            "faithfulness": config.thresholds.faithfulness,
            "answer_relevance": config.thresholds.answer_relevance,
            "context_precision": config.thresholds.context_precision,
            "context_recall": config.thresholds.context_recall,
        }

        summary = report_gen.generate_summary(
            results=results,
            adapter_name=adapter.name,
            endpoint=config.endpoint,
            duration_seconds=duration,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            thresholds=thresholds,
            slow_threshold_ms=config.http.slow_threshold * 1000,
        )

        report = EvaluationReport(
            summary=summary,
            results=results,
            thresholds=thresholds,
            weights=config.weights.normalized(),
        )

        # Write reports
        paths = report_gen.write_reports(report, config.output.formats)
        report_gen.append_to_log(summary)

        if not config.quiet:
            print(f"\nReports written to: {', '.join(paths)}")
            print(f"\nComposite Score: {summary.avg_composite:.2f}")
            print(f"Faithfulness: {summary.avg_faithfulness:.2f}")
            print(f"Passed: {summary.passed_count}/{summary.test_count}")

        # Determine exit code
        critical_failed = any(r.critical and not r.passed for r in results)
        if critical_failed:
            if not config.quiet:
                print("\nCRITICAL TEST FAILED", file=sys.stderr)
            return EXIT_CRITICAL_FAIL

        threshold_failed = not summary.composite_passed or not summary.faithfulness_passed
        if threshold_failed:
            if not config.quiet:
                print("\nTHRESHOLD NOT MET", file=sys.stderr)
            return EXIT_THRESHOLD_FAIL

        return EXIT_SUCCESS

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_FATAL_ERROR
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return EXIT_FATAL_ERROR
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_FATAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
