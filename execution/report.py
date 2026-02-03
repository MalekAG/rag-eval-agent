"""
Report generation for RAG evaluation.

Produces both Markdown (human-readable) and JSON (machine-readable) reports.
"""

import json
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Allowed directories for output (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ALLOWED_OUTPUT_DIRS = [
    PROJECT_ROOT / "results",
    PROJECT_ROOT / ".tmp",
]


def validate_output_path(dir_path: str) -> Path:
    """
    Validate that an output directory path is within allowed directories.

    Args:
        dir_path: The directory path to validate

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is outside allowed directories
    """
    path = Path(dir_path).resolve()

    # Check if path is within any allowed directory
    for allowed_dir in ALLOWED_OUTPUT_DIRS:
        try:
            path.relative_to(allowed_dir)
            return path
        except ValueError:
            continue

    # Also allow the exact allowed directories themselves
    if path in ALLOWED_OUTPUT_DIRS:
        return path

    # Path traversal attempt or path outside allowed dirs
    allowed_str = ", ".join(str(d) for d in ALLOWED_OUTPUT_DIRS)
    raise ValueError(
        f"Invalid output path: {dir_path}. "
        f"Path must be within allowed directories: {allowed_str}"
    )


@dataclass
class TestResult:
    """Result for a single test case."""
    id: str
    question: str
    ground_truth: str
    answer: str
    contexts: Optional[list[str]]
    faithfulness: float
    answer_relevance: float
    context_precision: Optional[float]
    context_recall: Optional[float]
    semantic_similarity: float
    composite_score: float
    latency_ms: float
    passed: bool
    critical: bool
    tags: list[str]
    judge_reasoning: str
    unsupported_claims: list[str]
    error: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary statistics for the entire evaluation run."""
    timestamp: str
    adapter: str
    endpoint: Optional[str]
    test_count: int
    passed_count: int
    failed_count: int
    error_count: int
    critical_count: int
    critical_passed: int
    duration_seconds: float
    estimated_cost: float
    actual_cost: float

    # Aggregate scores
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_precision: Optional[float]
    avg_context_recall: Optional[float]
    avg_composite: float

    # Latency stats
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    slow_count: int

    # Threshold checks
    composite_threshold: Optional[float]
    composite_passed: bool
    faithfulness_threshold: Optional[float]
    faithfulness_passed: bool


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    summary: EvaluationSummary
    results: list[TestResult]
    thresholds: dict[str, Optional[float]]
    weights: dict[str, float]


class ReportGenerator:
    """Generate evaluation reports in multiple formats."""

    def __init__(self, output_dir: str = "results"):
        # Validate output path is within allowed directories (prevents path traversal)
        self.output_dir = validate_output_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(
        self,
        results: list[TestResult],
        adapter_name: str,
        endpoint: Optional[str],
        duration_seconds: float,
        estimated_cost: float,
        actual_cost: float,
        thresholds: dict[str, Optional[float]],
        slow_threshold_ms: float = 5000,
    ) -> EvaluationSummary:
        """Generate summary statistics from test results."""
        valid_results = [r for r in results if r.error is None]

        # Basic counts
        test_count = len(results)
        passed_count = sum(1 for r in results if r.passed and r.error is None)
        failed_count = sum(1 for r in results if not r.passed and r.error is None)
        error_count = sum(1 for r in results if r.error is not None)
        critical_count = sum(1 for r in results if r.critical)
        critical_passed = sum(1 for r in results if r.critical and r.passed)

        # Latency stats
        latencies = [r.latency_ms for r in valid_results]
        avg_latency = statistics.mean(latencies) if latencies else 0
        p50_latency = statistics.median(latencies) if latencies else 0
        p95_latency = (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0
        )
        slow_count = sum(1 for l in latencies if l > slow_threshold_ms)

        # Score aggregates
        avg_faithfulness = statistics.mean([r.faithfulness for r in valid_results]) if valid_results else 0
        avg_answer_relevance = statistics.mean([r.answer_relevance for r in valid_results]) if valid_results else 0

        context_precisions = [r.context_precision for r in valid_results if r.context_precision is not None]
        avg_context_precision = statistics.mean(context_precisions) if context_precisions else None

        context_recalls = [r.context_recall for r in valid_results if r.context_recall is not None]
        avg_context_recall = statistics.mean(context_recalls) if context_recalls else None

        avg_composite = statistics.mean([r.composite_score for r in valid_results]) if valid_results else 0

        # Threshold checks
        composite_threshold = thresholds.get("composite")
        composite_passed = composite_threshold is None or avg_composite >= composite_threshold

        faithfulness_threshold = thresholds.get("faithfulness")
        faithfulness_passed = faithfulness_threshold is None or avg_faithfulness >= faithfulness_threshold

        return EvaluationSummary(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            adapter=adapter_name,
            endpoint=endpoint,
            test_count=test_count,
            passed_count=passed_count,
            failed_count=failed_count,
            error_count=error_count,
            critical_count=critical_count,
            critical_passed=critical_passed,
            duration_seconds=duration_seconds,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevance=avg_answer_relevance,
            avg_context_precision=avg_context_precision,
            avg_context_recall=avg_context_recall,
            avg_composite=avg_composite,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            slow_count=slow_count,
            composite_threshold=composite_threshold,
            composite_passed=composite_passed,
            faithfulness_threshold=faithfulness_threshold,
            faithfulness_passed=faithfulness_passed,
        )

    def generate_markdown(self, report: EvaluationReport) -> str:
        """Generate markdown report."""
        s = report.summary
        lines = [
            "# RAG Evaluation Report",
            "",
            f"**Date:** {s.timestamp}",
            f"**Adapter:** {s.adapter}",
        ]

        if s.endpoint:
            lines.append(f"**Endpoint:** {s.endpoint}")

        lines.extend([
            f"**Test Cases:** {s.test_count} ({s.critical_count} critical)",
            f"**Duration:** {self._format_duration(s.duration_seconds)}",
            f"**Estimated Cost:** ${s.estimated_cost:.2f} | **Actual Cost:** ${s.actual_cost:.2f}",
            "",
            "## Summary",
            "",
            "| Metric | Score | Threshold | Status |",
            "|--------|-------|-----------|--------|",
        ])

        # Composite score
        composite_status = "PASS" if s.composite_passed else "FAIL"
        composite_thresh = f"{s.composite_threshold:.2f}" if s.composite_threshold else "-"
        lines.append(f"| **Composite** | {s.avg_composite:.2f} | {composite_thresh} | {composite_status} |")

        # Faithfulness
        faith_status = "PASS" if s.faithfulness_passed else "FAIL"
        faith_thresh = f"{s.faithfulness_threshold:.2f}" if s.faithfulness_threshold else "-"
        lines.append(f"| Faithfulness | {s.avg_faithfulness:.2f} | {faith_thresh} | {faith_status} |")

        # Answer relevance
        ar_thresh = report.thresholds.get("answer_relevance")
        ar_status = "-" if ar_thresh is None else ("PASS" if s.avg_answer_relevance >= ar_thresh else "FAIL")
        ar_thresh_str = f"{ar_thresh:.2f}" if ar_thresh else "-"
        lines.append(f"| Answer Relevance | {s.avg_answer_relevance:.2f} | {ar_thresh_str} | {ar_status} |")

        # Context precision
        if s.avg_context_precision is not None:
            cp_thresh = report.thresholds.get("context_precision")
            cp_thresh_str = f"{cp_thresh:.2f}" if cp_thresh else "-"
            lines.append(f"| Context Precision | {s.avg_context_precision:.2f} | {cp_thresh_str} | - |")

        # Context recall
        if s.avg_context_recall is not None:
            cr_thresh = report.thresholds.get("context_recall")
            cr_thresh_str = f"{cr_thresh:.2f}" if cr_thresh else "-"
            lines.append(f"| Context Recall | {s.avg_context_recall:.2f} | {cr_thresh_str} | - |")

        lines.extend([
            "",
            f"**Latency:** avg {s.avg_latency_ms:.0f}ms | p50 {s.p50_latency_ms:.0f}ms | p95 {s.p95_latency_ms:.0f}ms | slow (>5s): {s.slow_count}",
            "",
        ])

        # Critical tests
        if s.critical_count > 0:
            crit_status = "PASSED" if s.critical_passed == s.critical_count else "FAILED"
            lines.append(f"## Critical Tests: {s.critical_passed}/{s.critical_count} {crit_status}")
            lines.append("")

        # Failed tests detail
        failed_results = [r for r in report.results if not r.passed or r.error]
        if failed_results:
            lines.append(f"## Issues Found: {len(failed_results)}")
            lines.append("")

            for r in failed_results:
                status = "ERROR" if r.error else "FAILED"
                lines.extend([
                    f"### {status}: {r.id} - {r.question[:50]}{'...' if len(r.question) > 50 else ''}",
                    "",
                    f"**Question:** {r.question}",
                    "",
                ])

                if r.error:
                    lines.extend([
                        f"**Error:** {r.error}",
                        "",
                    ])
                else:
                    lines.append("**Retrieved Contexts:**")
                    if r.contexts:
                        for i, ctx in enumerate(r.contexts[:3], 1):
                            ctx_preview = ctx[:200] + "..." if len(ctx) > 200 else ctx
                            lines.append(f"{i}. \"{ctx_preview}\"")
                    else:
                        lines.append("(none)")
                    lines.append("")

                    lines.extend([
                        f"**Generated Answer:** \"{r.answer}\"",
                        "",
                        f"**Ground Truth:** \"{r.ground_truth}\"",
                        "",
                        "**Scores:**",
                        f"- Faithfulness: {r.faithfulness:.2f}",
                        f"- Answer Relevance: {r.answer_relevance:.2f}",
                    ])

                    if r.context_precision is not None:
                        lines.append(f"- Context Precision: {r.context_precision:.2f}")
                    if r.context_recall is not None:
                        lines.append(f"- Context Recall: {r.context_recall:.2f}")

                    lines.extend([
                        "",
                        "**Judge Reasoning:**",
                        r.judge_reasoning[:500] + "..." if len(r.judge_reasoning) > 500 else r.judge_reasoning,
                        "",
                    ])

                    if r.unsupported_claims:
                        lines.append("**Unsupported Claims:**")
                        for claim in r.unsupported_claims[:5]:
                            lines.append(f"- \"{claim}\"")
                        lines.append("")

                lines.append("---")
                lines.append("")

        # Results by tag
        all_tags = set()
        for r in report.results:
            all_tags.update(r.tags)

        if all_tags:
            lines.extend([
                "## Test Results by Tag",
                "",
                "| Tag | Count | Avg Score |",
                "|-----|-------|-----------|",
            ])

            for tag in sorted(all_tags):
                tagged_results = [r for r in report.results if tag in r.tags and r.error is None]
                if tagged_results:
                    avg_score = statistics.mean([r.composite_score for r in tagged_results])
                    lines.append(f"| {tag} | {len(tagged_results)} | {avg_score:.2f} |")

            lines.append("")

        return "\n".join(lines)

    def generate_json(self, report: EvaluationReport) -> str:
        """Generate JSON report."""
        return json.dumps(asdict(report), indent=2, default=str)

    def write_reports(
        self,
        report: EvaluationReport,
        formats: list[str],
        base_name: str = "eval_report",
    ) -> list[str]:
        """Write reports to files and return paths."""
        paths = []

        if "markdown" in formats:
            md_path = self.output_dir / f"{base_name}.md"
            md_content = self.generate_markdown(report)
            md_path.write_text(md_content, encoding="utf-8")
            paths.append(str(md_path))

        if "json" in formats:
            json_path = self.output_dir / f"{base_name}.json"
            json_content = self.generate_json(report)
            json_path.write_text(json_content, encoding="utf-8")
            paths.append(str(json_path))

        return paths

    def append_to_log(self, summary: EvaluationSummary) -> str:
        """Append summary to JSONL log file."""
        log_path = self.output_dir / "results.jsonl"
        log_entry = json.dumps(asdict(summary), default=str)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        return str(log_path)

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
