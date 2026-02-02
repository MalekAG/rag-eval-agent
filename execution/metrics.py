"""
Metric calculations for RAG evaluation.

Implements RAGAS-based metrics using LLM-as-judge with 3-pass majority voting.
"""

import logging
import os
import statistics
import threading
from dataclasses import dataclass
from typing import Optional

import anthropic
from sentence_transformers import SentenceTransformer

# Judge model - hardcoded as per spec
JUDGE_MODEL = "claude-sonnet-4-20250514"


@dataclass
class MetricScores:
    """Scores for a single test case."""
    faithfulness: float
    answer_relevance: float
    context_precision: Optional[float]
    context_recall: Optional[float]
    semantic_similarity: float
    judge_reasoning: str
    unsupported_claims: list[str]


@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float:
        """
        Estimate cost in USD (Claude Sonnet 4 pricing as of 2025-01).

        Note: Verify current pricing at https://anthropic.com/pricing
        """
        input_cost = (self.input_tokens / 1_000_000) * 3.0  # $3 per 1M input tokens
        output_cost = (self.output_tokens / 1_000_000) * 15.0  # $15 per 1M output tokens
        return input_cost + output_cost

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class MetricsCalculator:
    """
    Calculate RAG evaluation metrics using LLM-as-judge.

    Uses 3-pass majority voting for consistency.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
    ):
        self.client = anthropic.Anthropic()
        self.similarity_threshold = similarity_threshold
        self.total_usage = TokenUsage()
        self._usage_lock = threading.Lock()  # Thread-safe token accumulation

        # Lazy load embedding model
        self._embedding_model_name = embedding_model
        self._embedding_model = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        """Thread-safe accumulation of token usage."""
        with self._usage_lock:
            self.total_usage = self.total_usage + usage

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        embeddings = self.embedding_model.encode([text1, text2])
        # Cosine similarity
        similarity = float(
            embeddings[0] @ embeddings[1]
            / (
                (embeddings[0] @ embeddings[0]) ** 0.5
                * (embeddings[1] @ embeddings[1]) ** 0.5
            )
        )
        return max(0.0, min(1.0, similarity))

    def _call_judge(self, prompt: str) -> tuple[str, TokenUsage]:
        """Make a single judge call and return response with token usage."""
        response = self.client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.content[0].text, usage

    def _majority_vote_score(self, prompt: str, passes: int = 3) -> tuple[float, str, TokenUsage]:
        """
        Run multiple judge passes and return majority-voted score.

        Returns:
            Tuple of (score, reasoning, total_usage)
        """
        scores = []
        reasonings = []
        total_usage = TokenUsage()

        for _ in range(passes):
            response, usage = self._call_judge(prompt)
            total_usage = total_usage + usage

            # Parse score from response
            score = self._parse_score(response)
            scores.append(score)
            reasonings.append(response)

        # Take median score
        final_score = statistics.median(scores)

        # Use reasoning from the pass closest to median
        closest_idx = min(range(len(scores)), key=lambda i: abs(scores[i] - final_score))
        final_reasoning = reasonings[closest_idx]

        return final_score, final_reasoning, total_usage

    def _parse_score(self, response: str) -> float:
        """Extract score from judge response."""
        import re

        # Look for patterns like "Score: 0.85" or "8.5/10" or just a decimal
        patterns = [
            r"[Ss]core:\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*/\s*10",
            r"(\d+\.?\d*)\s*/\s*1(?:\.0)?",
            r"Rating:\s*(\d+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                value = float(match.group(1))
                # Normalize to 0-1 scale
                if value > 1:
                    value = value / 10
                return max(0.0, min(1.0, value))

        # Default to 0.5 if no score found
        logger = logging.getLogger(__name__)
        logger.warning(
            "Could not parse score from judge response, defaulting to 0.5. "
            f"Response snippet: {response[:100]}..."
        )
        return 0.5

    def _parse_claims(self, response: str) -> list[str]:
        """Extract unsupported claims from judge response."""
        import re

        claims = []
        # Look for bullet points or numbered lists under "unsupported" section
        unsupported_section = re.search(
            r"[Uu]nsupported[^:]*:(.*?)(?:\n\n|\n[A-Z]|$)",
            response,
            re.DOTALL,
        )
        if unsupported_section:
            text = unsupported_section.group(1)
            # Extract bullet points
            bullets = re.findall(r"[-*]\s*(.+)", text)
            claims.extend(bullets)
            # Extract numbered items
            numbered = re.findall(r"\d+\.\s*(.+)", text)
            claims.extend(numbered)

        return claims

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Optional[list[str]] = None,
        expected_contexts: Optional[list[str]] = None,
    ) -> MetricScores:
        """
        Evaluate a single RAG response.

        Args:
            question: The original question
            answer: The RAG system's answer
            ground_truth: The expected correct answer
            contexts: Retrieved context documents (None if unavailable)
            expected_contexts: Expected document names/IDs for retrieval quality

        Returns:
            MetricScores with all calculated metrics
        """
        # Calculate semantic similarity (always available)
        semantic_sim = self.calculate_semantic_similarity(answer, ground_truth)

        # Handle empty context case
        if contexts is None or len(contexts) == 0:
            # Auto-fail faithfulness if no context
            return MetricScores(
                faithfulness=0.0,
                answer_relevance=self._evaluate_answer_relevance(question, answer),
                context_precision=None,
                context_recall=None,
                semantic_similarity=semantic_sim,
                judge_reasoning="Faithfulness auto-failed: no context retrieved",
                unsupported_claims=["Entire answer is unsupported (no context)"],
            )

        context_text = "\n\n---\n\n".join(contexts)

        # Evaluate faithfulness with majority voting
        faithfulness_prompt = f"""Evaluate whether the following answer is faithful to (grounded in) the provided context.
The answer should not contain information that cannot be inferred from the context.

Question: {question}

Context:
{context_text}

Answer: {answer}

Ground Truth (for reference): {ground_truth}

Evaluate faithfulness on a scale of 0 to 1:
- 1.0: Answer is completely grounded in context, no hallucinations
- 0.5: Answer is partially grounded, some claims unsupported
- 0.0: Answer contradicts context or makes unsupported claims

Provide your response in this format:
Score: [0-1]
Reasoning: [Your explanation]
Unsupported claims: [List any claims not supported by context]"""

        faithfulness, faith_reasoning, faith_usage = self._majority_vote_score(faithfulness_prompt)
        self._accumulate_usage(faith_usage)

        # Extract unsupported claims
        unsupported_claims = self._parse_claims(faith_reasoning)

        # Evaluate answer relevance
        answer_relevance = self._evaluate_answer_relevance(question, answer)

        # Evaluate context metrics if contexts available
        context_precision = self._evaluate_context_precision(question, contexts)
        context_recall = self._evaluate_context_recall(question, ground_truth, contexts)

        return MetricScores(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            semantic_similarity=semantic_sim,
            judge_reasoning=faith_reasoning,
            unsupported_claims=unsupported_claims,
        )

    def _evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """Evaluate if the answer addresses the question."""
        prompt = f"""Evaluate whether the following answer addresses the question asked.

Question: {question}

Answer: {answer}

Evaluate relevance on a scale of 0 to 1:
- 1.0: Answer directly and completely addresses the question
- 0.5: Answer partially addresses the question
- 0.0: Answer does not address the question at all

Provide your response in this format:
Score: [0-1]
Reasoning: [Your explanation]"""

        score, _, usage = self._majority_vote_score(prompt)
        self._accumulate_usage(usage)
        return score

    def _evaluate_context_precision(
        self, question: str, contexts: list[str]
    ) -> float:
        """Evaluate if retrieved contexts are relevant to the query."""
        context_text = "\n\n---\n\n".join(contexts)

        prompt = f"""Evaluate whether the retrieved context documents are relevant to answering the question.

Question: {question}

Retrieved Contexts:
{context_text}

Evaluate context precision on a scale of 0 to 1:
- 1.0: All retrieved contexts are highly relevant to the question
- 0.5: Some contexts are relevant, others are not
- 0.0: None of the contexts are relevant

Provide your response in this format:
Score: [0-1]
Reasoning: [Your explanation]"""

        score, _, usage = self._majority_vote_score(prompt)
        self._accumulate_usage(usage)
        return score

    def _evaluate_context_recall(
        self, question: str, ground_truth: str, contexts: list[str]
    ) -> float:
        """Evaluate if we retrieved all information needed to answer correctly."""
        context_text = "\n\n---\n\n".join(contexts)

        prompt = f"""Evaluate whether the retrieved contexts contain all the information needed to produce the correct answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Contexts:
{context_text}

Evaluate context recall on a scale of 0 to 1:
- 1.0: Contexts contain all information needed for the ground truth answer
- 0.5: Contexts contain some but not all needed information
- 0.0: Contexts are missing critical information for the answer

Provide your response in this format:
Score: [0-1]
Reasoning: [Your explanation]"""

        score, _, usage = self._majority_vote_score(prompt)
        self._accumulate_usage(usage)
        return score

    def estimate_cost(self, num_test_cases: int) -> float:
        """
        Estimate cost for running evaluation.

        Assumes ~2000 tokens per test case (input) and ~500 tokens output.
        Each test case runs 4 metrics x 3 passes = 12 judge calls.
        """
        input_tokens_per_case = 2000 * 12
        output_tokens_per_case = 500 * 12

        total_input = num_test_cases * input_tokens_per_case
        total_output = num_test_cases * output_tokens_per_case

        estimated_usage = TokenUsage(input_tokens=total_input, output_tokens=total_output)
        return estimated_usage.estimated_cost


def calculate_composite_score(
    scores: MetricScores,
    weights: dict[str, float],
) -> float:
    """
    Calculate weighted composite score.

    Args:
        scores: Individual metric scores
        weights: Normalized weights (should sum to 1.0)

    Returns:
        Weighted average score
    """
    weighted_sum = 0.0
    total_weight = 0.0

    # Faithfulness
    weighted_sum += scores.faithfulness * weights["faithfulness"]
    total_weight += weights["faithfulness"]

    # Answer relevance
    weighted_sum += scores.answer_relevance * weights["answer_relevance"]
    total_weight += weights["answer_relevance"]

    # Context metrics (only if available)
    if scores.context_precision is not None:
        weighted_sum += scores.context_precision * weights["context_precision"]
        total_weight += weights["context_precision"]

    if scores.context_recall is not None:
        weighted_sum += scores.context_recall * weights["context_recall"]
        total_weight += weights["context_recall"]

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight
