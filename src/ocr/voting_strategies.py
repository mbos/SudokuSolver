"""Voting strategies for combining results from multiple OCR models."""

from typing import List, Dict
from collections import Counter
from .base_recognizer import RecognitionResult


class VotingStrategy:
    """Base class for voting strategies."""

    def vote(self, results: List[RecognitionResult]) -> RecognitionResult:
        """
        Combine multiple recognition results into one.

        Args:
            results: List of RecognitionResult from different models

        Returns:
            Combined RecognitionResult
        """
        raise NotImplementedError


class MajorityVoting(VotingStrategy):
    """Simple majority voting: digit that appears most often wins."""

    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize majority voting.

        Args:
            min_confidence: Minimum confidence to consider a result valid
        """
        self.min_confidence = min_confidence

    def vote(self, results: List[RecognitionResult]) -> RecognitionResult:
        """
        Use majority voting to combine results.

        Args:
            results: List of RecognitionResult from different models

        Returns:
            Combined RecognitionResult
        """
        if not results:
            return RecognitionResult(
                digit=0,
                confidence=0.0,
                model_name="Ensemble-Majority",
                processing_time_ms=0.0
            )

        # Filter out low-confidence results
        valid_results = [r for r in results if r.confidence >= self.min_confidence]

        if not valid_results:
            # No confident results, return the best we have
            best = max(results, key=lambda r: r.confidence)
            return RecognitionResult(
                digit=best.digit,
                confidence=best.confidence,
                model_name="Ensemble-Majority(fallback)",
                processing_time_ms=sum(r.processing_time_ms for r in results)
            )

        # Count votes (excluding 0s which mean "uncertain")
        non_zero_results = [r for r in valid_results if r.digit != 0]

        if not non_zero_results:
            # All models returned 0 (empty/uncertain)
            return RecognitionResult(
                digit=0,
                confidence=1.0,
                model_name="Ensemble-Majority",
                processing_time_ms=sum(r.processing_time_ms for r in results)
            )

        # Count occurrences of each digit
        digit_counts = Counter(r.digit for r in non_zero_results)
        most_common_digit, count = digit_counts.most_common(1)[0]

        # Calculate average confidence for the winning digit
        winning_results = [r for r in non_zero_results if r.digit == most_common_digit]
        avg_confidence = sum(r.confidence for r in winning_results) / len(winning_results)

        # Boost confidence if multiple models agree
        agreement_boost = min(0.2 * (count - 1), 0.3)
        final_confidence = min(1.0, avg_confidence + agreement_boost)

        return RecognitionResult(
            digit=most_common_digit,
            confidence=final_confidence,
            model_name="Ensemble-Majority",
            processing_time_ms=sum(r.processing_time_ms for r in results)
        )


class WeightedVoting(VotingStrategy):
    """Weighted voting: models with higher weights have more influence."""

    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize weighted voting.

        Args:
            min_confidence: Minimum confidence to consider a result valid
        """
        self.min_confidence = min_confidence

    def vote(self, results: List[RecognitionResult]) -> RecognitionResult:
        """
        Use weighted voting to combine results.

        Args:
            results: List of RecognitionResult from different models (with weights)

        Returns:
            Combined RecognitionResult
        """
        if not results:
            return RecognitionResult(
                digit=0,
                confidence=0.0,
                model_name="Ensemble-Weighted",
                processing_time_ms=0.0
            )

        # Filter valid results
        valid_results = [r for r in results if r.confidence >= self.min_confidence]

        if not valid_results:
            # Return best available
            best = max(results, key=lambda r: r.confidence)
            return RecognitionResult(
                digit=best.digit,
                confidence=best.confidence,
                model_name="Ensemble-Weighted(fallback)",
                processing_time_ms=sum(r.processing_time_ms for r in results)
            )

        # Calculate weighted scores per digit
        # IMPORTANT: Include digit=0 as a vote for "empty/no digit"
        # This allows EasyOCR's high-confidence "no result" to veto false positives
        digit_scores: Dict[int, float] = {}

        for result in valid_results:
            if result.digit not in digit_scores:
                digit_scores[result.digit] = 0.0
            # Score = confidence * weight
            # EasyOCR (0.8 * 2.0 = 1.6) beats Tesseract false positive (0.7 * 1.0 = 0.7)
            digit_scores[result.digit] += result.confidence * result.weight

        # Find digit with highest score
        winning_digit = max(digit_scores, key=digit_scores.get)
        winning_score = digit_scores[winning_digit]

        # Normalize confidence
        total_score = sum(digit_scores.values())
        final_confidence = winning_score / total_score if total_score > 0 else 0.0

        return RecognitionResult(
            digit=winning_digit,
            confidence=final_confidence,
            model_name="Ensemble-Weighted",
            processing_time_ms=sum(r.processing_time_ms for r in results)
        )


class ConfidenceAggregation(VotingStrategy):
    """
    Confidence-based aggregation with adaptive thresholds.

    Accepts results based on agreement level and confidence.
    """

    def __init__(self,
                 all_agree_threshold: float = 0.5,
                 majority_threshold: float = 0.7,
                 single_threshold: float = 0.9):
        """
        Initialize confidence aggregation.

        Args:
            all_agree_threshold: Threshold when all models agree
            majority_threshold: Threshold when 2+ models agree
            single_threshold: Threshold for single model
        """
        self.all_agree_threshold = all_agree_threshold
        self.majority_threshold = majority_threshold
        self.single_threshold = single_threshold

    def vote(self, results: List[RecognitionResult]) -> RecognitionResult:
        """
        Use confidence aggregation to combine results.

        Args:
            results: List of RecognitionResult from different models

        Returns:
            Combined RecognitionResult
        """
        if not results:
            return RecognitionResult(
                digit=0,
                confidence=0.0,
                model_name="Ensemble-Confidence",
                processing_time_ms=0.0
            )

        # Filter non-zero results
        non_zero_results = [r for r in results if r.digit != 0]

        if not non_zero_results:
            return RecognitionResult(
                digit=0,
                confidence=1.0,
                model_name="Ensemble-Confidence",
                processing_time_ms=sum(r.processing_time_ms for r in results)
            )

        # Group by digit
        digit_groups: Dict[int, List[RecognitionResult]] = {}
        for result in non_zero_results:
            if result.digit not in digit_groups:
                digit_groups[result.digit] = []
            digit_groups[result.digit].append(result)

        # Find best digit based on agreement and confidence
        best_digit = 0
        best_confidence = 0.0

        for digit, group in digit_groups.items():
            avg_confidence = sum(r.confidence for r in group) / len(group)
            num_votes = len(group)

            # Determine threshold based on agreement
            if num_votes >= len(results):  # All agree
                threshold = self.all_agree_threshold
            elif num_votes >= 2:  # Majority
                threshold = self.majority_threshold
            else:  # Single model
                threshold = self.single_threshold

            # Check if this digit meets the threshold
            if avg_confidence >= threshold:
                # Boost confidence for agreement
                boosted_confidence = min(1.0, avg_confidence + 0.1 * (num_votes - 1))
                if boosted_confidence > best_confidence:
                    best_digit = digit
                    best_confidence = boosted_confidence

        return RecognitionResult(
            digit=best_digit,
            confidence=best_confidence,
            model_name="Ensemble-Confidence",
            processing_time_ms=sum(r.processing_time_ms for r in results)
        )


def get_voting_strategy(strategy_name: str, **kwargs) -> VotingStrategy:
    """
    Factory function to get a voting strategy by name.

    Args:
        strategy_name: Name of strategy ('majority', 'weighted', 'confidence')
        **kwargs: Additional arguments for the strategy

    Returns:
        VotingStrategy instance
    """
    strategies = {
        'majority': MajorityVoting,
        'weighted': WeightedVoting,
        'confidence': ConfidenceAggregation,
    }

    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown voting strategy: {strategy_name}. "
                        f"Available: {list(strategies.keys())}")

    return strategy_class(**kwargs)
