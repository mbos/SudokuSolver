"""Unit tests for voting strategies."""

import unittest
from src.ocr.base_recognizer import RecognitionResult
from src.ocr.voting_strategies import (
    MajorityVoting,
    WeightedVoting,
    ConfidenceAggregation,
    get_voting_strategy
)


class TestMajorityVoting(unittest.TestCase):
    """Test cases for MajorityVoting strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MajorityVoting(min_confidence=0.5)

    def test_all_agree(self):
        """Test when all models agree on the same digit."""
        results = [
            RecognitionResult(digit=5, confidence=0.8, model_name="Model1"),
            RecognitionResult(digit=5, confidence=0.9, model_name="Model2"),
            RecognitionResult(digit=5, confidence=0.7, model_name="Model3"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 5)
        self.assertGreater(result.confidence, 0.8)  # Should be boosted

    def test_majority_wins(self):
        """Test when majority of models agree."""
        results = [
            RecognitionResult(digit=5, confidence=0.8, model_name="Model1"),
            RecognitionResult(digit=5, confidence=0.9, model_name="Model2"),
            RecognitionResult(digit=3, confidence=0.7, model_name="Model3"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 5)

    def test_tie_highest_confidence_wins(self):
        """Test tie-breaking by confidence."""
        results = [
            RecognitionResult(digit=5, confidence=0.7, model_name="Model1"),
            RecognitionResult(digit=3, confidence=0.9, model_name="Model2"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 3)  # Higher confidence

    def test_all_zero(self):
        """Test when all models return 0 (empty/uncertain)."""
        results = [
            RecognitionResult(digit=0, confidence=0.9, model_name="Model1"),
            RecognitionResult(digit=0, confidence=0.8, model_name="Model2"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 0)

    def test_low_confidence_filtered(self):
        """Test that low confidence results are filtered out."""
        results = [
            RecognitionResult(digit=5, confidence=0.3, model_name="Model1"),  # Below threshold
            RecognitionResult(digit=7, confidence=0.8, model_name="Model2"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 7)

    def test_empty_results(self):
        """Test with empty results list."""
        results = []
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 0)
        self.assertEqual(result.confidence, 0.0)


class TestWeightedVoting(unittest.TestCase):
    """Test cases for WeightedVoting strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = WeightedVoting(min_confidence=0.5)

    def test_weighted_preference(self):
        """Test that higher confidence scores win."""
        results = [
            RecognitionResult(digit=5, confidence=0.6, model_name="Model1"),
            RecognitionResult(digit=7, confidence=0.9, model_name="Model2"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 7)

    def test_accumulated_score(self):
        """Test accumulated scores for same digit."""
        results = [
            RecognitionResult(digit=5, confidence=0.6, model_name="Model1"),
            RecognitionResult(digit=5, confidence=0.7, model_name="Model2"),
            RecognitionResult(digit=7, confidence=0.9, model_name="Model3"),
        ]
        result = self.strategy.vote(results)
        # Two models voting 5 with combined 1.3 should beat single 0.9 for 7
        self.assertEqual(result.digit, 5)


class TestConfidenceAggregation(unittest.TestCase):
    """Test cases for ConfidenceAggregation strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = ConfidenceAggregation(
            all_agree_threshold=0.5,
            majority_threshold=0.7,
            single_threshold=0.9
        )

    def test_all_agree_low_threshold(self):
        """Test when all models agree, lower threshold applies."""
        results = [
            RecognitionResult(digit=5, confidence=0.6, model_name="Model1"),
            RecognitionResult(digit=5, confidence=0.6, model_name="Model2"),
            RecognitionResult(digit=5, confidence=0.6, model_name="Model3"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 5)  # Passes 0.5 threshold

    def test_majority_medium_threshold(self):
        """Test when majority agree, medium threshold applies."""
        results = [
            RecognitionResult(digit=5, confidence=0.75, model_name="Model1"),
            RecognitionResult(digit=5, confidence=0.75, model_name="Model2"),
            RecognitionResult(digit=7, confidence=0.9, model_name="Model3"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 5)  # Passes 0.7 threshold

    def test_single_high_threshold(self):
        """Test when single model, high threshold required."""
        results = [
            RecognitionResult(digit=5, confidence=0.95, model_name="Model1"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 5)  # Passes 0.9 threshold

    def test_single_fails_threshold(self):
        """Test when single model doesn't meet threshold."""
        results = [
            RecognitionResult(digit=5, confidence=0.85, model_name="Model1"),
        ]
        result = self.strategy.vote(results)
        self.assertEqual(result.digit, 0)  # Fails 0.9 threshold


class TestVotingStrategyFactory(unittest.TestCase):
    """Test cases for voting strategy factory."""

    def test_get_majority_strategy(self):
        """Test getting majority voting strategy."""
        strategy = get_voting_strategy('majority')
        self.assertIsInstance(strategy, MajorityVoting)

    def test_get_weighted_strategy(self):
        """Test getting weighted voting strategy."""
        strategy = get_voting_strategy('weighted')
        self.assertIsInstance(strategy, WeightedVoting)

    def test_get_confidence_strategy(self):
        """Test getting confidence aggregation strategy."""
        strategy = get_voting_strategy('confidence')
        self.assertIsInstance(strategy, ConfidenceAggregation)

    def test_invalid_strategy(self):
        """Test that invalid strategy name raises error."""
        with self.assertRaises(ValueError):
            get_voting_strategy('invalid')


if __name__ == '__main__':
    unittest.main()
