"""Integration tests for ensemble OCR recognizer."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
from src.ocr.base_recognizer import RecognitionResult
from src.ocr.ensemble_recognizer import EnsembleRecognizer


class TestEnsembleRecognizer(unittest.TestCase):
    """Test cases for EnsembleRecognizer."""

    def test_initialization_default_config(self):
        """Test ensemble initializes with default config."""
        ensemble = EnsembleRecognizer()
        self.assertIsNotNone(ensemble.config)
        self.assertIn('models', ensemble.config)
        self.assertIn('thresholds', ensemble.config)

    def test_initialization_custom_voting_strategy(self):
        """Test ensemble with different voting strategies."""
        for strategy in ['majority', 'weighted', 'confidence']:
            ensemble = EnsembleRecognizer(voting_strategy=strategy)
            self.assertIsNotNone(ensemble.voting_strategy)

    def test_level1_fast_path(self):
        """Test that Level 1 accepts high-confidence results."""
        config = {
            'models': {
                'cnn': {'enabled': False},
                'tesseract': {'enabled': False},
                'easyocr': {'enabled': False}
            },
            'thresholds': {
                'level1_confidence': 0.75,
                'level2_confidence': 0.65,
                'min_confidence': 0.5
            }
        }

        ensemble = EnsembleRecognizer(config=config)

        # Mock a simple preprocessor
        mock_recognizer = Mock()
        mock_recognizer.name = "TestRecognizer"
        mock_recognizer.preprocess_cell.return_value = (np.zeros((28, 28)), False)
        mock_recognizer.recognize.return_value = RecognitionResult(
            digit=5,
            confidence=0.85,
            model_name="TestRecognizer"
        )

        ensemble.recognizers = [mock_recognizer]

        # Create a dummy cell
        cell = np.zeros((50, 50), dtype=np.uint8)
        result = ensemble.recognize_digit(cell)

        # Should accept at Level 1
        self.assertEqual(result, 5)
        self.assertEqual(ensemble.stats['level1_success'], 1)

    def test_empty_cell_detection(self):
        """Test that empty cells are properly detected."""
        config = {
            'models': {
                'cnn': {'enabled': False},
                'tesseract': {'enabled': False},
                'easyocr': {'enabled': False}
            },
            'thresholds': {
                'level1_confidence': 0.75,
                'level2_confidence': 0.65,
                'min_confidence': 0.5
            }
        }

        ensemble = EnsembleRecognizer(config=config)

        # Mock recognizer that returns empty
        mock_recognizer = Mock()
        mock_recognizer.name = "TestRecognizer"
        mock_recognizer.preprocess_cell.return_value = (np.zeros((28, 28)), True)  # is_empty = True

        ensemble.recognizers = [mock_recognizer]

        cell = np.zeros((50, 50), dtype=np.uint8)
        result = ensemble.recognize_digit(cell)

        self.assertEqual(result, 0)
        self.assertEqual(ensemble.stats['empty_cells'], 1)

    def test_recognize_grid_shape(self):
        """Test that recognize_grid returns correct shapes."""
        config = {
            'models': {
                'cnn': {'enabled': False},
                'tesseract': {'enabled': False},
                'easyocr': {'enabled': False}
            },
            'thresholds': {
                'level1_confidence': 0.75,
                'level2_confidence': 0.65,
                'min_confidence': 0.5
            }
        }

        ensemble = EnsembleRecognizer(config=config)

        # Mock recognizer
        mock_recognizer = Mock()
        mock_recognizer.name = "TestRecognizer"
        mock_recognizer.preprocess_cell.return_value = (np.zeros((28, 28)), True)

        ensemble.recognizers = [mock_recognizer]

        # Create 81 dummy cells
        cells = [np.zeros((50, 50), dtype=np.uint8) for _ in range(81)]

        grid, has_content = ensemble.recognize_grid(cells)

        # Check shapes
        self.assertEqual(grid.shape, (9, 9))
        self.assertEqual(has_content.shape, (9, 9))
        self.assertEqual(grid.dtype, int)
        self.assertEqual(has_content.dtype, bool)

    def test_stats_tracking(self):
        """Test that statistics are properly tracked."""
        config = {
            'models': {
                'cnn': {'enabled': False},
                'tesseract': {'enabled': False},
                'easyocr': {'enabled': False}
            },
            'thresholds': {
                'level1_confidence': 0.75,
                'level2_confidence': 0.65,
                'min_confidence': 0.5
            }
        }

        ensemble = EnsembleRecognizer(config=config)

        # Check initial stats
        self.assertEqual(ensemble.stats['total_cells'], 0)
        self.assertEqual(ensemble.stats['empty_cells'], 0)
        self.assertEqual(ensemble.stats['level1_success'], 0)


class TestEnsembleFallbackLevels(unittest.TestCase):
    """Test fallback level logic."""

    def test_level2_fallback(self):
        """Test that Level 2 is triggered when Level 1 has low confidence."""
        config = {
            'models': {
                'cnn': {'enabled': False},
                'tesseract': {'enabled': False},
                'easyocr': {'enabled': False}
            },
            'thresholds': {
                'level1_confidence': 0.75,  # High threshold
                'level2_confidence': 0.65,
                'min_confidence': 0.5
            }
        }

        ensemble = EnsembleRecognizer(config=config)

        # Mock Level 1 recognizer with low confidence
        mock_l1 = Mock()
        mock_l1.name = "CNN"
        mock_l1.preprocess_cell.return_value = (np.zeros((28, 28)), False)
        mock_l1.recognize.return_value = RecognitionResult(
            digit=5,
            confidence=0.60,  # Below Level 1 threshold
            model_name="CNN"
        )

        # Mock Level 2 recognizer with higher confidence
        mock_l2 = Mock()
        mock_l2.name = "EasyOCR"
        mock_l2.preprocess_cell.return_value = (np.zeros((28, 28)), False)
        mock_l2.recognize.return_value = RecognitionResult(
            digit=7,
            confidence=0.90,
            model_name="EasyOCR"
        )

        ensemble.recognizers = [mock_l1, mock_l2]

        cell = np.zeros((50, 50), dtype=np.uint8)
        result = ensemble.recognize_digit(cell)

        # Should use Level 2 result
        self.assertEqual(ensemble.stats['level2_success'], 1)


if __name__ == '__main__':
    unittest.main()
