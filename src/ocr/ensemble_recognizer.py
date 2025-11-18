"""Ensemble recognizer combining multiple OCR models with intelligent fallback."""

import numpy as np
from typing import List, Optional, Tuple
import time

from .base_recognizer import BaseRecognizer, RecognitionResult
from .cnn_recognizer import CNNRecognizer
from .tesseract_recognizer import TesseractRecognizer
from .easyocr_recognizer import EasyOCRRecognizer
from .voting_strategies import VotingStrategy, get_voting_strategy


class EnsembleRecognizer:
    """
    Ensemble recognizer combining multiple OCR models.

    Uses a multi-level fallback strategy:
    - Level 1 (Fast): CNN + Tesseract (PSM 10)
    - Level 2 (Medium): Add EasyOCR + more Tesseract modes
    - Level 3 (Full): All models with weighted voting
    """

    def __init__(self,
                 config: Optional[dict] = None,
                 voting_strategy: str = "weighted"):
        """
        Initialize ensemble recognizer.

        Args:
            config: Configuration dictionary (default: auto-configure)
            voting_strategy: Voting strategy to use ('majority', 'weighted', 'confidence')
        """
        self.config = config or self._default_config()
        self.voting_strategy = get_voting_strategy(voting_strategy)
        self.recognizers: List[BaseRecognizer] = []
        self.stats = {
            'total_cells': 0,
            'level1_success': 0,
            'level2_success': 0,
            'level3_success': 0,
            'empty_cells': 0,
        }

        # Initialize recognizers based on config
        self._initialize_recognizers()

    def _default_config(self) -> dict:
        """Get default configuration."""
        return {
            'models': {
                'cnn': {
                    'enabled': True,
                    'weight': 1.5,
                    'level': 1
                },
                'tesseract': {
                    'enabled': True,
                    'weight': 1.0,
                    'level': 1,
                    'psm_modes': [10, 8, 7, 13]
                },
                'easyocr': {
                    'enabled': True,
                    'weight': 2.0,
                    'level': 2,
                    'gpu': False
                }
            },
            'thresholds': {
                'level1_confidence': 0.75,  # If confidence > this, accept Level 1
                'level2_confidence': 0.65,  # If confidence > this, accept Level 2
                'min_confidence': 0.5       # Minimum to consider valid
            }
        }

    def _initialize_recognizers(self):
        """Initialize all configured recognizers."""
        models_config = self.config.get('models', {})

        # Initialize CNN
        if models_config.get('cnn', {}).get('enabled', True):
            cnn_config = models_config['cnn']
            recognizer = CNNRecognizer(
                enabled=True,
                weight=cnn_config.get('weight', 1.5)
            )
            if recognizer.is_available():
                self.recognizers.append(recognizer)
                print(f"[Ensemble] Added CNN recognizer (weight: {recognizer.weight})")

        # Initialize Tesseract
        if models_config.get('tesseract', {}).get('enabled', True):
            tess_config = models_config['tesseract']
            recognizer = TesseractRecognizer(
                enabled=True,
                weight=tess_config.get('weight', 1.0),
                psm_modes=tess_config.get('psm_modes', [10, 8, 7, 13])
            )
            if recognizer.is_available():
                self.recognizers.append(recognizer)
                print(f"[Ensemble] Added Tesseract recognizer (weight: {recognizer.weight})")

        # Initialize EasyOCR
        if models_config.get('easyocr', {}).get('enabled', True):
            easy_config = models_config['easyocr']
            recognizer = EasyOCRRecognizer(
                enabled=True,
                weight=easy_config.get('weight', 2.0),
                gpu=easy_config.get('gpu', False)
            )
            if recognizer.is_available():
                self.recognizers.append(recognizer)
                print(f"[Ensemble] Added EasyOCR recognizer (weight: {recognizer.weight})")

        if not self.recognizers:
            print("[Ensemble] WARNING: No recognizers available!")

        print(f"[Ensemble] Initialized with {len(self.recognizers)} recognizer(s)")

    def recognize_digit(self, cell: np.ndarray, verbose: bool = False) -> int:
        """
        Recognize digit using ensemble with fallback levels.

        Args:
            cell: Cell image
            verbose: Print debug information

        Returns:
            Recognized digit (0-9), where 0 means empty
        """
        self.stats['total_cells'] += 1

        # Preprocess once for all recognizers
        if not self.recognizers:
            return 0

        preprocessed, is_empty = self.recognizers[0].preprocess_cell(cell)

        if is_empty:
            self.stats['empty_cells'] += 1
            return 0

        # Level 1: Fast path (CNN + basic Tesseract)
        level1_results = self._run_level1(cell, preprocessed)
        result = self.voting_strategy.vote(level1_results)

        threshold = self.config['thresholds']['level1_confidence']
        if result.digit != 0 and result.confidence >= threshold:
            self.stats['level1_success'] += 1
            if verbose:
                print(f"  [L1] Digit: {result.digit}, Confidence: {result.confidence:.2f}")
            return result.digit

        # Level 2: Medium path (add EasyOCR)
        level2_results = self._run_level2(cell, preprocessed, level1_results)
        result = self.voting_strategy.vote(level2_results)

        threshold = self.config['thresholds']['level2_confidence']
        if result.digit != 0 and result.confidence >= threshold:
            self.stats['level2_success'] += 1
            if verbose:
                print(f"  [L2] Digit: {result.digit}, Confidence: {result.confidence:.2f}")
            return result.digit

        # Level 3: Full ensemble
        level3_results = self._run_level3(cell, preprocessed, level2_results)
        result = self.voting_strategy.vote(level3_results)

        self.stats['level3_success'] += 1
        if verbose:
            print(f"  [L3] Digit: {result.digit}, Confidence: {result.confidence:.2f}")

        return result.digit

    def _run_level1(self, cell: np.ndarray, preprocessed: np.ndarray) -> List[RecognitionResult]:
        """Run Level 1 recognizers (fast path)."""
        results = []

        for recognizer in self.recognizers:
            if recognizer.name in ['CNN', 'Tesseract']:
                result = recognizer.recognize(cell, preprocessed)
                results.append(result)

        return results

    def _run_level2(self, cell: np.ndarray, preprocessed: np.ndarray,
                    existing_results: List[RecognitionResult]) -> List[RecognitionResult]:
        """Run Level 2 recognizers (medium path)."""
        results = existing_results.copy()

        for recognizer in self.recognizers:
            if recognizer.name == 'EasyOCR':
                result = recognizer.recognize(cell, preprocessed)
                results.append(result)

        return results

    def _run_level3(self, cell: np.ndarray, preprocessed: np.ndarray,
                    existing_results: List[RecognitionResult]) -> List[RecognitionResult]:
        """Run Level 3 - all recognizers."""
        # For now, level 3 = level 2 (all available models)
        # In future, could add more advanced models here
        return existing_results

    def recognize_grid(self, cells: list, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recognize all digits in a grid of cells.

        Args:
            cells: List of 81 cell images
            verbose: Print progress information

        Returns:
            Tuple of (9x9 grid with recognized digits,
                     9x9 boolean array indicating which cells have visual content)
        """
        grid = np.zeros((9, 9), dtype=int)
        has_content = np.zeros((9, 9), dtype=bool)

        print(f"\n[Ensemble] Recognizing grid with {len(self.recognizers)} model(s)...")

        for i, cell in enumerate(cells):
            row = i // 9
            col = i % 9

            # Check if cell has content
            if self.recognizers:
                _, is_empty = self.recognizers[0].preprocess_cell(cell)
                has_content[row, col] = not is_empty
            else:
                has_content[row, col] = False

            # Recognize digit
            digit = self.recognize_digit(cell, verbose=verbose)
            grid[row, col] = digit

            # Progress indicator
            if (i + 1) % 9 == 0 and not verbose:
                print(f"  Row {row + 1}/9 complete")

        self._print_stats()

        return grid, has_content

    def _print_stats(self):
        """Print recognition statistics."""
        total = self.stats['total_cells']
        if total == 0:
            return

        print(f"\n[Ensemble] Recognition Statistics:")
        print(f"  Total cells:     {total}")
        print(f"  Empty cells:     {self.stats['empty_cells']}")
        print(f"  Level 1 success: {self.stats['level1_success']} "
              f"({100*self.stats['level1_success']/total:.1f}%)")
        print(f"  Level 2 success: {self.stats['level2_success']} "
              f"({100*self.stats['level2_success']/total:.1f}%)")
        print(f"  Level 3 success: {self.stats['level3_success']} "
              f"({100*self.stats['level3_success']/total:.1f}%)")
