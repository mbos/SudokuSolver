"""Ensemble recognizer combining multiple OCR models with intelligent fallback."""

import numpy as np
from typing import List, Optional, Tuple

from .base_recognizer import BaseRecognizer, RecognitionResult
from .cnn_recognizer import CNNRecognizer
from .tesseract_recognizer import TesseractRecognizer
from .easyocr_recognizer import EasyOCRRecognizer
from .voting_strategies import get_voting_strategy


class EnsembleRecognizer:
    """
    Ensemble recognizer combining multiple OCR models.

    Uses a multi-level fallback strategy:
    - Level 1 (Fast): CNN + Tesseract (PSM 10)
    - Level 2 (Medium): Add EasyOCR + more Tesseract modes
    - Level 3 (Full): All models with weighted voting
    """

    def __init__(self,
                 config: Optional[dict] = None):
        """
        Initialize ensemble recognizer.

        Args:
            config: Configuration dictionary (default: auto-configure)
        """
        self.config = config or self._default_config()
        
        # Use the voting strategy from the config, with a default fallback
        voting_strategy_name = self.config.get('voting_strategy', 'weighted')
        self.voting_strategy = get_voting_strategy(voting_strategy_name)

        self.recognizers_by_level: dict[int, List[BaseRecognizer]] = {}
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
            'voting_strategy': 'weighted',
            'models': {
                'cnn': {
                    'enabled': True,
                    'weight': 1.0,
                    'level': 1
                },
                'tesseract': {
                    'enabled': True,
                    'weight': 1.0,
                    'level': 1,
                    'psm_modes': [10]
                },
                'easyocr': {
                    'enabled': True,
                    'weight': 2.5,
                    'level': 2, # Slower, so run as a fallback
                    'gpu': False
                }
            },
            'thresholds': {
                'level1_confidence': 0.85,
                'level2_confidence': 0.70,
                'min_confidence': 0.40
            }
        }

    def _initialize_recognizers(self):
        """Initialize all configured recognizers and group them by level."""
        models_config = self.config.get('models', {})
        
        # Temporarily collect all created recognizers before grouping
        all_recognizers_with_level = []

        # Initialize CNN
        if models_config.get('cnn', {}).get('enabled', True):
            cnn_config = models_config['cnn']
            level = cnn_config.get('level', 1)
            recognizer = CNNRecognizer(
                enabled=True,
                weight=cnn_config.get('weight', 1.0),
                model_path=cnn_config.get('model_path', 'models/digit_cnn.h5')
            )
            if recognizer.is_available():
                all_recognizers_with_level.append((recognizer, level))
                print(f"[Ensemble] Loaded CNN recognizer (level: {level}, weight: {recognizer.weight})")

        # Initialize Tesseract
        if models_config.get('tesseract', {}).get('enabled', True):
            tess_config = models_config['tesseract']
            level = tess_config.get('level', 1)
            recognizer = TesseractRecognizer(
                enabled=True,
                weight=tess_config.get('weight', 1.0),
                psm_modes=tess_config.get('psm_modes', [10])
            )
            if recognizer.is_available():
                all_recognizers_with_level.append((recognizer, level))
                print(f"[Ensemble] Loaded Tesseract recognizer (level: {level}, weight: {recognizer.weight})")

        # Initialize EasyOCR
        if models_config.get('easyocr', {}).get('enabled', True):
            easy_config = models_config['easyocr']
            level = easy_config.get('level', 2)
            recognizer = EasyOCRRecognizer(
                enabled=True,
                weight=easy_config.get('weight', 2.0),
                gpu=easy_config.get('gpu', False),
                languages=easy_config.get('languages', ['en'])
            )
            if recognizer.is_available():
                all_recognizers_with_level.append((recognizer, level))
                print(f"[Ensemble] Loaded EasyOCR recognizer (level: {level}, weight: {recognizer.weight})")

        if not all_recognizers_with_level:
            print("[Ensemble] WARNING: No recognizers available!")
        
        # Group recognizers by level
        for recognizer, level in all_recognizers_with_level:
            if level not in self.recognizers_by_level:
                self.recognizers_by_level[level] = []
            self.recognizers_by_level[level].append(recognizer)

        total_recognizers = len(all_recognizers_with_level)
        print(f"[Ensemble] Initialized with {total_recognizers} recognizer(s) grouped into {len(self.recognizers_by_level)} level(s).")


    def recognize_digit(self, cell: np.ndarray, verbose: bool = False) -> Tuple[int, float]:
        """
        Recognize digit using an efficient, multi-level ensemble with fallback.

        Args:
            cell: Cell image
            verbose: Print debug information

        Returns:
            Tuple of (recognized digit (0-9), confidence score 0-1), where 0 means empty
        """
        self.stats['total_cells'] += 1

        # Get a flat list of all available recognizers to find one for preprocessing
        all_recognizers = [rec for level_recs in self.recognizers_by_level.values() for rec in level_recs]
        if not all_recognizers:
            return 0, 0.0

        # Preprocess once for all recognizers
        preprocessed, is_empty = all_recognizers[0].preprocess_cell(cell)

        if is_empty:
            self.stats['empty_cells'] += 1
            return 0, 1.0  # High confidence for empty cells

        all_results: List[RecognitionResult] = []
        
        # Iterate through levels (1, 2, 3...) in sorted order
        for level in sorted(self.recognizers_by_level.keys()):
            recognizers_at_level = self.recognizers_by_level[level]
            
            if verbose:
                print(f"  Running Level {level} recognizers...")

            for recognizer in recognizers_at_level:
                result = recognizer.recognize(cell, preprocessed)
                all_results.append(result)

            # Vote after each level is complete
            final_result = self.voting_strategy.vote(all_results)
            
            # Check if confidence is high enough to stop early
            level_threshold = self.config.get('thresholds', {}).get(f'level{level}_confidence')
            if level_threshold and final_result.confidence >= level_threshold:
                if verbose:
                    print(f"  [L{level}] Success! Digit: {final_result.digit}, Confidence: {final_result.confidence:.2f} (>{level_threshold})")
                self.stats[f'level{level}_success'] += 1
                return final_result.digit, final_result.confidence

        # If no level produced a high-confidence result, return the best result we have
        final_result = self.voting_strategy.vote(all_results)
        if verbose:
            print(f"  [Fallback] Low confidence. Digit: {final_result.digit}, Confidence: {final_result.confidence:.2f}")

        # Use a final minimum confidence threshold to avoid returning complete noise
        min_confidence = self.config.get('thresholds', {}).get('min_confidence', 0.0)
        if final_result.confidence < min_confidence:
            return 0, final_result.confidence # Return 0 if confidence is too low

        return final_result.digit, final_result.confidence


    def recognize_grid(self, cells: list, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recognize all digits in a grid of cells.

        Args:
            cells: List of 81 cell images
            verbose: Print progress information

        Returns:
            Tuple of (9x9 grid with recognized digits,
                     9x9 boolean array indicating which cells have visual content,
                     9x9 float array with confidence scores for each cell)
        """
        grid = np.zeros((9, 9), dtype=int)
        has_content = np.zeros((9, 9), dtype=bool)
        confidence_matrix = np.zeros((9, 9), dtype=float)

        # Get a flat list of all available recognizers to find one for preprocessing
        all_recognizers = [rec for level_recs in self.recognizers_by_level.values() for rec in level_recs]
        total_models = len(all_recognizers)
        print(f"\n[Ensemble] Recognizing grid with {total_models} model(s)...")

        for i, cell in enumerate(cells):
            row = i // 9
            col = i % 9

            # Check if cell has content
            if all_recognizers:
                _, is_empty = all_recognizers[0].preprocess_cell(cell)
                has_content[row, col] = not is_empty
            else:
                has_content[row, col] = False

            # Recognize digit
            digit, confidence = self.recognize_digit(cell, verbose=verbose)
            grid[row, col] = digit
            confidence_matrix[row, col] = confidence

            # Progress indicator
            if (i + 1) % 9 == 0 and not verbose:
                print(f"  Row {row + 1}/9 complete")

        self._print_stats()

        return grid, has_content, confidence_matrix

    def _print_stats(self):
        """Print recognition statistics."""
        total = self.stats['total_cells']
        if total == 0:
            return

        print("\n[Ensemble] Recognition Statistics:")
        print(f"  Total cells:     {total}")
        print(f"  Empty cells:     {self.stats['empty_cells']}")
        print(f"  Level 1 success: {self.stats['level1_success']} "
              f"({100*self.stats['level1_success']/total:.1f}%)")
        print(f"  Level 2 success: {self.stats['level2_success']} "
              f"({100*self.stats['level2_success']/total:.1f}%)")
        print(f"  Level 3 success: {self.stats['level3_success']} "
              f"({100*self.stats['level3_success']/total:.1f}%)")
