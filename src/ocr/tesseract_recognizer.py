"""Tesseract OCR-based digit recognizer."""

import time
import cv2
import numpy as np
from typing import Optional, List

from .base_recognizer import BaseRecognizer, RecognitionResult


class TesseractRecognizer(BaseRecognizer):
    """Recognizes digits using Tesseract OCR with multiple PSM modes."""

    def __init__(self, enabled: bool = True, weight: float = 1.0,
                 psm_modes: Optional[List[int]] = None):
        """
        Initialize Tesseract recognizer.

        Args:
            enabled: Whether this recognizer is enabled
            weight: Weight for ensemble voting
            psm_modes: PSM modes to try (default: [10, 8, 7, 13])
        """
        super().__init__(name="Tesseract", enabled=enabled, weight=weight)
        self.pytesseract = None
        self.psm_modes = psm_modes or [10, 8, 7, 13]

        # Try to import pytesseract
        if enabled:
            try:
                import pytesseract
                self.pytesseract = pytesseract
                print(f"[Tesseract] Initialized with PSM modes: {self.psm_modes}")
            except ImportError:
                print("[Tesseract] Warning: pytesseract not installed")
                self.enabled = False

    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        return self.pytesseract is not None

    def recognize(self, cell_image: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> RecognitionResult:
        """
        Recognize digit using Tesseract OCR.

        Args:
            cell_image: Original cell image
            preprocessed: Optional preprocessed image

        Returns:
            RecognitionResult with digit and confidence
        """
        start_time = time.time()

        if not self.is_available():
            return RecognitionResult(
                digit=0,
                confidence=0.0,
                model_name=self.name,
                processing_time_ms=0.0,
                weight=self.weight
            )

        # Preprocess if not provided
        if preprocessed is None:
            preprocessed, is_empty = self.preprocess_cell(cell_image)
            if is_empty:
                processing_time = (time.time() - start_time) * 1000
                return RecognitionResult(
                    digit=0,
                    confidence=1.0,
                    model_name=self.name,
                    processing_time_ms=processing_time,
                    weight=self.weight
                )

        # Resize for better OCR (8x upscale)
        height, width = preprocessed.shape
        scale = 8
        resized = cv2.resize(
            preprocessed, (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC
        )

        # Base Tesseract config
        base_config = (
            '--oem 3 '
            '-c tessedit_char_whitelist=123456789 '
            '-c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
            '-c load_system_dawg=0 '
            '-c load_freq_dawg=0 '
            '-c matcher_bad_match_pad=0.15'
        )

        # Try each PSM mode
        for psm in self.psm_modes:
            config = f'--psm {psm} {base_config}'

            try:
                # Use image_to_data for actual confidence scores
                data = self.pytesseract.image_to_data(
                    resized, config=config, output_type=self.pytesseract.Output.DICT
                )

                # Find best digit result with confidence
                best_digit = None
                best_conf = 0

                for i, text in enumerate(data['text']):
                    text = text.strip()
                    if text.isdigit() and len(text) == 1:
                        digit = int(text)
                        if 1 <= digit <= 9:
                            conf = int(data['conf'][i])
                            if conf > best_conf:
                                best_digit = digit
                                best_conf = conf

                if best_digit is not None and best_conf > 30:  # Minimum 30% confidence
                    # Scale confidence from 0-100 to 0-1
                    # Apply scaling factor since Tesseract confidence is often inflated
                    confidence = min(0.95, best_conf / 100.0 * 0.9)
                    processing_time = (time.time() - start_time) * 1000
                    return RecognitionResult(
                        digit=best_digit,
                        confidence=confidence,
                        model_name=self.name,
                        processing_time_ms=processing_time,
                        weight=self.weight
                    )
            except Exception:
                continue  # Try next PSM mode

        # No digit recognized
        processing_time = (time.time() - start_time) * 1000
        return RecognitionResult(
            digit=0,
            confidence=0.5,  # Moderate confidence for no result
            model_name=self.name,
            processing_time_ms=processing_time,
            weight=self.weight
        )
