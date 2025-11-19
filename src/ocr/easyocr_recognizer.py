"""EasyOCR-based digit recognizer."""

import time
import warnings
import numpy as np
from typing import Optional

from .base_recognizer import BaseRecognizer, RecognitionResult


class EasyOCRRecognizer(BaseRecognizer):
    """Recognizes digits using EasyOCR (deep learning-based OCR)."""

    def __init__(self, enabled: bool = True, weight: float = 2.0,
                 languages: Optional[list] = None, gpu: bool = False):
        """
        Initialize EasyOCR recognizer.

        Args:
            enabled: Whether this recognizer is enabled
            weight: Weight for ensemble voting (higher due to better accuracy)
            languages: Languages to support (default: ['en'])
            gpu: Whether to use GPU acceleration
        """
        super().__init__(name="EasyOCR", enabled=enabled, weight=weight)
        self.reader = None
        self.languages = languages or ['en']
        self.gpu = gpu

        # Try to initialize EasyOCR
        if enabled:
            try:
                import easyocr
                print("[EasyOCR] Initializing reader (this may take a moment)...")
                # Suppress PyTorch MPS pin_memory warning (not supported on Apple Silicon)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*pin_memory.*")
                    self.reader = easyocr.Reader(
                        self.languages,
                        gpu=self.gpu,
                        verbose=False
                    )
                print(f"[EasyOCR] Initialized successfully (GPU: {self.gpu})")
            except ImportError:
                print("[EasyOCR] Warning: easyocr not installed")
                self.enabled = False
            except Exception as e:
                print(f"[EasyOCR] Warning: Failed to initialize: {e}")
                self.enabled = False

    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        return self.reader is not None

    def recognize(self, cell_image: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> RecognitionResult:
        """
        Recognize digit using EasyOCR.

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

        try:
            # EasyOCR expects image in standard format (not binary threshold)
            # Use the preprocessed but convert back if needed
            # For better results, use the original grayscale
            if len(cell_image.shape) == 3:
                import cv2
                gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cell_image

            # Run EasyOCR with allowlist for digits only
            # Suppress PyTorch MPS pin_memory warning during inference
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*pin_memory.*")
                results = self.reader.readtext(
                    gray,
                    allowlist='123456789',
                    detail=1,  # Return bounding boxes and confidence
                    paragraph=False
                )

            # Process results
            if results:
                # Get the result with highest confidence
                best_result = max(results, key=lambda x: x[2])  # x[2] is confidence
                text = best_result[1].strip()
                confidence = float(best_result[2])

                # Validate it's a single digit
                if text.isdigit() and len(text) == 1:
                    digit = int(text)
                    if 1 <= digit <= 9:
                        processing_time = (time.time() - start_time) * 1000
                        return RecognitionResult(
                            digit=digit,
                            confidence=confidence,
                            model_name=self.name,
                            processing_time_ms=processing_time,
                            weight=self.weight
                        )

        except Exception:
            # Silently fail and return no result
            pass

        # No digit recognized - moderate confidence vote for "empty"
        # This acts as a veto only when other models are also uncertain
        # Score: 0.6 * 2.0 = 1.2, beats Tesseract alone (0.7 * 1.0 = 0.7)
        # but CNN + Tesseract agreeing (1.35 + 0.7 = 2.05) can override it
        processing_time = (time.time() - start_time) * 1000
        return RecognitionResult(
            digit=0,
            confidence=0.6,  # Moderate confidence - veto only uncertain cases
            model_name=self.name,
            processing_time_ms=processing_time,
            weight=self.weight
        )
