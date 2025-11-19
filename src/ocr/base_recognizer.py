"""Base class for digit recognizers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RecognitionResult:
    """Result from a digit recognition attempt.

    Attributes:
        digit: Recognized digit (0-9), where 0 means empty/uncertain
        confidence: Confidence score (0.0-1.0)
        model_name: Name of the model that produced this result
        processing_time_ms: Time taken for recognition in milliseconds
        weight: Weight for ensemble voting
    """
    digit: int
    confidence: float
    model_name: str
    processing_time_ms: float = 0.0
    weight: float = 1.0

    def __post_init__(self):
        """Validate result fields."""
        if not (0 <= self.digit <= 9):
            raise ValueError(f"Digit must be 0-9, got {self.digit}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


class BaseRecognizer(ABC):
    """Abstract base class for digit recognizers."""

    def __init__(self, name: str, enabled: bool = True, weight: float = 1.0):
        """
        Initialize recognizer.

        Args:
            name: Name of this recognizer
            enabled: Whether this recognizer is enabled
            weight: Weight for ensemble voting (higher = more trusted)
        """
        self.name = name
        self.enabled = enabled
        self.weight = weight

    @abstractmethod
    def recognize(self, cell_image: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> RecognitionResult:
        """
        Recognize a digit in a cell image.

        Args:
            cell_image: Original cell image
            preprocessed: Optional preprocessed image (if already preprocessed)

        Returns:
            RecognitionResult with digit and confidence
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this recognizer is available (dependencies installed, model loaded, etc.).

        Returns:
            True if recognizer can be used
        """
        pass

    def preprocess_cell(self, cell: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Preprocess a cell image for digit recognition.
        Shared preprocessing logic for all recognizers.

        Args:
            cell: Cell image (grayscale or BGR)

        Returns:
            Tuple of (preprocessed image, is_empty flag)
        """
        import cv2

        # Convert to grayscale if needed
        if len(cell.shape) == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell.copy()

        # Detect if cell has dark background (inverted/night theme)
        mean_brightness = np.mean(gray)

        # If cell is predominantly dark (mean < 128), invert it
        if mean_brightness < 128:
            gray = cv2.bitwise_not(gray)
            # After inversion, apply stronger CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
        else:
            # Normal CLAHE for regular images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)

        # Bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 5, 50, 50)

        # Use adaptive thresholding for better handling of uneven lighting
        # This works better than Otsu for varying backgrounds
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Check if cell is empty (too few white pixels)
        white_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size
        fill_ratio = white_pixels / total_pixels

        # Less conservative empty detection to catch faint digits
        if fill_ratio < 0.03:  # Less than 3% filled
            return thresh, True

        # Too much fill likely means noise or grid lines
        if fill_ratio > 0.60:  # More than 60% filled
            # Try Otsu as fallback for high-contrast images
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            fill_ratio = np.sum(thresh == 255) / total_pixels
            if fill_ratio < 0.03 or fill_ratio > 0.60:
                return thresh, True

        # Remove noise with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find the largest contour (the digit)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return thresh, True

        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Check if contour is too small
        if w < 4 or h < 6:
            return thresh, True

        # Check aspect ratio - digits should be taller than wide (mostly)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 5.0:
            # Likely noise, not a digit
            return thresh, True

        # Check if contour area is reasonable compared to bounding box
        contour_area = cv2.contourArea(largest_contour)
        bbox_area = w * h
        if bbox_area > 0 and contour_area / bbox_area < 0.15:
            # Too sparse, likely noise
            return thresh, True

        # Crop to digit with padding
        padding = 6
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(thresh.shape[1] - x, w + 2 * padding)
        h = min(thresh.shape[0] - y, h + 2 * padding)

        digit = thresh[y:y+h, x:x+w]

        return digit, False
