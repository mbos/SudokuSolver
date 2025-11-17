"""CNN-based digit recognizer using MNIST-trained model."""

import time
import cv2
import numpy as np
from typing import Optional
import os

from .base_recognizer import BaseRecognizer, RecognitionResult


class CNNRecognizer(BaseRecognizer):
    """Recognizes digits using a CNN model trained on MNIST."""

    def __init__(self, model_path: str = "models/digit_cnn.h5",
                 enabled: bool = True, weight: float = 1.5):
        """
        Initialize CNN recognizer.

        Args:
            model_path: Path to trained Keras model
            enabled: Whether this recognizer is enabled
            weight: Weight for ensemble voting
        """
        super().__init__(name="CNN", enabled=enabled, weight=weight)
        self.model_path = model_path
        self.model = None

        # Try to load model
        if enabled and os.path.exists(model_path):
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(model_path)
                print(f"[CNN] Loaded model from {model_path}")
            except Exception as e:
                print(f"[CNN] Warning: Could not load model: {e}")
                self.enabled = False

    def is_available(self) -> bool:
        """Check if CNN model is available."""
        return self.model is not None

    def resize_to_mnist_format(self, digit_image: np.ndarray) -> np.ndarray:
        """
        Resize digit to 28x28 while preserving aspect ratio and centering.

        Args:
            digit_image: Binary digit image

        Returns:
            28x28 image with centered digit
        """
        h, w = digit_image.shape

        # Calculate aspect ratio
        if h > w:
            new_h = 20  # Leave margin like MNIST
            new_w = int(w * 20 / h)
        else:
            new_w = 20
            new_h = int(h * 20 / w)

        # Avoid zero dimensions
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # Resize maintaining aspect ratio
        resized = cv2.resize(digit_image, (new_w, new_h))

        # Create 28x28 black canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)

        # Calculate position to center digit
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2

        # Place digit on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def recognize(self, cell_image: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> RecognitionResult:
        """
        Recognize digit using CNN.

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
                processing_time_ms=0.0
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
                    processing_time_ms=processing_time
                )

        # Resize to 28x28 MNIST format
        resized = self.resize_to_mnist_format(preprocessed)

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        # Reshape for model input
        input_data = normalized.reshape(1, 28, 28, 1)

        # Predict
        predictions = self.model.predict(input_data, verbose=0)
        digit = np.argmax(predictions[0])
        confidence = float(predictions[0][digit])

        # Adaptive confidence thresholds per digit
        confidence_thresholds = {
            1: 0.70,  # Can be confused with 7
            6: 0.65,  # Can be confused with 0, 5, 8
            8: 0.65,  # Can be confused with 3, 6
            9: 0.65,  # Can be confused with 4, 7
            0: 0.85,  # Higher threshold for "empty"
        }
        threshold = confidence_thresholds.get(digit, 0.75)

        # Only return if confident enough
        if confidence < threshold:
            digit = 0
            confidence = 0.0

        processing_time = (time.time() - start_time) * 1000

        return RecognitionResult(
            digit=int(digit) if digit != 0 else 0,
            confidence=confidence,
            model_name=self.name,
            processing_time_ms=processing_time
        )
