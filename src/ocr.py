"""OCR module for digit recognition using CNN and Tesseract fallback."""

import cv2
import numpy as np
from typing import Optional, Tuple
import os


class DigitRecognizer:
    """Recognizes digits in Sudoku cells using CNN or Tesseract."""

    def __init__(self, model_path: Optional[str] = None, use_tesseract: bool = False):
        """
        Initialize the digit recognizer.

        Args:
            model_path: Path to trained CNN model (.h5 file)
            use_tesseract: If True, use Tesseract instead of CNN
        """
        self.use_tesseract = use_tesseract
        self.model = None

        if not use_tesseract and model_path and os.path.exists(model_path):
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(model_path)
                print(f"Loaded CNN model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load CNN model: {e}")
                print("Falling back to Tesseract")
                self.use_tesseract = True

        # Always try to import pytesseract for fallback
        try:
            import pytesseract
            self.pytesseract = pytesseract
            if self.use_tesseract:
                print("Using Tesseract OCR")
        except ImportError:
            self.pytesseract = None
            if self.use_tesseract:
                print("Error: pytesseract not installed")
                raise

    def preprocess_cell(self, cell: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Preprocess a cell image for digit recognition.
        Handles both normal (light background) and inverted/dark theme cells.

        Args:
            cell: Cell image (grayscale or BGR)

        Returns:
            Tuple of (preprocessed image, is_empty flag)
        """
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

            # After inversion, apply stronger CLAHE for better contrast on dark theme images
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
        else:
            # Normal CLAHE for regular images
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            gray = clahe.apply(gray)

        # Bilateral filter to reduce noise while preserving edges (lighter filtering)
        gray = cv2.bilateralFilter(gray, 3, 40, 40)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Check if cell is empty (too few white pixels)
        white_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size
        fill_ratio = white_pixels / total_pixels

        # TIER 1: More conservative empty detection to reduce false positives
        if fill_ratio < 0.04:  # Less than 4% filled (increased from 3%)
            return thresh, True

        # Remove noise with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Dilate to thicken thin digits (helps with 1's and 7's)
        dilation_kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, dilation_kernel, iterations=1)

        # Find the largest contour (the digit)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return thresh, True

        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Check if contour is too small
        if w < 5 or h < 5:
            return thresh, True

        # Crop to digit with more padding for better centering
        padding = 8
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(thresh.shape[1] - x, w + 2 * padding)
        h = min(thresh.shape[0] - y, h + 2 * padding)

        digit = thresh[y:y+h, x:x+w]

        return digit, False

    def resize_to_mnist_format(self, digit_image: np.ndarray) -> np.ndarray:
        """
        Resize digit to 28x28 while preserving aspect ratio and centering,
        similar to MNIST preprocessing.

        Args:
            digit_image: Binary digit image

        Returns:
            28x28 image with centered digit
        """
        # Get dimensions
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

    def recognize_with_cnn(self, cell_image: np.ndarray) -> int:
        """
        Recognize digit using CNN model.

        Args:
            cell_image: Preprocessed cell image

        Returns:
            Recognized digit (1-9) or 0 if empty/uncertain
        """
        if self.model is None:
            return 0

        # Resize to 28x28 preserving aspect ratio (MNIST-style)
        resized = self.resize_to_mnist_format(cell_image)

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        # Reshape for model input
        input_data = normalized.reshape(1, 28, 28, 1)

        # Predict
        predictions = self.model.predict(input_data, verbose=0)
        digit = np.argmax(predictions[0])
        confidence = predictions[0][digit]

        # TIER 1: Adaptive confidence thresholds per digit
        # Some digits (6, 8, 9) are harder to distinguish
        # Balanced to minimize false positives while catching real digits
        confidence_thresholds = {
            1: 0.70,  # Can be confused with 7
            6: 0.65,  # Can be confused with 0, 5, 8
            8: 0.65,  # Can be confused with 3, 6
            9: 0.65,  # Can be confused with 4, 7
            0: 0.85,  # Higher threshold for "empty" (more conservative)
        }
        threshold = confidence_thresholds.get(digit, 0.75)  # Default: 0.75 (increased from 0.70)

        # Only return if confident enough
        if confidence > threshold:
            return int(digit) if digit != 0 else 0
        else:
            return 0

    def recognize_with_tesseract(self, cell_image: np.ndarray) -> int:
        """
        Recognize digit using Tesseract OCR with optimized configuration (TIER 1).

        Args:
            cell_image: Preprocessed cell image

        Returns:
            Recognized digit (1-9) or 0 if empty/uncertain
        """
        if self.pytesseract is None:
            return 0

        # Resize for better OCR (8x upscale for small digits)
        height, width = cell_image.shape
        scale = 8  # Increased from 4
        resized = cv2.resize(
            cell_image, (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC
        )

        # TIER 1: Try multiple PSM modes (ordered by likely success)
        psm_modes = [
            10,  # Single character (current)
            8,   # Single word
            7,   # Single text line
            13   # Raw line
        ]

        # TIER 1: Optimized Tesseract config
        base_config = (
            '--oem 3 '
            '-c tessedit_char_whitelist=123456789 '
            '-c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
            '-c load_system_dawg=0 '
            '-c load_freq_dawg=0 '
            '-c matcher_bad_match_pad=0.15'
        )

        # Try each PSM mode
        for psm in psm_modes:
            config = f'--psm {psm} {base_config}'

            try:
                text = self.pytesseract.image_to_string(resized, config=config)
                text = text.strip()

                if text.isdigit() and len(text) == 1:
                    digit = int(text)
                    if 1 <= digit <= 9:
                        return digit
            except Exception:
                continue  # Try next PSM mode

        return 0

    def recognize_digit(self, cell: np.ndarray) -> int:
        """
        Recognize digit in a cell.

        Args:
            cell: Cell image

        Returns:
            Recognized digit (1-9) or 0 if empty
        """
        # Preprocess
        processed, is_empty = self.preprocess_cell(cell)

        if is_empty:
            return 0

        # Use appropriate method
        if self.use_tesseract:
            return self.recognize_with_tesseract(processed)
        else:
            result = self.recognize_with_cnn(processed)
            # If CNN fails and tesseract is available, try tesseract
            if result == 0:
                try:
                    return self.recognize_with_tesseract(processed)
                except:
                    return 0
            return result

    def recognize_grid(self, cells: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recognize all digits in a grid of cells.

        Args:
            cells: List of 81 cell images

        Returns:
            Tuple of (9x9 numpy array with recognized digits,
                     9x9 boolean array indicating which cells have visual content)
        """
        grid = np.zeros((9, 9), dtype=int)
        has_content = np.zeros((9, 9), dtype=bool)

        for i, cell in enumerate(cells):
            row = i // 9
            col = i % 9

            # Check if cell has content first
            _, is_empty = self.preprocess_cell(cell)
            has_content[row, col] = not is_empty

            # Then recognize the digit
            digit = self.recognize_digit(cell)
            grid[row, col] = digit

        return grid, has_content


def train_cnn_model(save_path: str = "models/digit_cnn.h5"):
    """
    Train a CNN model on MNIST dataset for digit recognition.

    Args:
        save_path: Path to save the trained model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow not installed")
        return

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("Building model...")
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training model...")
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Train the model if run directly
    train_cnn_model()
