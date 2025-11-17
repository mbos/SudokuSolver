"""Grid detection and extraction module using OpenCV."""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class GridDetector:
    """Detects and extracts Sudoku grid from an image."""

    def __init__(self, debug: bool = False):
        """
        Initialize the GridDetector.

        Args:
            debug: If True, display intermediate processing steps
        """
        self.debug = debug

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for grid detection.
        Handles both normal (light background) and inverted/dark theme images.

        Args:
            image: Input BGR image

        Returns:
            Binary image suitable for contour detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect if image has dark background (inverted/night theme)
        # Calculate mean brightness of the image
        mean_brightness = np.mean(gray)

        # If image is predominantly dark (mean < 128), it's likely inverted
        # Invert it so we have standard light background
        if mean_brightness < 128:
            if self.debug:
                print(f"Dark theme detected (brightness: {mean_brightness:.1f}), inverting image...")
            gray = cv2.bitwise_not(gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)

        # Apply adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        if self.debug:
            cv2.imshow("Preprocessed", thresh)
            cv2.waitKey(0)

        return thresh

    def find_grid_contour(self, thresh: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the largest contour which should be the Sudoku grid.

        Args:
            thresh: Binary thresholded image

        Returns:
            Contour of the grid or None if not found
        """
        # Find all contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Sort contours by area (descending) and find the largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find the first contour that is roughly square
        for contour in contours[:10]:  # Check top 10 largest contours
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Grid should have 4 corners
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                # Filter small contours
                if area > 1000:
                    return approx

        return None

    def detect_synthetic_grid(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect grid in synthetic/clean images where grid lines are thin.

        Args:
            image: Input BGR image

        Returns:
            4 corner points or None if not found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try multiple thresholding approaches
        methods = []

        # Method 1: Simple threshold
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        methods.append(thresh1)

        # Method 2: Otsu's threshold (adaptive to image)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        methods.append(thresh2)

        # Method 3: Adaptive threshold (for noisy images)
        thresh3 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        methods.append(thresh3)

        # Try each method
        for thresh in methods:
            # Dilate to connect grid lines
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Look for a large 4-corner contour
            for contour in contours[:5]:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    if area > 10000:  # Larger minimum for synthetic images
                        return approx

        return None

    def order_corner_points(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corner points in consistent order: top-left, top-right,
        bottom-right, bottom-left.

        Args:
            corners: Array of 4 corner points

        Returns:
            Ordered corner points
        """
        corners = corners.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Sum and diff will help us identify corners
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        ordered[0] = corners[np.argmin(s)]      # Top-left (smallest sum)
        ordered[2] = corners[np.argmax(s)]      # Bottom-right (largest sum)
        ordered[1] = corners[np.argmin(diff)]   # Top-right (smallest diff)
        ordered[3] = corners[np.argmax(diff)]   # Bottom-left (largest diff)

        return ordered

    def apply_perspective_transform(
        self, image: np.ndarray, corners: np.ndarray, size: int = 450
    ) -> np.ndarray:
        """
        Apply perspective transform to get bird's eye view of the grid.

        Args:
            image: Input image
            corners: 4 corner points of the grid
            size: Output size (should be multiple of 9)

        Returns:
            Warped image showing grid from top-down view
        """
        # Order the corners
        ordered_corners = self.order_corner_points(corners)

        # Destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, dst)

        # Apply the transform
        warped = cv2.warpPerspective(image, matrix, (size, size))

        if self.debug:
            cv2.imshow("Warped Grid", warped)
            cv2.waitKey(0)

        return warped

    def extract_cells(self, grid_image: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual cells from the grid.

        Args:
            grid_image: Top-down view of the grid

        Returns:
            List of 81 cell images (row-wise, left to right, top to bottom)
        """
        cells = []
        cell_size = grid_image.shape[0] // 9

        for row in range(9):
            for col in range(9):
                # Calculate cell boundaries
                y1 = row * cell_size
                y2 = (row + 1) * cell_size
                x1 = col * cell_size
                x2 = (col + 1) * cell_size

                # Extract cell with small margin to avoid grid lines
                margin = 5
                cell = grid_image[y1 + margin:y2 - margin, x1 + margin:x2 - margin]
                cells.append(cell)

        return cells

    def detect_and_extract(
        self, image_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """
        Full pipeline: detect grid, extract, and return cells.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (original_image, list of 81 cells, warped_grid) or
            (None, None, None) if detection fails
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, None, None

        # Store original for later
        original = image.copy()

        # Try synthetic grid detection first (for clean images)
        grid_contour = self.detect_synthetic_grid(image)

        # If that fails, use traditional method (for photos)
        if grid_contour is None:
            # Preprocess
            thresh = self.preprocess_image(image)

            # Find grid contour
            grid_contour = self.find_grid_contour(thresh)

        if grid_contour is None:
            print("Error: Could not find grid in image")
            return None, None, None

        if self.debug:
            debug_img = original.copy()
            cv2.drawContours(debug_img, [grid_contour], -1, (0, 255, 0), 3)
            cv2.imshow("Grid Detected", debug_img)
            cv2.waitKey(0)

        # Apply perspective transform
        warped = self.apply_perspective_transform(image, grid_contour)

        # Extract individual cells
        cells = self.extract_cells(warped)

        return original, cells, warped
