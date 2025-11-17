"""Image generation module to draw solved Sudoku on original image."""

import cv2
import numpy as np
from typing import Tuple


class SolutionDrawer:
    """Draws the solved Sudoku numbers on the original image."""

    def __init__(self, font_scale: float = 1.0, thickness: int = 2):
        """
        Initialize the solution drawer.

        Args:
            font_scale: Scale factor for font size
            thickness: Thickness of the drawn numbers
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.color = (0, 0, 255)  # Red color for solved digits (BGR)

    def draw_on_warped(
        self,
        warped_grid: np.ndarray,
        has_content: np.ndarray,
        solution: np.ndarray
    ) -> np.ndarray:
        """
        Draw solution on the warped grid image.

        Args:
            warped_grid: Top-down view of the grid
            has_content: Boolean mask indicating which cells have visual content (9x9)
            solution: Solved grid (9x9)

        Returns:
            Warped grid with solution drawn
        """
        result = warped_grid.copy()

        # Convert to BGR if grayscale
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        cell_size = result.shape[0] // 9

        for row in range(9):
            for col in range(9):
                # Only draw if this cell was originally empty (no visual content)
                if not has_content[row, col]:
                    digit = solution[row, col]

                    # Calculate position to center the text
                    x = col * cell_size
                    y = row * cell_size

                    # Get text size for centering
                    text = str(digit)
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, self.font, self.font_scale, self.thickness
                    )

                    # Center the text in the cell
                    text_x = x + (cell_size - text_width) // 2
                    text_y = y + (cell_size + text_height) // 2

                    # Draw the digit
                    cv2.putText(
                        result, text, (text_x, text_y),
                        self.font, self.font_scale, self.color, self.thickness
                    )

        return result

    def overlay_solution(
        self,
        original_image: np.ndarray,
        warped_solution: np.ndarray,
        corners: np.ndarray
    ) -> np.ndarray:
        """
        Overlay the solved grid back onto the original image using perspective transform.

        Args:
            original_image: Original input image
            warped_solution: Solved grid in warped view
            corners: 4 corner points of the grid in original image

        Returns:
            Original image with solution overlaid
        """
        # Order corners
        corners = self._order_corners(corners)

        # Source points (corners of warped image)
        h, w = warped_solution.shape[:2]
        src = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        # Calculate inverse perspective transform
        matrix = cv2.getPerspectiveTransform(src, corners)

        # Warp the solution back to original perspective
        warped_back = cv2.warpPerspective(
            warped_solution, matrix,
            (original_image.shape[1], original_image.shape[0])
        )

        # Create mask for blending
        gray_warped = cv2.cvtColor(warped_back, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black out the grid area in original
        result = original_image.copy()
        result = cv2.bitwise_and(result, result, mask=mask_inv)

        # Add the warped solution
        result = cv2.add(result, warped_back)

        return result

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corner points consistently.

        Args:
            corners: Array of 4 corner points

        Returns:
            Ordered corner points (TL, TR, BR, BL)
        """
        corners = corners.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)

        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        ordered[0] = corners[np.argmin(s)]      # Top-left
        ordered[2] = corners[np.argmax(s)]      # Bottom-right
        ordered[1] = corners[np.argmin(diff)]   # Top-right
        ordered[3] = corners[np.argmax(diff)]   # Bottom-left

        return ordered

    def save_result(self, image: np.ndarray, output_path: str) -> None:
        """
        Save the result image.

        Args:
            image: Image to save
            output_path: Path to save the image
        """
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")

    def display_result(self, image: np.ndarray, window_name: str = "Solved Sudoku") -> None:
        """
        Display the result image.

        Args:
            image: Image to display
            window_name: Name of the display window
        """
        # Resize if too large
        max_height = 800
        h, w = image.shape[:2]
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, max_height))

        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_side_by_side_comparison(
    original: np.ndarray,
    solved: np.ndarray
) -> np.ndarray:
    """
    Create a side-by-side comparison of original and solved images.

    Args:
        original: Original image
        solved: Solved image

    Returns:
        Combined image showing both side by side
    """
    # Ensure both images have same height
    h1, w1 = original.shape[:2]
    h2, w2 = solved.shape[:2]

    if h1 != h2:
        if h1 > h2:
            solved = cv2.resize(solved, (int(w2 * h1 / h2), h1))
        else:
            original = cv2.resize(original, (int(w1 * h2 / h1), h2))

    # Concatenate horizontally
    comparison = np.hstack([original, solved])

    return comparison
