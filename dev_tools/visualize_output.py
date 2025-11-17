#!/usr/bin/env python3
"""
Visualize the output image as ASCII to see exactly what was drawn.
"""

import cv2
import numpy as np
import sys


def extract_grid_from_image(image_path: str) -> np.ndarray:
    """
    Extract the digit grid from the solved image.

    Args:
        image_path: Path to the solved image

    Returns:
        9x9 numpy array with the extracted digits
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        sys.exit(1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get cell size
    cell_size = gray.shape[0] // 9

    grid = np.zeros((9, 9), dtype=int)

    print("Analyzing image cells...")
    print("=" * 60)

    for row in range(9):
        for col in range(9):
            # Extract cell
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size

            cell = img[y1:y2, x1:x2]

            # Check for red pixels (BGR format, so red is high in index 2)
            red_channel = cell[:, :, 2]
            blue_channel = cell[:, :, 0]
            green_channel = cell[:, :, 1]

            # Red pixels have high red and low blue/green
            red_mask = (red_channel > 150) & (blue_channel < 100) & (green_channel < 100)
            has_red = np.sum(red_mask) > 100

            if has_red:
                print(f"Cell ({row},{col}) has RED content (added by solver)")

    return None


def print_color_grid(image_path: str):
    """
    Print what's in the image, distinguishing between original (black) and added (red) digits.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        sys.exit(1)

    print("\nAnalyzing solved image...")
    print("Legend: [BLACK] = original digits, [RED] = solver-added digits")
    print("=" * 60)

    cell_size = img.shape[0] // 9

    for row in range(9):
        row_str = ""
        for col in range(9):
            # Extract cell
            y1 = row * cell_size + 10
            y2 = (row + 1) * cell_size - 10
            x1 = col * cell_size + 10
            x2 = (col + 1) * cell_size - 10

            cell = img[y1:y2, x1:x2]

            # Check for red pixels
            red_channel = cell[:, :, 2]
            blue_channel = cell[:, :, 0]
            green_channel = cell[:, :, 1]

            red_mask = (red_channel > 150) & (blue_channel < 100) & (green_channel < 100)
            has_red = np.sum(red_mask) > 50

            if has_red:
                row_str += "[R] "
            else:
                row_str += "[B] "

        print(f"Row {row}: {row_str}")

    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_output.py <solved_image.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    extract_grid_from_image(image_path)
    print_color_grid(image_path)
