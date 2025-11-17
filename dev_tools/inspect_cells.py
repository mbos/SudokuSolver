#!/usr/bin/env python3
"""Inspect specific cells from the original image."""

import cv2
import numpy as np
from src.grid_detector import GridDetector

# Load and process image
detector = GridDetector(debug=False)
original_image, cells, warped_grid = detector.detect_and_extract("testplaatje.png")

# Cells that weren't recognized
problem_cells = [(0, 7), (1, 4), (8, 3), (8, 6)]

print("Inspecting cells with content that weren't recognized:")
print("=" * 60)

for row, col in problem_cells:
    cell_index = row * 9 + col
    cell = cells[cell_index]

    # Save cell for inspection
    filename = f"cell_{row}_{col}.png"
    cv2.imwrite(filename, cell)
    print(f"Cell ({row},{col}) saved to {filename}")

    # Show some stats
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    fill_ratio = white_pixels / total_pixels

    print(f"  Fill ratio: {fill_ratio:.2%}")
    print()

print("\nYou can now open these PNG files to see what the OCR couldn't recognize.")
