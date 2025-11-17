#!/usr/bin/env python3
"""Visualize the digits that CNN fails to recognize."""

import cv2
import numpy as np
from src.grid_detector import GridDetector
from src.ocr import DigitRecognizer

# Extract cells
detector = GridDetector(debug=False)
_, cells, _ = detector.detect_and_extract("testplaatje.png")

# Failed cells from analysis
failed_1s = [(1,3), (3,0), (5,4), (6,5), (7,6)]  # All predicted as 7
failed_6s = [(0,7), (3,4), (8,0)]  # Predicted as 5 or 0

print("Saving images of failed digit recognitions...")
print("="*60)

# Save images of 1's (predicted as 7)
print("\nDigit 1 (predicted as 7):")
for i, (row, col) in enumerate(failed_1s):
    cell_idx = row * 9 + col
    cell = cells[cell_idx]
    filename = f"failed_1_{i+1}_at_{row}_{col}.png"
    cv2.imwrite(filename, cell)
    print(f"  Saved: {filename} (cell {row},{col})")

# Save images of 6's (predicted as 5 or 0)
print("\nDigit 6 (predicted as 5 or 0):")
for i, (row, col) in enumerate(failed_6s):
    cell_idx = row * 9 + col
    cell = cells[cell_idx]
    filename = f"failed_6_{i+1}_at_{row}_{col}.png"
    cv2.imwrite(filename, cell)
    print(f"  Saved: {filename} (cell {row},{col})")

# Now preprocess and save what CNN sees
recognizer = DigitRecognizer(model_path="models/digit_cnn.h5", use_tesseract=False)

print("\n" + "="*60)
print("Saving preprocessed versions (what CNN sees)...")
print("="*60)

print("\nPreprocessed digit 1:")
for i, (row, col) in enumerate(failed_1s):
    cell_idx = row * 9 + col
    cell = cells[cell_idx]
    processed, is_empty = recognizer.preprocess_cell(cell)

    # Resize to 28x28 like CNN input
    resized = cv2.resize(processed, (28, 28))

    filename = f"preprocessed_1_{i+1}_at_{row}_{col}.png"
    cv2.imwrite(filename, resized)
    print(f"  Saved: {filename}")

print("\nPreprocessed digit 6:")
for i, (row, col) in enumerate(failed_6s):
    cell_idx = row * 9 + col
    cell = cells[cell_idx]
    processed, is_empty = recognizer.preprocess_cell(cell)

    # Resize to 28x28 like CNN input
    resized = cv2.resize(processed, (28, 28))

    filename = f"preprocessed_6_{i+1}_at_{row}_{col}.png"
    cv2.imwrite(filename, resized)
    print(f"  Saved: {filename}")

print("\n" + "="*60)
print("✓ All images saved. Open them to see what CNN sees.")
print("="*60)
