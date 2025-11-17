#!/usr/bin/env python3
"""
Analyze CNN prediction errors to identify patterns.
"""

import numpy as np
from collections import defaultdict
from src.grid_detector import GridDetector
from src.ocr import DigitRecognizer

# Ground truth from testplaatje.png
GROUND_TRUTH = np.array([
    [0, 0, 0, 0, 0, 0, 9, 6, 5],
    [0, 0, 0, 1, 9, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 8],
    [1, 0, 0, 7, 6, 0, 0, 0, 0],
    [0, 9, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 1, 0, 5, 3, 0],
    [0, 0, 3, 0, 2, 1, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 1, 5, 0],
    [6, 0, 0, 9, 0, 0, 8, 0, 0]
])

# Extract cells
detector = GridDetector(debug=False)
_, cells, _ = detector.detect_and_extract("testplaatje.png")

# Run CNN
recognizer = DigitRecognizer(model_path="models/digit_cnn.h5", use_tesseract=False)
detected_grid, _ = recognizer.recognize_grid(cells)

print("="*80)
print("CNN ERROR ANALYSIS")
print("="*80)

# Build confusion matrix
confusion = defaultdict(lambda: defaultdict(int))
errors = []

for row in range(9):
    for col in range(9):
        true_digit = GROUND_TRUTH[row, col]
        pred_digit = detected_grid[row, col]

        if true_digit != 0:  # Only check filled cells
            confusion[true_digit][pred_digit] += 1

            if true_digit != pred_digit:
                errors.append({
                    'cell': (row, col),
                    'true': true_digit,
                    'pred': pred_digit
                })

# Print confusion matrix
print("\nCONFUSION MATRIX (True digit → Predicted digit)")
print("-"*80)
print(f"{'True':<6}", end="")
for pred in range(1, 10):
    print(f"{pred:>6}", end="")
print()
print("-"*80)

for true_digit in range(1, 10):
    print(f"{true_digit:<6}", end="")
    for pred_digit in range(1, 10):
        count = confusion[true_digit][pred_digit]
        if count > 0:
            if true_digit == pred_digit:
                print(f"{count:>6}", end="")  # Correct predictions
            else:
                print(f"[{count:>3}]", end=" ")  # Errors in brackets
        else:
            print(f"{'':>6}", end="")
    print()

print("\n" + "="*80)
print("DETAILED ERRORS")
print("="*80)

# Group errors by true digit
error_groups = defaultdict(list)
for error in errors:
    error_groups[error['true']].append(error)

for true_digit in sorted(error_groups.keys()):
    errors_for_digit = error_groups[true_digit]
    print(f"\nDigit {true_digit} (misclassified {len(errors_for_digit)} times):")

    # Count predictions
    pred_counts = defaultdict(int)
    for error in errors_for_digit:
        pred_counts[error['pred']] += 1

    for pred, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / len(errors_for_digit)
        print(f"  → Predicted as {pred}: {count} times ({percentage:.0f}%)")

    # Show cell locations
    print(f"  Cells: ", end="")
    for error in errors_for_digit:
        print(f"({error['cell'][0]},{error['cell'][1]})", end=" ")
    print()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_filled = np.count_nonzero(GROUND_TRUTH)
total_errors = len(errors)
accuracy = (total_filled - total_errors) / total_filled

print(f"Total filled cells:  {total_filled}")
print(f"Correct predictions: {total_filled - total_errors} ({100*(1-total_errors/total_filled):.1f}%)")
print(f"Incorrect:           {total_errors} ({100*total_errors/total_filled:.1f}%)")

# Identify most problematic digits
print("\nMost confused digits:")
for true_digit in sorted(error_groups.keys(), key=lambda x: len(error_groups[x]), reverse=True):
    errors_count = len(error_groups[true_digit])
    total_count = sum(confusion[true_digit].values())
    error_rate = errors_count / total_count if total_count > 0 else 0
    print(f"  Digit {true_digit}: {errors_count}/{total_count} errors ({100*error_rate:.0f}% error rate)")

print("="*80)
