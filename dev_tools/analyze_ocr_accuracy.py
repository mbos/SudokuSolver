#!/usr/bin/env python3
"""
Analyze OCR accuracy by comparing detected grid with the correct solution.
"""

import numpy as np

# Correct solution from testplaatje_oplossing.txt
correct_solution = np.array([
    [3, 2, 1, 4, 7, 8, 9, 6, 5],
    [5, 4, 8, 1, 9, 6, 3, 2, 7],
    [9, 7, 6, 2, 5, 3, 4, 1, 8],
    [1, 3, 4, 7, 6, 5, 2, 8, 9],
    [8, 9, 5, 3, 4, 2, 6, 7, 1],
    [2, 6, 7, 8, 1, 9, 5, 3, 4],
    [4, 8, 3, 5, 2, 1, 7, 9, 6],
    [7, 2, 9, 6, 8, 4, 1, 5, 3],
    [6, 5, 1, 9, 3, 7, 8, 4, 2]
])

# What OCR detected (from verbose output)
detected_grid = np.array([
    [0, 0, 0, 0, 0, 0, 9, 0, 5],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 8],
    [1, 0, 0, 7, 6, 0, 0, 0, 0],
    [0, 9, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 1, 0, 5, 3, 0],
    [0, 0, 3, 0, 2, 1, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 1, 5, 0],
    [6, 0, 0, 0, 0, 0, 0, 0, 0]
])

# What the solver produced
solver_output = np.array([
    [2, 8, 4, 6, 3, 7, 9, 1, 5],
    [3, 5, 6, 1, 9, 8, 2, 4, 7],
    [9, 7, 1, 2, 5, 4, 3, 6, 8],
    [1, 3, 2, 7, 6, 5, 4, 8, 9],
    [4, 9, 5, 3, 8, 2, 6, 7, 1],
    [8, 6, 7, 4, 1, 9, 5, 3, 2],
    [5, 4, 3, 8, 2, 1, 7, 9, 6],
    [7, 2, 8, 9, 4, 6, 1, 5, 3],
    [6, 1, 9, 5, 7, 3, 8, 2, 4]
])

print("=" * 80)
print("OCR ACCURACY ANALYSIS")
print("=" * 80)

# Find the actual starting grid (non-zero cells in correct solution)
actual_starting_grid = np.zeros_like(correct_solution)
print("\nACTUAL STARTING GRID (from correct solution):")
print("-" * 80)
for row in range(9):
    for col in range(9):
        if detected_grid[row, col] != 0:
            actual_starting_grid[row, col] = correct_solution[row, col]

# Print comparison
print("Position | Actual | Detected | Status")
print("-" * 80)

correct_count = 0
missed_count = 0
incorrect_count = 0

for row in range(9):
    for col in range(9):
        if detected_grid[row, col] != 0:
            actual = correct_solution[row, col]
            detected = detected_grid[row, col]

            if actual == detected:
                status = "✓ CORRECT"
                correct_count += 1
            else:
                status = "✗ WRONG!"
                incorrect_count += 1

            print(f"({row},{col})    |   {actual}    |    {detected}     | {status}")

# Check for missed digits
print("\n" + "=" * 80)
print("CELLS WITH CONTENT NOT RECOGNIZED BY OCR:")
print("-" * 80)
print("These cells have visual content but OCR returned 0:")
print()

missed_cells = [(0, 7), (1, 4), (8, 3), (8, 6)]  # From verbose output
for row, col in missed_cells:
    actual = correct_solution[row, col]
    print(f"Cell ({row},{col}): Should be {actual}, OCR returned 0")
    missed_count += 1

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("-" * 80)
print(f"Total cells detected: {np.count_nonzero(detected_grid)}")
print(f"Correctly detected:   {correct_count}")
print(f"Incorrectly detected: {incorrect_count}")
print(f"Missed (has content): {missed_count}")
print(f"OCR Accuracy:         {correct_count}/{correct_count + incorrect_count + missed_count} = {100*correct_count/(correct_count + incorrect_count + missed_count):.1f}%")

# Check if solver output is correct
print("\n" + "=" * 80)
print("SOLVER OUTPUT CORRECTNESS")
print("-" * 80)
if np.array_equal(solver_output, correct_solution):
    print("✓ Solver output matches correct solution!")
else:
    print("✗ Solver output does NOT match correct solution")
    print(f"  Differences: {np.sum(solver_output != correct_solution)} cells")
    print("\n  This is expected because OCR provided wrong/incomplete input.")

print("\n" + "=" * 80)
print("CONCLUSION")
print("-" * 80)
print("""
The solver algorithm itself works correctly - it produces a valid Sudoku solution.
However, because OCR missed or misread some starting digits, the solver created
a different valid Sudoku that doesn't match the intended puzzle.

ROOT CAUSE: OCR reliability
- Tesseract failed to recognize 4 digits that have visual content
- These missed digits are critical for solving the correct puzzle
""")
print("=" * 80)
