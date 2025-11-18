"""
Analyze OCR errors in testplaat2 by comparing detected vs actual
"""

# From the original testplaat2.png image (manually verified)
original = [
    [1, 0, 0, 9, 0, 6, 0, 0, 4],
    [0, 0, 0, 4, 0, 0, 0, 0, 2],
    [0, 3, 0, 1, 0, 0, 0, 6, 0],
    [0, 1, 0, 0, 6, 0, 0, 5, 9],
    [0, 5, 0, 7, 0, 0, 3, 0, 0],
    [0, 0, 4, 0, 0, 0, 0, 0, 6],
    [0, 0, 8, 0, 0, 0, 0, 4, 0],
    [7, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 7, 8, 0, 0]
]

# From the script output (Detected puzzle section)
# Reading the ASCII art carefully:
# 1 . . 9 6 . . 9
# . . . 4 . . . 2
# . 3 . 1 . . . 6 .
# (blank line)
# . 1 . . 8 . . 5 9
# . 5 . 7 . . 3 . .
# . . 4 . . . . . 6
# (blank line)
# . . 8 . . . . 4 .
# 7 . . . 3 . . . .
# . 6 . . . 7 8 . .

detected = [
    [1, 0, 0, 9, 6, 0, 0, 9, 0],  # Row 0: extra 6 at pos 4, wrong digit at pos 7
    [0, 0, 0, 4, 0, 0, 0, 2, 0],  # Row 1
    [0, 3, 0, 1, 0, 0, 0, 6, 0],  # Row 2
    [0, 1, 0, 0, 8, 0, 0, 5, 9],  # Row 3: wrong digit at pos 4 (8 instead of 6)
    [0, 5, 0, 7, 0, 0, 3, 0, 0],  # Row 4
    [0, 0, 4, 0, 0, 0, 0, 0, 6],  # Row 5
    [0, 0, 8, 0, 0, 0, 0, 4, 0],  # Row 6
    [7, 0, 0, 0, 3, 0, 0, 0, 0],  # Row 7
    [0, 6, 0, 0, 0, 7, 8, 0, 0]   # Row 8
]

print("="*80)
print("TESTPLAAT2 OCR ERROR ANALYSIS")
print("="*80)
print()

# Count original digits
original_count = sum(1 for row in original for cell in row if cell != 0)
print(f"Total starting digits in puzzle: {original_count}")
print()

# Find all errors
errors = []
missed = []  # False negatives
wrong = []   # Wrong digit detected
extra = []   # False positives

for i in range(9):
    for j in range(9):
        orig = original[i][j]
        det = detected[i][j]

        if orig != 0 and det == 0:
            missed.append((i, j, orig))
            errors.append(f"Cell ({i},{j}): MISSED digit {orig} (detected as empty)")
        elif orig == 0 and det != 0:
            extra.append((i, j, det))
            errors.append(f"Cell ({i},{j}): FALSE POSITIVE - detected {det}, should be empty")
        elif orig != 0 and det != 0 and orig != det:
            wrong.append((i, j, orig, det))
            errors.append(f"Cell ({i},{j}): WRONG DIGIT - detected {det}, should be {orig}")

print("ERRORS FOUND:")
print("-" * 80)
if errors:
    for error in errors:
        print(f"  {error}")
else:
    print("  No errors - perfect OCR!")

print()
print("ERROR SUMMARY:")
print("-" * 80)
print(f"  Correctly detected:        {original_count - len(missed) - len(wrong)}")
print(f"  Missed digits:             {len(missed)}")
print(f"  Wrong digits:              {len(wrong)}")
print(f"  False positives:           {len(extra)}")
print(f"  Total errors:              {len(errors)}")
print(f"  Accuracy:                  {100 * (original_count - len(missed) - len(wrong)) / original_count:.1f}%")
print()

# Analyze why solver failed
print("SOLVER VALIDATION ERRORS:")
print("-" * 80)
print("The solver reported duplicate values:")
print("  - Row 0: Duplicate values [9]")
print("  - Column 8: Duplicate values [9]")
print()

print(f"Row 0 original:  {original[0]}")
print(f"Row 0 detected:  {detected[0]}")
print(f"  Problem: Position 4 has extra 6 (should be empty)")
print(f"  Problem: Position 7 has 9 (should be 4)")
print(f"  Result: Two 9s in row 0 → INVALID PUZZLE")
print()

col_8_orig = [original[i][8] for i in range(9)]
col_8_det = [detected[i][8] for i in range(9)]
print(f"Column 8 original: {col_8_orig}")
print(f"Column 8 detected: {col_8_det}")
print(f"  Problem: detected has no 9s in column 8")
print()

print("="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)
print()
print("The OCR errors cause the puzzle to become INVALID:")
print("  1. False positive at (0,4): Empty cell detected as 6")
print("  2. Wrong digit at (0,7): 4 detected as 9")
print("  3. Wrong digit at (3,4): 6 detected as 8")
print()
print("These errors create constraint violations (duplicates) that make")
print("the puzzle unsolvable by the constraint-based solver.")
print()
print("="*80)
