"""
Analyze OCR errors in testplaat2 by comparing detected vs actual solution
"""

# From the script output image, the detected puzzle was:
detected_str = """1..96..9
...4...2
.3.1...6.

.1..8..59
.5.7..3..
..4.....6

..8....4.
7...3....
.6...78.."""

# From testplaat2_oplossing.txt (original digits only, 0 = empty)
solution_str = """1..9.6..4
...4....2
.3.1...6.

.1..6..59
.5.7..3..
..4.....6

..8....4.
7...3....
.6...78.."""

def parse_puzzle(puzzle_str):
    """Parse puzzle string into 9x9 grid"""
    lines = [line.strip() for line in puzzle_str.strip().split('\n') if line.strip()]
    grid = []
    for line in lines:
        row = []
        for char in line:
            if char == '.' or char == ' ':
                row.append(0)
            elif char.isdigit():
                row.append(int(char))
        if len(row) == 9:
            grid.append(row)
    return grid

# Parse both grids
detected = parse_puzzle(detected_str)
solution = parse_puzzle(solution_str)

print("="*70)
print("TESTPLAAT2 OCR ERROR ANALYSIS")
print("="*70)
print()

# Count cells with content in original (from solution, non-zero = original digit)
original_cells = []
for i in range(9):
    for j in range(9):
        if solution[i][j] != 0:
            original_cells.append((i, j, solution[i][j]))

print(f"Total starting digits in puzzle: {len(original_cells)}")
print()

# Compare detected vs actual
errors = []
false_positives = []
false_negatives = []

for i in range(9):
    for j in range(9):
        det = detected[i][j]
        sol = solution[i][j]

        if sol != 0:  # This cell should have a digit
            if det == 0:
                false_negatives.append((i, j, sol))
                errors.append(f"Cell ({i},{j}): Missed digit {sol} (detected as empty)")
            elif det != sol:
                errors.append(f"Cell ({i},{j}): Wrong digit - detected {det}, should be {sol}")
        else:  # This cell should be empty
            if det != 0:
                false_positives.append((i, j, det))
                errors.append(f"Cell ({i},{j}): False positive - detected {det}, should be empty")

print("ERRORS FOUND:")
print("-" * 70)
if errors:
    for error in errors:
        print(f"  {error}")
else:
    print("  No errors!")

print()
print("SUMMARY:")
print("-" * 70)
print(f"  Total starting digits: {len(original_cells)}")
print(f"  Correctly detected: {len(original_cells) - len(false_negatives) - len([e for e in errors if 'Wrong digit' in e])}")
print(f"  Missed digits (false negatives): {len(false_negatives)}")
print(f"  Wrong digits: {len([e for e in errors if 'Wrong digit' in e])}")
print(f"  False positives (empty detected as digit): {len(false_positives)}")
print(f"  Total errors: {len(errors)}")
print()

# Analyze the duplicate errors mentioned in the output
print("DUPLICATE ANALYSIS (from solver error):")
print("-" * 70)
print("  The solver reported:")
print("    - Row 0: Duplicate values [9]")
print("    - Column 8: Duplicate values [9]")
print()
print("  Let's check row 0 and column 8:")
print(f"    Detected row 0: {detected[0]}")
print(f"    Actual row 0:   {solution[0]}")
print()
print(f"    Detected col 8: {[detected[i][8] for i in range(9)]}")
print(f"    Actual col 8:   {[solution[i][8] for i in range(9)]}")
print()

print("="*70)
