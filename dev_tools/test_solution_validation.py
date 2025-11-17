#!/usr/bin/env python3
"""
Test solution validation by simulating a successful solve.
"""

import numpy as np
from src.solver import SudokuSolver
from src.validator import SudokuValidator

# Easy valid puzzle
puzzle = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

print("Testing Solution Validation Pipeline")
print("=" * 60)
print("\nOriginal puzzle:")
print(puzzle)

# Solve
solver = SudokuSolver()
solver.load_puzzle(puzzle)

if solver.solve():
    solution = solver.get_solution()

    print("\n[✓] Puzzle solved successfully!")
    print("\nSolution:")
    print(solution)

    # Validate solution
    print("\n" + "=" * 60)
    print("[3.5/4] Validating solution...")
    print("=" * 60)

    is_valid, errors = SudokuValidator.validate_solution(solution, verbose=False)

    if is_valid:
        print("✅ Solution validation passed!")
    else:
        print(f"⚠️  Solution validation failed ({len(errors)} errors)")
        for error in errors[:5]:
            print(f"  - {error}")

    # Compare grids
    print("\n" + "=" * 60)
    print("Checking for overwrites...")
    print("=" * 60)

    comparison = SudokuValidator.compare_grids(puzzle, solution, verbose=True)

    if comparison['overwrite_count'] > 0:
        print(f"\n⚠️  Warning: {comparison['overwrite_count']} original values were changed!")
    else:
        print("\n✅ No original values overwritten")

    print("\n" + "=" * 60)
    print("VALIDATION TEST COMPLETE")
    print("=" * 60)
    print(f"Solution valid: {is_valid}")
    print(f"Overwrites: {comparison['overwrite_count']}")
    print(f"Cells filled by solver: {comparison['cells_filled_by_solver']}")

else:
    print("\n[✗] Failed to solve puzzle")
