#!/usr/bin/env python3
"""Test the solver independently without OCR."""

import numpy as np
from src.solver import SudokuSolver


def test_solver():
    """Test the solver with a known puzzle."""
    # Simple valid test puzzle (not from image, just to test solver)
    # 0 represents empty cells
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

    print("Testing Sudoku Solver")
    print("=" * 50)
    print("\nInput puzzle:")
    print_grid(puzzle)

    solver = SudokuSolver()
    solver.load_puzzle(puzzle)

    if not solver.is_valid_puzzle():
        print("\nError: Puzzle is invalid!")
        # Debug: Check which constraint is violated
        print("\nChecking rows...")
        for row in range(9):
            nums = [puzzle[row, col] for col in range(9) if puzzle[row, col] != 0]
            if len(nums) != len(set(nums)):
                print(f"  Row {row+1} has duplicates: {nums}")

        print("\nChecking columns...")
        for col in range(9):
            nums = [puzzle[row, col] for row in range(9) if puzzle[row, col] != 0]
            if len(nums) != len(set(nums)):
                print(f"  Column {col+1} has duplicates: {nums}")

        print("\nChecking boxes...")
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                nums = [
                    puzzle[r, c]
                    for r in range(box_row, box_row + 3)
                    for c in range(box_col, box_col + 3)
                    if puzzle[r, c] != 0
                ]
                if len(nums) != len(set(nums)):
                    print(f"  Box at ({box_row//3+1}, {box_col//3+1}) has duplicates: {nums}")
        return False

    print("\nSolving...")
    if solver.solve():
        solution = solver.get_solution()
        print("\nSolved puzzle:")
        print_grid(solution)
        print("\n✓ Solver works correctly!")
        return True
    else:
        print("\nError: Could not solve puzzle")
        return False


def print_grid(grid: np.ndarray) -> None:
    """Print a Sudoku grid in a nice format."""
    print("\n┌─────────┬─────────┬─────────┐")
    for i, row in enumerate(grid):
        if i > 0 and i % 3 == 0:
            print("├─────────┼─────────┼─────────┤")

        row_str = "│"
        for j, val in enumerate(row):
            if j > 0 and j % 3 == 0:
                row_str += "│"
            row_str += f" {val if val != 0 else '.'}"

        row_str += " │"
        print(row_str)

    print("└─────────┴─────────┴─────────┘")


if __name__ == "__main__":
    import sys
    success = test_solver()
    sys.exit(0 if success else 1)
