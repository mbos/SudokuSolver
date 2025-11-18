"""Sudoku solver using constraint propagation and backtracking."""

import numpy as np
from typing import List, Set, Tuple, Optional


class SudokuSolver:
    """
    Solves Sudoku puzzles using hybrid approach:
    1. Constraint propagation with arc consistency
    2. Backtracking with MRV (Minimum Remaining Value) heuristic
    """

    def __init__(self):
        """Initialize the Sudoku solver."""
        self.grid = np.zeros((9, 9), dtype=int)
        # Possible values for each cell
        self.possibilities = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]

    def load_puzzle(self, puzzle: np.ndarray) -> None:
        """
        Load a Sudoku puzzle.

        Args:
            puzzle: 9x9 numpy array where 0 represents empty cells
        """
        self.grid = puzzle.copy()
        self.possibilities = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]

        # Initialize possibilities based on filled cells
        for row in range(9):
            for col in range(9):
                if self.grid[row, col] != 0:
                    self.possibilities[row][col] = set()
                    self._update_constraints(row, col, self.grid[row, col])

    def _update_constraints(self, row: int, col: int, value: int) -> None:
        """
        Update possibilities after placing a value.

        Args:
            row: Row index
            col: Column index
            value: Placed value
        """
        # Remove value from row
        for c in range(9):
            if c != col:
                self.possibilities[row][c].discard(value)

        # Remove value from column
        for r in range(9):
            if r != row:
                self.possibilities[r][col].discard(value)

        # Remove value from 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if r != row or c != col:
                    self.possibilities[r][c].discard(value)

    def _propagate_constraints(self) -> bool:
        """
        Apply constraint propagation techniques.

        Returns:
            True if successful, False if contradiction found
        """
        changed = True
        while changed:
            changed = False

            for row in range(9):
                for col in range(9):
                    if self.grid[row, col] == 0:
                        # Naked single: only one possibility left
                        if len(self.possibilities[row][col]) == 0:
                            return False  # Contradiction
                        elif len(self.possibilities[row][col]) == 1:
                            value = list(self.possibilities[row][col])[0]
                            self.grid[row, col] = value
                            self.possibilities[row][col] = set()
                            self._update_constraints(row, col, value)
                            changed = True

            # Hidden singles: check rows
            changed = changed or self._find_hidden_singles()

        return True

    def _find_hidden_singles(self) -> bool:
        """
        Find hidden singles in rows, columns, and boxes.

        Returns:
            True if any hidden single was found
        """
        found = False

        # Check rows
        for row in range(9):
            for num in range(1, 10):
                positions = [col for col in range(9)
                           if num in self.possibilities[row][col]]
                if len(positions) == 1:
                    col = positions[0]
                    if self.grid[row, col] == 0:
                        self.grid[row, col] = num
                        self.possibilities[row][col] = set()
                        self._update_constraints(row, col, num)
                        found = True

        # Check columns
        for col in range(9):
            for num in range(1, 10):
                positions = [row for row in range(9)
                           if num in self.possibilities[row][col]]
                if len(positions) == 1:
                    row = positions[0]
                    if self.grid[row, col] == 0:
                        self.grid[row, col] = num
                        self.possibilities[row][col] = set()
                        self._update_constraints(row, col, num)
                        found = True

        # Check boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                for num in range(1, 10):
                    positions = [
                        (r, c) for r in range(box_row, box_row + 3)
                        for c in range(box_col, box_col + 3)
                        if num in self.possibilities[r][c]
                    ]
                    if len(positions) == 1:
                        row, col = positions[0]
                        if self.grid[row, col] == 0:
                            self.grid[row, col] = num
                            self.possibilities[row][col] = set()
                            self._update_constraints(row, col, num)
                            found = True

        return found

    def _find_best_cell(self) -> Optional[Tuple[int, int]]:
        """
        Find empty cell with minimum remaining values (MRV heuristic).

        Returns:
            Tuple of (row, col) or None if no empty cells
        """
        min_possibilities = 10
        best_cell = None

        for row in range(9):
            for col in range(9):
                if self.grid[row, col] == 0:
                    num_poss = len(self.possibilities[row][col])
                    if num_poss < min_possibilities:
                        min_possibilities = num_poss
                        best_cell = (row, col)

        return best_cell

    def _is_valid(self, row: int, col: int, num: int) -> bool:
        """
        Check if placing num at (row, col) is valid.

        Args:
            row: Row index
            col: Column index
            num: Number to place

        Returns:
            True if valid, False otherwise
        """
        # Check row
        if num in self.grid[row, :]:
            return False

        # Check column
        if num in self.grid[:, col]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in self.grid[box_row:box_row + 3, box_col:box_col + 3]:
            return False

        return True

    def _solve_backtrack(self) -> bool:
        """
        Solve using backtracking with MRV heuristic.

        Returns:
            True if solved, False if no solution
        """
        # Try constraint propagation first
        if not self._propagate_constraints():
            return False

        # Find best cell to fill (MRV heuristic)
        cell = self._find_best_cell()

        if cell is None:
            # No empty cells, puzzle is solved
            return True

        row, col = cell

        # Try each possibility
        for num in list(self.possibilities[row][col]):
            if self._is_valid(row, col, num):
                # Save state
                old_grid = self.grid.copy()
                old_poss = [row_poss.copy() for row_poss in
                           [[cell.copy() for cell in row] for row in self.possibilities]]

                # Make move
                self.grid[row, col] = num
                self.possibilities[row][col] = set()
                self._update_constraints(row, col, num)

                # Recurse
                if self._solve_backtrack():
                    return True

                # Restore state
                self.grid = old_grid
                self.possibilities = old_poss

        return False

    def solve(self) -> bool:
        """
        Solve the loaded puzzle.

        Returns:
            True if solved successfully, False otherwise
        """
        return self._solve_backtrack()

    def get_solution(self) -> np.ndarray:
        """
        Get the solved grid.

        Returns:
            9x9 numpy array with solution
        """
        return self.grid.copy()

    def is_valid_puzzle(self) -> bool:
        """
        Check if the loaded puzzle is valid.

        Returns:
            True if valid, False otherwise
        """
        # Check no duplicates in rows
        for row in range(9):
            nums = [self.grid[row, col] for col in range(9) if self.grid[row, col] != 0]
            if len(nums) != len(set(nums)):
                return False

        # Check no duplicates in columns
        for col in range(9):
            nums = [self.grid[row, col] for row in range(9) if self.grid[row, col] != 0]
            if len(nums) != len(set(nums)):
                return False

        # Check no duplicates in boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                nums = [
                    self.grid[r, c]
                    for r in range(box_row, box_row + 3)
                    for c in range(box_col, box_col + 3)
                    if self.grid[r, c] != 0
                ]
                if len(nums) != len(set(nums)):
                    return False

        return True

    def find_constraint_violations(self) -> List[Tuple[str, int, List[int], List[Tuple[int, int]]]]:
        """
        Find all constraint violations in the puzzle.

        Returns:
            List of violations, each as (violation_type, index, duplicate_values, cell_positions)
            - violation_type: "row", "column", or "box"
            - index: row/column number (0-8) or box number (0-8)
            - duplicate_values: list of values that appear multiple times
            - cell_positions: list of (row, col) tuples for cells involved in violations
        """
        violations = []

        # Check rows for duplicates
        for row in range(9):
            cells_with_values = [(row, col, self.grid[row, col])
                                for col in range(9) if self.grid[row, col] != 0]

            value_counts = {}
            for r, c, val in cells_with_values:
                if val not in value_counts:
                    value_counts[val] = []
                value_counts[val].append((r, c))

            duplicates = {val: positions for val, positions in value_counts.items()
                         if len(positions) > 1}

            if duplicates:
                duplicate_values = list(duplicates.keys())
                cell_positions = []
                for positions in duplicates.values():
                    cell_positions.extend(positions)
                violations.append(("row", row, duplicate_values, cell_positions))

        # Check columns for duplicates
        for col in range(9):
            cells_with_values = [(row, col, self.grid[row, col])
                                for row in range(9) if self.grid[row, col] != 0]

            value_counts = {}
            for r, c, val in cells_with_values:
                if val not in value_counts:
                    value_counts[val] = []
                value_counts[val].append((r, c))

            duplicates = {val: positions for val, positions in value_counts.items()
                         if len(positions) > 1}

            if duplicates:
                duplicate_values = list(duplicates.keys())
                cell_positions = []
                for positions in duplicates.values():
                    cell_positions.extend(positions)
                violations.append(("column", col, duplicate_values, cell_positions))

        # Check boxes for duplicates
        for box_idx in range(9):
            box_row = 3 * (box_idx // 3)
            box_col = 3 * (box_idx % 3)

            cells_with_values = [
                (r, c, self.grid[r, c])
                for r in range(box_row, box_row + 3)
                for c in range(box_col, box_col + 3)
                if self.grid[r, c] != 0
            ]

            value_counts = {}
            for r, c, val in cells_with_values:
                if val not in value_counts:
                    value_counts[val] = []
                value_counts[val].append((r, c))

            duplicates = {val: positions for val, positions in value_counts.items()
                         if len(positions) > 1}

            if duplicates:
                duplicate_values = list(duplicates.keys())
                cell_positions = []
                for positions in duplicates.values():
                    cell_positions.extend(positions)
                violations.append(("box", box_idx, duplicate_values, cell_positions))

        return violations
