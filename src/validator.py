"""
Sudoku validation module.

Validates Sudoku puzzles and solutions for logical correctness.
"""

import numpy as np
from typing import Tuple, List, Dict


class SudokuValidator:
    """Validates Sudoku puzzles and solutions."""

    @staticmethod
    def validate_solution(grid: np.ndarray, verbose: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate a complete Sudoku solution.

        Checks:
        1. All rows contain digits 1-9 exactly once
        2. All columns contain digits 1-9 exactly once
        3. All 3x3 boxes contain digits 1-9 exactly once
        4. No empty cells (all filled)

        Args:
            grid: 9x9 numpy array representing the Sudoku grid
            verbose: If True, print detailed validation messages

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if grid.shape != (9, 9):
            return False, ["Grid must be 9x9"]

        errors = []

        # Check if all cells are filled
        if np.any(grid == 0):
            empty_count = np.sum(grid == 0)
            errors.append(f"Solution incomplete: {empty_count} empty cells")
            if verbose:
                print(f"❌ Solution incomplete: {empty_count} empty cells")

        # Check if all values are in valid range
        if np.any((grid < 0) | (grid > 9)):
            invalid_values = grid[(grid < 0) | (grid > 9)]
            errors.append(f"Invalid values found: {invalid_values}")
            if verbose:
                print(f"❌ Invalid values found: {invalid_values}")

        # Validate rows
        row_errors = SudokuValidator._validate_rows(grid, verbose)
        errors.extend(row_errors)

        # Validate columns
        col_errors = SudokuValidator._validate_columns(grid, verbose)
        errors.extend(col_errors)

        # Validate 3x3 boxes
        box_errors = SudokuValidator._validate_boxes(grid, verbose)
        errors.extend(box_errors)

        is_valid = len(errors) == 0

        if verbose:
            if is_valid:
                print("✅ Sudoku solution is VALID!")
            else:
                print(f"\n❌ Sudoku solution is INVALID ({len(errors)} errors)")

        return is_valid, errors

    @staticmethod
    def _validate_rows(grid: np.ndarray, verbose: bool = False) -> List[str]:
        """Validate all rows contain 1-9 exactly once."""
        errors = []

        for row_idx in range(9):
            row = grid[row_idx, :]
            # Remove zeros (empty cells) for partial validation
            filled = row[row != 0]

            # Check for duplicates in filled cells
            unique, counts = np.unique(filled, return_counts=True)
            duplicates = unique[counts > 1]

            if len(duplicates) > 0:
                error = f"Row {row_idx}: Duplicate values {duplicates.tolist()}"
                errors.append(error)
                if verbose:
                    print(f"❌ {error}")

            # For complete solutions, check if all 1-9 are present
            if len(filled) == 9:
                missing = set(range(1, 10)) - set(filled)
                if missing:
                    error = f"Row {row_idx}: Missing values {sorted(missing)}"
                    errors.append(error)
                    if verbose:
                        print(f"❌ {error}")

        return errors

    @staticmethod
    def _validate_columns(grid: np.ndarray, verbose: bool = False) -> List[str]:
        """Validate all columns contain 1-9 exactly once."""
        errors = []

        for col_idx in range(9):
            col = grid[:, col_idx]
            filled = col[col != 0]

            # Check for duplicates
            unique, counts = np.unique(filled, return_counts=True)
            duplicates = unique[counts > 1]

            if len(duplicates) > 0:
                error = f"Column {col_idx}: Duplicate values {duplicates.tolist()}"
                errors.append(error)
                if verbose:
                    print(f"❌ {error}")

            # For complete solutions, check if all 1-9 are present
            if len(filled) == 9:
                missing = set(range(1, 10)) - set(filled)
                if missing:
                    error = f"Column {col_idx}: Missing values {sorted(missing)}"
                    errors.append(error)
                    if verbose:
                        print(f"❌ {error}")

        return errors

    @staticmethod
    def _validate_boxes(grid: np.ndarray, verbose: bool = False) -> List[str]:
        """Validate all 3x3 boxes contain 1-9 exactly once."""
        errors = []

        for box_row in range(3):
            for box_col in range(3):
                # Extract 3x3 box
                row_start = box_row * 3
                col_start = box_col * 3
                box = grid[row_start:row_start+3, col_start:col_start+3]
                box_flat = box.flatten()
                filled = box_flat[box_flat != 0]

                # Check for duplicates
                unique, counts = np.unique(filled, return_counts=True)
                duplicates = unique[counts > 1]

                if len(duplicates) > 0:
                    error = f"Box ({box_row},{box_col}): Duplicate values {duplicates.tolist()}"
                    errors.append(error)
                    if verbose:
                        print(f"❌ {error}")

                # For complete solutions, check if all 1-9 are present
                if len(filled) == 9:
                    missing = set(range(1, 10)) - set(filled)
                    if missing:
                        error = f"Box ({box_row},{box_col}): Missing values {sorted(missing)}"
                        errors.append(error)
                        if verbose:
                            print(f"❌ {error}")

        return errors

    @staticmethod
    def validate_puzzle(grid: np.ndarray, verbose: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate a Sudoku puzzle (can be incomplete).

        Only checks for conflicts, not completeness.

        Args:
            grid: 9x9 numpy array (0 = empty)
            verbose: If True, print validation messages

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if grid.shape != (9, 9):
            return False, ["Grid must be 9x9"]

        errors = []

        # Only check for duplicates in filled cells
        row_errors = SudokuValidator._validate_rows(grid, verbose)
        col_errors = SudokuValidator._validate_columns(grid, verbose)
        box_errors = SudokuValidator._validate_boxes(grid, verbose)

        errors.extend(row_errors)
        errors.extend(col_errors)
        errors.extend(box_errors)

        is_valid = len(errors) == 0

        if verbose:
            if is_valid:
                print("✅ Sudoku puzzle is valid (no conflicts)")
            else:
                print(f"❌ Sudoku puzzle has conflicts ({len(errors)} errors)")

        return is_valid, errors

    @staticmethod
    def get_validation_report(grid: np.ndarray) -> Dict:
        """
        Get detailed validation report.

        Args:
            grid: 9x9 numpy array

        Returns:
            Dictionary with validation statistics
        """
        is_valid, errors = SudokuValidator.validate_solution(grid, verbose=False)

        filled_cells = np.sum(grid != 0)
        empty_cells = np.sum(grid == 0)

        report = {
            'is_valid': is_valid,
            'total_errors': len(errors),
            'errors': errors,
            'filled_cells': int(filled_cells),
            'empty_cells': int(empty_cells),
            'completion': float(filled_cells / 81 * 100),
        }

        # Count errors by type
        row_errors = [e for e in errors if e.startswith('Row')]
        col_errors = [e for e in errors if e.startswith('Column')]
        box_errors = [e for e in errors if e.startswith('Box')]

        report['row_errors'] = len(row_errors)
        report['column_errors'] = len(col_errors)
        report['box_errors'] = len(box_errors)

        return report

    @staticmethod
    def compare_grids(original: np.ndarray, solved: np.ndarray, verbose: bool = False) -> Dict:
        """
        Compare original puzzle with solved solution.

        Args:
            original: Original puzzle (0 = empty)
            solved: Solved puzzle
            verbose: Print comparison details

        Returns:
            Dictionary with comparison statistics
        """
        # Check if solved overwrote any original values
        overwrites = []
        for row in range(9):
            for col in range(9):
                if original[row, col] != 0 and original[row, col] != solved[row, col]:
                    overwrites.append({
                        'position': (row, col),
                        'original': int(original[row, col]),
                        'solved': int(solved[row, col])
                    })

        # Count cells filled by solver
        filled_by_solver = np.sum((original == 0) & (solved != 0))

        comparison = {
            'overwrites': overwrites,
            'overwrite_count': len(overwrites),
            'cells_filled_by_solver': int(filled_by_solver),
            'original_filled': int(np.sum(original != 0)),
            'solved_filled': int(np.sum(solved != 0)),
        }

        if verbose:
            print("\n" + "="*60)
            print("GRID COMPARISON")
            print("="*60)
            print(f"Original cells filled: {comparison['original_filled']}")
            print(f"Solved cells filled:   {comparison['solved_filled']}")
            print(f"Cells filled by solver: {comparison['cells_filled_by_solver']}")

            if overwrites:
                print(f"\n❌ WARNING: {len(overwrites)} original values were overwritten!")
                for ow in overwrites:
                    pos = ow['position']
                    print(f"  Cell ({pos[0]},{pos[1]}): {ow['original']} → {ow['solved']}")
            else:
                print("\n✅ No original values overwritten")

        return comparison


def print_validation_report(report: Dict):
    """Pretty print validation report."""
    print("\n" + "="*60)
    print("SUDOKU VALIDATION REPORT")
    print("="*60)

    print(f"\nCompletion: {report['completion']:.1f}% ({report['filled_cells']}/81 cells)")

    if report['is_valid']:
        print("\n✅ SOLUTION IS VALID!")
    else:
        print(f"\n❌ SOLUTION IS INVALID ({report['total_errors']} errors)")

        if report['row_errors'] > 0:
            print(f"  • Row errors: {report['row_errors']}")
        if report['column_errors'] > 0:
            print(f"  • Column errors: {report['column_errors']}")
        if report['box_errors'] > 0:
            print(f"  • Box errors: {report['box_errors']}")

        if report['errors']:
            print("\nErrors:")
            for error in report['errors'][:10]:  # Show first 10
                print(f"  - {error}")
            if len(report['errors']) > 10:
                print(f"  ... and {len(report['errors']) - 10} more")

    print("="*60)
