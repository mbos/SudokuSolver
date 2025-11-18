"""OCR error correction module for Sudoku puzzles."""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from .solver import SudokuSolver
import copy


class OCRErrorCorrector:
    """
    Corrects OCR errors in Sudoku puzzles by identifying constraint violations
    and using confidence scores to determine which cells to re-interpret.
    """

    def __init__(self, max_corrections: int = 10, max_attempts: int = 100):
        """
        Initialize the OCR error corrector.

        Args:
            max_corrections: Maximum number of cells to correct
            max_attempts: Maximum correction attempts to prevent infinite loops
        """
        self.max_corrections = max_corrections
        self.max_attempts = max_attempts

    def suggest_alternatives(self, detected_digit: int, confidence: float) -> List[int]:
        """
        Suggest alternative digit interpretations based on common OCR confusions.

        Args:
            detected_digit: The digit that was detected (or 0 for empty)
            confidence: Confidence score for the detection

        Returns:
            List of alternative digits to try, ordered by likelihood
        """
        # Common OCR confusions based on visual similarity
        confusion_matrix = {
            0: [8, 6],           # Empty confused with 8 or 6
            1: [7, 4],           # 1 confused with 7 or 4
            2: [7],              # 2 confused with 7
            3: [8, 5],           # 3 confused with 8 or 5
            4: [9, 1],           # 4 confused with 9 or 1
            5: [6, 3],           # 5 confused with 6 or 3
            6: [8, 5, 0],        # 6 confused with 8, 5, or empty
            7: [1, 2],           # 7 confused with 1 or 2
            8: [6, 3, 0],        # 8 confused with 6, 3, or empty
            9: [4, 7],           # 9 confused with 4 or 7
        }

        alternatives = confusion_matrix.get(detected_digit, [])

        # If confidence is very low, consider all digits
        if confidence < 0.4:
            # Try all digits except the detected one
            alternatives = [d for d in range(1, 10) if d != detected_digit]

        return alternatives

    def identify_suspect_cells(
        self,
        grid: np.ndarray,
        confidence_matrix: np.ndarray,
        violations: List[Tuple[str, int, List[int], List[Tuple[int, int]]]]
    ) -> List[Tuple[int, int, float]]:
        """
        Identify cells most likely to contain OCR errors based on violations and confidence.

        Args:
            grid: 9x9 puzzle grid
            confidence_matrix: 9x9 confidence scores
            violations: List of constraint violations

        Returns:
            List of (row, col, confidence) tuples, sorted by suspicion (lowest confidence first)
        """
        suspect_cells = set()

        # Collect all cells involved in violations
        for violation_type, index, duplicate_values, cell_positions in violations:
            for row, col in cell_positions:
                suspect_cells.add((row, col, confidence_matrix[row, col]))

        # Sort by confidence (lowest first - most suspicious)
        return sorted(suspect_cells, key=lambda x: x[2])

    def try_correction(
        self,
        grid: np.ndarray,
        row: int,
        col: int,
        new_value: int
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Try correcting a cell and check if it makes the puzzle valid and solvable.

        Args:
            grid: 9x9 puzzle grid
            row: Row of cell to correct
            col: Column of cell to correct
            new_value: New value to try (0 for empty, 1-9 for digits)

        Returns:
            Tuple of (success, corrected_grid or None)
        """
        # Create a copy with the correction
        corrected_grid = grid.copy()
        corrected_grid[row, col] = new_value

        # Check if puzzle is now valid
        solver = SudokuSolver()
        solver.load_puzzle(corrected_grid)

        if not solver.is_valid_puzzle():
            return False, None

        # Try to solve it
        if solver.solve():
            return True, corrected_grid
        else:
            return False, None

    def correct_errors(
        self,
        grid: np.ndarray,
        confidence_matrix: np.ndarray,
        has_content: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[bool, np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Attempt to correct OCR errors in the puzzle.

        Args:
            grid: 9x9 puzzle grid with suspected OCR errors
            confidence_matrix: 9x9 confidence scores
            has_content: Optional 9x9 boolean array indicating cells with visual content
            verbose: If True, print correction attempts

        Returns:
            Tuple of (success, corrected_grid, corrections_made)
            - success: True if puzzle was successfully corrected and solved
            - corrected_grid: The corrected grid (may be partially corrected even if not solved)
            - corrections_made: List of (row, col, old_value, new_value) tuples
        """
        # First check if puzzle is already valid and solvable
        solver = SudokuSolver()
        solver.load_puzzle(grid)

        if solver.is_valid_puzzle() and solver.solve():
            if verbose:
                print("Puzzle is already valid and solvable, no corrections needed")
            return True, grid.copy(), []

        # Find violations
        violations = solver.find_constraint_violations()

        if not violations:
            # Puzzle is valid but not solvable - might be underspecified
            if verbose:
                print("Puzzle has no constraint violations but is not solvable")
            return False, grid.copy(), []

        if verbose:
            print(f"\nFound {len(violations)} constraint violation(s):")
            for vtype, idx, dups, cells in violations:
                print(f"  {vtype.capitalize()} {idx}: Duplicate {dups} at cells {cells}")

        # Identify suspect cells
        suspect_cells = self.identify_suspect_cells(grid, confidence_matrix, violations)

        if verbose:
            print(f"\nIdentified {len(suspect_cells)} suspect cell(s):")
            for row, col, conf in suspect_cells:
                print(f"  ({row},{col}): value={grid[row,col]}, confidence={conf:.2f}")

        # Try to correct errors
        corrections_made = []
        current_grid = grid.copy()
        attempts = 0

        for row, col, confidence in suspect_cells:
            if len(corrections_made) >= self.max_corrections:
                break

            old_value = current_grid[row, col]

            # Get alternative interpretations
            alternatives = self.suggest_alternatives(old_value, confidence)

            # If cell has visual content, don't try empty (0)
            if has_content is not None and has_content[row, col]:
                alternatives = [a for a in alternatives if a != 0]
            # If cell appears empty in OCR, also try empty
            elif old_value == 0:
                alternatives = list(range(1, 10))

            if verbose:
                print(f"\nTrying alternatives for ({row},{col}) [current={old_value}, conf={confidence:.2f}]:")
                print(f"  Alternatives: {alternatives}")

            # Try each alternative
            for new_value in alternatives:
                attempts += 1
                if attempts > self.max_attempts:
                    if verbose:
                        print(f"  Reached maximum attempts ({self.max_attempts})")
                    break

                success, corrected_grid = self.try_correction(current_grid, row, col, new_value)

                if success:
                    if verbose:
                        print(f"  ✓ SUCCESS: Changed ({row},{col}) from {old_value} to {new_value}")
                    corrections_made.append((row, col, old_value, new_value))
                    current_grid = corrected_grid
                    return True, corrected_grid, corrections_made

            if verbose:
                print(f"  No working alternative found for ({row},{col})")

        if verbose:
            print(f"\nCorrected {len(corrections_made)} cell(s) but puzzle still not solvable")

        return False, current_grid, corrections_made

    def correct_multiple_errors(
        self,
        grid: np.ndarray,
        confidence_matrix: np.ndarray,
        has_content: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[bool, np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Attempt to correct multiple OCR errors using iterative refinement.

        This method tries to correct errors one by one, rechecking violations after each correction.

        Args:
            grid: 9x9 puzzle grid with suspected OCR errors
            confidence_matrix: 9x9 confidence scores
            has_content: Optional 9x9 boolean array indicating cells with visual content
            verbose: If True, print correction attempts

        Returns:
            Tuple of (success, corrected_grid, corrections_made)
        """
        current_grid = grid.copy()
        all_corrections = []
        iterations = 0
        max_iterations = 5

        while iterations < max_iterations:
            iterations += 1

            if verbose:
                print(f"\n{'='*70}")
                print(f"Correction Iteration {iterations}")
                print(f"{'='*70}")

            # Check current state
            solver = SudokuSolver()
            solver.load_puzzle(current_grid)

            # If valid and solvable, we're done
            if solver.is_valid_puzzle() and solver.solve():
                if verbose:
                    print(f"\n✓ Puzzle successfully corrected after {iterations} iteration(s)")
                    print(f"  Total corrections: {len(all_corrections)}")
                return True, solver.get_solution(), all_corrections

            # Find violations
            violations = solver.find_constraint_violations()

            if not violations:
                if verbose:
                    print("No more violations but puzzle still not solvable")
                break

            # Identify suspect cells for this iteration
            suspect_cells = self.identify_suspect_cells(current_grid, confidence_matrix, violations)

            if not suspect_cells:
                if verbose:
                    print("No suspect cells identified")
                break

            # Try to correct the most suspicious cell
            row, col, confidence = suspect_cells[0]
            old_value = current_grid[row, col]

            alternatives = self.suggest_alternatives(old_value, confidence)

            if has_content is not None and has_content[row, col]:
                alternatives = [a for a in alternatives if a != 0]
            elif old_value == 0:
                alternatives = list(range(1, 10))

            if verbose:
                print(f"\nAttempting to correct ({row},{col}): value={old_value}, conf={confidence:.2f}")

            # Try each alternative
            correction_made = False
            for new_value in alternatives:
                test_grid = current_grid.copy()
                test_grid[row, col] = new_value

                solver = SudokuSolver()
                solver.load_puzzle(test_grid)

                # Check if this improves the situation (valid or reduces violations)
                test_violations = solver.find_constraint_violations()

                if len(test_violations) < len(violations):
                    # Improvement! Use this correction
                    if verbose:
                        print(f"  Corrected ({row},{col}): {old_value} → {new_value} (reduced violations)")
                    current_grid = test_grid
                    all_corrections.append((row, col, old_value, new_value))
                    correction_made = True
                    break

            if not correction_made:
                if verbose:
                    print(f"  Could not improve cell ({row},{col})")
                # Try next suspect cell
                if len(suspect_cells) > 1:
                    # Move to next iteration to re-evaluate
                    continue
                else:
                    break

        # Final check
        solver = SudokuSolver()
        solver.load_puzzle(current_grid)
        if solver.is_valid_puzzle() and solver.solve():
            return True, solver.get_solution(), all_corrections

        return False, current_grid, all_corrections
