"""OCR error correction module for Sudoku puzzles."""

import numpy as np
from typing import List, Tuple, Optional
from .solver import SudokuSolver


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

        For duplicate violations, only add the cell with LOWER confidence (more suspicious).

        Args:
            grid: 9x9 puzzle grid
            confidence_matrix: 9x9 confidence scores
            violations: List of constraint violations

        Returns:
            List of (row, col, confidence) tuples, sorted by suspicion (lowest confidence first)
        """
        suspect_cells = set()

        # Collect cells involved in violations, prioritizing low confidence
        for violation_type, index, duplicate_values, cell_positions in violations:
            # For each duplicate value, only mark the cell(s) with lowest confidence as suspect
            # Group cells by their value
            cells_by_value = {}
            for row, col in cell_positions:
                val = grid[row, col]
                if val not in cells_by_value:
                    cells_by_value[val] = []
                cells_by_value[val].append((row, col, confidence_matrix[row, col]))

            # For each value that appears multiple times, mark the lower-confidence ones as suspect
            for val, cells in cells_by_value.items():
                if len(cells) > 1:
                    # Sort by confidence (lowest first)
                    cells_sorted = sorted(cells, key=lambda x: x[2])
                    # Add only the lowest confidence cell(s) - keep the highest confidence one
                    # If confidences are similar, add all but the highest
                    highest_conf = cells_sorted[-1][2]
                    for row, col, conf in cells_sorted[:-1]:
                        # Only add if significantly lower confidence OR if similar confidence
                        if conf < highest_conf or (highest_conf - conf) < 0.1:
                            suspect_cells.add((row, col, conf))
                    # If all confidences are very similar, also add the "highest" as it might be wrong too
                    if len(cells_sorted) > 1 and (cells_sorted[-1][2] - cells_sorted[0][2]) < 0.15:
                        suspect_cells.add((cells_sorted[-1][0], cells_sorted[-1][1], cells_sorted[-1][2]))

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
        verbose: bool = False,
        low_confidence_threshold: float = 0.7
    ) -> Tuple[bool, np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Attempt to correct multiple OCR errors using iterative refinement.

        This method tries to correct errors one by one, rechecking violations after each correction.
        After resolving constraint violations, also checks low-confidence cells even if puzzle is valid.

        Args:
            grid: 9x9 puzzle grid with suspected OCR errors
            confidence_matrix: 9x9 confidence scores
            has_content: Optional 9x9 boolean array indicating cells with visual content
            verbose: If True, print correction attempts
            low_confidence_threshold: Confidence threshold for checking cells (default: 0.7)

        Returns:
            Tuple of (success, corrected_grid, corrections_made)
        """
        current_grid = grid.copy()
        all_corrections = []
        iterations = 0
        max_iterations = 10  # Increased from 5
        failed_attempts = set()  # Track cells that didn't lead to improvement

        while iterations < max_iterations:
            iterations += 1

            if verbose:
                print(f"\n{'='*70}")
                print(f"Correction Iteration {iterations}")
                print(f"{'='*70}")

            # Check current state
            solver = SudokuSolver()
            solver.load_puzzle(current_grid)

            # If valid and solvable, check low-confidence cells before returning
            if solver.is_valid_puzzle() and solver.solve():
                if verbose:
                    print(f"\n✓ Puzzle valid and solvable after {iterations} iteration(s)")
                    print(f"  Corrections so far: {len(all_corrections)}")

                # Break out of constraint violation loop, proceed to Phase 2
                baseline_solution = solver.get_solution()
                break

            # Find violations
            violations = solver.find_constraint_violations()

            if not violations:
                if verbose:
                    print("No more violations but puzzle still not solvable")
                break

            # Identify suspect cells for this iteration
            suspect_cells = self.identify_suspect_cells(current_grid, confidence_matrix, violations)

            # Filter out cells we've already failed to correct
            suspect_cells = [(r, c, conf) for r, c, conf in suspect_cells if (r, c) not in failed_attempts]

            if not suspect_cells:
                if verbose:
                    print("No more suspect cells to try (all previously attempted)")
                break

            # Try to correct suspect cells in order of confidence
            correction_made = False
            for row, col, confidence in suspect_cells:
                old_value = current_grid[row, col]

                alternatives = self.suggest_alternatives(old_value, confidence)

                if has_content is not None and has_content[row, col]:
                    alternatives = [a for a in alternatives if a != 0]
                elif old_value == 0:
                    alternatives = list(range(1, 10))

                if verbose:
                    print(f"\nAttempting to correct ({row},{col}): value={old_value}, conf={confidence:.2f}")
                    print(f"  Alternatives: {alternatives}")

                # Try each alternative
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
                            print(f"  ✓ Corrected ({row},{col}): {old_value} → {new_value} (reduced violations)")
                        current_grid = test_grid
                        all_corrections.append((row, col, old_value, new_value))
                        correction_made = True
                        # Clear failed attempts since grid changed
                        failed_attempts.clear()
                        break

                if correction_made:
                    break

                # This cell didn't help, mark it as failed
                failed_attempts.add((row, col))
                if verbose:
                    print(f"  No improvement from correcting ({row},{col})")

            if not correction_made:
                if verbose:
                    print("\nNo corrections made in this iteration")
                break

        # Phase 2: Check low-confidence cells even if puzzle seems valid
        # Get baseline solution first (may already be set from while loop break)
        if 'baseline_solution' not in locals():
            solver = SudokuSolver()
            solver.load_puzzle(current_grid)
            if solver.is_valid_puzzle() and solver.solve():
                baseline_solution = solver.get_solution()
            else:
                # If still not solvable, return failure
                return False, current_grid, all_corrections

        if verbose:
            print(f"\n{'='*70}")
            print("Phase 2: Checking low-confidence cells")
            print(f"{'='*70}")

        # Find all cells with low confidence OR easily confused digits (excluding already corrected cells)
        low_conf_cells = []
        corrected_positions = {(r, c) for r, c, _, _ in all_corrections}

        # Highly confusable digit pairs that should always be checked
        high_confusion_digits = {6, 8, 9}  # These are often confused with each other

        for row in range(9):
            for col in range(9):
                if (row, col) not in corrected_positions and current_grid[row, col] != 0:
                    val = current_grid[row, col]
                    conf = confidence_matrix[row, col]

                    # Include if: low confidence OR high-confusion digit with medium-low confidence
                    if conf < low_confidence_threshold or (val in high_confusion_digits and conf < 0.9):
                        low_conf_cells.append((row, col, conf))

        # Sort by confidence (lowest first)
        low_conf_cells.sort(key=lambda x: x[2])

        if verbose:
            if low_conf_cells:
                print(f"\nFound {len(low_conf_cells)} low-confidence cells (threshold < {low_confidence_threshold}):")
                for row, col, conf in low_conf_cells[:10]:  # Show first 10
                    print(f"  ({row},{col}): value={current_grid[row,col]}, confidence={conf:.2f}")
            else:
                print(f"\nNo low-confidence cells found (all above threshold {low_confidence_threshold})")

        # Try correcting low-confidence cells
        # Strategy: If a low-confidence cell has an alternative from confusion matrix
        # that ALSO gives a valid solution, prefer that alternative
        for row, col, confidence in low_conf_cells:
            old_value = current_grid[row, col]

            # Get alternatives from confusion matrix (visual similarity)
            alternatives = self.suggest_alternatives(old_value, confidence)

            if has_content is not None and has_content[row, col]:
                alternatives = [a for a in alternatives if a != 0]

            if not alternatives:
                continue

            if verbose:
                print(f"\nChecking low-confidence cell ({row},{col}): value={old_value}, conf={confidence:.2f}, alternatives={alternatives}")

            # Try each alternative
            found_better = False
            for new_value in alternatives:
                if new_value == old_value:
                    continue

                test_grid = current_grid.copy()
                test_grid[row, col] = new_value

                solver = SudokuSolver()
                solver.load_puzzle(test_grid)

                if solver.is_valid_puzzle() and solver.solve():
                    # Alternative also gives valid solution!
                    # Since confidence is low AND alternative is from confusion matrix,
                    # the alternative is likely more correct
                    if verbose:
                        print(f"  ✓ Alternative {new_value} also gives valid solution (confusion: {old_value}↔{new_value})")
                        print(f"    Given low confidence ({confidence:.2f}), preferring alternative")

                    # Use alternative
                    current_grid = test_grid
                    baseline_solution = solver.get_solution()
                    all_corrections.append((row, col, old_value, new_value))
                    found_better = True
                    break
                else:
                    if verbose:
                        print(f"  ✗ Alternative {new_value} does not give valid solution")

            if found_better and verbose:
                print(f"  → Corrected ({row},{col}): {old_value} → {all_corrections[-1][3]}")

        # Final check with corrections
        solver = SudokuSolver()
        solver.load_puzzle(current_grid)
        if solver.is_valid_puzzle() and solver.solve():
            return True, solver.get_solution(), all_corrections

        return False, current_grid, all_corrections
