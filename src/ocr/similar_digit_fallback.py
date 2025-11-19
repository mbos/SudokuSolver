"""
Similar-Digit Fallback Recognizer

This module implements a fallback strategy for OCR errors by trying visually similar digits
when confidence is low or when Sudoku constraints are violated.

Strategy: When a digit is recognized with low confidence or creates conflicts,
try alternative digits that look similar (e.g., 6/8/9, 1/7) and check if they:
1. Have reasonable confidence in the model
2. Don't violate Sudoku constraints
3. Potentially resolve existing conflicts
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


# Visual similarity matrix: for each digit, list similar-looking digits in order of similarity
DIGIT_SIMILARITY = {
    0: [6, 8, 9],           # Round shapes
    1: [7, 4],              # Vertical lines
    2: [3, 7],              # Curved tops
    3: [8, 5, 2],           # Stacked curves
    4: [9, 1],              # Angled lines
    5: [6, 8, 3],           # Top line with curve
    6: [8, 9, 5, 0],        # Closed loops, most confusion
    7: [1, 9, 4],           # Diagonal/vertical
    8: [6, 9, 3, 0],        # Double loops, most confusion
    9: [8, 6, 4, 7],        # Open/closed top, most confusion
}


class SimilarDigitFallbackRecognizer:
    """
    Recognizer that tries visually similar digits when initial recognition
    has low confidence or creates Sudoku constraint violations.
    """

    def __init__(self,
                 base_recognizer,
                 confidence_threshold: float = 0.70,
                 enable_constraint_checking: bool = True,
                 similarity_matrix: Optional[Dict[int, List[int]]] = None):
        """
        Initialize the fallback recognizer.

        Args:
            base_recognizer: Base OCR recognizer (CNN, Tesseract, or Ensemble)
            confidence_threshold: Trigger fallback if confidence below this
            enable_constraint_checking: Use Sudoku constraints to guide decisions
            similarity_matrix: Custom similarity matrix (default: DIGIT_SIMILARITY)
        """
        self.base = base_recognizer
        self.confidence_threshold = confidence_threshold
        self.enable_constraint_checking = enable_constraint_checking
        self.similarity_matrix = similarity_matrix or DIGIT_SIMILARITY

        self.stats = {
            'total_fallbacks': 0,
            'successful_corrections': 0,
            'constraint_violations_fixed': 0,
            'low_confidence_retries': 0,
        }

    def recognize_digit_with_fallback(self,
                                     cell: np.ndarray,
                                     row: int,
                                     col: int,
                                     current_puzzle: np.ndarray,
                                     verbose: bool = False) -> Tuple[int, float, str]:
        """
        Recognize digit with similar-digit fallback strategy.

        Args:
            cell: Cell image
            row: Row position in puzzle (0-8)
            col: Column position in puzzle (0-8)
            current_puzzle: Current state of puzzle (9x9 array)
            verbose: Print debug information

        Returns:
            Tuple of (digit, confidence, reasoning)
        """
        # Get base prediction
        if hasattr(self.base, 'recognize_digit'):
            base_digit = self.base.recognize_digit(cell)
            # Try to extract confidence if available
            base_confidence = self._get_base_confidence(cell, base_digit)
        else:
            # Fallback for simpler recognizers
            base_digit = self.base.recognize_digit(cell)
            base_confidence = 0.8  # Assume reasonable confidence

        if verbose:
            print(f"  Cell ({row},{col}): Base prediction = {base_digit}, "
                  f"confidence = {base_confidence:.2f}")

        # Check if we should try fallback
        should_retry = False
        reason = ""

        # Reason 1: Low confidence
        if base_confidence < self.confidence_threshold:
            should_retry = True
            reason = f"low confidence ({base_confidence:.2f})"
            self.stats['low_confidence_retries'] += 1

        # Reason 2: Creates conflict
        if self.enable_constraint_checking and base_digit != 0:
            if self._creates_conflict(base_digit, row, col, current_puzzle):
                should_retry = True
                reason = f"constraint violation (digit {base_digit})"

        if not should_retry:
            return base_digit, base_confidence, "base_prediction"

        if verbose:
            print(f"    -> Triggering fallback: {reason}")

        # Try fallback
        self.stats['total_fallbacks'] += 1
        fallback_result = self._try_similar_digits(
            cell, base_digit, row, col, current_puzzle, verbose
        )

        if fallback_result is not None:
            digit, confidence, fallback_reason = fallback_result
            if digit != base_digit:
                self.stats['successful_corrections'] += 1
                if self._creates_conflict(base_digit, row, col, current_puzzle):
                    self.stats['constraint_violations_fixed'] += 1

            if verbose:
                print(f"    -> Fallback result: {digit} "
                      f"(confidence: {confidence:.2f}, reason: {fallback_reason})")

            return digit, confidence, f"fallback_{fallback_reason}"

        # No better alternative found
        if verbose:
            print(f"    -> No better alternative, keeping {base_digit}")

        return base_digit, base_confidence, "base_prediction_no_fallback"

    def _try_similar_digits(self,
                           cell: np.ndarray,
                           base_digit: int,
                           row: int,
                           col: int,
                           puzzle: np.ndarray,
                           verbose: bool = False) -> Optional[Tuple[int, float, str]]:
        """
        Try recognizing as similar-looking digits.

        Args:
            cell: Cell image
            base_digit: Originally predicted digit
            row, col: Position in puzzle
            puzzle: Current puzzle state
            verbose: Debug output

        Returns:
            Tuple of (digit, confidence, reason) or None if no better alternative
        """
        # Get list of similar digits to try
        candidates = self.similarity_matrix.get(base_digit, [])

        if not candidates:
            return None

        # Score each candidate
        scores = {}
        reasons = {}

        for candidate_digit in candidates:
            # Skip if would create conflict
            if self.enable_constraint_checking:
                if self._creates_conflict(candidate_digit, row, col, puzzle):
                    if verbose:
                        print(f"      Candidate {candidate_digit}: creates conflict, skip")
                    continue

            # Get model's confidence for this specific digit
            likelihood = self._get_digit_likelihood(cell, candidate_digit)

            if likelihood < 0.3:  # Too low to consider
                if verbose:
                    print(f"      Candidate {candidate_digit}: "
                          f"likelihood too low ({likelihood:.2f}), skip")
                continue

            # Base score is the likelihood
            score = likelihood
            reason_parts = [f"likelihood={likelihood:.2f}"]

            # Bonus: Resolves existing conflicts elsewhere in puzzle
            if self.enable_constraint_checking:
                if self._would_help_resolve_conflicts(candidate_digit, puzzle):
                    score *= 1.15
                    reason_parts.append("helps_resolve_conflicts")

            # Bonus: Much more confident than base prediction
            base_likelihood = self._get_digit_likelihood(cell, base_digit)
            if likelihood > base_likelihood * 1.3:
                score *= 1.1
                reason_parts.append("much_more_confident")

            scores[candidate_digit] = score
            reasons[candidate_digit] = ",".join(reason_parts)

            if verbose:
                print(f"      Candidate {candidate_digit}: score={score:.3f} "
                      f"({reasons[candidate_digit]})")

        if not scores:
            return None

        # Return highest scoring candidate
        best_digit = max(scores, key=scores.get)
        best_score = scores[best_digit]

        # Need minimum threshold to accept fallback
        if best_score < 0.4:
            return None

        return best_digit, best_score, reasons[best_digit]

    def _get_digit_likelihood(self, cell: np.ndarray, target_digit: int) -> float:
        """
        Get the model's confidence that this cell contains target_digit.

        Args:
            cell: Cell image
            target_digit: Digit to check likelihood for

        Returns:
            Confidence score (0.0-1.0)
        """
        # Try to extract from CNN model
        if hasattr(self.base, 'model') and self.base.model is not None:
            try:
                preprocessed, is_empty = self.base.preprocess_cell(cell)
                if is_empty:
                    return 0.0

                resized = self.base.resize_to_mnist_format(preprocessed)
                normalized = resized.astype(np.float32) / 255.0
                input_data = normalized.reshape(1, 28, 28, 1)

                predictions = self.base.model.predict(input_data, verbose=0)
                return float(predictions[0][target_digit])
            except Exception:
                # Fallback to heuristics
                pass

        # For ensemble or other recognizers, use heuristics
        # This is a simplified approach - could be improved
        # by caching predictions from the initial recognition

        # Basic heuristic: if target is similar to base prediction, give it some credit
        base_digit = self.base.recognize_digit(cell)
        if target_digit == base_digit:
            return 0.7
        elif target_digit in self.similarity_matrix.get(base_digit, []):
            # Similar digit gets moderate score
            similarity_index = self.similarity_matrix[base_digit].index(target_digit)
            return 0.6 - (similarity_index * 0.1)
        else:
            return 0.2

    def _get_base_confidence(self, cell: np.ndarray, digit: int) -> float:
        """Extract confidence from base recognizer if possible."""
        # This is a simplified version - would need to be adapted
        # based on the actual recognizer being used
        return 0.75  # Default moderate confidence

    def _creates_conflict(self, digit: int, row: int, col: int, puzzle: np.ndarray) -> bool:
        """
        Check if placing digit at (row, col) creates a Sudoku violation.

        Args:
            digit: Digit to check
            row, col: Position in puzzle
            puzzle: Current puzzle state (9x9 array)

        Returns:
            True if creates conflict, False otherwise
        """
        if digit == 0:
            return False

        # Check row
        for c in range(9):
            if c != col and puzzle[row, c] == digit:
                return True

        # Check column
        for r in range(9):
            if r != row and puzzle[r, col] == digit:
                return True

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col) and puzzle[r, c] == digit:
                    return True

        return False

    def _would_help_resolve_conflicts(self, digit: int, puzzle: np.ndarray) -> bool:
        """
        Check if this digit choice might help resolve conflicts elsewhere.

        This is a heuristic bonus for digits that don't appear much in the puzzle yet.
        """
        if digit == 0:
            return False

        # Count how many times this digit appears
        count = np.sum(puzzle == digit)

        # Digits that appear less frequently might help fill gaps
        return count < 3  # Arbitrary threshold

    def refine_grid(self,
                   cells: List[np.ndarray],
                   initial_grid: np.ndarray,
                   has_content: np.ndarray,
                   verbose: bool = False) -> np.ndarray:
        """
        Refine an initially recognized grid using similar-digit fallback.

        Args:
            cells: List of 81 cell images
            initial_grid: Initial OCR recognition result (9x9)
            has_content: Boolean mask of cells with content (9x9)
            verbose: Print debug information

        Returns:
            Refined grid (9x9 array)
        """
        print("\n[Similar-Digit Fallback] Refining grid...")

        refined_grid = initial_grid.copy()

        # First pass: Fix cells that create obvious conflicts
        for row in range(9):
            for col in range(9):
                if not has_content[row, col]:
                    continue

                digit = refined_grid[row, col]

                # Skip empty cells
                if digit == 0:
                    continue

                # Check if creates conflict
                if self._creates_conflict(digit, row, col, refined_grid):
                    idx = row * 9 + col
                    cell = cells[idx]

                    if verbose:
                        print(f"  Conflict at ({row},{col}): digit {digit}")

                    # Temporarily set to 0 to avoid self-conflict
                    refined_grid[row, col] = 0

                    new_digit, confidence, reason = self.recognize_digit_with_fallback(
                        cell, row, col, refined_grid, verbose=verbose
                    )

                    refined_grid[row, col] = new_digit

                    if new_digit != digit:
                        print(f"    -> Corrected: {digit} → {new_digit} "
                              f"(confidence: {confidence:.2f}, {reason})")

        # Second pass: Try to recognize previously unrecognized cells
        for row in range(9):
            for col in range(9):
                if has_content[row, col] and refined_grid[row, col] == 0:
                    idx = row * 9 + col
                    cell = cells[idx]

                    if verbose:
                        print(f"  Unrecognized cell at ({row},{col})")

                    new_digit, confidence, reason = self.recognize_digit_with_fallback(
                        cell, row, col, refined_grid, verbose=verbose
                    )

                    if new_digit != 0:
                        refined_grid[row, col] = new_digit
                        print(f"    -> Recognized: 0 → {new_digit} "
                              f"(confidence: {confidence:.2f}, {reason})")

        self._print_stats()

        return refined_grid

    def _print_stats(self):
        """Print statistics about fallback performance."""
        print("\n[Fallback Stats]")
        print(f"  Total fallback attempts: {self.stats['total_fallbacks']}")
        print(f"  Successful corrections: {self.stats['successful_corrections']}")
        print(f"  Constraint violations fixed: {self.stats['constraint_violations_fixed']}")
        print(f"  Low confidence retries: {self.stats['low_confidence_retries']}")
