"""
Text file parser for Sudoku puzzles.

Supports text format where:
- 0 = empty cell
- 1-9 = digit
- Spaces separate groups of 3
- Empty lines separate 3x3 blocks
"""

import numpy as np
from typing import Optional


def parse_sudoku_text(file_path: str) -> Optional[np.ndarray]:
    """
    Parse a Sudoku puzzle from a text file.

    Expected format:
        016 000 070
        000 096 100
        204 800 090

        700 080 000
        030 070 000
        000 300 400

        000 000 000
        000 769 302
        001 003 009

    Where:
        - 0 = empty cell
        - 1-9 = filled cell
        - Spaces separate groups of 3
        - Empty lines separate 3x3 blocks

    Args:
        file_path: Path to text file

    Returns:
        9x9 numpy array or None if parsing fails
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Remove trailing whitespace and filter out empty lines for processing
        lines = [line.rstrip() for line in lines]

        # Extract only lines with digits
        digit_lines = []
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            # Extract digits (ignore spaces)
            digits = ''.join(c for c in line if c.isdigit())
            if digits:
                digit_lines.append(digits)

        # Should have exactly 9 lines with digits
        if len(digit_lines) != 9:
            print(f"Error: Expected 9 rows, found {len(digit_lines)}")
            return None

        # Parse into 9x9 grid
        grid = np.zeros((9, 9), dtype=int)

        for row_idx, line in enumerate(digit_lines):
            if len(line) != 9:
                print(f"Error: Row {row_idx + 1} has {len(line)} digits, expected 9")
                return None

            for col_idx, char in enumerate(line):
                digit = int(char)
                if digit < 0 or digit > 9:
                    print(f"Error: Invalid digit '{digit}' at row {row_idx + 1}, col {col_idx + 1}")
                    return None
                grid[row_idx, col_idx] = digit

        return grid

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error parsing text file: {e}")
        return None


def write_sudoku_text(grid: np.ndarray, output_path: str, original: Optional[np.ndarray] = None):
    """
    Write a Sudoku solution to a text file.

    Args:
        grid: 9x9 numpy array with solution
        output_path: Path to output file
        original: Optional original puzzle to mark original vs solved digits
    """
    with open(output_path, 'w') as f:
        for row in range(9):
            # Add blank line between 3x3 blocks
            if row > 0 and row % 3 == 0:
                f.write('\n')

            line_parts = []
            for block in range(3):
                # Get 3 digits for this block
                start_col = block * 3
                digits = []
                for col in range(start_col, start_col + 3):
                    digit = grid[row, col]

                    # Mark solution digits differently if we have original
                    if original is not None and original[row, col] == 0:
                        # Solution digit (was empty in original)
                        digits.append(f"[{digit}]" if digit != 0 else "[ ]")
                    else:
                        # Original digit
                        digits.append(str(digit))

                line_parts.append(''.join(digits))

            f.write(' '.join(line_parts) + '\n')


def format_sudoku_text(grid: np.ndarray, original: Optional[np.ndarray] = None) -> str:
    """
    Format a Sudoku grid as a nice text representation.

    Args:
        grid: 9x9 numpy array
        original: Optional original puzzle to mark solved digits

    Returns:
        Formatted string
    """
    lines = []
    lines.append("┌─────────┬─────────┬─────────┐")

    for row in range(9):
        if row > 0 and row % 3 == 0:
            lines.append("├─────────┼─────────┼─────────┤")

        row_str = "│"
        for col in range(9):
            if col > 0 and col % 3 == 0:
                row_str += "│"

            digit = grid[row, col]

            # Mark solution digits
            if original is not None and original[row, col] == 0 and digit != 0:
                # Solved digit
                row_str += f" {digit}"
            else:
                # Original digit or empty
                row_str += f" {digit if digit != 0 else '.'}"

        row_str += " │"
        lines.append(row_str)

    lines.append("└─────────┴─────────┴─────────┘")

    return '\n'.join(lines)


if __name__ == "__main__":
    # Test the parser
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        grid = parse_sudoku_text(file_path)

        if grid is not None:
            print("Successfully parsed:")
            print(format_sudoku_text(grid))
            print(f"\nFilled cells: {np.count_nonzero(grid)}/81")
        else:
            print("Failed to parse file")
    else:
        print("Usage: python text_parser.py <sudoku_file.txt>")
