#!/usr/bin/env python3
"""
Sudoku Solver - Main entry point

This script takes an image of a Sudoku puzzle, extracts the grid,
recognizes the digits using OCR, solves the puzzle, and outputs
an image with the solution filled in.
"""

import argparse
import sys
import os
import yaml
import numpy as np

from src.grid_detector import GridDetector
from src.ocr import DigitRecognizer
from src.ocr.ensemble_recognizer import EnsembleRecognizer
from src.solver import SudokuSolver
from src.image_generator import SolutionDrawer
from src.validator import SudokuValidator, print_validation_report
from src.training_data_collector import TrainingDataCollector, print_collection_summary
from src.text_parser import parse_sudoku_text, write_sudoku_text, format_sudoku_text
from src.error_corrector import OCRErrorCorrector


def solve_sudoku_from_image(
    image_path: str,
    output_path: str,
    model_path: str = "models/digit_cnn.h5",
    use_tesseract: bool = False,
    use_ensemble: bool = True,
    debug: bool = False,
    show_result: bool = False,
    verbose: bool = False,
    collect_training_data: bool = True
) -> bool:
    """
    Complete pipeline to solve Sudoku from an image.

    Args:
        image_path: Path to input image
        output_path: Path to save result image
        model_path: Path to CNN model for OCR
        use_tesseract: Use Tesseract instead of CNN
        use_ensemble: Use ensemble of multiple OCR models (default: True)
        debug: Show debug information (with image windows)
        show_result: Display result in window
        verbose: Show verbose ASCII output without image windows
        collect_training_data: Collect labeled samples for CNN training (default: True)

    Returns:
        True if successful, False otherwise
    """
    print(f"Processing image: {image_path}")
    print("=" * 50)

    # Step 1: Detect and extract grid
    print("\n[1/4] Detecting and extracting Sudoku grid...")
    detector = GridDetector(debug=debug)
    original_image, cells, warped_grid = detector.detect_and_extract(image_path)

    if cells is None:
        print("Error: Could not detect Sudoku grid in image")
        return False

    print(f"✓ Grid detected and extracted into {len(cells)} cells")

    # Step 2: Recognize digits using OCR
    print("\n[2/4] Recognizing digits using OCR...")
    if use_ensemble:
        print("Using ensemble of multiple OCR models...")
        # Load ensemble configuration from YAML file
        try:
            with open("config/ocr_config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Loaded OCR configuration from config/ocr_config.yaml")
        except FileNotFoundError:
            print("⚠️ Warning: config/ocr_config.yaml not found. Using default OCR configuration.")
            config = None
        except Exception as e:
            print(f"⚠️ Warning: Error loading config/ocr_config.yaml: {e}. Using default OCR configuration.")
            config = None

        ensemble = EnsembleRecognizer(config=config)
        detected_grid, has_content, confidence_matrix = ensemble.recognize_grid(cells, verbose=verbose)
    else:
        recognizer = DigitRecognizer(model_path=model_path, use_tesseract=use_tesseract)
        detected_grid, has_content, confidence_matrix = recognizer.recognize_grid(cells)

    # Display detected grid
    print("\nDetected puzzle:")
    print_grid(detected_grid)

    # Count detected digits
    filled_cells = np.count_nonzero(detected_grid)
    print(f"✓ Recognized {filled_cells} digits")

    # Count cells with content (might be more if OCR failed on some)
    content_cells = np.count_nonzero(has_content)
    if content_cells > filled_cells:
        print(f"  (Warning: {content_cells - filled_cells} cells have content but couldn't be recognized)")
        print("\n  Cells with content that weren't recognized:")
        for row in range(9):
            for col in range(9):
                if has_content[row, col] and detected_grid[row, col] == 0:
                    print(f"    - Cell ({row},{col})")

    # Step 3: Solve the puzzle
    print("\n[3/4] Solving Sudoku puzzle...")
    solver = SudokuSolver()
    solver.load_puzzle(detected_grid)

    # Try to solve directly first
    if solver.is_valid_puzzle() and solver.solve():
        solution = solver.get_solution()
        corrections_made = []
        print("\nSolved puzzle:")
        print_grid(solution)
        print("✓ Puzzle solved successfully!")
    else:
        # Puzzle is invalid or unsolvable - try error correction
        print("⚠️  Puzzle is invalid or unsolvable (likely due to OCR errors)")
        print("\n[3.5/4] Attempting OCR error correction...")

        corrector = OCRErrorCorrector(max_corrections=10, max_attempts=100)
        success, corrected_solution, corrections_made = corrector.correct_multiple_errors(
            detected_grid,
            confidence_matrix,
            has_content,
            verbose=verbose or debug,
            low_confidence_threshold=0.8  # Check cells with confidence < 0.8
        )

        if success:
            solution = corrected_solution
            print(f"\n✅ Puzzle successfully corrected and solved!")
            print(f"   Made {len(corrections_made)} correction(s):")
            for row, col, old_val, new_val in corrections_made:
                conf = confidence_matrix[row, col]
                print(f"   - Cell ({row},{col}): {old_val} → {new_val} (confidence was {conf:.2f})")

            print("\nCorrected and solved puzzle:")
            print_grid(solution)
        else:
            print("\n❌ Error correction failed - could not find valid solution")
            if corrections_made:
                print(f"   Attempted {len(corrections_made)} correction(s) but puzzle still unsolvable")
            print("\nTips:")
            print("  - Try with --debug to inspect OCR preprocessing")
            print("  - Check that the image has good quality and lighting")
            print("  - Ensure all digits are clearly visible")
            return False

    # Validate solution
    print("\n[3.6/4] Validating solution...")
    is_valid, errors = SudokuValidator.validate_solution(solution, verbose=False)

    if is_valid:
        print("✅ Solution validation passed!")

        # Collect training data from successfully solved puzzle
        if collect_training_data:
            collector = TrainingDataCollector()
            samples_collected = collector.collect_from_solved_puzzle(
                cells=cells,
                solution=solution,
                source_image=image_path,
                validation_passed=True
            )
            if verbose or debug:
                print_collection_summary(samples_collected, image_path)
    else:
        print(f"⚠️  Solution validation failed ({len(errors)} errors)")
        if verbose or debug:
            for error in errors[:5]:
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")

    # Compare grids (check if original values were overwritten)
    comparison = SudokuValidator.compare_grids(detected_grid, solution, verbose=verbose or debug)

    if comparison['overwrite_count'] > 0:
        print(f"⚠️  Warning: {comparison['overwrite_count']} original values were changed!")
        if verbose or debug:
            for ow in comparison['overwrites'][:3]:
                pos = ow['position']
                print(f"  Cell ({pos[0]},{pos[1]}): {ow['original']} → {ow['solved']}")

    # Step 4: Draw solution on original image
    print("\n[4/4] Generating output image...")
    drawer = SolutionDrawer(font_scale=0.9, thickness=2)

    # Draw on warped grid - use has_content to avoid overwriting original digits
    warped_with_solution = drawer.draw_on_warped(warped_grid, has_content, solution)

    # For simplicity, save the warped solution
    # (Overlaying back to original with perspective would need corner coordinates)
    drawer.save_result(warped_with_solution, output_path)
    print(f"✓ Solution saved to {output_path}")

    # Show final visualization
    if verbose or debug:
        print("\n" + "=" * 60)
        print("FINAL OUTPUT VISUALIZATION")
        print("=" * 60)
        print("Legend:")
        print("  [B] = Black (original) digit will remain visible")
        print("  [R] = Red (solver) digit will be drawn")
        print("  (empty) = no digit was drawn\n")

        for row in range(9):
            if row > 0 and row % 3 == 0:
                print("-" * 60)
            row_str = ""
            for col in range(9):
                if col > 0 and col % 3 == 0:
                    row_str += " | "

                if has_content[row, col]:
                    # Original digit (black)
                    row_str += f"[B]{detected_grid[row, col] if detected_grid[row, col] != 0 else '?'} "
                else:
                    # Solver digit (red)
                    row_str += f"[R]{solution[row, col]} "
            print(row_str)
        print("=" * 60)

    if show_result:
        drawer.display_result(warped_with_solution)

    print("\n" + "=" * 50)
    print("Success! Sudoku solved and saved.")
    return True


def print_grid(grid: np.ndarray) -> None:
    """
    Print a Sudoku grid in a nice format.

    Args:
        grid: 9x9 numpy array
    """
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


def solve_sudoku_from_text(
    text_path: str,
    verbose: bool = False
) -> bool:
    """
    Solve a Sudoku puzzle from a text file and display the result.

    Args:
        text_path: Path to input text file
        verbose: Show verbose output

    Returns:
        True if successful, False otherwise
    """
    print(f"Processing text file: {text_path}")
    print("=" * 50)

    # Step 1: Parse text file
    print("\n[1/3] Parsing Sudoku from text file...")
    puzzle = parse_sudoku_text(text_path)

    if puzzle is None:
        print("Error: Could not parse Sudoku from text file")
        return False

    filled_cells = np.count_nonzero(puzzle)
    print(f"✓ Parsed puzzle with {filled_cells} filled cells")

    # Display puzzle
    print("\nInput puzzle:")
    print_grid(puzzle)

    # Step 2: Solve the puzzle
    print("\n[2/3] Solving Sudoku puzzle...")
    solver = SudokuSolver()
    solver.load_puzzle(puzzle)

    if not solver.is_valid_puzzle():
        print("Error: Puzzle is invalid (contains duplicates)")
        is_valid, errors = SudokuValidator.validate_puzzle(puzzle, verbose=False)
        if not is_valid:
            print(f"\nValidation found {len(errors)} error(s):")
            for error in errors[:5]:
                print(f"  - {error}")
        return False

    if not solver.solve():
        print("Error: Could not solve the puzzle")
        print("The puzzle might be invalid or unsolvable")
        return False

    solution = solver.get_solution()
    print("\nSolved puzzle:")
    print_grid(solution)
    print("✓ Puzzle solved successfully!")

    # Step 3: Validate solution
    print("\n[3/3] Validating solution...")
    is_valid, errors = SudokuValidator.validate_solution(solution, verbose=False)

    if is_valid:
        print("✅ Solution validation passed!")
    else:
        print(f"⚠️  Solution validation failed ({len(errors)} errors)")
        if verbose:
            for error in errors[:5]:
                print(f"  - {error}")

    # Compare grids
    comparison = SudokuValidator.compare_grids(puzzle, solution, verbose=verbose)

    if comparison['overwrite_count'] > 0:
        print(f"⚠️  Warning: {comparison['overwrite_count']} original values were changed!")

    # Display formatted output
    print("\n" + "=" * 50)
    print("SUDOKU SOLUTION")
    print("=" * 50)
    print(f"\nValidation: {'✅ VALID' if is_valid else '⚠️  INVALID'}")
    print(f"Cells filled by solver: {comparison['cells_filled_by_solver']}")

    if verbose:
        print("\n" + "-" * 50)
        print("Solution with original digits marked:")
        print(format_sudoku_text(solution, original=puzzle))

    print("\n" + "=" * 50)
    print("Success! Sudoku solved.")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Solve Sudoku puzzles from images or text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve from image using ensemble OCR (default, best accuracy)
  python main.py testplaatje.png -o solved.png

  # Solve from text file
  python main.py sudoku.txt -o solved.txt

  # Disable ensemble and use only CNN
  python main.py testplaatje.png -o solved.png --no-ensemble

  # Use Tesseract OCR instead of ensemble
  python main.py testplaatje.png -o solved.png --tesseract

  # Enable debug mode to see processing steps
  python main.py testplaatje.png -o solved.png --debug

  # Display result in window
  python main.py testplaatje.png -o solved.png --show

Text file format:
  016 000 070
  000 096 100
  204 800 090

  700 080 000
  030 070 000
  000 300 400

  000 000 000
  000 769 302
  001 003 009

  Where 0 = empty cell, spaces separate groups, blank lines separate 3x3 blocks.
        """
    )

    parser.add_argument(
        "input",
        help="Path to input image or text file containing Sudoku puzzle"
    )

    parser.add_argument(
        "-o", "--output",
        default="solved_sudoku.png",
        help="Path to save output image (default: solved_sudoku.png)"
    )

    parser.add_argument(
        "-m", "--model",
        default="models/digit_cnn.h5",
        help="Path to CNN model for digit recognition (default: models/digit_cnn.h5)"
    )

    parser.add_argument(
        "-t", "--tesseract",
        action="store_true",
        help="Use Tesseract OCR instead of CNN"
    )

    parser.add_argument(
        "--no-ensemble",
        action="store_false",
        dest="ensemble",
        default=True,
        help="Disable ensemble mode and use only CNN or Tesseract (ensemble is enabled by default)"
    )

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode (shows intermediate steps with images)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose ASCII output without image windows"
    )

    parser.add_argument(
        "-s", "--show",
        action="store_true",
        help="Display result in window"
    )

    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Don't collect training data from successfully solved puzzles"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1

    # Check if input is a text file
    input_ext = os.path.splitext(args.input)[1].lower()
    is_text_file = input_ext == '.txt'

    if is_text_file:
        # Text file mode
        print("Text file mode detected")

        # Run text solver (output only to shell, no files created)
        success = solve_sudoku_from_text(
            text_path=args.input,
            verbose=args.verbose
        )
    else:
        # Image mode
        # Check if model exists (if not using tesseract)
        if not args.tesseract and not os.path.exists(args.model):
            print(f"Warning: CNN model '{args.model}' not found")
            print("Please train the model first by running:")
            print("  python -m src.ocr")
            print("Or use --tesseract flag to use Tesseract OCR instead")
            return 1

        # Run image solver
        success = solve_sudoku_from_image(
            image_path=args.input,
            output_path=args.output,
            model_path=args.model,
            use_tesseract=args.tesseract,
            use_ensemble=args.ensemble,
            debug=args.debug,
            show_result=args.show,
            verbose=args.verbose,
            collect_training_data=not args.no_collect
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
