#!/usr/bin/env python3
"""
A general-purpose OCR accuracy analyzer.

Compares the OCR results from an image against a ground truth text file.

Usage:
  python dev_tools/new_ocr_analyzer.py path/to/image.png path/to/ground_truth.txt
"""
import sys
import os
import yaml
import numpy as np

# Add src directory to path to import from sibling directories
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.grid_detector import GridDetector
from src.ocr.ensemble_recognizer import EnsembleRecognizer
from src.text_parser import parse_sudoku_text

def analyze_ocr_accuracy(image_path: str, ground_truth_path: str):
    """
    Runs the OCR pipeline and compares the result to a ground truth file.
    """
    print("=" * 80)
    print("OCR ACCURACY ANALYZER")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Ground Truth: {ground_truth_path}\n")

    # 1. Load Ground Truth
    if not os.path.exists(ground_truth_path):
        print(f"ERROR: Ground truth file not found at {ground_truth_path}")
        return
    actual_grid = parse_sudoku_text(ground_truth_path)
    print("✓ Loaded Ground Truth")

    # 2. Run OCR on the image
    print("Running OCR pipeline...")
    detector = GridDetector()
    _, cells, _ = detector.detect_and_extract(image_path)
    if cells is None:
        print("ERROR: Sudoku grid could not be detected in the image.")
        return

    # Load ensemble configuration from YAML
    try:
        with open("config/ocr_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Loaded OCR configuration from config/ocr_config.yaml")
    except Exception as e:
        print(f"⚠️ Warning: Could not load config. Using default. Error: {e}")
        config = None

    ensemble = EnsembleRecognizer(config=config)
    detected_grid, _, _ = ensemble.recognize_grid(cells)
    print("✓ OCR processing complete.\n")

    # 3. Compare and report
    print("=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    total_digits = 0
    correct_count = 0
    incorrect_count = 0 # False positives (e.g., empty cell read as a 5)
    missed_count = 0    # False negatives (e.g., a 5 read as empty)

    print("Position | Actual | Detected | Status")
    print("-" * 45)

    for r in range(9):
        for c in range(9):
            actual = actual_grid[r, c]
            detected = detected_grid[r, c]

            if actual != 0:
                total_digits += 1
                if actual == detected:
                    correct_count += 1
                else:
                    if detected == 0:
                        missed_count += 1
                        status = f"✗ MISSED (should be {actual})"
                        print(f"({r},{c})    |   {actual}    |    {detected}     | {status}")
                    else:
                        incorrect_count += 1
                        status = f"✗ WRONG (should be {actual})"
                        print(f"({r},{c})    |   {actual}    |    {detected}     | {status}")
            elif detected != 0:
                # This is an empty cell in ground truth that OCR found a digit in
                incorrect_count += 1
                status = f"✗ WRONG (should be empty)"
                print(f"({r},{c})    |   {actual}    |    {detected}     | {status}")


    # 4. Print Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 45)
    if total_digits == 0:
        print("No digits found in ground truth file.")
        return
        
    accuracy = correct_count / total_digits

    print(f"Total Digits in Ground Truth: {total_digits}")
    print(f"✓ Correctly Identified:       {correct_count}")
    print(f"✗ Incorrectly Identified:     {incorrect_count} (False Positives)")
    print(f"✗ Missed Digits:              {missed_count} (False Negatives)")
    print("-" * 45)
    print(f"OCR Accuracy:                 {accuracy:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dev_tools/new_ocr_analyzer.py <image_path> <ground_truth_path>")
        sys.exit(1)

    image_path_arg = sys.argv[1]
    ground_truth_path_arg = sys.argv[2]

    analyze_ocr_accuracy(image_path_arg, ground_truth_path_arg)
