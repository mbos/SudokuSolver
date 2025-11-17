#!/usr/bin/env python3
"""
Quick test script to establish baseline OCR performance.
Run this before and after improvements to measure impact.
"""

import numpy as np
from src.grid_detector import GridDetector
from src.ocr import DigitRecognizer
import time

# Ground truth for testplaatje.png
GROUND_TRUTH = np.array([
    [0, 0, 0, 0, 0, 0, 9, 6, 5],
    [0, 0, 0, 1, 9, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 8],
    [1, 0, 0, 7, 6, 0, 0, 0, 0],
    [0, 9, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 1, 0, 5, 3, 0],
    [0, 0, 3, 0, 2, 1, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 1, 5, 0],
    [6, 0, 0, 9, 0, 0, 8, 0, 0]
])

def test_ocr_accuracy(image_path="testplaatje.png", use_tesseract=True):
    """Test OCR accuracy against ground truth"""

    print("="*60)
    print("OCR ACCURACY TEST")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Engine: {'Tesseract' if use_tesseract else 'CNN'}")
    print()

    # Extract cells
    start = time.time()
    detector = GridDetector(debug=False)
    _, cells, _ = detector.detect_and_extract(image_path)
    grid_time = time.time() - start

    # Run OCR
    start = time.time()
    recognizer = DigitRecognizer(
        model_path="models/digit_cnn.h5",
        use_tesseract=use_tesseract
    )
    detected_grid, has_content = recognizer.recognize_grid(cells)
    ocr_time = time.time() - start

    # Calculate metrics
    filled_mask = GROUND_TRUTH != 0
    total_filled = np.sum(filled_mask)
    correct = np.sum(detected_grid[filled_mask] == GROUND_TRUTH[filled_mask])
    incorrect = np.sum(
        (detected_grid[filled_mask] != 0) &
        (detected_grid[filled_mask] != GROUND_TRUTH[filled_mask])
    )
    missed = np.sum(
        (has_content[filled_mask]) &
        (detected_grid[filled_mask] == 0)
    )

    # False positives
    empty_mask = GROUND_TRUTH == 0
    false_positives = np.sum(detected_grid[empty_mask] != 0)

    # Print results
    print("RESULTS")
    print("-"*60)
    print(f"Total digits in puzzle: {total_filled}")
    print(f"Correctly recognized:   {correct} ({100*correct/total_filled:.1f}%)")
    print(f"Incorrectly recognized: {incorrect} ({100*incorrect/total_filled:.1f}%)")
    print(f"Missed (has content):   {missed} ({100*missed/total_filled:.1f}%)")
    print(f"False positives:        {false_positives}")
    print()
    print(f"Overall Accuracy:       {100*correct/total_filled:.1f}%")
    print()
    print("PERFORMANCE")
    print("-"*60)
    print(f"Grid detection time:    {grid_time:.3f}s")
    print(f"OCR time:               {ocr_time:.3f}s")
    print(f"Total time:             {grid_time + ocr_time:.3f}s")
    print()

    # Show which cells were missed
    if missed > 0:
        print("MISSED CELLS (have content but OCR returned 0):")
        print("-"*60)
        for row in range(9):
            for col in range(9):
                if (has_content[row, col] and
                    detected_grid[row, col] == 0 and
                    GROUND_TRUTH[row, col] != 0):
                    actual = GROUND_TRUTH[row, col]
                    print(f"  Cell ({row},{col}): Should be {actual}")
        print()

    # Show incorrect recognitions
    if incorrect > 0:
        print("INCORRECT RECOGNITIONS:")
        print("-"*60)
        for row in range(9):
            for col in range(9):
                if (detected_grid[row, col] != 0 and
                    detected_grid[row, col] != GROUND_TRUTH[row, col] and
                    GROUND_TRUTH[row, col] != 0):
                    actual = GROUND_TRUTH[row, col]
                    detected = detected_grid[row, col]
                    print(f"  Cell ({row},{col}): Detected {detected}, should be {actual}")
        print()

    print("="*60)

    return {
        'accuracy': correct / total_filled,
        'false_positive_rate': false_positives / np.sum(empty_mask),
        'false_negative_rate': missed / total_filled,
        'total_time': grid_time + ocr_time
    }


if __name__ == "__main__":
    # Test with Tesseract
    print("\n")
    tesseract_results = test_ocr_accuracy(use_tesseract=True)

    # Test with CNN (if model exists)
    try:
        print("\n\n")
        cnn_results = test_ocr_accuracy(use_tesseract=False)

        # Compare
        print("\n")
        print("="*60)
        print("COMPARISON")
        print("="*60)
        print(f"{'Metric':<30} {'Tesseract':<15} {'CNN':<15}")
        print("-"*60)
        tess_acc = f"{tesseract_results['accuracy']:.1%}"
        cnn_acc = f"{cnn_results['accuracy']:.1%}"
        print(f"{'Accuracy':<30} {tess_acc:<15} {cnn_acc:<15}")

        tess_fp = f"{tesseract_results['false_positive_rate']:.1%}"
        cnn_fp = f"{cnn_results['false_positive_rate']:.1%}"
        print(f"{'False Positive Rate':<30} {tess_fp:<15} {cnn_fp:<15}")

        tess_fn = f"{tesseract_results['false_negative_rate']:.1%}"
        cnn_fn = f"{cnn_results['false_negative_rate']:.1%}"
        print(f"{'False Negative Rate':<30} {tess_fn:<15} {cnn_fn:<15}")

        tess_time = f"{tesseract_results['total_time']:.3f}"
        cnn_time = f"{cnn_results['total_time']:.3f}"
        print(f"{'Total Time (s)':<30} {tess_time:<15} {cnn_time:<15}")
        print("="*60)
    except Exception as e:
        print(f"\nNote: Could not test CNN (model not found or error: {e})")
