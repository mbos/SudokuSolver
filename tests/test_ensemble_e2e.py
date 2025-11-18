#!/usr/bin/env python3
"""
End-to-end test for ensemble OCR with testplaatje.png.

This test validates the ensemble OCR implementation by:
1. Running OCR on testplaatje.png with ensemble mode
2. Comparing results against the known correct solution
3. Measuring accuracy improvements over single-model approaches
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid_detector import GridDetector
from src.ocr import DigitRecognizer
from src.ocr.ensemble_recognizer import EnsembleRecognizer


# Ground truth from testplaatje_oplossing.txt
CORRECT_SOLUTION = np.array([
    [3, 2, 1, 4, 7, 8, 9, 6, 5],
    [5, 4, 8, 1, 9, 6, 3, 2, 7],
    [9, 7, 6, 2, 5, 3, 4, 1, 8],
    [1, 3, 4, 7, 6, 5, 2, 8, 9],
    [8, 9, 5, 3, 4, 2, 6, 7, 1],
    [2, 6, 7, 8, 1, 9, 5, 3, 4],
    [4, 8, 3, 5, 2, 1, 7, 9, 6],
    [7, 2, 9, 6, 8, 4, 1, 5, 3],
    [6, 5, 1, 9, 3, 7, 8, 4, 2]
])


def get_starting_grid():
    """Get the actual starting grid (cells that should have content)."""
    # These are the cells that have visual content in testplaatje.png
    starting_positions = [
        (0, 6), (0, 8),  # Row 0
        (1, 3), (1, 4),  # Row 1 (note: (1,4) was missed by old OCR)
        (2, 3), (2, 8),  # Row 2
        (3, 0), (3, 3), (3, 4),  # Row 3
        (4, 1), (4, 2),  # Row 4
        (5, 2), (5, 4), (5, 6), (5, 7),  # Row 5
        (6, 2), (6, 4), (6, 5),  # Row 6
        (7, 0), (7, 6), (7, 7),  # Row 7
        (8, 0), (8, 3), (8, 6),  # Row 8 (note: (8,3) and (8,6) were missed)
    ]

    starting_grid = np.zeros_like(CORRECT_SOLUTION)
    for row, col in starting_positions:
        starting_grid[row, col] = CORRECT_SOLUTION[row, col]

    return starting_grid, starting_positions


def calculate_accuracy(detected, correct_starting):
    """
    Calculate OCR accuracy.

    Args:
        detected: Detected grid from OCR
        correct_starting: Ground truth starting grid

    Returns:
        Dict with accuracy metrics
    """
    # Find cells that should have content
    has_content_mask = correct_starting != 0

    # Count correct detections
    correct_detections = np.sum((detected == correct_starting) & has_content_mask)
    total_content_cells = np.sum(has_content_mask)

    # Find missed cells (has content but OCR returned 0)
    missed = np.sum((detected == 0) & has_content_mask)

    # Find incorrect cells (OCR returned wrong digit)
    incorrect = np.sum((detected != correct_starting) & (detected != 0) & has_content_mask)

    accuracy = (correct_detections / total_content_cells * 100) if total_content_cells > 0 else 0

    return {
        'correct': correct_detections,
        'missed': missed,
        'incorrect': incorrect,
        'total': total_content_cells,
        'accuracy': accuracy
    }


def test_ensemble_ocr():
    """Test ensemble OCR on testplaatje.png."""
    print("=" * 80)
    print("END-TO-END ENSEMBLE OCR TEST")
    print("=" * 80)

    image_path = "testplaatje.png"

    if not os.path.exists(image_path):
        print(f"\n❌ Error: {image_path} not found")
        print("This test requires testplaatje.png in the current directory")
        return False

    # Get ground truth
    correct_starting, starting_positions = get_starting_grid()
    print(f"\nGround truth: {len(starting_positions)} cells with content")

    # Step 1: Extract grid
    print("\n[1/4] Extracting grid from image...")
    detector = GridDetector(debug=False)
    original_image, cells, warped_grid = detector.detect_and_extract(image_path)

    if cells is None:
        print("❌ Failed to extract grid")
        return False

    print(f"✓ Extracted {len(cells)} cells")

    # Step 2: Test with old single-model OCR (baseline)
    print("\n[2/4] Testing baseline (Tesseract only)...")
    baseline_recognizer = DigitRecognizer(use_tesseract=True)
    baseline_grid, _ = baseline_recognizer.recognize_grid(cells)
    baseline_metrics = calculate_accuracy(baseline_grid, correct_starting)

    print(f"  Correct:    {baseline_metrics['correct']}/{baseline_metrics['total']}")
    print(f"  Missed:     {baseline_metrics['missed']}")
    print(f"  Incorrect:  {baseline_metrics['incorrect']}")
    print(f"  Accuracy:   {baseline_metrics['accuracy']:.1f}%")

    # Step 3: Test with CNN (if available)
    print("\n[3/4] Testing CNN model...")
    cnn_model_path = "models/digit_cnn.h5"

    if os.path.exists(cnn_model_path):
        cnn_recognizer = DigitRecognizer(model_path=cnn_model_path, use_tesseract=False)
        cnn_grid, _ = cnn_recognizer.recognize_grid(cells)
        cnn_metrics = calculate_accuracy(cnn_grid, correct_starting)

        print(f"  Correct:    {cnn_metrics['correct']}/{cnn_metrics['total']}")
        print(f"  Missed:     {cnn_metrics['missed']}")
        print(f"  Incorrect:  {cnn_metrics['incorrect']}")
        print(f"  Accuracy:   {cnn_metrics['accuracy']:.1f}%")
    else:
        print("  ⚠️  CNN model not found, skipping CNN test")
        cnn_metrics = None

    # Step 4: Test with Ensemble
    print("\n[4/4] Testing Ensemble OCR...")
    ensemble = EnsembleRecognizer(voting_strategy="weighted")
    ensemble_grid, _ = ensemble.recognize_grid(cells, verbose=False)
    ensemble_metrics = calculate_accuracy(ensemble_grid, correct_starting)

    print(f"  Correct:    {ensemble_metrics['correct']}/{ensemble_metrics['total']}")
    print(f"  Missed:     {ensemble_metrics['missed']}")
    print(f"  Incorrect:  {ensemble_metrics['incorrect']}")
    print(f"  Accuracy:   {ensemble_metrics['accuracy']:.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON")
    print("=" * 80)
    print(f"Baseline (Tesseract): {baseline_metrics['accuracy']:.1f}%")
    if cnn_metrics:
        print(f"CNN Model:            {cnn_metrics['accuracy']:.1f}%")
    print(f"Ensemble:             {ensemble_metrics['accuracy']:.1f}%")

    # Calculate improvement
    improvement = ensemble_metrics['accuracy'] - baseline_metrics['accuracy']
    print(f"\nImprovement:          +{improvement:.1f}% over baseline")

    # Show missed digits
    if ensemble_metrics['missed'] > 0:
        print("\n" + "-" * 80)
        print("MISSED DIGITS (Ensemble still couldn't recognize):")
        print("-" * 80)
        for row in range(9):
            for col in range(9):
                if correct_starting[row, col] != 0 and ensemble_grid[row, col] == 0:
                    expected = correct_starting[row, col]
                    print(f"  Cell ({row},{col}): Expected {expected}, got 0")

    # Show incorrect digits
    if ensemble_metrics['incorrect'] > 0:
        print("\n" + "-" * 80)
        print("INCORRECT DIGITS:")
        print("-" * 80)
        for row in range(9):
            for col in range(9):
                if (correct_starting[row, col] != 0 and
                    ensemble_grid[row, col] != 0 and
                    ensemble_grid[row, col] != correct_starting[row, col]):
                    expected = correct_starting[row, col]
                    detected = ensemble_grid[row, col]
                    print(f"  Cell ({row},{col}): Expected {expected}, got {detected}")

    # Success criteria
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    success = ensemble_metrics['accuracy'] >= 90.0  # 90% threshold

    if success:
        print(f"✅ TEST PASSED: {ensemble_metrics['accuracy']:.1f}% accuracy (≥90% required)")
    else:
        print(f"⚠️  TEST NEEDS IMPROVEMENT: {ensemble_metrics['accuracy']:.1f}% accuracy (<90%)")

    # Additional success: Must be better than baseline
    if ensemble_metrics['accuracy'] > baseline_metrics['accuracy']:
        print(f"✅ IMPROVEMENT VERIFIED: +{improvement:.1f}% over baseline")
    else:
        print(f"⚠️  NO IMPROVEMENT over baseline")

    print("=" * 80)

    return success


if __name__ == '__main__':
    success = test_ensemble_ocr()
    sys.exit(0 if success else 1)
