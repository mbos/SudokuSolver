#!/usr/bin/env python3
"""
Comprehensive OCR test suite using test_dataset.json
"""

import json
import numpy as np
import time
from src.grid_detector import GridDetector
from src.ocr import DigitRecognizer


class OCRTestSuite:
    """Test suite for OCR accuracy evaluation."""

    def __init__(self, dataset_path='test_dataset.json'):
        with open(dataset_path) as f:
            self.dataset = json.load(f)

        self.detector = GridDetector(debug=False)

    def test_single_image(self, image_data, recognizer):
        """Test OCR on a single image."""
        filename = image_data['filename']
        image_path = image_data.get('path', filename)  # Use path if available
        ground_truth = np.array(image_data['ground_truth'])

        try:
            # Extract cells
            _, cells, _ = self.detector.detect_and_extract(image_path)

            if cells is None:
                return {
                    'success': False,
                    'error': 'Grid detection failed',
                    'filename': filename
                }

            # Run OCR
            detected_grid, has_content = recognizer.recognize_grid(cells)

            # Calculate metrics
            filled_mask = ground_truth != 0
            total_filled = np.sum(filled_mask)

            if total_filled == 0:
                return {
                    'success': False,
                    'error': 'No digits in ground truth',
                    'filename': filename
                }

            correct = np.sum(detected_grid[filled_mask] == ground_truth[filled_mask])
            incorrect = np.sum(
                (detected_grid[filled_mask] != 0) &
                (detected_grid[filled_mask] != ground_truth[filled_mask])
            )
            missed = np.sum(
                (has_content[filled_mask]) &
                (detected_grid[filled_mask] == 0)
            )

            # False positives
            empty_mask = ground_truth == 0
            false_positives = np.sum(detected_grid[empty_mask] != 0)

            return {
                'success': True,
                'filename': filename,
                'total_digits': int(total_filled),
                'correct': int(correct),
                'incorrect': int(incorrect),
                'missed': int(missed),
                'false_positives': int(false_positives),
                'accuracy': float(correct / total_filled) if total_filled > 0 else 0,
                'config': image_data.get('config', {})
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }

    def test_all(self, recognizer, verbose=True):
        """Test OCR on all images in dataset."""

        results = []
        total_start = time.time()

        print("="*80)
        print("OCR TEST SUITE")
        print("="*80)
        print(f"Dataset: {len(self.dataset['images'])} images")
        print(f"Engine: {'Tesseract' if recognizer.use_tesseract else 'CNN'}")
        print()

        for i, image_data in enumerate(self.dataset['images'], 1):
            if verbose:
                print(f"[{i}/{len(self.dataset['images'])}] Testing {image_data['filename']}...", end=' ')

            start = time.time()
            result = self.test_single_image(image_data, recognizer)
            result['time'] = time.time() - start

            results.append(result)

            if verbose:
                if result['success']:
                    print(f"✓ {result['accuracy']:.1%} ({result['time']:.2f}s)")
                else:
                    print(f"✗ {result['error']}")

        total_time = time.time() - total_start

        # Calculate aggregate statistics
        successful = [r for r in results if r['success']]

        if not successful:
            print("\n❌ No successful tests!")
            return results

        stats = self._calculate_statistics(successful, total_time)

        # Print summary
        self._print_summary(stats, successful, results)

        return results, stats

    def _calculate_statistics(self, successful_results, total_time):
        """Calculate aggregate statistics."""

        total_digits = sum(r['total_digits'] for r in successful_results)
        total_correct = sum(r['correct'] for r in successful_results)
        total_incorrect = sum(r['incorrect'] for r in successful_results)
        total_missed = sum(r['missed'] for r in successful_results)
        total_false_positives = sum(r['false_positives'] for r in successful_results)

        accuracies = [r['accuracy'] for r in successful_results]

        return {
            'num_tests': len(successful_results),
            'total_digits': total_digits,
            'total_correct': total_correct,
            'total_incorrect': total_incorrect,
            'total_missed': total_missed,
            'total_false_positives': total_false_positives,
            'overall_accuracy': total_correct / total_digits if total_digits > 0 else 0,
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'false_positive_rate': total_false_positives / (len(successful_results) * 81 - total_digits) if total_digits > 0 else 0,
            'false_negative_rate': total_missed / total_digits if total_digits > 0 else 0,
            'total_time': total_time,
            'avg_time': total_time / len(successful_results)
        }

    def _print_summary(self, stats, successful, all_results):
        """Print test summary."""

        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)

        print(f"\nTests Run:              {len(all_results)}")
        print(f"Successful:             {len(successful)} ({100*len(successful)/len(all_results):.1f}%)")
        print(f"Failed:                 {len(all_results) - len(successful)}")

        print(f"\nTotal Digits:           {stats['total_digits']}")
        print(f"Correctly Recognized:   {stats['total_correct']} ({100*stats['overall_accuracy']:.1f}%)")
        print(f"Incorrectly Recognized: {stats['total_incorrect']} ({100*stats['total_incorrect']/stats['total_digits']:.1f}%)")
        print(f"Missed (has content):   {stats['total_missed']} ({100*stats['false_negative_rate']:.1f}%)")
        print(f"False Positives:        {stats['total_false_positives']}")

        print(f"\n{'OVERALL ACCURACY:':<25} {stats['overall_accuracy']:.1%}")
        print(f"{'Average Accuracy:':<25} {stats['avg_accuracy']:.1%} (±{stats['std_accuracy']:.1%})")
        print(f"{'Range:':<25} {stats['min_accuracy']:.1%} - {stats['max_accuracy']:.1%}")

        print(f"\n{'False Positive Rate:':<25} {stats['false_positive_rate']:.1%}")
        print(f"{'False Negative Rate:':<25} {stats['false_negative_rate']:.1%}")

        print(f"\n{'Total Time:':<25} {stats['total_time']:.2f}s")
        print(f"{'Average Time/Image:':<25} {stats['avg_time']:.2f}s")

        # Breakdown by quality
        self._print_breakdown_by_quality(successful)

        print("="*80)

    def _print_breakdown_by_quality(self, results):
        """Print accuracy breakdown by image quality."""

        print("\nACCURACY BY IMAGE QUALITY:")
        print("-"*80)

        quality_groups = {}
        for result in results:
            quality = result.get('config', {}).get('quality', 'unknown')
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(result['accuracy'])

        for quality in sorted(quality_groups.keys()):
            accuracies = quality_groups[quality]
            avg_acc = np.mean(accuracies)
            print(f"  {quality.capitalize():<15} {avg_acc:.1%} ({len(accuracies)} images)")

    def compare_engines(self, verbose=False):
        """Compare Tesseract vs CNN performance."""

        print("\n" + "="*80)
        print("COMPARING OCR ENGINES")
        print("="*80)

        # Test Tesseract
        print("\n[1/2] Testing Tesseract OCR...")
        tesseract = DigitRecognizer(use_tesseract=True)
        tess_results, tess_stats = self.test_all(tesseract, verbose=verbose)

        # Test CNN
        print("\n[2/2] Testing CNN...")
        try:
            cnn = DigitRecognizer(
                model_path="models/digit_cnn.h5",
                use_tesseract=False
            )
            cnn_results, cnn_stats = self.test_all(cnn, verbose=verbose)

            # Print comparison
            print("\n" + "="*80)
            print("ENGINE COMPARISON")
            print("="*80)

            print(f"\n{'Metric':<30} {'Tesseract':<20} {'CNN':<20}")
            print("-"*80)

            metrics = [
                ('Overall Accuracy', 'overall_accuracy', '%'),
                ('False Negative Rate', 'false_negative_rate', '%'),
                ('False Positive Rate', 'false_positive_rate', '%'),
                ('Avg Time per Image', 'avg_time', 's'),
            ]

            for label, key, unit in metrics:
                tess_val = tess_stats[key]
                cnn_val = cnn_stats[key]

                if unit == '%':
                    tess_str = f"{tess_val:.1%}"
                    cnn_str = f"{cnn_val:.1%}"
                else:
                    tess_str = f"{tess_val:.3f}{unit}"
                    cnn_str = f"{cnn_val:.3f}{unit}"

                # Add winner indicator
                if key in ['overall_accuracy', 'max_accuracy']:
                    winner = " ✓" if tess_val > cnn_val else ""
                    loser = " ✓" if cnn_val > tess_val else ""
                elif key in ['false_negative_rate', 'false_positive_rate', 'avg_time']:
                    winner = " ✓" if tess_val < cnn_val else ""
                    loser = " ✓" if cnn_val < tess_val else ""
                else:
                    winner = loser = ""

                print(f"{label:<30} {tess_str:<20}{winner} {cnn_str:<20}{loser}")

            print("="*80)

        except Exception as e:
            print(f"\n⚠ Could not test CNN: {e}")
            print("=" *80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OCR test suite")
    parser.add_argument(
        "-d", "--dataset",
        default="test_dataset.json",
        help="Path to test dataset JSON (default: test_dataset.json)"
    )
    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        help="Compare Tesseract vs CNN"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--tesseract",
        action="store_true",
        help="Test only Tesseract"
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Test only CNN"
    )

    args = parser.parse_args()

    suite = OCRTestSuite(args.dataset)

    if args.compare:
        suite.compare_engines(verbose=args.verbose)
    elif args.cnn:
        recognizer = DigitRecognizer(
            model_path="models/digit_cnn.h5",
            use_tesseract=False
        )
        suite.test_all(recognizer, verbose=True)
    else:
        # Default: test Tesseract
        recognizer = DigitRecognizer(use_tesseract=True)
        suite.test_all(recognizer, verbose=True)
