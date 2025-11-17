#!/usr/bin/env python3
"""
Demo script voor Ensemble OCR implementatie.

Dit script demonstreert de werking van het ensemble systeem
zonder dat alle dependencies geïnstalleerd hoeven te zijn.
"""

from src.ocr.base_recognizer import RecognitionResult
from src.ocr.voting_strategies import (
    MajorityVoting,
    WeightedVoting,
    ConfidenceAggregation
)


def demo_voting_strategies():
    """Demonstreer verschillende voting strategieën."""
    print("=" * 80)
    print("ENSEMBLE OCR - VOTING STRATEGIES DEMO")
    print("=" * 80)

    # Simuleer results van verschillende models voor dezelfde cell
    print("\nScenario: Drie models detecteren een digit")
    print("-" * 80)

    results = [
        RecognitionResult(digit=5, confidence=0.8, model_name="CNN", processing_time_ms=10),
        RecognitionResult(digit=5, confidence=0.7, model_name="Tesseract", processing_time_ms=15),
        RecognitionResult(digit=7, confidence=0.6, model_name="EasyOCR", processing_time_ms=50),
    ]

    for r in results:
        print(f"  {r.model_name:12} → Digit: {r.digit}, Confidence: {r.confidence:.2f}")

    # Test 1: Majority Voting
    print("\n[1] Majority Voting")
    print("-" * 80)
    strategy = MajorityVoting(min_confidence=0.5)
    result = strategy.vote(results)
    print(f"Result: Digit {result.digit} with confidence {result.confidence:.2f}")
    print(f"Reason: Majority voted for 5 (2 votes vs 1)")

    # Test 2: Weighted Voting
    print("\n[2] Weighted Voting (CNN=1.5, Tesseract=1.0, EasyOCR=2.0)")
    print("-" * 80)
    strategy = WeightedVoting(min_confidence=0.5)
    result = strategy.vote(results)
    print(f"Result: Digit {result.digit} with confidence {result.confidence:.2f}")
    print(f"Scores: Digit 5 = {0.8+0.7:.1f}, Digit 7 = {0.6:.1f}")

    # Test 3: Confidence Aggregation
    print("\n[3] Confidence Aggregation")
    print("-" * 80)
    strategy = ConfidenceAggregation(
        all_agree_threshold=0.5,
        majority_threshold=0.7,
        single_threshold=0.9
    )
    result = strategy.vote(results)
    print(f"Result: Digit {result.digit} with confidence {result.confidence:.2f}")
    print(f"Reason: 2+ models agree on 5, avg confidence {(0.8+0.7)/2:.2f} > threshold 0.7")

    # Scenario 2: Alle models akkoord
    print("\n" + "=" * 80)
    print("\nScenario: Alle models zijn het eens")
    print("-" * 80)

    results_unanimous = [
        RecognitionResult(digit=9, confidence=0.85, model_name="CNN", processing_time_ms=10),
        RecognitionResult(digit=9, confidence=0.75, model_name="Tesseract", processing_time_ms=15),
        RecognitionResult(digit=9, confidence=0.90, model_name="EasyOCR", processing_time_ms=50),
    ]

    for r in results_unanimous:
        print(f"  {r.model_name:12} → Digit: {r.digit}, Confidence: {r.confidence:.2f}")

    print("\nAll strategies agree:")
    for strategy_name, strategy in [
        ("Majority", MajorityVoting()),
        ("Weighted", WeightedVoting()),
        ("Confidence", ConfidenceAggregation())
    ]:
        result = strategy.vote(results_unanimous)
        print(f"  {strategy_name:12} → Digit {result.digit}, Confidence {result.confidence:.2f}")

    # Scenario 3: Onzeker (alle low confidence)
    print("\n" + "=" * 80)
    print("\nScenario: Alle models onzeker")
    print("-" * 80)

    results_uncertain = [
        RecognitionResult(digit=3, confidence=0.4, model_name="CNN", processing_time_ms=10),
        RecognitionResult(digit=8, confidence=0.3, model_name="Tesseract", processing_time_ms=15),
        RecognitionResult(digit=0, confidence=0.9, model_name="EasyOCR", processing_time_ms=50),
    ]

    for r in results_uncertain:
        digit_str = "empty" if r.digit == 0 else str(r.digit)
        print(f"  {r.model_name:12} → Digit: {digit_str}, Confidence: {r.confidence:.2f}")

    print("\nVoting results:")
    strategy = MajorityVoting(min_confidence=0.5)
    result = strategy.vote(results_uncertain)
    digit_str = "empty/uncertain" if result.digit == 0 else str(result.digit)
    print(f"  Final result: {digit_str}")
    print(f"  Reason: Low confidence results filtered out, EasyOCR says empty")


def demo_fallback_levels():
    """Demonstreer fallback levels."""
    print("\n" + "=" * 80)
    print("FALLBACK LEVELS DEMO")
    print("=" * 80)

    print("\nLevel 1: Fast Path (CNN + Tesseract PSM 10)")
    print("-" * 80)
    print("Target: 80% van cellen in <50ms")
    print("Threshold: confidence >= 0.75")
    print()
    print("Example:")
    print("  CNN:       Digit 5, Confidence 0.82  ✓")
    print("  Tesseract: Digit 5, Confidence 0.78  ✓")
    print("  → Accept at Level 1 (avg confidence 0.80 > 0.75)")

    print("\nLevel 2: Medium Path (+ EasyOCR)")
    print("-" * 80)
    print("Target: 95% van cellen in <200ms")
    print("Threshold: confidence >= 0.65")
    print()
    print("Example (Level 1 failed):")
    print("  CNN:       Digit 6, Confidence 0.65")
    print("  Tesseract: Digit 8, Confidence 0.60")
    print("  → Disagreement, low confidence, try Level 2")
    print()
    print("  EasyOCR:   Digit 6, Confidence 0.88  ✓")
    print("  → Weighted vote: Digit 6 wins")
    print("  → Accept at Level 2")

    print("\nLevel 3: Full Ensemble (all models)")
    print("-" * 80)
    print("Target: 99% van cellen in <500ms")
    print("Return: Best available prediction")
    print()
    print("Example (Level 2 still uncertain):")
    print("  All models return 0 (empty/uncertain)")
    print("  → Accept as empty cell")


def demo_model_comparison():
    """Demonstreer expected improvements."""
    print("\n" + "=" * 80)
    print("EXPECTED ACCURACY IMPROVEMENT")
    print("=" * 80)

    print("\nTest case: testplaatje.png (25 digits)")
    print("-" * 80)

    print("\n1. Tesseract alleen (baseline):")
    print("   Correct:   21/25 (84.0%)")
    print("   Missed:    4 digits")
    print("   - Cell (0,7): Expected 6, got 0")
    print("   - Cell (1,4): Expected 9, got 0")
    print("   - Cell (8,3): Expected 9, got 0")
    print("   - Cell (8,6): Expected 8, got 0")

    print("\n2. CNN Model (if trained):")
    print("   Correct:   22-23/25 (88-92%)")
    print("   Missed:    2-3 digits")

    print("\n3. Ensemble (CNN + Tesseract + EasyOCR):")
    print("   Expected:  24-25/25 (96-100%)")
    print("   Missed:    0-1 digits")
    print("   Improvement: +12-16% over baseline")

    print("\n" + "=" * 80)
    print("Why ensemble works:")
    print("-" * 80)
    print("• Different models have complementary strengths")
    print("• CNN: Good with standard MNIST-like digits")
    print("• Tesseract: Good with printed text")
    print("• EasyOCR: Good with difficult/unclear digits")
    print("• Voting eliminates isolated errors")
    print("• Consensus increases confidence")


if __name__ == '__main__':
    demo_voting_strategies()
    demo_fallback_levels()
    demo_model_comparison()

    print("\n" + "=" * 80)
    print("DEMO COMPLETED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run tests: ./run_tests.sh")
    print("3. Try ensemble: python main.py testplaatje.png -o solved.png --ensemble")
    print()
