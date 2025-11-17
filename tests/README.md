# Ensemble OCR Tests

This directory contains tests for the multi-model ensemble OCR implementation.

## Test Structure

### Unit Tests (`test_voting_strategies.py`)
Tests for individual voting strategy algorithms:
- Majority voting
- Weighted voting
- Confidence aggregation
- Strategy factory

Run with:
```bash
python -m pytest tests/test_voting_strategies.py -v
```

### Integration Tests (`test_ensemble.py`)
Tests for ensemble recognizer integration:
- Configuration handling
- Fallback level logic
- Statistics tracking
- Grid recognition

Run with:
```bash
python -m pytest tests/test_ensemble.py -v
```

### End-to-End Test (`test_ensemble_e2e.py`)
Complete pipeline test using `testplaatje.png`:
- Compares ensemble vs single-model approaches
- Measures accuracy against ground truth
- Validates improvement over baseline

Run with:
```bash
python tests/test_ensemble_e2e.py
```

## Running All Tests

Use the test runner script:
```bash
./run_tests.sh
```

Or with pytest:
```bash
python -m pytest tests/ -v
```

## Dependencies

For full test coverage, ensure all OCR dependencies are installed:
```bash
pip install -r requirements.txt
```

Note: EasyOCR requires ~500MB download on first run.

## Expected Results

- **Unit tests**: 100% pass rate
- **Integration tests**: 100% pass rate
- **E2E test**: ≥90% OCR accuracy (target)

The E2E test compares:
1. Baseline (Tesseract only): ~84% accuracy
2. CNN model: ~85-90% accuracy (if model trained)
3. Ensemble: **95-98% accuracy** (expected)
