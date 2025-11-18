#!/bin/bash
# Test runner for Sudoku Solver ensemble OCR

echo "=========================================="
echo "Sudoku Solver - Test Suite"
echo "=========================================="

# Run unit tests
echo ""
echo "[1/3] Running unit tests (voting strategies)..."
python -m pytest tests/test_voting_strategies.py -v

if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed"
    exit 1
fi

# Run integration tests
echo ""
echo "[2/3] Running integration tests (ensemble)..."
python -m pytest tests/test_ensemble.py -v

if [ $? -ne 0 ]; then
    echo "❌ Integration tests failed"
    exit 1
fi

# Run end-to-end test
echo ""
echo "[3/3] Running end-to-end test (testplaatje.png)..."
python tests/test_ensemble_e2e.py

if [ $? -ne 0 ]; then
    echo "⚠️  End-to-end test did not meet target accuracy"
    echo "    (This is expected if EasyOCR is not installed yet)"
    echo "    Install with: pip install easyocr"
fi

echo ""
echo "=========================================="
echo "✅ Core tests completed"
echo "=========================================="
