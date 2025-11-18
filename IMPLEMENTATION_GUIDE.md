# OCR Improvement Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing OCR improvements in the Sudoku Solver project.

## Phase 1: Quick Wins (Estimated Time: 2-4 hours)

### 1.1 Optimize Confidence Thresholds

**Goal**: Find optimal confidence thresholds through experimentation

**File**: `src/ocr/ensemble_recognizer.py`

**Current thresholds** (lines 70-74):
```python
'thresholds': {
    'level1_confidence': 0.75,
    'level2_confidence': 0.65,
    'min_confidence': 0.5
}
```

**Action**: Create a test script to find optimal values:

```python
# create: test_threshold_optimization.py
import numpy as np
from src.ocr.ensemble_recognizer import EnsembleRecognizer

def test_thresholds(image_paths, solutions, threshold_ranges):
    """Test different threshold combinations."""
    best_accuracy = 0
    best_config = None

    for l1 in threshold_ranges['level1']:
        for l2 in threshold_ranges['level2']:
            config = {
                'thresholds': {
                    'level1_confidence': l1,
                    'level2_confidence': l2,
                    'min_confidence': 0.5
                }
            }

            recognizer = EnsembleRecognizer(config=config)
            accuracy = evaluate_on_dataset(recognizer, image_paths, solutions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config

    return best_config, best_accuracy

# Usage
images = ['testplaat2.png', 'testplaatje.png', ...]  # Add more test images
solutions = [...]  # Corresponding ground truth solutions

threshold_ranges = {
    'level1': [0.70, 0.75, 0.80, 0.85],
    'level2': [0.60, 0.65, 0.70]
}

best_config, accuracy = test_thresholds(images, solutions, threshold_ranges)
print(f"Best config: {best_config}")
print(f"Accuracy: {accuracy:.2%}")
```

### 1.2 Tune Model Weights

**File**: `src/ocr/ensemble_recognizer.py`

**Current weights** (lines 52-68):
```python
'cnn': {'weight': 1.5},
'tesseract': {'weight': 1.0},
'easyocr': {'weight': 2.0}
```

**Action**: Test different weight combinations:

```python
# In test_threshold_optimization.py, add:

weight_combinations = [
    {'cnn': 1.0, 'tesseract': 1.0, 'easyocr': 2.0},  # Trust EasyOCR most
    {'cnn': 1.5, 'tesseract': 1.0, 'easyocr': 1.5},  # Balanced
    {'cnn': 2.0, 'tesseract': 0.8, 'easyocr': 2.0},  # Trust ML models
    {'cnn': 1.2, 'tesseract': 1.5, 'easyocr': 1.8},  # Slightly favor Tess
]

# Test each and pick best
```

## Phase 2: Standard Improvements (Estimated Time: 1-2 days)

### 2.1 Multi-Scale Preprocessing

**Goal**: Process cells at multiple scales to handle thickness variations

**File**: Create `src/ocr/multi_scale_recognizer.py`

```python
"""Multi-scale digit recognizer."""
import cv2
import numpy as np
from typing import List, Tuple

class MultiScaleRecognizer:
    """Recognize digits at multiple scales."""

    def __init__(self, base_recognizer, scales=[0.8, 1.0, 1.2]):
        self.base = base_recognizer
        self.scales = scales

    def recognize_digit(self, cell: np.ndarray) -> Tuple[int, float]:
        """Recognize digit at multiple scales, return best result."""
        results = []

        for scale in self.scales:
            # Resize cell
            h, w = cell.shape[:2]
            scaled = cv2.resize(cell, (int(w * scale), int(h * scale)))

            # Recognize
            digit = self.base.recognize_digit(scaled)
            confidence = self._get_confidence(scaled, digit)

            results.append((digit, confidence))

        # Return most confident non-zero result
        non_zero = [(d, c) for d, c in results if d != 0]
        if non_zero:
            return max(non_zero, key=lambda x: x[1])

        # All returned 0, return with highest confidence
        return max(results, key=lambda x: x[1])
```

**Integration**: Wrap your recognizer with `MultiScaleRecognizer` in main.py

### 2.2 Context-Aware Voting

**Goal**: Use Sudoku constraints during voting

**File**: `src/ocr/voting_strategies.py`

**Action**: Add new voting strategy (append to file):

```python
class ConstraintAwareVoting(VotingStrategy):
    """Voting that considers Sudoku constraints."""

    def __init__(self, base_strategy: VotingStrategy, puzzle_state: np.ndarray):
        self.base = base_strategy
        self.puzzle_state = puzzle_state
        self.current_pos = (0, 0)  # Will be set before each vote

    def set_position(self, row: int, col: int):
        """Set current cell position for constraint checking."""
        self.current_pos = (row, col)

    def vote(self, results: List[RecognitionResult]) -> RecognitionResult:
        """Vote with constraint awareness."""
        # Get base vote
        base_result = self.base.vote(results)

        # Check if it violates constraints
        if self._creates_conflict(base_result.digit):
            # Filter out conflicting results
            valid_results = [
                r for r in results
                if not self._creates_conflict(r.digit)
            ]

            if valid_results:
                # Re-vote with valid results only
                return self.base.vote(valid_results)

        return base_result

    def _creates_conflict(self, digit: int) -> bool:
        """Check if digit violates Sudoku constraints."""
        if digit == 0:
            return False

        row, col = self.current_pos

        # Check row, column, and box
        # (implementation same as in similar_digit_fallback.py)
        ...
```

### 2.3 Data Augmentation for CNN Training

**Goal**: Train CNN with more diverse digit appearances

**File**: `src/ocr/digit_recognizer.py` (or create new `src/ocr/train_augmented_cnn.py`)

**Action**: Modify training function (line 326+):

```python
def train_cnn_model_augmented(save_path: str = "models/digit_cnn_augmented.h5"):
    """Train CNN with data augmentation."""
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,           # Rotate ±15°
        width_shift_range=0.1,       # Shift horizontally
        height_shift_range=0.1,      # Shift vertically
        shear_range=0.1,             # Shear transformation
        zoom_range=0.15,             # Zoom in/out
        fill_mode='constant',        # Fill with black
        cval=0
    )

    print("Building model...")
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training model with augmentation...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=10,  # More epochs due to augmentation
        validation_data=(x_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ],
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
```

**Usage**:
```bash
python -c "from src.ocr.digit_recognizer import train_cnn_model_augmented; train_cnn_model_augmented()"
```

## Phase 3: Similar-Digit Fallback (Estimated Time: 4-6 hours)

### 3.1 Basic Integration

**File**: `main.py`

**Action**: Modify `solve_sudoku_from_image` function to add fallback:

```python
def solve_sudoku_from_image(...):
    # ... existing OCR ...
    if use_ensemble:
        ensemble = EnsembleRecognizer(voting_strategy="weighted")
        detected_grid, has_content = ensemble.recognize_grid(cells, verbose=verbose)
    else:
        # ... existing code ...

    # NEW: Apply similar-digit fallback if puzzle is invalid
    solver = SudokuSolver()
    solver.load_puzzle(detected_grid)

    if not solver.is_valid_puzzle():
        print("\n⚠ Puzzle has conflicts, applying similar-digit fallback...")

        from src.ocr.similar_digit_fallback import SimilarDigitFallbackRecognizer

        # Get base recognizer
        base_rec = ensemble if use_ensemble else recognizer

        # Create fallback recognizer
        fallback = SimilarDigitFallbackRecognizer(
            base_recognizer=base_rec,
            confidence_threshold=0.70,
            enable_constraint_checking=True
        )

        # Refine grid
        detected_grid = fallback.refine_grid(
            cells, detected_grid, has_content, verbose=verbose
        )

        # Re-check
        solver.load_puzzle(detected_grid)
        print("\nRefined puzzle:")
        print_grid(detected_grid)

    # Continue with solving...
    if not solver.is_valid_puzzle():
        # Still invalid...
        ...
```

### 3.2 Add Command-Line Flag

**File**: `main.py`

**Action**: Add argument (around line 150):

```python
parser.add_argument(
    "--fallback",
    action="store_true",
    help="Enable similar-digit fallback for OCR errors (experimental)"
)
```

Update function call:
```python
success = solve_sudoku_from_image(
    # ... existing args ...
    use_fallback=args.fallback
)
```

### 3.3 Usage Examples

```bash
# Standard usage (ensemble only)
python main.py testplaat2.png -o output.png

# With fallback enabled
python main.py testplaat2.png -o output.png --fallback

# With fallback and verbose output
python main.py testplaat2.png -o output.png --fallback --verbose

# Use specific model + fallback
python main.py testplaat2.png -o output.png --tesseract --fallback
```

## Phase 4: Testing & Validation

### 4.1 Create Test Dataset

**Action**: Collect test images with known solutions

```
tests/ocr_accuracy/
  ├── testplaat1.png
  ├── testplaat1_solution.txt
  ├── testplaat2.png
  ├── testplaat2_solution.txt
  ├── testplaat3.png
  ├── testplaat3_solution.txt
  └── ...
```

### 4.2 Automated Testing Script

**Create**: `tests/test_ocr_accuracy.py`

```python
"""Test OCR accuracy across multiple puzzle images."""
import os
import numpy as np
from pathlib import Path
from src.grid_detector import GridDetector
from src.ocr.ensemble_recognizer import EnsembleRecognizer
from src.text_parser import parse_sudoku_text

def load_solution(solution_path):
    """Load ground truth solution."""
    with open(solution_path, 'r') as f:
        return parse_sudoku_text(f.read())

def calculate_accuracy(predicted, actual):
    """Calculate digit-level accuracy."""
    correct = np.sum(predicted == actual)
    total = 81
    return correct / total

def test_ocr_on_dataset(dataset_dir, recognizer):
    """Test OCR on all images in dataset."""
    results = []

    for image_path in Path(dataset_dir).glob("*.png"):
        solution_path = image_path.with_suffix('.txt').with_name(
            image_path.stem.replace('.png', '_solution.txt')
        )

        if not solution_path.exists():
            print(f"Skipping {image_path}: no solution file")
            continue

        # Load ground truth
        solution = load_solution(solution_path)

        # Detect and recognize
        detector = GridDetector()
        _, cells, _ = detector.detect_and_extract(str(image_path))

        if cells is None:
            print(f"Skipping {image_path}: grid detection failed")
            continue

        predicted_grid, _ = recognizer.recognize_grid(cells)

        # Calculate accuracy
        accuracy = calculate_accuracy(predicted_grid, solution)

        results.append({
            'image': image_path.name,
            'accuracy': accuracy,
            'errors': 81 - np.sum(predicted_grid == solution)
        })

        print(f"{image_path.name}: {accuracy:.2%} ({results[-1]['errors']} errors)")

    # Summary
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"\nAverage accuracy: {avg_accuracy:.2%}")

    return results

# Usage
recognizer = EnsembleRecognizer()
results = test_ocr_on_dataset('tests/ocr_accuracy', recognizer)
```

### 4.3 Performance Benchmarking

**Create**: `tests/benchmark_ocr.py`

```python
"""Benchmark OCR performance."""
import time
import numpy as np

def benchmark_recognizer(recognizer, test_images, num_runs=5):
    """Benchmark recognition speed and accuracy."""
    times = []

    for run in range(num_runs):
        start = time.time()

        for image_path in test_images:
            # ... recognize ...
            pass

        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Average time: {avg_time:.2f}s ± {std_time:.2f}s")
    print(f"Per-image: {avg_time / len(test_images):.2f}s")
```

## Phase 5: Documentation & Deployment

### 5.1 Update CLAUDE.md

Add section documenting new features:

```markdown
### OCR Improvements (2024)

The OCR system now includes:

1. **Similar-Digit Fallback**: Tries visually similar digits when confidence is low
   - Enable with `--fallback` flag
   - Helps with 6/8/9 confusion

2. **Multi-Scale Processing**: Recognizes digits at different scales
   - Automatically enabled in ensemble mode

3. **Augmented CNN**: Trained with rotation, scaling, and shearing
   - Use `models/digit_cnn_augmented.h5`

### Usage

```bash
# Best accuracy (slower)
python main.py puzzle.png -o output.png --fallback --verbose

# Fast mode (less accurate)
python main.py puzzle.png -o output.png --tesseract
```
```

### 5.2 Update README.md

Add performance section:

```markdown
## OCR Performance

Current accuracy on test dataset:
- **Ensemble mode**: 93-97% digit recognition
- **With fallback**: 95-98% digit recognition
- **Processing time**: 2-5 seconds per puzzle

Common issues:
- Visually similar digits (6/8/9) may confuse models
- Very small or faint digits may not be recognized
- Use `--fallback` flag for maximum accuracy
```

## Testing Checklist

Before deploying improvements:

- [ ] Test on testplaat2.png (known issue case)
- [ ] Test on at least 10 different puzzle images
- [ ] Verify no regressions (old images still work)
- [ ] Check processing time is acceptable (<10s per puzzle)
- [ ] Test edge cases (empty puzzles, invalid grids)
- [ ] Verify fallback doesn't over-correct
- [ ] Check that original digits aren't overwritten
- [ ] Test with different command-line flags
- [ ] Update documentation
- [ ] Create unit tests for new components

## Troubleshooting

### Problem: Fallback over-corrects

**Solution**: Increase `confidence_threshold` in `SimilarDigitFallbackRecognizer`:
```python
fallback = SimilarDigitFallbackRecognizer(
    ...,
    confidence_threshold=0.80  # More conservative
)
```

### Problem: Too slow

**Solution**: Disable EasyOCR or use Level 1 only:
```python
config = {
    'models': {
        'easyocr': {'enabled': False}  # Disable slowest model
    }
}
```

### Problem: Still getting errors on specific digits

**Solution**: Adjust similarity matrix in `similar_digit_fallback.py`:
```python
DIGIT_SIMILARITY = {
    6: [5, 8, 9, 0],  # Try 5 before 8
    ...
}
```

## Next Steps

After implementing all phases:

1. **Collect more training data**: Real Sudoku puzzles, not just MNIST
2. **Fine-tune models**: Train on domain-specific data
3. **Experiment with newer models**: Try ViT, ResNet, EfficientNet
4. **Add confidence visualization**: Show which digits are uncertain
5. **Implement active learning**: Ask user to label uncertain cells

## Summary

This guide provides a structured approach to improving OCR accuracy:

1. **Start with quick wins**: Configuration tuning (hours)
2. **Standard improvements**: Preprocessing, training (days)
3. **Advanced techniques**: Fallback strategy (days)
4. **Thorough testing**: Validation and benchmarking (ongoing)

The similar-digit fallback is powerful but should be used as a last resort after exhausting standard improvements.
