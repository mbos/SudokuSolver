# OCR Accuracy Improvement Analysis

## Problem Statement

Testplaat2 analysis shows OCR errors causing invalid puzzles:
- **Error**: 6 being recognized as 8 in row 4
- **Result**: Duplicate 9s in Row 0 and Column 8
- **Root Cause**: Visually similar digits (6, 8, 9) have overlapping features

From the output:
```
Error: Detected puzzle is invalid (contains duplicates)

Validation found 2 error(s):
  - Row 0: Duplicate values [9]
  - Column 8: Duplicate values [9]

This is likely due to OCR errors.
```

## Current OCR Architecture

The system already has a sophisticated **3-level ensemble architecture**:

### Level 1 (Fast Path)
- **CNN** (custom MNIST-trained, weight 1.5)
- **Tesseract** (PSM mode 10, weight 1.0)
- Threshold: 0.75 confidence

### Level 2 (Medium Path)
- Add **EasyOCR** (weight 2.0)
- Threshold: 0.65 confidence

### Level 3 (Full Ensemble)
- Weighted voting across all models
- Uses confidence-based aggregation

### Existing Optimizations

**Preprocessing (digit_recognizer.py:45-124)**:
- Dark theme detection (inverted cell handling)
- CLAHE contrast enhancement
- Bilateral filtering for noise reduction
- Otsu's thresholding
- Morphological operations
- Empty cell detection (4% fill ratio)
- Contour-based digit extraction
- MNIST-style padding and centering

**CNN Optimizations (digit_recognizer.py:194-210)**:
- Adaptive confidence thresholds per digit:
  - Digits 6, 8, 9: 0.65 (lower threshold due to similarity)
  - Digit 1: 0.70 (can confuse with 7)
  - Digit 0: 0.85 (higher to avoid false positives)
  - Default: 0.75

**Tesseract Optimizations (digit_recognizer.py:212-266)**:
- 8x upscaling (increased from 4x)
- Multiple PSM modes [10, 8, 7, 13]
- Strict whitelist: digits 1-9 only
- Custom matching parameters

**Voting Strategies**:
- Majority voting with agreement boost
- Weighted voting with model-specific weights
- Confidence aggregation with adaptive thresholds

## Strategy 1: Standard OCR Improvements

Before implementing advanced fallback strategies, exhaust these standard improvements:

### 1.1 Preprocessing Enhancements

**A. Multi-Scale Processing**
- Currently processes cells at one scale
- Proposal: Try recognition at 3 different scales (0.8x, 1.0x, 1.2x)
- Helps with:
  - Digits that are too thick/thin
  - OCR sensitivity to stroke width

**B. Advanced Noise Removal**
- Add **Gaussian blur** before thresholding (reduce salt-and-pepper noise)
- Try **median filter** as alternative to bilateral filter
- Experiment with adaptive morphological operations based on digit thickness

**C. Rotation Correction**
- Detect slight digit rotation (±5°)
- Apply minor rotation corrections
- Helps with digits that aren't perfectly upright

**D. Contrast Enhancement Alternatives**
- Try **histogram equalization** instead of CLAHE for some cases
- Experiment with **gamma correction**
- Test multiple preprocessing variations per cell

### 1.2 Model Improvements

**A. CNN Model Enhancement**
- Current: 5 epochs on vanilla MNIST
- Proposal: **Data augmentation** during training:
  - Rotation: ±15°
  - Scaling: 0.8x - 1.2x
  - Slight shearing/perspective transform
  - Noise injection
  - Stroke thickness variation
- Train for **10 epochs** instead of 5
- Add **early stopping** with validation monitoring
- Experiment with **ensemble of multiple CNNs** trained with different augmentations

**B. Fine-Tuning on Real Sudoku Data**
- Collect failed recognition cases (using `has_content` mask)
- Manually label 100-500 real Sudoku digits
- Fine-tune CNN on this domain-specific data
- Addresses: printed font differences, grid line artifacts

**C. Model Architecture Improvements**
- Current: Simple 3-layer CNN
- Proposal: Try **ResNet-style skip connections**
- Or: Use **EfficientNet-B0** pre-trained and fine-tuned
- Or: **Ensemble multiple CNN architectures**

### 1.3 Voting Strategy Enhancements

**A. Context-Aware Voting**
- **Sudoku constraint checking** during voting
- If a digit creates a duplicate, **downweight that prediction**
- Example: If row already has a 9, and models vote [9, 8, 9], prefer 8

**B. Uncertainty-Based Retry**
- Track which cells have **low voting consensus** (close vote splits)
- Re-process those cells with:
  - Different preprocessing parameters
  - Additional models
  - Higher quality scaling

**C. Iterative Refinement**
- After first pass, identify cells creating conflicts
- Re-recognize only those cells with stricter thresholds
- Use Sudoku constraints to guide second attempt

### 1.4 Configuration Tuning

**A. Confidence Threshold Optimization**
- Current thresholds may be suboptimal
- **Grid search** optimal thresholds:
  - Level 1: try 0.70, 0.75, 0.80
  - Level 2: try 0.60, 0.65, 0.70
  - Per-digit thresholds: optimize for 6/8/9 specifically

**B. Model Weight Optimization**
- Current: CNN=1.5, Tesseract=1.0, EasyOCR=2.0
- Find optimal weights using **validation dataset**
- Consider per-digit weights (EasyOCR might be better for 6/8/9)

**C. Fallback Order Optimization**
- Test different model orderings
- Maybe EasyOCR should be Level 1 (it has weight 2.0)

## Strategy 2: Similar-Digit Fallback (Novel Approach)

**When to use**: After exhausting Strategy 1 improvements

### 2.1 Concept

When OCR confidence is low or results cause Sudoku conflicts, retry recognition by **biasing toward visually similar digits**.

**Similar digit groups**:
- **{6, 8, 9, 5}**: Rounded shapes, closed loops
- **{1, 7}**: Vertical strokes
- **{3, 8}**: Stacked curves
- **{2, 3}**: Similar curvature
- **{5, 6}**: Top horizontal line with curves

### 2.2 Implementation Strategy

**A. Define Similarity Matrix**
```python
DIGIT_SIMILARITY = {
    6: [8, 9, 5, 0],  # Most to least similar
    8: [6, 9, 3, 0],
    9: [8, 6, 4, 7],
    5: [6, 8, 3],
    1: [7, 4],
    7: [1, 9, 4],
    3: [8, 5, 2],
    2: [3, 7],
    4: [9, 1],
    0: [6, 8, 9],
}
```

**B. Confidence-Based Trigger**
```python
class SimilarDigitFallbackRecognizer:
    def __init__(self, base_recognizer, confidence_threshold=0.70):
        self.base = base_recognizer
        self.threshold = confidence_threshold

    def recognize_with_fallback(self, cell, row, col, current_puzzle):
        # Get base prediction
        result = self.base.recognize(cell)

        # Check if we should try fallback
        should_retry = (
            result.confidence < self.threshold or
            self._creates_conflict(result.digit, row, col, current_puzzle)
        )

        if not should_retry:
            return result.digit

        # Try similar digits
        candidates = DIGIT_SIMILARITY.get(result.digit, [])
        return self._try_similar_digits(cell, candidates, row, col, current_puzzle)
```

**C. Biased Recognition**
```python
def _try_similar_digits(self, cell, candidates, row, col, puzzle):
    """Try recognizing as similar-looking digits."""
    scores = {}

    for candidate_digit in candidates:
        # Skip if would create conflict
        if self._creates_conflict(candidate_digit, row, col, puzzle):
            continue

        # Get model confidence for this specific digit
        score = self._get_digit_likelihood(cell, candidate_digit)

        # Boost score if resolves conflicts
        if self._resolves_conflicts(candidate_digit, row, col, puzzle):
            score *= 1.2

        scores[candidate_digit] = score

    if not scores:
        return 0  # No valid candidates

    # Return highest scoring candidate
    best_digit = max(scores, key=scores.get)
    if scores[best_digit] > 0.5:  # Minimum threshold
        return best_digit
    return 0
```

**D. Extract Per-Digit Confidence**
```python
def _get_digit_likelihood(self, cell, target_digit):
    """Get model's confidence that this cell contains target_digit."""
    # For CNN
    if hasattr(self.base, 'model'):
        preprocessed = self.base.preprocess_cell(cell)[0]
        resized = self.base.resize_to_mnist_format(preprocessed)
        normalized = resized.astype(np.float32) / 255.0
        input_data = normalized.reshape(1, 28, 28, 1)
        predictions = self.base.model.predict(input_data, verbose=0)
        return float(predictions[0][target_digit])

    # For Tesseract/EasyOCR: use heuristics or confidence data
    return 0.0
```

**E. Sudoku Constraint Checking**
```python
def _creates_conflict(self, digit, row, col, puzzle):
    """Check if placing digit at (row,col) creates Sudoku violation."""
    if digit == 0:
        return False

    # Check row
    if digit in puzzle[row, :]:
        return True

    # Check column
    if digit in puzzle[:, col]:
        return True

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    box = puzzle[box_row:box_row+3, box_col:box_col+3]
    if digit in box:
        return True

    return False
```

### 2.3 Integration Points

**Option A: Post-Processing Layer**
```python
def solve_sudoku_from_image(...):
    # ... existing OCR ...
    detected_grid, has_content = ensemble.recognize_grid(cells)

    # NEW: Apply similar-digit fallback
    if not solver.is_valid_puzzle():
        print("Applying similar-digit fallback for conflicts...")
        fallback = SimilarDigitFallbackRecognizer(ensemble)
        detected_grid = fallback.refine_grid(
            cells, detected_grid, has_content
        )
```

**Option B: Integrated into Ensemble Level 4**
```python
class EnsembleRecognizer:
    def recognize_digit(self, cell, ...):
        # ... existing Level 1-3 ...

        # Level 4: Similar-digit fallback
        if result.confidence < threshold:
            similar_result = self._try_similar_digits(cell, result)
            if similar_result.confidence > result.confidence:
                return similar_result
```

### 2.4 Advantages & Disadvantages

**Advantages**:
- Leverages visual similarity between digits
- Uses Sudoku constraints as validation
- Minimal computational overhead (only on low-confidence cells)
- Can correct OCR errors without retraining models
- Addresses the 6/8/9 confusion directly

**Disadvantages**:
- Relies on puzzle constraints (doesn't work for non-puzzle OCR)
- May mask underlying OCR model weaknesses
- Risk of overfitting to Sudoku-specific patterns
- Harder to debug (adds another layer of complexity)

## Strategy 3: Hybrid Approach (Recommended)

**Phase 1: Standard Improvements (Weeks 1-2)**
1. Implement multi-scale processing
2. Add data augmentation to CNN training
3. Optimize confidence thresholds via grid search
4. Tune model weights
5. Add context-aware voting

**Phase 2: Evaluation (Week 3)**
- Test on testplaat2 and collect 10-20 more test images
- Measure accuracy improvement
- Identify remaining failure patterns

**Phase 3: Similar-Digit Fallback (Week 4)**
- If accuracy still <95%, implement fallback
- Start with conservative similarity matrix
- Integrate as post-processing layer first
- Monitor for overcorrection

**Phase 4: Fine-Tuning (Week 5)**
- Collect real-world failures
- Fine-tune CNN on domain-specific data
- Adjust similarity matrix based on results

## Implementation Priority

### High Priority (Try First)
1. ✅ **Confidence threshold optimization** - Quick win, no code changes
2. ✅ **Model weight tuning** - Quick win, config only
3. ✅ **Context-aware voting** - Moderate effort, high impact
4. ✅ **Multi-scale preprocessing** - Moderate effort, medium impact

### Medium Priority (Try Second)
5. ✅ **Data augmentation training** - High effort, high impact
6. ✅ **Iterative refinement** - Moderate effort, medium impact
7. ✅ **PSM mode optimization** - Low effort, low-medium impact

### Low Priority (Try Last / If Needed)
8. **Similar-digit fallback** - High effort, medium-high impact (risk)
9. **Fine-tuning on real data** - Very high effort, high impact (needs data collection)
10. **Model architecture changes** - Very high effort, high impact (risky)

## Testing Protocol

For each improvement:
1. **Baseline**: Run on testplaat2 without changes, record errors
2. **Apply**: Implement single improvement
3. **Test**: Re-run on testplaat2, measure:
   - Digit recognition accuracy
   - Puzzle validity rate
   - Confidence scores per digit
   - Processing time
4. **Validate**: Test on 5+ other puzzle images
5. **Document**: Record what works and what doesn't

## Metrics to Track

- **Per-digit accuracy**: Especially 6, 8, 9
- **Puzzle validity rate**: % of puzzles without conflicts
- **Confidence distribution**: Are we too conservative/aggressive?
- **False positive rate**: Empty cells incorrectly filled
- **False negative rate**: Digit cells left empty
- **Processing time**: Ensure improvements don't slow down too much

## Expected Outcomes

With Strategy 1 (Standard Improvements):
- **Target**: 95-98% digit recognition accuracy
- **Realistic**: 90-95% (some puzzles will always be hard)
- **Processing time**: <5 seconds per puzzle

With Strategy 1 + 2 (Similar-Digit Fallback):
- **Target**: 98-99% digit recognition accuracy
- **Realistic**: 95-98%
- **Risk**: May introduce false corrections in edge cases

## Conclusion

**Recommendation**: Start with Strategy 1 (Standard Improvements), focusing on:
1. Configuration tuning (quick wins)
2. Context-aware voting (leverages Sudoku constraints without risks)
3. Multi-scale preprocessing (addresses root cause of OCR failures)

Only move to Strategy 2 (Similar-Digit Fallback) if Strategy 1 doesn't achieve >93% accuracy on testplaat2 and other test images.

The similar-digit fallback is clever but should be a last resort, as it:
- Adds complexity
- May mask model weaknesses
- Is Sudoku-specific (not generalizable)

Better to fix the underlying OCR models first through proper preprocessing, training, and configuration.
