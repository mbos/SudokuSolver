# Strategy for Handling OCR Errors in Sudoku Solver

## Problem Analysis

### Current Performance (testplaat2.png)
- **Total starting digits**: 25
- **Correctly detected**: 21 (84% accuracy)
- **OCR errors**: 7 errors
  - Missed digits: 3
  - Wrong digits: 1
  - False positives: 3

### Critical Issue
OCR errors create **invalid puzzles** with constraint violations (duplicate digits in rows/columns), making them unsolvable by the constraint-based solver.

### Error Pattern
Errors cluster at the edges of the grid (positions 7-8 in rows 0-1), suggesting potential **grid alignment issues**.

---

## Strategy Options

### Option 1: Improve OCR Accuracy (Preventive Approach)
**Goal**: Reduce OCR errors at the source

**Approach 1A: Use CNN Instead of Tesseract**
- ✅ **Pros**:
  - CNN already exists in codebase, trained on MNIST
  - Typically more accurate for printed digits
  - Confidence scores help identify uncertain predictions
- ❌ **Cons**:
  - Requires trained model
  - May still have errors on low-quality images

**Approach 1B: Ensemble OCR Voting**
- Use multiple OCR methods (CNN + Tesseract + EasyOCR)
- Vote on final result, flag disagreements as uncertain
- ✅ **Pros**: Higher accuracy through consensus
- ❌ **Cons**: Slower, more complex

**Approach 1C: Improve Grid Detection & Alignment**
- Fine-tune perspective transform
- Add sub-pixel alignment
- Better cell extraction margins
- ✅ **Pros**: Fixes root cause of edge errors
- ❌ **Cons**: Complex, may not fix all errors

---

### Option 2: Error Correction in Solver (Reactive Approach)
**Goal**: Handle invalid puzzles gracefully

**Approach 2A: Invalid Puzzle Detection with Error Correction**
- **Step 1**: Detect invalid puzzle (duplicates in rows/cols/boxes)
- **Step 2**: Identify conflicting cells
- **Step 3**: Use OCR confidence scores to determine which digits to question
- **Step 4**: Try alternative interpretations (e.g., if "9" has low confidence, try "4", "7", "8")
- **Step 5**: Re-validate puzzle after each correction

- ✅ **Pros**:
  - Handles OCR errors without perfect OCR
  - User gets feedback on corrections made
  - Works even with new image types
- ❌ **Cons**:
  - Computationally expensive
  - May not always find correct interpretation
  - Could mask underlying OCR issues

**Approach 2B: Constraint-Based Error Correction**
- When solver finds invalid puzzle:
  1. Identify cells involved in violations
  2. Mark them as "uncertain"
  3. Use constraint propagation to determine correct values
  4. Example: If row has duplicate 9s, check which cells make puzzle solvable

- ✅ **Pros**:
  - Uses puzzle structure to self-correct
  - Elegant solution
- ❌ **Cons**:
  - May have multiple valid corrections
  - Computationally complex

**Approach 2C: Multiple Hypothesis Testing**
- For cells with low OCR confidence, track multiple possibilities
- Solver tries each combination until valid solution found
- ✅ **Pros**: Guaranteed to find solution if one exists
- ❌ **Cons**: Exponential complexity, very slow

---

## Recommended Strategy: Hybrid Approach

### Phase 1: Improve OCR (Quick Win)
1. **Switch from Tesseract to CNN by default**
   - CNN is already trained and available
   - Should improve accuracy immediately
   - Keep Tesseract as fallback option

2. **Add OCR confidence tracking**
   - Modify `recognize_digit()` to return (digit, confidence)
   - Store confidence scores for each cell
   - Flag cells with confidence < 0.7 as "uncertain"

### Phase 2: Add Validation & Error Correction (Robustness)
3. **Enhance puzzle validation**
   - Current: `is_valid_puzzle()` detects duplicates
   - Add: Return list of conflicting cells, not just True/False
   - Add: Suggest which cells might be errors based on confidence

4. **Implement Smart Error Correction**
   - When invalid puzzle detected:
     - Identify conflicting cells
     - Sort by OCR confidence (lowest first)
     - Try alternative digit interpretations for low-confidence cells
     - Re-validate after each change
     - Limit attempts to avoid infinite loops

### Phase 3: User Feedback (Transparency)
5. **Report corrections to user**
   - Show which cells had OCR errors corrected
   - Display confidence scores in verbose mode
   - Allow user to review corrections before solving

---

## Implementation Plan

### Step 1: Modify OCR to Track Confidence
**File**: `src/ocr/digit_recognizer.py`
- Change `recognize_digit()` to return `(digit, confidence)`
- Update `recognize_grid()` to return `(grid, has_content, confidence_matrix)`

### Step 2: Enhance Solver Validation
**File**: `src/solver.py`
- Add `find_invalid_cells()` method to return conflicting cells
- Add `get_constraint_violations()` to detail duplicates

### Step 3: Add Error Correction Module
**New File**: `src/error_corrector.py`
- Class `OCRErrorCorrector`
- Methods:
  - `detect_errors(grid, confidence_matrix)` → list of suspect cells
  - `try_corrections(grid, suspect_cells)` → corrected grid or None
  - `suggest_alternatives(digit, confidence)` → list of likely alternatives

### Step 4: Integrate into Main Pipeline
**File**: `main.py`
- After OCR, check if puzzle is valid
- If invalid, run error correction
- Report corrections to user
- Proceed with solving

---

## Expected Outcomes

### With CNN OCR (Phase 1)
- Expected accuracy improvement: **84% → 92-95%**
- Reduces errors from 7 to 2-3 per puzzle

### With Error Correction (Phase 2)
- Can handle remaining 5-8% of errors
- Most puzzles become solvable even with OCR errors
- User gets transparency on corrections

### Overall Success Rate
- Current: ~60% of puzzles solvable (estimate based on testplaat2 failure)
- After Phase 1: ~85-90% solvable
- After Phase 2: ~95-98% solvable

---

## Alternative: Quick Fix for Immediate Testing

If you need a quick solution for testing:
1. **Manual correction mode**: When puzzle invalid, show user the detected grid and ask for corrections
2. **Pre-validated test set**: Use known-good puzzle files for testing solver separately

---

## Recommendation

**Implement Phases 1 & 2** for a robust, production-ready solution that:
- Improves OCR accuracy immediately (CNN switch)
- Gracefully handles remaining errors (error correction)
- Provides transparency to users (confidence scores + correction reports)

This balances **development effort** with **real-world robustness**.
