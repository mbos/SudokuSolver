# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Sudoku solver that processes images of Sudoku puzzles and outputs solved versions. It uses computer vision for grid detection, OCR for digit recognition, and a hybrid constraint propagation + backtracking algorithm for solving.

## Common Commands

### Setup and Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Train the CNN model for digit recognition (one-time setup)
python -m src.ocr
```

### Running the Solver

```bash
# Basic usage with CNN
python main.py testplaatje.png -o solved.png

# Using Tesseract OCR (no training needed)
python main.py testplaatje.png -o solved.png --tesseract

# Verbose mode (ASCII output showing what will be drawn)
python main.py testplaatje.png -o solved.png --tesseract --verbose

# Debug mode (shows intermediate steps with image windows)
python main.py testplaatje.png -o solved.png --debug

# Display result in window
python main.py testplaatje.png -o solved.png --show
```

### Testing

```bash
# Run tests (when implemented)
pytest tests/

# Test individual modules
python -m src.grid_detector
python -m src.solver
```

## Architecture

The codebase follows a modular pipeline architecture with four main components:

### 1. Grid Detection (`src/grid_detector.py`)
- **Purpose**: Extract Sudoku grid from images
- **Key Methods**:
  - `preprocess_image()`: Grayscale → Blur → Adaptive Threshold
  - `find_grid_contour()`: Detects largest 4-corner contour
  - `apply_perspective_transform()`: Gets bird's eye view using `cv2.getPerspectiveTransform`
  - `extract_cells()`: Divides 450x450 grid into 81 cells (9x9)
- **Note**: Debug mode shows intermediate visualizations

### 2. OCR Module (`src/ocr.py`)
- **Purpose**: Recognize digits in extracted cells
- **Two Recognition Methods**:
  1. **CNN (Primary)**: Custom model trained on MNIST
     - Input: 28x28 grayscale images
     - Architecture: 3 Conv2D layers + Dense layers + Dropout
     - Confidence threshold: 0.7
  2. **Tesseract (Fallback)**: pytesseract with config `--psm 10 --oem 3`
     - Upscales images 4x before recognition
     - Whitelist: digits 1-9 only
- **Key Methods**:
  - `preprocess_cell()`: Thresholding, morphology, contour extraction
    - Returns tuple: `(preprocessed_image, is_empty_flag)`
    - Empty detection: < 3% white pixels after thresholding
  - `recognize_digit()`: Returns 0 for empty cells, 1-9 for digits
  - `recognize_grid()`: Processes all 81 cells
    - Returns tuple: `(detected_grid, has_content_mask)`
    - `has_content_mask` indicates which cells have visual content (bool array)
    - This separates "cell has content" from "OCR recognized content"
- **Training**: Run `python -m src.ocr` to train on MNIST (5 epochs, ~99% accuracy)

### 3. Solver (`src/solver.py`)
- **Purpose**: Solve Sudoku using hybrid algorithm
- **Algorithm**: Constraint Propagation + Backtracking with MRV heuristic
- **Key Techniques**:
  1. **Constraint Propagation**:
     - Naked singles: Cells with only one possibility
     - Hidden singles: Numbers with only one valid position in row/col/box
  2. **Backtracking**:
     - MRV heuristic: Always fill cell with fewest possibilities
     - Forward checking: Update constraints after each placement
- **Key Methods**:
  - `load_puzzle()`: Initialize from 9x9 array (0 = empty)
  - `solve()`: Main solving method
  - `is_valid_puzzle()`: Check for duplicate violations
- **Performance**: 1.3-3x faster than pure backtracking

### 4. Image Generator (`src/image_generator.py`)
- **Purpose**: Visualize solution on original image
- **Key Methods**:
  - `draw_on_warped()`: Draws solution digits in red on warped grid
    - Parameters: `(warped_grid, has_content_mask, solution)`
    - Only draws where `has_content[row, col] == False`
    - This prevents overwriting original digits, even if OCR failed to read them
  - `overlay_solution()`: (Future) Inverse perspective transform back to original
  - `save_result()`: Saves output image
- **Note**: Currently outputs warped view; perspective overlay can be added

## Data Flow

```
Input Image
  → GridDetector.detect_and_extract()
    → [original_image, 81 cells, warped_grid]
  → DigitRecognizer.recognize_grid()
    → (detected_grid, has_content_mask)
      - detected_grid: 9x9 numpy array (0 = empty/unrecognized)
      - has_content_mask: 9x9 bool array (True = cell has visual content)
  → SudokuSolver.solve()
    → 9x9 solved array
  → SolutionDrawer.draw_on_warped(warped_grid, has_content_mask, solution)
    → Output Image (only draws red digits where has_content == False)
```

## Key Implementation Details

### Grid Detection
- Uses `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C` for varying lighting
- Looks for 4-corner contours with area > 1000 pixels
- Perspective transform outputs 450x450 image (divisible by 9)
- Cell extraction includes 5-pixel margin to avoid grid lines

### OCR Preprocessing
- Empty cell detection: < 3% white pixels after thresholding
- Morphological operations to remove noise
- Extracts largest contour per cell (the digit)
- CNN confidence threshold prevents false positives

### Solver State Management
- Maintains `possibilities` set for each cell (1-9)
- Deep copies state before backtracking moves
- Validates puzzle before solving (no duplicate violations)
- Returns False if contradiction found during propagation

### Protection Against Overwriting Original Digits
**IMPORTANT**: The system now protects original digits from being overwritten, even when OCR fails to recognize them.

- `recognize_grid()` returns a tuple: `(detected_grid, has_content_mask)`
- `has_content_mask` tracks which cells have visual content (regardless of OCR success)
- `draw_on_warped()` only draws red solution digits where `has_content[row, col] == False`
- This prevents OCR failures from causing original digits to be overwritten

### Common Pitfalls
- **Grid not detected**: Ensure grid is largest object, good contrast
- **OCR failures**: Small/faint digits fail with Tesseract, use CNN
- **Unsolvable puzzle**: Usually due to OCR errors, check debug output
- **Model not found**: Run `python -m src.ocr` to train CNN first

## Dependencies

- `opencv-python`: Image processing and computer vision
- `numpy`: Array operations
- `tensorflow/keras`: CNN training and inference
- `pytesseract`: Tesseract OCR wrapper
- `Pillow`: Image I/O
- `matplotlib`: Visualization (optional)

## Known Issues and Performance Analysis

### OCR Accuracy on testplaatje.png

**Test Results** (analyzed against testplaatje_oplossing.txt):
- **Total starting digits**: 25
- **Correctly detected**: 21 (84% accuracy)
- **Missed digits**: 4 (16%)
- **Incorrectly detected**: 0

**Missed Digits** (have visual content but OCR returned 0):
- Cell (0,7): Should be 6
- Cell (1,4): Should be 9
- Cell (8,3): Should be 9
- Cell (8,6): Should be 8

**Root Cause Analysis**:
1. **OCR Module** (`src/ocr.py`):
   - Tesseract fails to recognize some digits despite clear visual content
   - `preprocess_cell()` detects content (>3% fill ratio), but OCR confidence is too low
   - All 4 missed cells have ~12-13% fill ratio (well above 3% threshold)

2. **Solver Module** (`src/solver.py`):
   - **Solver algorithm works correctly** - produces a valid Sudoku solution
   - However, with incomplete input (4 missing digits), it solves a **different puzzle**
   - Result: 38 cells differ from the intended solution

3. **Image Generation** (`src/image_generator.py`):
   - **Protection works correctly** - original digits are NOT overwritten
   - Cells with unrecognized content remain visible (shown as [B]? in verbose mode)
   - Only truly empty cells receive red solution digits

**Conclusion**: The pipeline's accuracy bottleneck is OCR reliability, not the solver or protection logic.

### Debugging Tools

```bash
# Show verbose output with ASCII visualization
python main.py testplaatje.png -o solved.png --tesseract --verbose

# Inspect cells that OCR couldn't recognize
python inspect_cells.py

# Analyze OCR accuracy against known solution
python analyze_ocr_accuracy.py
```

The verbose flag shows:
- Which cells have content but weren't recognized
- Final visualization: `[B]` = black (original), `[R]` = red (solver-added)
- Warnings when OCR misses digits

## Code Change Protocol

When making code changes, **always check if documentation needs to be updated**. This ensures consistency between code and documentation.

### Documentation Update Checklist

After making code changes, verify if any of the following need updates:

1. **Command-line interfaces changed?**
   - [ ] Update README.md usage examples
   - [ ] Update CLAUDE.md "Common Commands" section
   - [ ] Update `main.py` help text and error messages
   - [ ] Update any shell scripts that call the changed commands

2. **File structure changed?** (files moved, renamed, or reorganized)
   - [ ] Update README.md "Project Structure" section
   - [ ] Update CLAUDE.md "Architecture" section
   - [ ] Update import statements in all affected files
   - [ ] Update any documentation references to file paths

3. **API/function signatures changed?**
   - [ ] Update CLAUDE.md "Key Methods" documentation
   - [ ] Update inline code comments
   - [ ] Update any example code in documentation

4. **New dependencies added?**
   - [ ] Update `requirements.txt`
   - [ ] Update README.md "Dependencies" section
   - [ ] Update installation instructions

5. **Algorithm or behavior changed?**
   - [ ] Update CLAUDE.md architecture descriptions
   - [ ] Update README.md "Algorithm Details" section
   - [ ] Update comments explaining the logic

6. **New features or options added?**
   - [ ] Add to README.md feature list
   - [ ] Update command-line help text
   - [ ] Add usage examples
   - [ ] Update CLAUDE.md with implementation details

**Standard practice**: Always use `grep -r` to search for references to changed elements before considering a change complete.

Example:
```bash
# If you renamed src/ocr.py, search for all references:
grep -r "src/ocr.py" .
grep -r "python src/ocr" .
```

## Note on Repository Name

The directory is named "sudoka_solver" - this is intentional
