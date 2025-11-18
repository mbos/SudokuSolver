# Sudoku Solver

An intelligent Sudoku solver that processes images of Sudoku puzzles and outputs solved versions. Uses computer vision for grid detection, OCR for digit recognition, and advanced algorithms for solving.

> **Note**: This project was vibe coded with AI assistance, demonstrating rapid prototyping and iterative development.

## Features

- **Automatic Grid Detection**: Detects Sudoku grids from images using OpenCV
- **Perspective Correction**: Handles photos taken at angles
- **Ensemble OCR (Default)**: Multi-model OCR system with intelligent fallback for superior accuracy
  - Combines CNN (MNIST-trained) and Tesseract OCR
  - Weighted voting strategy for robust digit recognition
  - Can also use single OCR method if preferred
- **Text File Input**: Solve puzzles from text files with formatted output in terminal
- **Advanced Solving Algorithm**: Hybrid approach using constraint propagation and backtracking with MRV heuristic (1.3-3x faster than pure backtracking)
- **Solution Validation**: Comprehensive validation with detailed error reporting
- **Training Data Collection**: Automatically collects labeled samples from successfully solved puzzles to improve CNN accuracy
- **Visual Output**: Generates solved puzzle image with solution filled in

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sudoka_solver
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Train the CNN model for better accuracy:
```bash
python src/ocr.py
```

This will download the MNIST dataset and train a CNN model, saving it to `models/digit_cnn.h5`.

## Usage

### Basic Usage

**Recommended - Ensemble OCR (default, best accuracy):**
```bash
python main.py testplaatje.png -o solved.png
```

**From text file:**
```bash
python main.py sudoku.txt
```

**Using only Tesseract OCR:**
```bash
python main.py testplaatje.png -o solved.png --tesseract
```

**Using only CNN (disable ensemble):**
```bash
python main.py testplaatje.png -o solved.png --no-ensemble
```

### Command Line Options

```bash
python main.py INPUT [-o OUTPUT] [-m MODEL] [OPTIONS]

Arguments:
  INPUT                 Path to input image or text file containing Sudoku puzzle

Options:
  -o, --output OUTPUT   Path to save output image (default: solved_sudoku.png)
  -m, --model MODEL     Path to CNN model (default: models/digit_cnn.h5)
  -t, --tesseract       Use Tesseract OCR instead of ensemble
  --no-ensemble         Disable ensemble mode, use only CNN or Tesseract
  -d, --debug           Enable debug mode (shows intermediate steps with images)
  -v, --verbose         Show verbose ASCII output without image windows
  -s, --show            Display result in window
  --no-collect          Don't collect training data from solved puzzles
```

**Text File Format:**
```
016 000 070
000 096 100
204 800 090

700 080 000
030 070 000
000 300 400

000 000 000
000 769 302
001 003 009
```
Where `0` = empty cell, spaces separate groups, blank lines separate 3x3 blocks.

### Examples

**Solve from image using ensemble OCR (default, best accuracy):**
```bash
python main.py testplaatje.png -o solved.png
```

**Solve from text file (displays output in terminal):**
```bash
python main.py puzzle.txt --verbose
```

**Use only Tesseract OCR:**
```bash
python main.py testplaatje.png -o solved.png --tesseract
```

**Debug mode with visualization:**
```bash
python main.py testplaatje.png -o solved.png --debug --show
```

**Verbose mode (ASCII visualization without image windows):**
```bash
python main.py testplaatje.png -o solved.png --verbose
```

## Architecture

The solver consists of four main modules:

### 1. Grid Detection (`src/grid_detector.py`)
- Preprocesses image (grayscale, blur, adaptive threshold)
- Detects largest contour (the Sudoku grid)
- Applies perspective transform for bird's eye view
- Extracts 81 individual cells

### 2. OCR (`src/ocr/`)
- **Ensemble Recognizer (default)**: Combines multiple OCR models for superior accuracy
  - Weighted voting strategy to resolve disagreements
  - Automatic fallback on low confidence
- **CNN-based recognition**: Custom model trained on MNIST dataset
- **Tesseract OCR**: Pytesseract integration for printed digits
- Preprocessing: thresholding, noise removal, digit extraction
- Empty cell detection
- Automatic training data collection from solved puzzles

### 3. Solver (`src/solver.py`)
- Constraint propagation with arc consistency
- Backtracking with MRV (Minimum Remaining Value) heuristic
- Forward checking
- Hidden singles detection
- Validates puzzle before solving

### 4. Image Generator (`src/image_generator.py`)
- Draws solution on warped grid
- Color-codes solved vs original digits
- Can overlay back onto original image with perspective transform

## Algorithm Details

### Solving Strategy

The solver uses a hybrid approach that combines:

1. **Constraint Propagation**: Eliminates impossible values based on Sudoku rules
   - Naked singles
   - Hidden singles in rows, columns, and boxes

2. **Backtracking with Heuristics**:
   - MRV (Minimum Remaining Value): Always fills cells with fewest possibilities first
   - Forward checking: Updates constraints after each move

This approach is 1.3-3x faster than pure backtracking, especially on harder puzzles.

### OCR Strategy

Three methods are available:

1. **Ensemble OCR (Default & Recommended)**:
   - Combines CNN and Tesseract OCR
   - Weighted voting for robust digit recognition
   - Best overall accuracy across different puzzle types
   - Automatic confidence-based fallback

2. **CNN Only**:
   - Trained on MNIST dataset (~99% accuracy)
   - Use with `--no-ensemble` flag
   - Good for handwritten digits
   - Requires one-time training

3. **Tesseract Only**:
   - Use with `--tesseract` flag
   - No training needed
   - Works well with printed/app fonts
   - Good for high-contrast digital images

## Dependencies

- OpenCV (cv2): Image processing and computer vision
- NumPy: Numerical operations
- TensorFlow/Keras: Neural network for digit recognition
- pytesseract: OCR fallback
- Pillow: Image handling
- matplotlib: Visualization (optional)

## Troubleshooting

### Grid Not Detected
- Ensure the grid is the largest rectangular object in the image
- Try adjusting lighting/contrast in the source image
- Use `--debug` flag to see intermediate processing steps

### Poor Digit Recognition
- **Ensemble mode (default) usually provides best accuracy**
- If using CNN only: Ensure model is trained (`python src/ocr.py`)
- Try different OCR modes: ensemble (default), `--tesseract`, or `--no-ensemble`
- Use `--debug` or `--verbose` to see which digits are being misrecognized
- The system automatically collects training data to improve accuracy over time
- Use `dev_tools/analyze_ocr_accuracy.py` to analyze OCR performance

### Puzzle Unsolvable
- Check OCR results in debug output
- Verify that detected digits are correct
- Some puzzles may be invalid or have multiple solutions

## Project Structure

```
sudoka_solver/
├── src/
│   ├── ocr/                          # OCR module
│   │   ├── ensemble_recognizer.py   # Multi-model ensemble OCR (default)
│   │   ├── cnn_recognizer.py        # CNN-based digit recognition
│   │   ├── tesseract_recognizer.py  # Tesseract OCR wrapper
│   │   ├── easyocr_recognizer.py    # EasyOCR integration
│   │   ├── voting_strategies.py     # Ensemble voting logic
│   │   └── base_recognizer.py       # Base class for recognizers
│   ├── grid_detector.py             # Grid detection & extraction
│   ├── solver.py                    # Sudoku solving algorithm
│   ├── image_generator.py           # Solution visualization
│   ├── validator.py                 # Solution validation
│   ├── text_parser.py               # Text file input/output
│   └── training_data_collector.py   # Auto-collect labeled data
├── dev_tools/                        # Development and analysis tools
│   ├── analyze_ocr_accuracy.py      # OCR accuracy analysis
│   ├── inspect_cells.py             # Cell-by-cell inspection
│   └── ...                          # Other debugging tools
├── models/
│   └── digit_cnn.h5                 # Trained CNN model (generated)
├── tests/                           # Unit and integration tests
│   ├── test_ensemble.py
│   └── test_voting_strategies.py
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
└── testplaatje.png                  # Sample puzzle image
```

