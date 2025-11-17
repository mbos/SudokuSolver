# Sudoku Solver

An intelligent Sudoku solver that processes images of Sudoku puzzles and outputs solved versions. Uses computer vision for grid detection, OCR for digit recognition, and advanced algorithms for solving.

## Features

- **Automatic Grid Detection**: Detects Sudoku grids from images using OpenCV
- **Perspective Correction**: Handles photos taken at angles
- **Multiple OCR Methods**:
  - Custom CNN trained on MNIST dataset
  - Tesseract OCR fallback
- **Advanced Solving Algorithm**: Hybrid approach using constraint propagation and backtracking with MRV heuristic (1.3-3x faster than pure backtracking)
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

**Recommended for printed/app Sudoku (like testplaatje.png):**
```bash
python main.py testplaatje.png -o solved.png --tesseract
```

**For handwritten Sudoku:**
```bash
python main.py handwritten.png -o solved.png
```

### Command Line Options

```bash
python main.py INPUT [-o OUTPUT] [-m MODEL] [-t] [-d] [-s]

Arguments:
  INPUT                 Path to input image containing Sudoku puzzle

Options:
  -o, --output OUTPUT   Path to save output image (default: solved_sudoku.png)
  -m, --model MODEL     Path to CNN model (default: models/digit_cnn.h5)
  -t, --tesseract       Use Tesseract OCR instead of CNN
  -d, --debug           Enable debug mode (shows intermediate steps)
  -s, --show            Display result in window
```

### Examples

Using CNN model (requires training first):
```bash
python main.py testplaatje.png -o solved.png
```

Using Tesseract OCR (no training needed):
```bash
python main.py testplaatje.png -o solved.png --tesseract
```

With debug visualization:
```bash
python main.py testplaatje.png -o solved.png --debug --show
```

## Architecture

The solver consists of four main modules:

### 1. Grid Detection (`src/grid_detector.py`)
- Preprocesses image (grayscale, blur, adaptive threshold)
- Detects largest contour (the Sudoku grid)
- Applies perspective transform for bird's eye view
- Extracts 81 individual cells

### 2. OCR (`src/ocr.py`)
- CNN-based digit recognition (trained on MNIST)
- Tesseract OCR fallback option
- Preprocessing: thresholding, noise removal, digit extraction
- Empty cell detection

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

Two methods are available:

1. **CNN (Recommended)**:
   - Trained on MNIST dataset
   - ~99% accuracy on test data
   - Better with printed digits
   - Requires one-time training

2. **Tesseract OCR**:
   - No training needed
   - Works well with printed/app fonts
   - Good for high-contrast digital images
   - **Recommended for printed Sudoku puzzles**

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
- If using CNN: Ensure model is trained (`python src/ocr.py`)
- Try Tesseract with `--tesseract` flag
- Use `--debug` to see which digits are being misrecognized
- Consider retraining CNN on custom dataset if dealing with handwritten puzzles

### Puzzle Unsolvable
- Check OCR results in debug output
- Verify that detected digits are correct
- Some puzzles may be invalid or have multiple solutions

## Project Structure

```
sudoka_solver/
├── src/
│   ├── __init__.py
│   ├── grid_detector.py      # Grid detection & extraction
│   ├── ocr.py                # Digit recognition
│   ├── solver.py             # Sudoku solving algorithm
│   └── image_generator.py    # Solution visualization
├── models/
│   └── digit_cnn.h5          # Trained CNN model (generated)
├── tests/
│   └── test_solver.py        # Unit tests
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
└── testplaatje.png          # Sample puzzle image
```

## Future Improvements

- Support for handwritten Sudoku puzzles
- Web interface
- Mobile app
- Support for other grid sizes (4x4, 16x16)
- Real-time solving from video stream
- Multiple solution detection
