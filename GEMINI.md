# Gemini Code Assistant Context: Sudoku Solver

This document provides context for the Gemini Code Assistant to effectively assist with development in the `sudoka_solver` project.

## Project Overview

This is a Python-based Sudoku solver that can process puzzles from both image and text files. It uses computer vision to detect the Sudoku grid in an image, an advanced ensemble OCR system to recognize the digits, a high-performance algorithm to solve the puzzle, and then generates an image of the solved puzzle.

The project is well-structured, with a clear separation of concerns between image processing, OCR, solving logic, and output generation. It includes a suite of development and testing tools.

### Core Technologies

*   **Language:** Python 3
*   **Image Processing:** OpenCV (`opencv-python`)
*   **OCR Engine:** A custom-built ensemble system combining:
    *   A custom Convolutional Neural Network (CNN) trained on the MNIST dataset (using `tensorflow`/`keras`).
    *   Tesseract (`pytesseract`).
    *   EasyOCR (`easyocr`).
    The ensemble uses a configurable weighted voting strategy (`config/ocr_config.yaml`).
*   **Core Libraries:** NumPy, Pillow
*   **Testing:** `pytest`

### Architecture

The application pipeline is as follows:

1.  **Input:** Takes an image (`.png`, `.jpg`) or a text file (`.txt`) as input.
2.  **Grid Detection (`src/grid_detector.py`):** Locates the Sudoku grid in the image, corrects for perspective, and extracts the 81 cells.
3.  **OCR (`src/ocr/`):** The `EnsembleRecognizer` (`src/ocr/ensemble_recognizer.py`) processes each cell. It runs multiple OCR models (CNN, Tesseract, EasyOCR) and uses a `weighted` voting strategy to determine the most likely digit. It is designed to be robust and accurate.
4.  **Error Correction (`src/error_corrector.py`):** This is a key feature that makes the solver robust against OCR mistakes. If the initial puzzle is invalid or unsolvable (a strong indicator of OCR errors), this module automatically attempts to fix it. It uses a combination of Sudoku's constraint rules and the confidence scores from the OCR step to identify the most likely incorrect digits and intelligently tries alternatives. This process significantly increases the puzzle success rate.
5.  **Solver (`src/solver.py`):** Employs a hybrid algorithm combining constraint propagation and backtracking with the Minimum Remaining Value (MRV) heuristic for efficient solving.
6.  **Validation (`src/validator.py`):** The final solution is validated against Sudoku rules.
7.  **Data Collection (`src/training_data_collector.py`):** On a successful solve, the original cells are saved as labeled training data to be used for future fine-tuning of the CNN model.
8.  **Output (`src/image_generator.py`):** Generates an output image with the solution digits drawn onto the grid.

## Building and Running

### 1. Installation

Install the required Python packages from `requirements.txt`.

```bash
# It is recommended to use a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the CNN Model

Before first use, or to improve accuracy, the CNN model should be trained. This script downloads the MNIST dataset and saves the trained model to `models/digit_cnn.h5`.

```bash
python -m src.ocr
```

### 3. Running the Solver

The main entry point is `main.py`.

**To solve from an image:**
```bash
# Uses the default ensemble OCR mode for best accuracy
python main.py path/to/your/puzzle.png -o solved.png
```

**To solve from a text file:**
```bash
python main.py path/to/puzzle.txt
```

**Key Command-Line Flags:**
*   `--debug`: Enables debug mode, showing intermediate image processing steps.
*   `--verbose`: Prints detailed logs and ASCII visualizations to the console.
*   `--no-ensemble`: Disables the ensemble and uses the CNN model directly.
*   `--tesseract`: Disables the ensemble and uses Tesseract directly.

## Development and Testing

### Running Tests

The project has a dedicated test script that runs unit, integration, and end-to-end tests using `pytest`.

```bash
./run_tests.sh
```

### Development Conventions

*   The code is well-documented with docstrings and comments.
*   The project follows a modular structure, located under the `src/` directory.
*   Development and analysis tools are kept separate in the `dev_tools/` directory.
*   There is a strong emphasis on improving OCR accuracy, as detailed in `docs/OCR_IMPROVEMENT_PLAN.md`. This plan has guided the implementation of the current ensemble system.
*   New features or bug fixes should ideally be accompanied by tests. The existing tests in `tests/` serve as a good reference.
*   The project uses type hints.
*   **Documentation Protocol:** There is a strict convention to keep documentation (`README.md`, `CLAUDE.md`, etc.) synchronized with code changes. Any modifications to the CLI, file structure, APIs, or project behavior should be reflected in the relevant documentation.

### Key Files for Context

*   `main.py`: The main application entry point and argument parser.
*   `src/ocr/ensemble_recognizer.py`: The core of the OCR logic, combining multiple models.
*   `config/ocr_config.yaml`: Configuration file for the OCR ensemble weights and thresholds.
*   `src/solver.py`: The Sudoku solving algorithm.
*   `src/error_corrector.py`: Logic for automatically fixing OCR errors.
*   `run_tests.sh`: The main test execution script.
*   `docs/OCR_IMPROVEMENT_PLAN.md`: Provides historical context and the rationale for the current OCR architecture.
