# Development Tools

Deze directory bevat development en testing scripts die niet nodig zijn voor normale productie gebruik.

## Analysis Tools

- `analyze_cnn_errors.py` - Analyseert CNN fouten met confusion matrix
- `analyze_ocr_accuracy.py` - OCR accuracy testing

## Testing Tools

- `test_validator.py` - Tests voor de Sudoku validator (10 comprehensive tests)
- `test_solution_validation.py` - Test solution validation pipeline
- `test_solver_only.py` - Test alleen de Sudoku solver
- `test_ocr_suite.py` - Comprehensive OCR test suite (16 images)
- `test_training_collection.py` - Test training data collection system
- `quick_test.py` - Snelle baseline test op testplaatje.png

## Visualization Tools

- `visualize_failed_digits.py` - Visualiseer preprocessing stappen voor failed digits
- `visualize_output.py` - Visualiseer output images

## Data Generation

- `generate_test_dataset.py` - Genereer synthetic Sudoku test images
- `test_dataset.json` - Ground truth labels voor test images

## Training Tools

- `train_improved_cnn.py` - Advanced CNN training met data augmentation
- `inspect_cells.py` - Inspect geëxtraheerde cell images

## Gebruik

Deze tools zijn voornamelijk gebruikt tijdens development en kunnen nuttig zijn voor:
- Debugging OCR problemen
- Performance analysis
- Model training experiments
- Test data generation

Voor normale productie gebruik zijn alleen de scripts in de root directory nodig.
