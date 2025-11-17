#!/usr/bin/env python3
"""
Test training data collection with a simulated successful solve.
"""

import numpy as np
import cv2
import os
from src.training_data_collector import TrainingDataCollector

# Create dummy cell images (simple 50x50 black images with white numbers)
def create_dummy_cells(solution):
    """Create 81 dummy cell images based on solution."""
    cells = []
    for i in range(81):
        row = i // 9
        col = i % 9
        digit = solution[row, col]

        # Create 50x50 black image
        cell = np.zeros((50, 50), dtype=np.uint8)

        # If digit > 0, draw white text
        if digit > 0:
            cv2.putText(cell, str(digit), (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        cells.append(cell)

    return cells

# Test solution
solution = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
])

print("Testing Training Data Collection")
print("=" * 60)

# Create dummy cells
print("\nCreating 81 dummy cell images...")
cells = create_dummy_cells(solution)
print(f"✓ Created {len(cells)} cell images")

# Initialize collector
collector = TrainingDataCollector()

# Collect training data
print("\nCollecting training data from solved puzzle...")
samples_collected = collector.collect_from_solved_puzzle(
    cells=cells,
    solution=solution,
    source_image="test_image.png",
    validation_passed=True
)

print(f"✓ Collected {samples_collected} samples")

# Show statistics
collector.print_statistics()

# Verify files were created
print("\nVerifying saved files...")
images_dir = "training_data/sudoku_digits/images"
if os.path.exists(images_dir):
    image_count = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
    print(f"✓ Found {image_count} images in {images_dir}")
else:
    print(f"✗ Directory {images_dir} not found")

metadata_file = "training_data/sudoku_digits/metadata.json"
if os.path.exists(metadata_file):
    print(f"✓ Metadata file created: {metadata_file}")
else:
    print(f"✗ Metadata file not found")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
