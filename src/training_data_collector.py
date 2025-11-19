"""
Training Data Collector - Automatically collect labeled Sudoku digits
from successfully solved puzzles to improve CNN accuracy.
"""

import os
import cv2
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple


class TrainingDataCollector:
    """Collects and saves labeled digit images from solved Sudoku puzzles."""

    def __init__(self, data_dir: str = "training_data/sudoku_digits"):
        """
        Initialize the training data collector.

        Args:
            data_dir: Directory to save training data
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.metadata_file = os.path.join(data_dir, "metadata.json")

        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)

        # Load existing metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load existing metadata or create new."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'total_samples': 0,
            'samples_by_digit': {str(i): 0 for i in range(1, 10)},
            'sources': [],
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    def _save_metadata(self):
        """Save metadata to file."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, indent=2, fp=f)

    def collect_from_solved_puzzle(
        self,
        cells: List[np.ndarray],
        solution: np.ndarray,
        source_image: str,
        validation_passed: bool = True
    ) -> int:
        """
        Collect training data from a successfully solved puzzle.

        Args:
            cells: List of 81 cell images (from GridDetector)
            solution: 9x9 solved grid (ground truth labels)
            source_image: Path to original image (for tracking)
            validation_passed: Only collect if validation passed

        Returns:
            Number of samples collected
        """
        if not validation_passed:
            return 0

        if len(cells) != 81:
            print(f"Warning: Expected 81 cells, got {len(cells)}")
            return 0

        samples_collected = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, cell in enumerate(cells):
            row = i // 9
            col = i % 9
            digit = solution[row, col]

            # Skip empty cells (0)
            if digit == 0:
                continue

            # Generate unique filename
            sample_id = f"{timestamp}_{i:02d}_digit{digit}"
            image_path = os.path.join(self.images_dir, f"{sample_id}.png")

            # Save cell image
            cv2.imwrite(image_path, cell)

            # Update metadata
            samples_collected += 1
            self.metadata['total_samples'] += 1
            self.metadata['samples_by_digit'][str(digit)] += 1

        # Add source info
        source_info = {
            'timestamp': timestamp,
            'source_image': source_image,
            'samples_collected': samples_collected,
            'date': datetime.now().isoformat()
        }
        self.metadata['sources'].append(source_info)

        # Save metadata
        self._save_metadata()

        return samples_collected

    def get_statistics(self) -> dict:
        """Get statistics about collected training data."""
        return {
            'total_samples': self.metadata['total_samples'],
            'samples_by_digit': self.metadata['samples_by_digit'],
            'total_sources': len(self.metadata['sources']),
            'created': self.metadata['created'],
            'last_updated': self.metadata['last_updated']
        }

    def print_statistics(self):
        """Print collection statistics."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("TRAINING DATA COLLECTION STATISTICS")
        print("=" * 60)
        print(f"Total samples collected: {stats['total_samples']}")
        print(f"From {stats['total_sources']} source images")
        print("\nSamples per digit:")

        for digit in range(1, 10):
            count = stats['samples_by_digit'][str(digit)]
            bar = "█" * (count // 10)
            print(f"  Digit {digit}: {count:4d} {bar}")

        print(f"\nCreated: {stats['created'][:10]}")
        print(f"Last updated: {stats['last_updated'][:19]}")
        print("=" * 60)

    def prepare_training_data(self, target_size: Tuple[int, int] = (28, 28)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all collected images and prepare for CNN training.

        Args:
            target_size: Target image size (default: 28x28 for MNIST-compatible)

        Returns:
            Tuple of (images array, labels array)
        """
        images = []
        labels = []

        # Load all images from images directory
        for filename in os.listdir(self.images_dir):
            if not filename.endswith('.png'):
                continue

            # Extract digit from filename (format: timestamp_idx_digitN.png)
            try:
                digit = int(filename.split('_digit')[1].split('.')[0])
            except ValueError:
                continue

            # Load and preprocess image
            image_path = os.path.join(self.images_dir, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Resize to target size
                img_resized = cv2.resize(img, target_size)
                images.append(img_resized)
                labels.append(digit)

        if not images:
            return np.array([]), np.array([])

        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)

        # Normalize
        X = X.astype('float32') / 255.0

        # Reshape for CNN (add channel dimension)
        X = X.reshape(-1, target_size[0], target_size[1], 1)

        return X, y


def print_collection_summary(samples_collected: int, source_image: str):
    """Print summary after collecting training data."""
    if samples_collected > 0:
        print(f"\n📚 Collected {samples_collected} labeled samples from {source_image}")
        print("   Training data saved to: training_data/sudoku_digits/")
        print("   Use 'python src/training_data_collector.py' to view statistics")
    else:
        print("\n📚 No training data collected (puzzle not solved or validation failed)")


if __name__ == "__main__":
    # Show statistics when run directly
    collector = TrainingDataCollector()
    collector.print_statistics()
