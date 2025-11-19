#!/usr/bin/env python3
"""
Generate additional training data for CNN by:
1. Using MNIST dataset
2. Creating synthetic Sudoku-style digits
3. Augmenting existing samples
"""

import os
import cv2
import numpy as np
from pathlib import Path


def generate_synthetic_digits(output_dir: str, samples_per_digit: int = 500):
    """
    Generate synthetic digit images with various fonts and styles.

    Args:
        output_dir: Directory to save images
        samples_per_digit: Number of samples per digit (1-9)
    """
    os.makedirs(output_dir, exist_ok=True)

    # OpenCV font options
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]

    total_generated = 0

    for digit in range(1, 10):
        print(f"Generating digit {digit}...")

        for i in range(samples_per_digit):
            # Create image
            img_size = 28
            img = np.zeros((img_size, img_size), dtype=np.uint8)

            # Random parameters
            font = fonts[i % len(fonts)]
            scale = np.random.uniform(0.6, 1.0)
            thickness = np.random.randint(1, 3)

            # Get text size
            text = str(digit)
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, scale, thickness
            )

            # Center text with random offset
            x = (img_size - text_width) // 2 + np.random.randint(-3, 4)
            y = (img_size + text_height) // 2 + np.random.randint(-3, 4)

            # Ensure text is visible
            x = max(0, min(x, img_size - text_width))
            y = max(text_height, min(y, img_size))

            # Draw digit
            cv2.putText(img, text, (x, y), font, scale, 255, thickness)

            # Apply random augmentations
            img = augment_image(img)

            # Save image
            filename = f"synthetic_{digit}_{i:04d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), img)
            total_generated += 1

    print(f"Generated {total_generated} synthetic digit images")
    return total_generated


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random augmentations to an image."""
    augmented = img.copy()
    h, w = augmented.shape

    # Random rotation (-15 to 15 degrees)
    if np.random.random() < 0.7:
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=0)

    # Random translation
    if np.random.random() < 0.5:
        tx = np.random.randint(-2, 3)
        ty = np.random.randint(-2, 3)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=0)

    # Random scaling
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=0)

    # Random blur
    if np.random.random() < 0.3:
        ksize = np.random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

    # Random erosion/dilation
    if np.random.random() < 0.3:
        kernel = np.ones((2, 2), np.uint8)
        if np.random.random() < 0.5:
            augmented = cv2.erode(augmented, kernel, iterations=1)
        else:
            augmented = cv2.dilate(augmented, kernel, iterations=1)

    # Random noise
    if np.random.random() < 0.2:
        noise = np.random.normal(0, 10, augmented.shape).astype(np.uint8)
        augmented = cv2.add(augmented, noise)

    return augmented


def load_mnist_subset(samples_per_digit: int = 1000):
    """
    Load MNIST dataset and return a balanced subset.

    Args:
        samples_per_digit: Number of samples per digit (1-9)

    Returns:
        Tuple of (images, labels)
    """
    try:
        from tensorflow import keras
    except ImportError:
        print("TensorFlow not installed")
        return None, None

    print("Loading MNIST dataset...")
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    # Filter to digits 1-9 only
    mask = y_train > 0
    x_train = x_train[mask]
    y_train = y_train[mask]

    # Sample balanced subset
    images = []
    labels = []

    for digit in range(1, 10):
        digit_mask = y_train == digit
        digit_images = x_train[digit_mask]

        # Random sample
        n_samples = min(samples_per_digit, len(digit_images))
        indices = np.random.choice(len(digit_images), n_samples, replace=False)

        for idx in indices:
            images.append(digit_images[idx])
            labels.append(digit)

    print(f"Loaded {len(images)} MNIST samples")
    return np.array(images), np.array(labels)


def save_mnist_as_images(images: np.ndarray, labels: np.ndarray, output_dir: str):
    """Save MNIST images as PNG files."""
    os.makedirs(output_dir, exist_ok=True)

    for i, (img, label) in enumerate(zip(images, labels)):
        filename = f"mnist_{label}_{i:05d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), img)

    print(f"Saved {len(images)} MNIST images to {output_dir}")


def combine_all_training_data(base_dir: str = "training_data"):
    """
    Combine all training data sources into one directory.

    Returns:
        Tuple of (images, labels) arrays
    """
    images = []
    labels = []

    # Source directories
    sources = [
        os.path.join(base_dir, "sudoku_digits", "images"),  # Real Sudoku
        os.path.join(base_dir, "synthetic"),                 # Synthetic
        os.path.join(base_dir, "mnist"),                     # MNIST
    ]

    for source_dir in sources:
        if not os.path.exists(source_dir):
            continue

        print(f"Loading from {source_dir}...")

        for filename in os.listdir(source_dir):
            if not filename.endswith('.png'):
                continue

            # Extract digit from filename
            try:
                if 'digit' in filename:
                    digit = int(filename.split('digit')[1].split('.')[0].split('_')[0])
                elif filename.startswith('synthetic_') or filename.startswith('mnist_'):
                    digit = int(filename.split('_')[1])
                else:
                    continue
            except:
                continue

            if digit < 1 or digit > 9:
                continue

            # Load image
            img_path = os.path.join(source_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Resize to 28x28
                if img.shape != (28, 28):
                    img = cv2.resize(img, (28, 28))
                images.append(img)
                labels.append(digit)

    if images:
        images = np.array(images)
        labels = np.array(labels)
        print(f"\nTotal combined samples: {len(images)}")

        # Print distribution
        unique, counts = np.unique(labels, return_counts=True)
        for digit, count in zip(unique, counts):
            print(f"  Digit {digit}: {count}")

    return images, labels


def main():
    """Generate and combine all training data."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--synthetic", type=int, default=300,
                        help="Synthetic samples per digit")
    parser.add_argument("--mnist", type=int, default=500,
                        help="MNIST samples per digit")
    parser.add_argument("--output", type=str, default="training_data",
                        help="Output directory")

    args = parser.parse_args()

    # Generate synthetic digits
    print("\n=== Generating Synthetic Digits ===")
    synthetic_dir = os.path.join(args.output, "synthetic")
    generate_synthetic_digits(synthetic_dir, args.synthetic)

    # Load and save MNIST subset
    print("\n=== Loading MNIST Dataset ===")
    mnist_images, mnist_labels = load_mnist_subset(args.mnist)
    if mnist_images is not None:
        mnist_dir = os.path.join(args.output, "mnist")
        save_mnist_as_images(mnist_images, mnist_labels, mnist_dir)

    # Combine all data
    print("\n=== Combining All Training Data ===")
    all_images, all_labels = combine_all_training_data(args.output)

    print(f"\n✓ Training data generation complete!")
    print(f"  Total samples available: {len(all_labels)}")


if __name__ == "__main__":
    main()
