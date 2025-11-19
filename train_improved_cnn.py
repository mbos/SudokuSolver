#!/usr/bin/env python3
"""
Train improved CNN model on real Sudoku data + MNIST.

This script trains the digit recognition CNN using:
1. Real Sudoku digit samples from training_data/sudoku_digits/
2. MNIST data for additional generalization
3. Data augmentation for robustness
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path


def load_sudoku_digits(data_dir: str = "training_data/sudoku_digits"):
    """
    Load real Sudoku digit images from the training data directory.

    Returns:
        Tuple of (images, labels) where images are 28x28 grayscale
    """
    images_dir = Path(data_dir) / "images"
    metadata_path = Path(data_dir) / "metadata.json"

    if not images_dir.exists():
        print(f"Error: {images_dir} not found")
        return None, None

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"Loading {metadata['total_samples']} Sudoku digit samples...")

    images = []
    labels = []

    # Load all images
    for img_path in sorted(images_dir.glob("*.png")):
        # Extract digit from filename (e.g., "20251117_200002_00_digit2.png")
        filename = img_path.stem
        parts = filename.split("_")
        if len(parts) >= 4 and parts[-1].startswith("digit"):
            digit = int(parts[-1].replace("digit", ""))

            # Load and preprocess image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize to 28x28 if needed
            if img.shape != (28, 28):
                img = resize_to_mnist_format(img)

            images.append(img)
            labels.append(digit)

    if not images:
        print("No images loaded!")
        return None, None

    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images")
    print(f"Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return images, labels


def resize_to_mnist_format(digit_image: np.ndarray) -> np.ndarray:
    """
    Resize digit to 28x28 while preserving aspect ratio and centering.
    """
    h, w = digit_image.shape

    # Calculate aspect ratio
    if h > w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))

    # Resize maintaining aspect ratio
    resized = cv2.resize(digit_image, (new_w, new_h))

    # Create 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Calculate position to center digit
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    # Place digit on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply random augmentation to an image.
    """
    augmented = image.copy()

    # Random rotation (-10 to 10 degrees)
    angle = np.random.uniform(-10, 10)
    h, w = augmented.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=0)

    # Random translation (-2 to 2 pixels)
    tx = np.random.randint(-2, 3)
    ty = np.random.randint(-2, 3)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=0)

    # Random scaling (0.9 to 1.1)
    scale = np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=0)

    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    augmented = np.clip(augmented * brightness, 0, 255).astype(np.uint8)

    # Random Gaussian blur (occasionally)
    if np.random.random() < 0.3:
        ksize = np.random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

    return augmented


def create_augmented_dataset(images: np.ndarray, labels: np.ndarray,
                             augmentation_factor: int = 5) -> tuple:
    """
    Create augmented dataset by applying transformations.

    Args:
        images: Original images
        labels: Original labels
        augmentation_factor: Number of augmented copies per original

    Returns:
        Tuple of (augmented_images, augmented_labels)
    """
    print(f"Augmenting dataset with factor {augmentation_factor}...")

    aug_images = [images]  # Include originals
    aug_labels = [labels]

    for i in range(augmentation_factor):
        batch_images = []
        for img in images:
            batch_images.append(augment_image(img))
        aug_images.append(np.array(batch_images))
        aug_labels.append(labels.copy())

    final_images = np.concatenate(aug_images)
    final_labels = np.concatenate(aug_labels)

    # Shuffle
    indices = np.random.permutation(len(final_images))
    final_images = final_images[indices]
    final_labels = final_labels[indices]

    print(f"Augmented dataset size: {len(final_images)}")

    return final_images, final_labels


def load_all_training_data(base_dir: str = "training_data"):
    """
    Load all training data from all sources.

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
        print(f"Total loaded samples: {len(images)}")

    return images, labels


def train_improved_cnn(save_path: str = "models/digit_cnn.h5",
                       augmentation_factor: int = 3,
                       epochs: int = 20):
    """
    Train improved CNN on all available training data.

    Args:
        save_path: Path to save trained model
        augmentation_factor: Number of augmented copies
        epochs: Number of training epochs
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        print("Error: TensorFlow not installed")
        return

    # Load all training data
    print("Loading all training data...")
    all_images, all_labels = load_all_training_data()

    if all_images is None or len(all_images) == 0:
        print("Failed to load training data")
        return

    # Augment data
    x_train, y_train = create_augmented_dataset(
        all_images, all_labels, augmentation_factor
    )

    # Normalize
    x_train = x_train.astype("float32") / 255.0

    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)

    # Shuffle final dataset
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    print(f"\nTotal training samples: {len(x_train)}")
    print(f"Label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Shift labels from 1-9 to 0-8 for proper 9-class classification
    # This avoids the problem of having an untrained class 0
    y_train = y_train - 1  # Now labels are 0-8

    # Build improved model with 9 output classes (digits 1-9)
    print("\nBuilding improved model (9 classes for digits 1-9)...")
    model = keras.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(9, activation='softmax')  # 9 classes for digits 1-9
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001
        )
    ]

    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    # Final evaluation
    val_loss = min(history.history['val_loss'])
    val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {val_acc:.4f}")
    print(f"Best validation loss: {val_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train improved CNN model")
    parser.add_argument("--augmentation", type=int, default=3,
                        help="Augmentation factor")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--output", type=str, default="models/digit_cnn.h5",
                        help="Output model path")

    args = parser.parse_args()

    train_improved_cnn(
        save_path=args.output,
        augmentation_factor=args.augmentation,
        epochs=args.epochs
    )
