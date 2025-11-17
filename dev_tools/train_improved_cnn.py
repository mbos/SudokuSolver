#!/usr/bin/env python3
"""
Train an improved CNN model with data augmentation for better Sudoku digit recognition.
"""

import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_augmented_model():
    """Create CNN model with better architecture for digit recognition."""

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),

        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model


def train_improved_cnn(save_path="models/digit_cnn_improved.h5"):
    """
    Train improved CNN model with data augmentation.
    """

    print("="*80)
    print("TRAINING IMPROVED CNN FOR SUDOKU DIGIT RECOGNITION")
    print("="*80)

    # Load MNIST dataset
    print("\n[1/5] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print(f"✓ Training samples: {len(x_train)}")
    print(f"✓ Test samples: {len(x_test)}")

    # Create data augmentation generator
    print("\n[2/5] Setting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=10,           # ±10 degrees
        width_shift_range=0.1,       # 10% shift
        height_shift_range=0.1,      # 10% shift
        shear_range=0.1,            # Shear transformation
        zoom_range=0.1,             # Zoom in/out
        fill_mode='constant',        # Fill with black
        cval=0
    )

    datagen.fit(x_train)
    print("✓ Data augmentation configured")
    print("  - Rotation: ±10°")
    print("  - Shift: ±10%")
    print("  - Shear: 10%")
    print("  - Zoom: 90%-110%")

    # Build model
    print("\n[3/5] Building improved model...")
    model = create_augmented_model()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✓ Model built with architecture:")
    model.summary()

    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]

    # Train model
    print("\n[4/5] Training model with data augmentation...")
    print("This may take several minutes...")

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=15,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n[5/5] Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✓ Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"✓ Test loss: {test_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n✓ Model saved to {save_path}")

    # Compare with old model if it exists
    old_model_path = "models/digit_cnn.h5"
    if os.path.exists(old_model_path):
        print("\n" + "="*80)
        print("COMPARING WITH OLD MODEL")
        print("="*80)

        old_model = keras.models.load_model(old_model_path)
        old_loss, old_acc = old_model.evaluate(x_test, y_test, verbose=0)

        print(f"Old model accuracy:  {old_acc:.4f} ({old_acc*100:.2f}%)")
        print(f"New model accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Improvement:         {(test_acc-old_acc)*100:+.2f}%")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nTo use the new model, either:")
    print("  1. Rename it to models/digit_cnn.h5 (replace old model)")
    print("  2. Or pass model_path='models/digit_cnn_improved.h5' to DigitRecognizer")
    print("\nTest it with:")
    print("  python quick_test.py")
    print("="*80)

    return model, history


if __name__ == "__main__":
    train_improved_cnn()
