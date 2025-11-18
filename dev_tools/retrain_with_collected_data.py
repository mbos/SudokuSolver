#!/usr/bin/env python3
"""
Retrain CNN model using collected Sudoku digit samples.

This script:
1. Loads the existing CNN model
2. Loads collected Sudoku digit samples from successfully solved puzzles
3. Fine-tunes the model on this domain-specific data
4. Saves the improved model
"""

import numpy as np
import os
from src.training_data_collector import TrainingDataCollector


def retrain_model(
    model_path: str = "models/digit_cnn.h5",
    output_path: str = "models/digit_cnn_finetuned.h5",
    epochs: int = 3,
    batch_size: int = 32,
    auto_replace: bool = False,
    min_improvement: float = 0.0
):
    """
    Retrain CNN model with collected Sudoku samples.

    Args:
        model_path: Path to existing CNN model
        output_path: Path to save fine-tuned model
        epochs: Number of training epochs
        batch_size: Batch size for training
        auto_replace: If True, automatically replace original model if improved
        min_improvement: Minimum improvement (%) required to keep new model (default: 0.0)
    """
    try:
        from tensorflow import keras
        import tensorflow as tf
    except ImportError:
        print("Error: TensorFlow not installed")
        print("Install with: pip install tensorflow")
        return False

    print("=" * 60)
    print("CNN MODEL FINE-TUNING WITH COLLECTED DATA")
    print("=" * 60)

    # Load collected training data
    print("\n[1/4] Loading collected Sudoku digit samples...")
    collector = TrainingDataCollector()
    stats = collector.get_statistics()

    if stats['total_samples'] == 0:
        print("✗ No training data collected yet!")
        print("\nTo collect training data:")
        print("  1. Solve Sudoku puzzles using: python main.py <image> -o output.png")
        print("  2. Successfully solved puzzles automatically collect training data")
        print("  3. Run this script again to fine-tune the model")
        return False

    print(f"✓ Found {stats['total_samples']} samples from {stats['total_sources']} puzzles")

    # Load and prepare data
    X_sudoku, y_sudoku = collector.prepare_training_data(target_size=(28, 28))

    if len(X_sudoku) == 0:
        print("✗ Failed to load training data")
        return False

    print(f"✓ Loaded {len(X_sudoku)} digit images")
    print(f"\nSamples per digit:")
    for digit in range(1, 10):
        count = stats['samples_by_digit'][str(digit)]
        print(f"  Digit {digit}: {count:4d} samples")

    # Load existing model
    print(f"\n[2/4] Loading existing CNN model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        print("Please train the base model first: python -m src.ocr")
        return False

    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")

    # Evaluate on Sudoku data before fine-tuning
    print("\n[3/4] Evaluating current model on Sudoku data...")
    predictions = model.predict(X_sudoku, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    accuracy_before = np.mean(y_pred == y_sudoku) * 100
    print(f"Current accuracy on Sudoku digits: {accuracy_before:.2f}%")

    # Fine-tune model
    print(f"\n[4/4] Fine-tuning model on Sudoku data ({epochs} epochs)...")
    print("-" * 60)

    # Compile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train with validation split
    history = model.fit(
        X_sudoku, y_sudoku,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate after fine-tuning
    print("\n" + "-" * 60)
    predictions_after = model.predict(X_sudoku, verbose=0)
    y_pred_after = np.argmax(predictions_after, axis=1)
    accuracy_after = np.mean(y_pred_after == y_sudoku) * 100

    improvement = accuracy_after - accuracy_before

    print(f"\nAccuracy before fine-tuning: {accuracy_before:.2f}%")
    print(f"Accuracy after fine-tuning:  {accuracy_after:.2f}%")
    print(f"Improvement: {improvement:+.2f}%")

    # Decide whether to keep the new model
    print("\n" + "=" * 60)

    if improvement >= min_improvement:
        print("✅ MODEL IMPROVED - KEEPING NEW MODEL")
        print("=" * 60)

        # Save fine-tuned model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path)
        print(f"\n✓ Fine-tuned model saved to: {output_path}")

        # Auto-replace if requested
        if auto_replace:
            import shutil
            backup_path = model_path.replace('.h5', '_backup.h5')

            # Backup original
            if os.path.exists(model_path):
                shutil.copy2(model_path, backup_path)
                print(f"✓ Original model backed up to: {backup_path}")

            # Replace with improved model
            shutil.copy2(output_path, model_path)
            print(f"✓ Original model replaced with improved version")
            print(f"\n🎉 Model automatically updated!")
        else:
            print("\nTo use the improved model:")
            print(f"  python main.py <image> -o output.png -m {output_path}")
            print("\nTo replace the original model:")
            print(f"  cp {output_path} {model_path}")

        return True
    else:
        print("❌ NO IMPROVEMENT - DISCARDING NEW MODEL")
        print("=" * 60)
        print(f"\nRequired improvement: {min_improvement:+.2f}%")
        print(f"Actual improvement:   {improvement:+.2f}%")
        print("\n⚠️  New model NOT saved (no improvement over original)")
        print("✓ Original model kept unchanged")

        if improvement < 0:
            print("\n💡 Suggestions:")
            print("  - Try collecting more training data (currently {len(X_sudoku)} samples)")
            print("  - Increase number of epochs: -e 5")
            print("  - Ensure training data is diverse (different puzzles/fonts)")

        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune CNN model with collected Sudoku digit samples"
    )

    parser.add_argument(
        "-m", "--model",
        default="models/digit_cnn.h5",
        help="Path to existing CNN model (default: models/digit_cnn.h5)"
    )

    parser.add_argument(
        "-o", "--output",
        default="models/digit_cnn_finetuned.h5",
        help="Path to save fine-tuned model (default: models/digit_cnn_finetuned.h5)"
    )

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )

    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )

    parser.add_argument(
        "--auto-replace",
        action="store_true",
        help="Automatically replace original model if improved (creates backup first)"
    )

    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Minimum improvement (%%) required to keep new model (default: 0.0)"
    )

    args = parser.parse_args()

    success = retrain_model(
        model_path=args.model,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        auto_replace=args.auto_replace,
        min_improvement=args.min_improvement
    )

    exit(0 if success else 1)
