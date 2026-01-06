#!/usr/bin/env python3
"""
DigitClassifier.py

Provides functions for creating, training, and using a CNN-based digit classifier.
The classifier is trained on the MNIST and EMNIST Digits datasets to recognize handwritten digits (0-9).
"""

import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import albumentations as A

try:
    from emnist import extract_training_samples, extract_test_samples
    EMNIST_AVAILABLE = True
except ImportError:
    EMNIST_AVAILABLE = False
    print("Warning: 'emnist' package not available. Install with: pip install emnist")


def create_digit_classifier_model():
    """
    Create a CNN model for digit classification (0-9).
    Architecture similar to MNIST classifiers.
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='elu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='elu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_augmentation_pipeline(stats_tracker=None):
    """
    Create an albumentations augmentation pipeline for digit training.
    Includes rotation, stroke thickness variation, missing parts, and curved strokes.
    
    Args:
        stats_tracker: Optional dict to track augmentation statistics
    
    Returns:
        albumentations Compose object
    """
    # Create individual transforms - all can apply together for realistic combinations
    transforms = []
    
    # 1. Rotation OR Slant (mutually exclusive) - both include scale and position
    # Rotation: rotates entire digit (like tilting paper) + scale + position
    # Slant: vertical shear for forward/backward tilt + scale + position
    # These are mutually exclusive since they both affect orientation
    transforms.append(A.OneOf([
        A.ShiftScaleRotate(
            shift_limit=0.0,      # No shift
            scale_limit=0.0,      # No scale variation
            rotate_limit=48,       # 48 degrees max rotation
            p=1.0
        ),
        A.Affine(
            shear={'x': 0, 'y': (-15, 15)},  # Vertical shear for forward/backward tilt (±15 degrees)
            scale=(1, 1),                # No scale variation
            translate_percent={'x': (0, 0), 'y': (0, 0)},  # No position shift
            p=1.0
        )
    ], p=0.8))  # 80% chance of getting either rotation OR slant (both with scale + position)
    
    # 2. Image quality issues - can occur together (realistic for poor scans/photos)
    transforms.append(A.GaussianBlur(blur_limit=(1, 3), p=0.3))  # Light blur
    transforms.append(A.GaussNoise(p=0.2))  # Light noise
    
    # Note: Morphological operations (stroke thickness variation) are applied separately
    # in the data generator, so they can combine with the above augmentations
    
    transform = A.Compose(transforms, p=1.0)
    
    return transform


class ImageDataGeneratorWithAugmentation:
    """
    Custom data generator that applies albumentations augmentations to MNIST data.
    """
    def __init__(self, augmentation_pipeline, batch_size=64):
        self.augmentation_pipeline = augmentation_pipeline
        self.batch_size = batch_size
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'morphology_thicker': 0,
            'morphology_thinner': 0,
        }
    
    def flow(self, x, y, batch_size=None):
        """
        Generator that yields batches of augmented data.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        num_samples = len(x)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        while True:
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                
                # Process each image in the batch
                batch_x_aug = []
                for img in batch_x:
                    self.stats['total_samples'] += 1
                    
                    # Convert from (28, 28, 1) to (28, 28) for albumentations
                    img_2d = img.squeeze(axis=-1)
                    
                    # Convert from float [0,1] to uint8 [0,255] for albumentations
                    img_uint8 = (img_2d * 255).astype(np.uint8)
                    
                    # Apply augmentation to 50% of samples (reduce augmented samples by 50%)
                    # All original samples are used, but only 50% get augmented
                    if np.random.random() < 0.5:
                        # Apply augmentation (albumentations expects dict with 'image' key)
                        # Note: All augmentations can apply to the same image simultaneously
                        # based on their individual probabilities
                        augmented = self.augmentation_pipeline(image=img_uint8)
                        img_aug = augmented['image']
                        
                        # Apply stroke thickness variation using morphological operations
                        # For WHITE digits on BLACK background:
                        # - dilate() makes bright regions (digits) THICKER
                        # - erode() makes bright regions (digits) THINNER
                        if np.random.random() < 0.5:
                            kernel_size = np.random.choice([1, 2])
                            if np.random.random() < 0.5:
                                # Thicker strokes: dilate the white digits
                                self.stats['morphology_thicker'] += 1
                                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                img_aug = cv2.dilate(img_aug, kernel, iterations=1)
                            #else:
                                # Thinner strokes: erode the white digits
                                #self.stats['morphology_thinner'] += 1
                                #kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                #img_aug = cv2.erode(img_aug, kernel, iterations=1)
                    else:
                        # Keep original image (no augmentation) - 50% of samples remain original
                        img_aug = img_uint8
                    
                    # Convert back to float [0,1]
                    img_aug_float = (img_aug.astype(np.float32) / 255.0)
                    
                    # Reshape to (28, 28, 1)
                    img_aug_float = np.expand_dims(img_aug_float, axis=-1)
                    
                    # Ensure values are in [0, 1] range
                    img_aug_float = np.clip(img_aug_float, 0.0, 1.0)
                    
                    batch_x_aug.append(img_aug_float)
                
                batch_x_aug = np.array(batch_x_aug)
                
                yield batch_x_aug, batch_y


def load_or_create_digit_classifier(classifier_model_path=None):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Args:
        classifier_model_path: Path to saved classifier model (.h5 file)
    
    Returns:
        Trained Keras model for digit classification
    """
    # Create timestamped directory for model checkpoints
    base_dir = Path.home() / "data" / "modelForBBFY"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory: run_YYYY_MM_DD_HH_MM_SS
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine the model path to use
    if classifier_model_path:
        # If user specified a path, use it (but still save epoch models to run_dir)
        model_path_to_use = classifier_model_path
    else:
        # Default: save to the timestamped run directory
        model_path_to_use = str(run_dir / "digit_classifier_mnist.h5")
    
    print(f"Model checkpoints will be saved to: {run_dir}")
    
    # Try to load existing model (either specified path or default location)
    if os.path.exists(model_path_to_use):
        try:
            print(f"Loading digit classifier from: {model_path_to_use}")
            model = keras.models.load_model(model_path_to_use)
            print("Digit classifier loaded successfully")
            return model
        except Exception as e:
            print(f"Warning: Could not load classifier from {model_path_to_use}: {e}")
            print("Creating new classifier model...")
    
    # Create new model
    print("Creating new digit classifier model...")
    model = create_digit_classifier_model()
    
    # Try to train on MNIST + EMNIST Digits dataset
    try:
        print("Loading MNIST dataset...")
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()
        
        # Load EMNIST Digits if available
        x_train_emnist = None
        y_train_emnist = None
        x_test_emnist = None
        y_test_emnist = None
        
        if EMNIST_AVAILABLE:
            x_train_emnist = None
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    if attempt == 0:
                        print("Loading EMNIST Digits dataset (this may take a few minutes on first run)...")
                    else:
                        print(f"Retrying EMNIST Digits loading (attempt {attempt + 1}/{max_retries})...")
                    # Note: 'digits' parameter loads ONLY EMNIST Digits (0-9), not the full EMNIST dataset
                    # This gives us ~240,000 additional digit samples beyond MNIST's 60,000
                    x_train_emnist, y_train_emnist = extract_training_samples('digits')
                    x_test_emnist, y_test_emnist = extract_test_samples('digits')
                    print(f"Loaded EMNIST Digits: {len(x_train_emnist)} training, {len(x_test_emnist)} test samples")
                    if len(x_train_emnist) == 0:
                        print("Warning: EMNIST Digits loaded but has 0 samples. Check dataset installation.")
                        x_train_emnist = None
                    break  # Success, exit retry loop
                except Exception as e:
                    error_msg = str(e)
                    # Check if it's a corrupted zip file error
                    if ("not a zip file" in error_msg.lower() or "BadZipFile" in error_msg) and attempt < max_retries - 1:
                        print(f"Warning: EMNIST cache file is corrupted: {e}")
                        print("Attempting to clear corrupted cache file and retry...")
                        try:
                            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'emnist')
                            cache_file = os.path.join(cache_dir, 'emnist.zip')
                            if os.path.exists(cache_file):
                                os.remove(cache_file)
                                print(f"Deleted corrupted cache file: {cache_file}")
                                print("Will retry downloading EMNIST...")
                            else:
                                print(f"Cache file not found at: {cache_file}")
                        except Exception as cleanup_error:
                            print(f"Could not delete cache file automatically: {cleanup_error}")
                            print(f"Please manually delete: {os.path.join(os.path.expanduser('~'), '.cache', 'emnist', 'emnist.zip')}")
                    else:
                        # Not a corrupted file error, or we've exhausted retries
                        print(f"Warning: Could not load EMNIST Digits: {e}")
                        if attempt == max_retries - 1:
                            import traceback
                            print("Full error traceback:")
                            traceback.print_exc()
                        x_train_emnist = None
                        break  # Give up after showing error
        
        # Combine datasets
        if x_train_emnist is not None:
            print(f"Combining datasets: MNIST ({len(x_train_mnist)} samples) + EMNIST Digits ({len(x_train_emnist)} samples)")
            x_train = np.concatenate([x_train_mnist, x_train_emnist], axis=0)
            y_train = np.concatenate([y_train_mnist, y_train_emnist], axis=0)
            x_test = np.concatenate([x_test_mnist, x_test_emnist], axis=0)
            y_test = np.concatenate([y_test_mnist, y_test_emnist], axis=0)
            print(f"Combined dataset: {len(x_train)} training, {len(x_test)} test samples")
        else:
            if not EMNIST_AVAILABLE:
                print(f"Warning: EMNIST package not available. Install with: pip install emnist")
            print(f"Using MNIST only: {len(x_train_mnist)} training, {len(x_test_mnist)} test samples")
            x_train = x_train_mnist
            y_train = y_train_mnist
            x_test = x_test_mnist
            y_test = y_test_mnist
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN (28, 28, 1)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Create data augmentation pipeline
        # This will be applied on-the-fly during training
        print("Setting up data augmentation pipeline...")
        augmentation_pipeline = create_augmentation_pipeline()
        
        # Create data generator with augmentation
        train_datagen = ImageDataGeneratorWithAugmentation(
            augmentation_pipeline=augmentation_pipeline,
            batch_size=64
        )
        
        # Train the model with augmented data
        print("Starting training with data augmentation...")
        print("\n" + "="*60)
        print("Augmentation Configuration:")
        print("="*60)
        print("All augmentations can apply to the SAME image simultaneously")
        print("(Based on their individual probabilities)")
        print("\nAugmentation probabilities:")
        print("  50% of samples will be augmented, 50% will remain original")
        print("  Rotation OR Slant (mutually exclusive): 80% (p=0.8) of augmented samples")
        print("    - Rotation: ±48° rotation (no shift, no scale)")
        print("    - Slant: ±15° vertical shear (no shift, no scale)")
        print("  GaussianBlur (image quality): 30% (p=0.3) of augmented samples")
        print("  GaussNoise (image quality): 20% (p=0.2) of augmented samples")
        print("  Morphology - Stroke thickness variation: 50% (p=0.5) of augmented samples")
        print("\nNote: Augmentation is applied on-the-fly - each sample is augmented differently each epoch")
        print(f"Training samples per epoch: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        num_epochs = 30
        print(f"Total augmented samples over {num_epochs} epochs: ~{len(x_train) * num_epochs}")
        print("(Each sample is augmented uniquely each time it's seen)")
        print("="*60 + "\n")
        
        # Custom callback to print per-epoch statistics
        class AugmentationStatsCallback(keras.callbacks.Callback):
            def __init__(self, datagen, samples_per_epoch):
                self.datagen = datagen
                self.samples_per_epoch = samples_per_epoch
                self.last_total = 0
            
            def on_epoch_end(self, epoch, logs=None):
                current_total = self.datagen.stats['total_samples']
                samples_this_epoch = current_total - self.last_total
                self.last_total = current_total
                print(f"\n[Epoch {epoch+1}] Augmented samples processed: {samples_this_epoch}")
        
        stats_callback = AugmentationStatsCallback(train_datagen, len(x_train))
        
        # ModelCheckpoint callback to save model after each epoch with epoch number
        # Save all epoch models in the timestamped run directory
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "digit_classifier_epoch_{epoch:02d}.h5"),
            save_best_only=False,  # Save every epoch, not just best
            save_weights_only=False,  # Save full model
            verbose=0  # Don't print save messages (already verbose=1 in fit)
        )
        
        print(f"Epoch models will be saved as: {run_dir}/digit_classifier_epoch_XX.h5 (one per epoch)")
        
        model.fit(
            train_datagen.flow(x_train, y_train, batch_size=64),
            steps_per_epoch=len(x_train) // 64,
            epochs=30,
            validation_data=(x_test, y_test),
            verbose=1,
            callbacks=[stats_callback, checkpoint_callback]
        )
        
        # Print final augmentation statistics
        print("\n" + "="*60)
        print("Final Augmentation Statistics:")
        print("="*60)
        stats = train_datagen.stats
        total = stats['total_samples']
        if total > 0:
            print(f"Total samples processed across all epochs: {total}")
            print(f"Average samples per epoch: {total / num_epochs:.0f}")
            print(f"\nMorphology augmentation application rates:")
            print(f"  Morphology - Thicker strokes: {stats['morphology_thicker']}/{total} ({stats['morphology_thicker']/total*100:.1f}%)")
            print(f"  Morphology - Thinner strokes: {stats['morphology_thinner']}/{total} ({stats['morphology_thinner']/total*100:.1f}%)")
            print(f"\nNote: All augmentations (ShiftScaleRotate, GaussianBlur, GaussNoise, Morphology)")
            print(f"      can apply to the same image simultaneously based on their individual probabilities.")
            print(f"      (We can only track morphology stats since it's applied manually)")
        print("="*60)
        
        # Save the final model (also saved by checkpoint, but this ensures final state is saved)
        model.save(model_path_to_use)
        print(f"Final model also saved to: {model_path_to_use}")
        print(f"(Individual epoch models saved in: {run_dir})")
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("Evaluating model on test set (MNIST + EMNIST Digits)...")
        print("="*60)
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4%} ({test_accuracy*10000:.0f} out of 10,000 test images)")
        
        # Get per-class accuracy
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nPer-class accuracy on test set:")
        print("-" * 40)
        for digit in range(10):
            mask = y_test == digit
            if np.sum(mask) > 0:
                class_accuracy = np.mean(y_pred_classes[mask] == digit)
                correct = np.sum(y_pred_classes[mask] == digit)
                total = np.sum(mask)
                print(f"  Digit {digit}: {class_accuracy:.2%} ({correct}/{total})")
        
        print("="*60)
        print("Digit classifier trained and ready!")
        return model
        
    except Exception as e:
        print(f"Warning: Could not train on MNIST dataset: {e}")
        print("Using untrained model (predictions will be random)")
        return model


def classify_digit(classifier_model, digit_image):
    """
    Classify a single digit image using the CNN model.
    
    Args:
        classifier_model: Trained Keras model
        digit_image: 28x28 greyscale image (numpy array)
    
    Returns:
        Predicted digit (0-9) and confidence score
    """
    # Ensure image is the right shape and type
    if digit_image.shape != (28, 28):
        # Resize if needed
        digit_image = cv2.resize(digit_image, (28, 28))
    
    # Normalize pixel values to [0, 1]
    digit_normalized = digit_image.astype('float32') / 255.0
    
    # The input image should already be in MNIST format: white digits on black background
    # (ensured by BoundingBoxFromYolo.py preprocessing)
    # MNIST: white digits (high values ~1.0) on black background (low values ~0.0)
    
    # Reshape for model input: (1, 28, 28, 1)
    digit_input = digit_normalized.reshape(1, 28, 28, 1)
    
    # Predict
    predictions = classifier_model.predict(digit_input, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_digit])
    
    return predicted_digit, confidence


def main():
    """
    Standalone training function for the digit classifier.
    Can be run directly to train the model without running BoundingBoxFromYolo.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a digit classifier on MNIST dataset"
    )
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default=None,
        help="Path to save the trained model (.h5 file). Default: ~/.digit_classifier_mnist.h5"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if a model already exists"
    )
    
    args = parser.parse_args()
    
    # If force-retrain, remove existing model first
    if args.force_retrain:
        model_path = args.model_path if args.model_path else str(Path.home() / ".digit_classifier_mnist.h5")
        if os.path.exists(model_path):
            print(f"Removing existing model at: {model_path}")
            os.remove(model_path)
    
    # Train the model
    print("Starting digit classifier training...")
    model = load_or_create_digit_classifier(args.model_path)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

