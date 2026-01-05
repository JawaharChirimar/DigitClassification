#!/usr/bin/env python3
"""
DigitClassifier.py

Provides functions for creating, training, and using a CNN-based digit classifier.
The classifier is trained on the MNIST dataset to recognize handwritten digits (0-9).
"""

import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_digit_classifier_model():
    """
    Create a CNN model for digit classification (0-9).
    Architecture similar to MNIST classifiers.
    
    Returns:
        Compiled Keras model
    """
    #layers.Conv2D(64, (3, 3), activation='relu'),
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
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


def load_or_create_digit_classifier(classifier_model_path=None):
    """
    Load a pre-trained digit classifier or create/train a new one.
    
    Args:
        classifier_model_path: Path to saved classifier model (.h5 file)
    
    Returns:
        Trained Keras model for digit classification
    """
    # Determine the model path to use
    default_path = Path.home() / ".digit_classifier_mnist.h5"
    model_path_to_use = classifier_model_path if classifier_model_path else str(default_path)
    
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
    
    # Try to train on MNIST dataset
    try:
        print("Training classifier on MNIST dataset (this may take a few minutes)...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN (28, 28, 1)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Train the model
        model.fit(x_train, y_train, 
                 epochs=20,
                 batch_size=64, 
                 validation_data=(x_test, y_test),
                 verbose=1)
        
        # Save the trained model
        model.save(model_path_to_use)
        print(f"Trained model saved to: {model_path_to_use}")
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("Evaluating model on official MNIST test set...")
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
    
    # Check and correct polarity to match MNIST format (dark digits on light background)
    # MNIST has dark digits (low values ~0.0) on light background (high values ~1.0)
    # Check border pixels (typically background) to determine if image needs inversion
    border_pixels = np.concatenate([
        digit_normalized[0, :],   # top border
        digit_normalized[-1, :],  # bottom border
        digit_normalized[:, 0],   # left border
        digit_normalized[:, -1]   # right border
    ])
    border_mean = np.mean(border_pixels)
    
    # If background (border) is dark (mean < 0.5), the image is likely inverted
    # Invert it to match MNIST format (dark digits on light background)
    if border_mean < 0.5:
        digit_normalized = 1.0 - digit_normalized
    
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

