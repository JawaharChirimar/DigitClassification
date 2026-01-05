#!/usr/bin/env python3
"""
BoundingBoxFromYolo.py

Processes a JPEG image with handwritten digits using YOLO detection.
Extracts each detected digit region, normalizes to 28x28 greyscale, and saves
as individual JPEG files with naming pattern: file_L_D.jpg
where L = line number (0-indexed), D = digit number (1-indexed).

Output directory: ~/data/BBFY/run_YYYY_MM_DD_HH_MM_SS/
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from DigitClassifier import load_or_create_digit_classifier, classify_digit


def create_output_directory():
    """Create output directory with timestamp in ~/data/BBFY/"""
    home_dir = Path.home()
    output_base = home_dir / "data" / "BBFY"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = output_base / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def sort_detections_by_reading_order(detections, image_height, line_threshold=0.1):
    """
    Sort detections in reading order: top to bottom, left to right.
    
    Groups detections into lines based on y-coordinate similarity,
    then sorts each line by x-coordinate.
    
    Args:
        detections: List of detection boxes (x1, y1, x2, y2)
        image_height: Height of the image (for relative thresholding)
        line_threshold: Relative threshold for grouping into lines (fraction of image height)
    
    Returns:
        List of (line_number, digit_index, box) tuples
    """
    if not detections:
        return []
    
    # Calculate absolute threshold based on image height
    threshold = image_height * line_threshold
    
    # Group boxes by line (similar y-coordinates)
    lines = []
    for box in detections:
        x1, y1, x2, y2 = box
        center_y = (y1 + y2) / 2
        
        # Find existing line or create new one
        assigned = False
        for line_boxes in lines:
            # Check if this box belongs to this line (based on y-coordinate)
            if line_boxes:
                line_center_y = np.mean([(b[1] + b[3]) / 2 for b in line_boxes])
                if abs(center_y - line_center_y) < threshold:
                    line_boxes.append(box)
                    assigned = True
                    break
        
        if not assigned:
            lines.append([box])
    
    # Sort lines by topmost y-coordinate
    lines.sort(key=lambda line_boxes: min([b[1] for b in line_boxes]))
    
    # Sort boxes within each line by x-coordinate
    for line_boxes in lines:
        line_boxes.sort(key=lambda b: b[0])
    
    # Create result list with (line_number, digit_index, box)
    # line_number is 0-indexed, digit_index is 1-indexed
    result = []
    for line_idx, line_boxes in enumerate(lines):
        for digit_idx, box in enumerate(line_boxes, start=1):
            result.append((line_idx, digit_idx, box))
    
    return result


def detect_digits_with_contours(image, min_area=50, max_area=None, aspect_ratio_range=(0.2, 3.0)):
    """
    Detect digit regions using contour detection as a fallback method.
    
    This method uses OpenCV contour detection to find potential digit regions
    when YOLO doesn't detect anything (e.g., when using a model not trained on digits).
    
    Args:
        image: Input image (BGR format from cv2)
        min_area: Minimum contour area to consider (filters out noise)
        max_area: Maximum contour area (None = no limit)
        aspect_ratio_range: (min, max) aspect ratio for bounding boxes
    
    Returns:
        List of detection boxes (x1, y1, x2, y2)
    """
    # Convert to greyscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding or Otsu's thresholding
    # Try Otsu's first, fall back to adaptive if needed
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Alternative: use adaptive thresholding for varying lighting
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    h, w = gray.shape
    
    # Set default max_area if not provided (e.g., 10% of image area)
    if max_area is None:
        max_area = (w * h) * 0.1
    
    for contour in contours:
        # Get bounding rectangle
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Calculate aspect ratio
        if box_h > 0:
            aspect_ratio = box_w / box_h
        else:
            continue
        
        # Filter by aspect ratio (digits are usually not too wide or too tall)
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            continue
        
        # Add padding around the bounding box
        padding = 3
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + box_w + padding)
        y2 = min(h, y + box_h + padding)
        
        detections.append((float(x1), float(y1), float(x2), float(y2)))
    
    return detections


def extract_and_process_region(image, box, target_size=(28, 28)):
    """
    Extract region from image, resize to target size, and convert to greyscale.
    
    Args:
        image: Input image (BGR format from cv2)
        box: Bounding box (x1, y1, x2, y2)
        target_size: Target size (width, height)
    
    Returns:
        Processed 28x28 greyscale image
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Extract region
    region = image[y1:y2, x1:x2]
    
    # Convert to greyscale if not already
    if len(region.shape) == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        region_gray = region
    
    # Resize to 28x28 first (preserves shape better than thresholding before resize)
    region_resized = cv2.resize(region_gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to improve contrast without destroying shape
    # Use adaptive normalization to match MNIST's clean appearance
    # Invert if needed to ensure dark digits on light background
    mean_val = np.mean(region_resized)
    if mean_val < 127:  # If image is mostly dark, invert it
        region_resized = 255 - region_resized
    
    # Optional: Light contrast enhancement (without thresholding to preserve shape)
    # Normalize to full range to maximize contrast
    min_val = np.min(region_resized)
    max_val = np.max(region_resized)
    if max_val > min_val:
        region_resized = ((region_resized - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return region_resized


def process_image(input_path, model_path=None, output_dir=None, classifier_model_path=None, classify_digits=False):
    """
    Process input image with YOLO, extract digit regions, and save them.
    
    Args:
        input_path: Path to input JPEG file
        model_path: Path to YOLO model file (optional, uses default if None)
        output_dir: Output directory (created with timestamp if None)
    """
    # Load image
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image: {input_path}")
        sys.exit(1)
    
    image_height, image_width = image.shape[:2]
    
    # Load YOLO model
    try:
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"Loaded model from: {model_path}")
        else:
            # Use default YOLOv8 model (will download automatically on first use)
            if model_path:
                print(f"Warning: Model path '{model_path}' not found. Using default YOLOv8n model.")
            else:
                print("No model path provided. Using default YOLOv8n model (will download on first use).")
            print("Note: For best results with handwritten digits, use a trained digit detection model.")
            model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)
    
    # Run inference
    try:
        results = model(input_path)
    except Exception as e:
        print(f"Error running YOLO inference: {e}")
        sys.exit(1)
    
    # Extract bounding boxes
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append((x1, y1, x2, y2))
    
    if not detections:
        print("Warning: No detections found with YOLO model.")
        print("Note: Default YOLOv8n model is trained on COCO dataset (80 classes like 'person', 'car', etc.)")
        print("      and does NOT include handwritten digits.")
        print("\nTrying contour-based detection as fallback method...")
        
        # Try contour-based detection as fallback
        detections = detect_digits_with_contours(image, min_area=30, aspect_ratio_range=(0.15, 4.0))
        
        if not detections:
            print("Error: No digit regions found using contour detection either.")
            print("Please try:")
            print("  1. Using a YOLO model trained specifically for digit detection")
            print("  2. Adjusting image preprocessing (ensure good contrast)")
            return
        else:
            print(f"Found {len(detections)} potential digit regions using contour detection")
    else:
        print(f"Found {len(detections)} detections with YOLO")
    
    # Sort detections in reading order
    sorted_detections = sort_detections_by_reading_order(detections, image_height)
    
    # Create output directory
    if output_dir is None:
        output_dir = create_output_directory()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load digit classifier if classification is requested
    classifier_model = None
    if classify_digits:
        try:
            classifier_model = load_or_create_digit_classifier(classifier_model_path)
        except Exception as e:
            print(f"Warning: Could not load/create digit classifier: {e}")
            print("Saving regions without classification...")
            classify_digits = False
    
    # Process and save each region
    for line_num, digit_num, box in sorted_detections:
        # Extract and process region
        processed_region = extract_and_process_region(image, box)
        
        # Classify digit if requested
        predicted_digit = None
        confidence = None
        if classify_digits and classifier_model:
            try:
                predicted_digit, confidence = classify_digit(classifier_model, processed_region)
            except Exception as e:
                print(f"Warning: Classification failed for region {line_num}_{digit_num}: {e}")
        
        # Create filename: file_L_D.jpg (or file_L_D_classified_X.jpg if classifying)
        if predicted_digit is not None:
            filename = f"file_{line_num}_{digit_num}_classified_{predicted_digit}_conf_{confidence:.2f}.jpg"
        else:
            filename = f"file_{line_num}_{digit_num}.jpg"
        
        output_path = output_dir / filename
        
        # Save as JPEG
        cv2.imwrite(str(output_path), processed_region, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if predicted_digit is not None:
            print(f"Saved: {filename} (predicted: {predicted_digit}, confidence: {confidence:.2%})")
        else:
            print(f"Saved: {filename}")
    
    print(f"\nProcessing complete! Saved {len(sorted_detections)} digit regions to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract handwritten digits from JPEG image using YOLO detection"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input JPEG file with handwritten digits"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Path to YOLO model file (optional, uses default YOLOv8n if not provided)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (optional, creates timestamped directory if not provided)"
    )
    parser.add_argument(
        "-c", "--classify",
        action="store_true",
        help="Enable digit classification using CNN (requires TensorFlow, will train on MNIST if no model provided)"
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default=None,
        help="Path to pre-trained digit classifier model (.h5 file). If not provided and -c is used, will train on MNIST"
    )
    
    args = parser.parse_args()
    
    process_image(args.input_image, args.model, args.output, args.classifier_model, args.classify)


if __name__ == "__main__":
    main()

