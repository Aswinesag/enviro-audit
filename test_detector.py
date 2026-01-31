#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.grounding_dino_detector import GroundingDINODetector
from PIL import Image
import numpy as np

print("Testing GroundingDINO detector...")

# Create a simple test image
test_image = Image.new('RGB', (640, 480), color='blue')

# Initialize detector
detector = GroundingDINODetector()

# Check if available
if detector.is_available():
    print("✓ GroundingDINO detector is available!")
    
    # Test detection
    result = detector.detect_construction(test_image)
    print(f"Detection result: {result['available']}")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Total detections: {result['total_detections']}")
else:
    print("✗ GroundingDINO detector is not available")
