#!/usr/bin/env python3
"""
Test script to verify GroundingDINO integration
"""
import requests
import json
from PIL import Image
import io
import base64

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (640, 480), color='blue')
    # Add some red rectangles to simulate construction objects
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 200, 200], fill='red')  # Simulate a vehicle
    draw.rectangle([300, 150, 400, 250], fill='yellow')  # Simulate machinery
    return img

def test_groundingdino_api():
    """Test GroundingDINO through the API"""
    print("Testing GroundingDINO integration via API...")
    
    # Create test image
    img = create_test_image()
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    # Test the analysis endpoint
    url = "http://localhost:8000/analyze"
    
    payload = {
        "project_id": "groundingdino-test",
        "include_detection": True,
        "include_captioning": True,
        "include_classification": True
    }
    
    files = {
        'file': ('test.jpg', buffer.getvalue(), 'image/jpeg')
    }
    
    try:
        print("Sending request to API...")
        response = requests.post(url, files=files, data=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API request successful!")
            print(f"Full response: {json.dumps(result, indent=2)}")
            
            # Check various possible response structures
            if 'object_detection' in result:
                detection = result['object_detection']
                print(f"✓ Object detection results found!")
                print(f"Available: {detection.get('available', False)}")
                print(f"Total detections: {detection.get('total_detections', 0)}")
                print(f"Error: {detection.get('error', 'None')}")
                
                if detection.get('fallback_mode'):
                    print("⚠️ GroundingDINO is in fallback mode")
                elif detection.get('available'):
                    print("✅ GroundingDINO is fully functional!")
                else:
                    print("⚠️ GroundingDINO is not available")
                    
            elif 'detection_results' in result:
                detection = result['detection_results']
                print(f"✓ Detection results found: {detection.get('available', False)}")
                print(f"Total detections: {detection.get('total_detections', 0)}")
                print(f"GroundingDINO available: {detection.get('available', False)}")
                
                if detection.get('fallback_mode'):
                    print("⚠️ GroundingDINO is in fallback mode")
                else:
                    print("✅ GroundingDINO is fully functional!")
            else:
                print("⚠️ No detection results found")
                print(f"Available keys: {list(result.keys())}")
                
        else:
            print(f"✗ API request failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    test_groundingdino_api()
