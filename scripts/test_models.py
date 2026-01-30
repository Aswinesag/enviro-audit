# scripts/test_models.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.clip_classifier import CLIPZeroShotClassifier
from src.models.blip_captioner import BLIPCaptioner
from PIL import Image
import requests
from io import BytesIO

def test_models():
    print("Testing CLIP and BLIP models...")
    
    # Download a test image
    test_url = "https://images.unsplash.com/photo-1581094794329-c8112a89af12"
    response = requests.get(test_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    print(f"Test image: {image.size}")
    
    # Test CLIP
    print("\n1. Testing CLIP classifier...")
    try:
        clip = CLIPZeroShotClassifier()
        classification = clip.classify_construction(image)
        print(f"   ✓ CLIP loaded successfully")
        print(f"   Primary label: {classification['primary_label']}")
        print(f"   Confidence: {classification['primary_confidence']}")
        print(f"   Analysis: {classification['analysis']}")
    except Exception as e:
        print(f"   ✗ CLIP error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test BLIP
    print("\n2. Testing BLIP captioner...")
    try:
        blip = BLIPCaptioner()
        caption = blip.caption(image)
        print(f"   ✓ BLIP loaded successfully")
        print(f"   Caption: {caption}")
        
        caption_with_context = blip.caption_with_context(image)
        print(f"   Has construction terms: {caption_with_context['has_construction_terms']}")
    except Exception as e:
        print(f"   ✗ BLIP error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("If both models work, the API should run successfully!")
    print("="*60)

if __name__ == "__main__":
    test_models()