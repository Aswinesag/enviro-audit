# scripts/test_full_pipeline.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.pipelines.analysis_pipeline import AnalysisPipeline
from PIL import Image
import requests
from io import BytesIO
import json

def test_full_pipeline():
    print("Testing full EnviroAudit pipeline...")
    
    # Initialize pipeline
    pipeline = AnalysisPipeline()
    
    # Test with a construction image
    test_url = "https://images.unsplash.com/photo-1581094794329-c8112a89af12"
    print(f"\nTesting with image: {test_url}")
    
    # Download image
    response = requests.get(test_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    # Create metadata
    metadata = {
        "source": "test",
        "location": {
            "latitude": 40.7128,
            "longitude": -74.0060
        },
        "project_id": "test-001"
    }
    
    # Run analysis
    print("\nRunning analysis...")
    result = pipeline.analyze_image(image, metadata)
    
    # Display results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n1. Classification:")
    print(f"   Primary: {result['classification']['primary_label']}")
    print(f"   Confidence: {result['classification']['primary_confidence']}")
    print(f"   Construction detected: {result['classification']['is_construction_activity']}")
    
    print(f"\n2. Caption:")
    print(f"   {result['caption']['basic_caption']}")
    print(f"   Has construction terms: {result['caption']['has_construction_terms']}")
    
    print(f"\n3. Compliance Assessment:")
    print(f"   Risk Level: {result['compliance']['risk_level']}")
    print(f"   Action: {result['compliance']['recommended_action']}")
    
    print(f"\n4. Report Summary:")
    print(f"   {result['report']['summary']}")
    
    # Save full result to file
    output_file = "data/test_analysis_result.json"
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nFull results saved to: {output_file}")
    
    # Print the text report
    print("\n" + "="*60)
    print("TEXT REPORT")
    print("="*60)
    print(result['report']['text'])
    
    return result

if __name__ == "__main__":
    test_full_pipeline()