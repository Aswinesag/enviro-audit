# scripts/test_api_functional.py
import requests
import json
import time

def test_api_functionality():
    base_url = "http://localhost:8000"
    
    print("="*60)
    print("Testing EnviroAudit API Functionality")
    print("="*60)
    
    # Test 1: Analyze a construction image via URL
    print("\n1. Testing construction site image...")
    
    construction_url = "https://images.unsplash.com/photo-1581094794329-c8112a89af12"
    
    try:
        response = requests.post(
            f"{base_url}/analyze-url",
            json={
                "image_url": construction_url,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "project_id": "construction-site-001"
            },
            timeout=120  # Give it more time for first analysis
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Analysis successful!")
            print(f"   Analysis ID: {result.get('metadata', {}).get('analysis_id')}")
            print(f"   Primary Classification: {result.get('classification', {}).get('primary_label', 'N/A')}")
            print(f"   Confidence: {result.get('classification', {}).get('primary_confidence', 'N/A')}")
            print(f"   Risk Level: {result.get('compliance', {}).get('risk_level', 'N/A')}")
            print(f"   Caption: {result.get('caption', {}).get('basic_caption', 'N/A')[:60]}...")
            
            # Save the result
            with open("data/construction_analysis.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"   Results saved to: data/construction_analysis.json")
        else:
            print(f"   ✗ Failed with status: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"   Request failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Analyze a natural landscape
    print("\n2. Testing natural landscape image...")
    
    natural_url = "https://images.unsplash.com/photo-1501854140801-50d01698950b"
    
    try:
        response = requests.post(
            f"{base_url}/analyze-url",
            json={
                "image_url": natural_url,
                "latitude": 37.7749,
                "longitude": -122.4194,
                "project_id": "natural-landscape-001"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Analysis successful!")
            print(f"   Primary Classification: {result.get('classification', {}).get('primary_label', 'N/A')}")
            print(f"   Risk Level: {result.get('compliance', {}).get('risk_level', 'N/A')}")
            print(f"   Caption: {result.get('caption', {}).get('basic_caption', 'N/A')[:60]}...")
            
            # Save the result
            with open("data/natural_analysis.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"   Results saved to: data/natural_analysis.json")
        else:
            print(f"   ✗ Failed with status: {response.status_code}")
            
    except Exception as e:
        print(f"   Request failed: {e}")
    
    # Test 3: Test file upload (simulated)
    print("\n3. Testing Swagger UI endpoints...")
    print(f"   Visit: {base_url}/docs")
    print(f"   Try the '/analyze' endpoint with a file upload")
    print(f"   Try the '/analyze-url' endpoint with different images")
    
    print("\n" + "="*60)
    print("API Testing Complete!")
    print("="*60)

if __name__ == "__main__":
    test_api_functionality()