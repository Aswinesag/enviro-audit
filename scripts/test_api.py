# scripts/test_api_quick.py
import requests
import json

# Test the running API
def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing EnviroAudit API...")
    
    # 1. Check health
    print("\n1. Health check:")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Failed to connect: {e}")
        return
    
    # 2. Test with URL
    print("\n2. Testing analyze-url endpoint:")
    test_url = "https://images.unsplash.com/photo-1581094794329-c8112a89af12"
    
    try:
        response = requests.post(
            f"{base_url}/analyze-url",
            json={
                "image_url": test_url,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "project_id": "test-project-001"
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Success!")
            print(f"   Analysis ID: {result.get('metadata', {}).get('analysis_id')}")
            print(f"   Risk Level: {result.get('compliance', {}).get('risk_level')}")
            print(f"   Caption: {result.get('caption', {}).get('basic_caption', '')[:50]}...")
        else:
            print(f"   ✗ Failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"   Request failed: {e}")
    
    print("\n3. API Documentation available at:")
    print(f"   {base_url}/docs")

if __name__ == "__main__":
    test_api()