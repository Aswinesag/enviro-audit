# scripts/test_multiple_images.py
import requests
import json
import time

def test_multiple_images():
    base_url = "http://localhost:8000"
    
    test_cases = [
        {
            "name": "Construction Site",
            "url": "https://images.unsplash.com/photo-1581094794329-c8112a89af12",
            "expected": "construction/mining activity"
        },
        {
            "name": "Natural Landscape", 
            "url": "https://images.unsplash.com/photo-1501854140801-50d01698950b",
            "expected": "natural landscape"
        },
        {
            "name": "Excavator",
            "url": "https://images.unsplash.com/photo-1541888946425-d81bb19240f5",
            "expected": "construction equipment"
        },
        {
            "name": "Forest",
            "url": "https://images.unsplash.com/photo-1448375240586-882707db888b",
            "expected": "natural area"
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test['name']}")
        print(f"URL: {test['url']}")
        
        try:
            response = requests.post(
                f"{base_url}/analyze-url",
                json={
                    "image_url": test['url'],
                    "project_id": f"test-{test['name'].lower().replace(' ', '-')}"
                },
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                risk_level = result.get('compliance', {}).get('risk_level', 'UNKNOWN')
                primary_label = result.get('classification', {}).get('primary_label', 'N/A')
                
                print(f"✅ Success!")
                print(f"   Primary Label: {primary_label}")
                print(f"   Risk Level: {risk_level}")
                print(f"   Caption: {result.get('caption', {}).get('basic_caption', 'N/A')[:60]}...")
                
                results.append({
                    "name": test['name'],
                    "success": True,
                    "risk_level": risk_level,
                    "primary_label": primary_label,
                    "expected": test['expected']
                })
                
                # Save individual result
                filename = f"data/test_{test['name'].lower().replace(' ', '_')}.json"
                with open(filename, "w") as f:
                    json.dump(result, f, indent=2)
                    
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"   Error: {response.text[:100]}")
                results.append({
                    "name": test['name'],
                    "success": False,
                    "error": response.text[:100]
                })
                
        except Exception as e:
            print(f"❌ Exception: {str(e)[:100]}")
            results.append({
                "name": test['name'],
                "success": False,
                "error": str(e)[:100]
            })
        
        # Wait a bit between requests
        time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n✅ Successful: {len(successful)}/{len(test_cases)}")
    for r in successful:
        print(f"   - {r['name']}: {r['primary_label']} -> {r['risk_level']} risk")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(test_cases)}")
        for r in failed:
            print(f"   - {r['name']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    with open("data/test_summary.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(test_cases),
            "successful": len(successful),
            "failed": len(failed),
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Summary saved to: data/test_summary.json")
    
    return len(successful) == len(test_cases)

if __name__ == "__main__":
    success = test_multiple_images()
    if success:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the results above.")