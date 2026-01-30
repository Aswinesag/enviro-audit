import sys
import os
import time

# Ensure src is in path
sys.path.append(os.getcwd())

from src.utils.satellite import SatelliteImageryOrchestrator

def test_satellite_module():
    print("===========================================")
    print("   Testing Zero-Key Satellite Module       ")
    print("===========================================")
    
    orchestrator = SatelliteImageryOrchestrator()
    
    # Test Location: San Francisco (Urban)
    lat, lon = 37.7749, -122.4194
    print(f"\n1. Testing fetch for San Francisco ({lat}, {lon})...")
    
    start_time = time.time()
    result = orchestrator.get_satellite_image(lat, lon)
    duration = time.time() - start_time
    
    if result["success"]:
        print(f"   ✓ SUCCESS")
        print(f"   ✓ Provider Used: {result.get('provider')}")
        print(f"   ✓ Image Size: {result['image'].size}")
        print(f"   ✓ Time Taken: {duration:.2f}s")
        print(f"   ✓ Metadata: {result['metadata']}")
        
        # Save for manual inspection
        output_file = "test_satellite_output.png"
        result['image'].save(output_file)
        print(f"   ✓ Image saved to {output_file}")
    else:
        print(f"   ✗ FAILED: {result.get('error')}")

    # Test Map Generation
    print("\n2. Testing Map Visualization...")
    html = orchestrator.create_map_visualization(lat, lon)
    
    if "folium missing" not in html and "failed" not in html:
         print("   ✓ SUCCESS: Map HTML generated.")
         # Save sample map
         with open("test_map_output.html", "w") as f:
             f.write(html)
         print("   ✓ Map saved to test_map_output.html")
    else:
         print(f"   ✗ FAILED: {html}")

    print("\n===========================================")
    print("Verification Complete. Check the output files.")

if __name__ == "__main__":
    test_satellite_module()
