import os
import sys
from PIL import Image
import traceback

# Ensure src is in path
sys.path.append(os.getcwd())

try:
    print("Step 1: Initialize Pipeline...")
    from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline
    pipeline = EnhancedAnalysisPipeline()
    print("  Pipeline initialized.")

    print("\nStep 2: Create Dummy Image...")
    image = Image.new('RGB', (640, 640), color='red')
    metadata = {"filename": "test.jpg", "project_id": "test"}

    print("\nStep 3: Run Analysis...")
    result = pipeline.analyze_image(image, metadata)
    print("\nSUCCESS! Result generated.")
    print(result.keys())

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    traceback.print_exc()
