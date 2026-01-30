import os
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

try:
    print("Step 1: Importing Settings...")
    from src.core.config import settings
    print(f"  Settings Loaded. HF Token present: {bool(settings.huggingface_token)}")
    print(f"  SentinelHub ID present: {bool(settings.sentinelhub_client_id)}")

    print("\nStep 2: Initialize Satellite Client...")
    from src.utils.satellite_client import SatelliteImageryClient
    client = SatelliteImageryClient()
    print("  Satellite Client initialized.")

    print("\nStep 3: Initialize Analysis Pipeline...")
    from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline
    pipeline = EnhancedAnalysisPipeline()
    print("  Analysis Pipeline initialized.")
    
except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
