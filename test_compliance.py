import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure src is in path
sys.path.append(os.getcwd())

from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline

class TestCompliance(unittest.TestCase):
    def test_assess_compliance(self):
        # Mock models to avoid loading them
        pipeline = EnhancedAnalysisPipeline.__new__(EnhancedAnalysisPipeline)
        pipeline.classifier = MagicMock()
        pipeline.captioner = MagicMock()
        pipeline.detector = MagicMock()
        pipeline.satellite_client = MagicMock()
        
        # Mock data based on expected structure
        classification = {
            "predictions": [],
            "is_construction_activity": True,
            "primary_label": "construction site",
            "primary_confidence": "98%",
            "average_confidence": "95%",
            "analysis": "Test analysis"
        }
        
        caption = {
            "basic_caption": "a construction site",
            "enhanced_caption": "a construction site context",
            "has_construction_terms": True
        }
        
        detection = {
            "available": True,
            "detections": [],
            "statistics": {
                "heavy_machinery_count": 2,
                "vehicle_count": 1,
                "material_count": 5,
                "total_count": 8,
                "avg_confidence": 0.8
            }
        }
        
        print("\nTesting _assess_enhanced_compliance...")
        try:
            result = pipeline._assess_enhanced_compliance(classification, caption, detection)
            print("SUCCESS:", result)
        except Exception as e:
            print(f"FAILED with {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    unittest.main()
