import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure src is in path
sys.path.append(os.getcwd())

from src.pipelines.analysis_pipeline import EnhancedAnalysisPipeline

class TestReport(unittest.TestCase):
    def test_generate_report(self):
        # Mock models
        pipeline = EnhancedAnalysisPipeline.__new__(EnhancedAnalysisPipeline)
        
        # Mock data
        classification = {
            "predictions": [],
            "is_construction_activity": True,
            "primary_label": "construction site",
            "primary_confidence": "98%",
            "analysis": "Test analysis"
        }
        
        caption = {
            "basic_caption": "a construction site",
            "has_construction_terms": True
        }
        
        detection = {
            "available": True,
            "statistics": {
                "heavy_machinery_count": 2,
                "vehicle_count": 1,
                "material_count": 5,
                "total_count": 8,
                "avg_confidence": 0.8
            },
            "detections": [
                {"label": "excavator", "confidence": "0.9"},
                {"label": "truck", "confidence": "0.8"}
            ]
        }
        
        compliance = {
            "risk_level": "HIGH",
            "risk_score": 6.5,
            "recommended_action": "INSPECT",
            "confidence": "high",
            "indicators": {
                "heavy_machinery_count": 2,
                "vehicle_count": 1,
                "material_count": 5
            }
        }
        
        metadata = {"filename": "test.jpg"}
        
        print("\nTesting _generate_enhanced_report...")
        try:
            report = pipeline._generate_enhanced_report(classification, caption, detection, compliance, metadata)
            print("SUCCESS Report Generated")
        except Exception as e:
            print(f"FAILED with {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    unittest.main()
