# Update src/pipelines/analysis_pipeline.py
from typing import Dict, Any, Optional
from PIL import Image
from datetime import datetime
import numpy as np

from src.models.clip_classifier import CLIPZeroShotClassifier
from src.models.blip_captioner import BLIPCaptioner
from src.models.grounding_dino_detector import GroundingDINODetector
from src.utils.image_utils import ImagePreprocessor
from src.utils.satellite import SatelliteImageryOrchestrator

class EnhancedAnalysisPipeline:
    """Enhanced analysis pipeline with GroundingDINO and satellite imagery."""
    
    def __init__(self):
        print("Initializing Enhanced EnviroAudit Pipeline...")
        
        # Core models
        self.classifier = CLIPZeroShotClassifier()
        self.captioner = BLIPCaptioner()
        self.preprocessor = ImagePreprocessor()
        
        # Enhanced models
        self.detector = GroundingDINODetector()
        self.satellite_client = SatelliteImageryOrchestrator()
        
        print("✓ Enhanced pipeline initialized!")
        print(f"  ✓ CLIP Classifier: {'✅' if self.classifier else '❌'}")
        print(f"  ✓ BLIP Captioner: {'✅' if self.captioner else '❌'}")
        print(f"  ✓ GroundingDINO Detector: {'✅' if self.detector.is_available() else '❌'}")
        print(f"  ✓ Satellite Client: ✅")
    
    def analyze_image(self, image: Image.Image, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced image analysis with object detection."""
        print(f"Analyzing image: {image.size}")
        
        # Preprocess image
        processed_image = self.preprocessor.prepare_for_detection(image, target_size=(640, 640))
        
        # Step 1: Zero-shot classification
        print("  Running classification...")
        classification = self.classifier.classify_construction(processed_image)
        
        # Step 2: Image captioning
        print("  Generating caption...")
        caption_result = self.captioner.caption_with_context(processed_image)
        
        # Step 3: Object detection (if GroundingDINO is available)
        print("  Running object detection...")
        detection_result = self.detector.detect_construction(processed_image)
        
        # Step 4: Enhanced compliance assessment
        print("  Assessing compliance...")
        compliance = self._assess_enhanced_compliance(classification, caption_result, detection_result)
        
        # Step 5: Generate comprehensive report
        report = self._generate_enhanced_report(classification, caption_result, detection_result, compliance, metadata)
        
        # Prepare result
        result = {
            "metadata": metadata or {},
            "image_info": {
                "original_size": image.size,
                "processed_size": processed_image.size,
                "format": image.format if hasattr(image, 'format') else 'Unknown'
            },
            "classification": classification,
            "caption": caption_result,
            "object_detection": detection_result,
            "compliance": compliance,
            "report": report,
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "2.0"
        }
        
        # Add annotated image if available
        if detection_result.get("available") and detection_result.get("annotated_image"):
            result["annotated_image"] = self._image_to_base64(detection_result["annotated_image"])
        
        return result
    
    def analyze_location(
        self,
        latitude: float,
        longitude: float,
        date: Optional[str] = None,
        bbox_size: float = 0.01,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze a geographic location using satellite imagery.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            date: Date for satellite imagery (YYYY-MM-DD)
            bbox_size: Size of area to analyze (in degrees)
            metadata: Additional metadata
            
        Returns:
            Comprehensive analysis including satellite imagery
        """
        print(f"Analyzing location: {latitude}, {longitude}")
        
        # Step 1: Get satellite imagery
        print("  Fetching satellite imagery...")
        satellite_data = self.satellite_client.get_satellite_image(
            latitude=latitude,
            longitude=longitude,
            bbox_size=bbox_size,
            date=date
        )
        
        if not satellite_data.get("success"):
            return {
                "error": "Failed to obtain satellite imagery",
                "satellite_data": satellite_data
            }
        
        satellite_image = satellite_data["image"]
        
        # Step 2: Analyze the satellite image
        print("  Analyzing satellite image...")
        image_analysis = self.analyze_image(satellite_image, metadata)
        
        # Step 3: Create map visualization
        print("  Creating map visualization...")
        detections = image_analysis.get("object_detection", {}).get("detections", [])
        map_html = self.satellite_client.create_map_visualization(
            latitude=latitude,
            longitude=longitude,
            detections=detections,
            bbox_size=bbox_size
        )
        
        # Step 4: Combine results
        result = {
            **image_analysis,
            "satellite_data": {
                "success": satellite_data["success"],
                "provider": satellite_data.get("provider"),
                "coordinates": satellite_data.get("coordinates"),
                "metadata": satellite_data.get("metadata"),
                "map_html": map_html,
                "satellite_image": self._image_to_base64(satellite_image)
            },
            "location_analysis": {
                "latitude": latitude,
                "longitude": longitude,
                "bbox_size": bbox_size,
                "date": date or datetime.now().strftime("%Y-%m-%d"),
                "has_satellite_imagery": True
            }
        }
        
        return result
    
    def compare_locations_over_time(
        self,
        latitude: float,
        longitude: float,
        date1: str,
        date2: Optional[str] = None,
        bbox_size: float = 0.01
    ) -> Dict[str, Any]:
        """Compare changes at a location over time."""
        print(f"Comparing changes at {latitude}, {longitude}")
        
        # Get historical comparison
        comparison = self.satellite_client.get_historical_comparison(
            latitude=latitude,
            longitude=longitude,
            date1=date1,
            date2=date2,
            bbox_size=bbox_size
        )
        
        if not comparison.get("success"):
            return {
                "error": "Failed to get historical comparison",
                "comparison": comparison
            }
        
        # Analyze both images
        analysis1 = self.analyze_image(comparison["image1"])
        analysis2 = self.analyze_image(comparison["image2"])
        
        # Calculate changes
        changes = self._calculate_changes(analysis1, analysis2, comparison)
        
        return {
            "date1": date1,
            "date2": date2 or datetime.now().strftime("%Y-%m-%d"),
            "location": [latitude, longitude],
            "change_percentage": comparison.get("change_percentage", 0),
            "change_detected": comparison.get("change_detected", False),
            "analysis_date1": analysis1,
            "analysis_date2": analysis2,
            "changes": changes,
            "comparison_images": {
                "image1": self._image_to_base64(comparison["image1"]),
                "image2": self._image_to_base64(comparison["image2"]),
                "difference": self._image_to_base64(comparison["difference"]) if "difference" in comparison else None
            }
        }
    
    def _assess_enhanced_compliance(
        self,
        classification: Dict,
        caption: Dict,
        detection: Dict
    ) -> Dict[str, Any]:
        """Enhanced compliance assessment using object detection."""
        
        # Base assessment from classification
        is_construction = classification["is_construction_activity"]
        has_construction_terms = caption["has_construction_terms"]
        
        # Enhanced assessment from object detection
        detection_stats = detection.get("statistics", {})
        heavy_machinery_count = detection_stats.get("heavy_machinery_count", 0)
        vehicle_count = detection_stats.get("vehicle_count", 0)
        material_count = detection_stats.get("material_count", 0)
        
        # Calculate enhanced risk score
        base_risk = 0
        
        if is_construction and has_construction_terms:
            base_risk = 3
        elif is_construction or has_construction_terms:
            base_risk = 2
        else:
            base_risk = 1
        
        # Add detection-based risk
        detection_risk = min(heavy_machinery_count * 2 + vehicle_count + material_count * 0.5, 10)
        total_risk = base_risk + detection_risk
        
        # Determine risk level
        if total_risk >= 8:
            risk_level = "CRITICAL"
            action = "IMMEDIATE_ACTION"
            confidence = "high"
        elif total_risk >= 5:
            risk_level = "HIGH"
            action = "IMMEDIATE_INSPECTION"
            confidence = "high"
        elif total_risk >= 3:
            risk_level = "MEDIUM"
            action = "SCHEDULED_INSPECTION"
            confidence = "medium"
        else:
            risk_level = "LOW"
            action = "MONITOR_ONLY"
            confidence = "high"
        
        # Special cases based on detection
        if heavy_machinery_count >= 3:
            risk_level = "CRITICAL"
            action = "IMMEDIATE_ACTION"
        
        return {
            "risk_level": risk_level,
            "recommended_action": action,
            "confidence": confidence,
            "risk_score": total_risk,
            "indicators": {
                "construction_activity_detected": is_construction,
                "construction_terms_in_caption": has_construction_terms,
                "heavy_machinery_count": heavy_machinery_count,
                "vehicle_count": vehicle_count,
                "material_count": material_count,
                "primary_activity": classification["primary_label"],
                "object_detection_available": detection.get("available", False)
            },
            "detection_statistics": detection_stats
        }
    
    def _generate_enhanced_report(
        self,
        classification: Dict,
        caption: Dict,
        detection: Dict,
        compliance: Dict,
        metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate enhanced report with object detection details."""
        
        # Basic report components
        report_text = f"""
        ENVIRONMENTAL COMPLIANCE REPORT - ENHANCED ANALYSIS
        {'='*60}
        
        ANALYSIS SUMMARY:
        - Primary Classification: {classification['primary_label']}
        - Confidence: {classification['primary_confidence']}
        - Caption: {caption['basic_caption']}
        
        OBJECT DETECTION RESULTS:
        """
        
        # Add object detection details
        if detection.get("available"):
            stats = detection.get("statistics", {})
            report_text += f"""
        - Heavy Machinery Detected: {stats.get('heavy_machinery_count', 0)}
        - Construction Vehicles: {stats.get('vehicle_count', 0)}
        - Material Stockpiles: {stats.get('material_count', 0)}
        - Total Objects Detected: {stats.get('total_count', 0)}
        - Detection Confidence: {stats.get('avg_confidence', 0.0):.1%}
            """
            
            # List top detections
            detections = detection.get("detections", [])[:5]
            if detections:
                report_text += "\n        Top Detections:"
                for i, det in enumerate(detections[:3]):
                    report_text += f"\n          {i+1}. {det['label']} ({det['confidence']})"
        else:
            report_text += "\n        - Object detection not available"
        
        # Compliance assessment
        report_text += f"""
        
        COMPLIANCE ASSESSMENT:
        - Risk Level: {compliance['risk_level']}
        - Risk Score: {compliance['risk_score']:.1f}/10
        - Recommended Action: {compliance['recommended_action']}
        - Confidence: {compliance['confidence'].upper()}
        
        DETAILED INDICATORS:
        - Construction Activity: {'YES' if classification['is_construction_activity'] else 'NO'}
        - Construction Terms in Description: {'YES' if caption['has_construction_terms'] else 'NO'}
        - Heavy Machinery Count: {compliance['indicators']['heavy_machinery_count']}
        - Vehicle Count: {compliance['indicators']['vehicle_count']}
        - Material Count: {compliance['indicators']['material_count']}
        
        ADDITIONAL ANALYSIS:
        {classification['analysis']}
        
        RECOMMENDATIONS:
        {self._get_enhanced_recommendations(compliance)}
        
        {'='*60}
        Report generated: {datetime.now().isoformat()}
        Enhanced Analysis Pipeline v2.0
        """
        
        # HTML report with visualization
        html_report = self._generate_enhanced_html_report(
            classification, caption, detection, compliance, metadata
        )
        
        return {
            "text": report_text.strip(),
            "html": html_report,
            "summary": {
                "risk": compliance["risk_level"],
                "action": compliance["recommended_action"],
                "confidence": compliance["confidence"],
                "object_count": detection.get("statistics", {}).get("total_count", 0),
                "heavy_machinery": detection.get("statistics", {}).get("heavy_machinery_count", 0)
            }
        }
    
    def _get_enhanced_recommendations(self, compliance: Dict) -> str:
        """Get enhanced recommendations based on object detection."""
        risk_level = compliance["risk_level"]
        heavy_machinery = compliance["indicators"]["heavy_machinery_count"]
        
        recommendations = {
            "CRITICAL": f"""
        1. IMMEDIATELY dispatch inspection team (within 24 hours)
        2. Notify regulatory authorities immediately
        3. Issue stop-work order if unauthorized activity detected
        4. Set up 24/7 monitoring with aerial surveillance
        5. Document all heavy machinery ({heavy_machinery} pieces detected)
        6. Schedule follow-up inspection in 7 days""",
            
            "HIGH": f"""
        1. Schedule inspection within 48 hours
        2. Review all permits and documentation
        3. Document current site conditions with photos
        4. Set up bi-weekly monitoring
        5. Verify machinery counts and types
        6. Update risk assessment based on findings""",
            
            "MEDIUM": """
        1. Schedule inspection within 7 days
        2. Request updated project documentation
        3. Monitor for changes or expansion
        4. Verify compliance with existing permits
        5. Update GIS database with findings""",
            
            "LOW": """
        1. Continue routine monitoring schedule
        2. No immediate action required
        3. Document analysis for records
        4. Review in next quarterly cycle
        5. Maintain baseline imagery for future comparison"""
        }
        
        return recommendations.get(risk_level, "No specific recommendations available.")
    
    def _generate_enhanced_html_report(self, classification: Dict, caption: Dict,
                                      detection: Dict, compliance: Dict, metadata: Optional[Dict]) -> str:
        """Generate enhanced HTML report."""
        risk_class = f"risk-{compliance['risk_level'].lower()}"
        
        # Detection statistics
        detection_stats = ""
        if detection.get("available"):
            stats = detection.get("statistics", {})
            detection_stats = f"""
            <div class="detection-stats">
                <h3>Object Detection Statistics</h3>
                <table>
                    <tr><td>Heavy Machinery:</td><td>{stats.get('heavy_machinery_count', 0)}</td></tr>
                    <tr><td>Construction Vehicles:</td><td>{stats.get('vehicle_count', 0)}</td></tr>
                    <tr><td>Material Stockpiles:</td><td>{stats.get('material_count', 0)}</td></tr>
                    <tr><td>Total Detections:</td><td>{stats.get('total_count', 0)}</td></tr>
                    <tr><td>Average Confidence:</td><td>{stats.get('avg_confidence', 0.0):.1%}</td></tr>
                </table>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Environmental Compliance Report - Enhanced</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #1E3A8A; color: white; padding: 30px; border-radius: 10px; }}
                .risk-critical {{ color: #DC2626; font-weight: bold; background-color: #FEE2E2; padding: 10px; border-radius: 5px; }}
                .risk-high {{ color: #EA580C; font-weight: bold; background-color: #FFEDD5; padding: 10px; border-radius: 5px; }}
                .risk-medium {{ color: #CA8A04; font-weight: bold; background-color: #FEF3C7; padding: 10px; border-radius: 5px; }}
                .risk-low {{ color: #16A34A; font-weight: bold; background-color: #DCFCE7; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 25px 0; padding: 20px; background-color: #F8FAFC; border-radius: 8px; border-left: 5px solid #3B82F6; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .detection-stats table {{ width: 100%; border-collapse: collapse; }}
                .detection-stats td {{ padding: 8px; border-bottom: 1px solid #E5E7EB; }}
                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #E5E7EB; color: #6B7280; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌍 EnviroAudit - Enhanced Compliance Report</h1>
                <p>AI-Powered Environmental Monitoring with Object Detection</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Compliance Assessment</h2>
                <div class="{risk_class}">
                    <h3>Risk Level: {compliance['risk_level']} (Score: {compliance['risk_score']:.1f}/10)</h3>
                    <p><strong>Recommended Action:</strong> {compliance['recommended_action']}</p>
                    <p><strong>Confidence:</strong> {compliance['confidence'].upper()}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Analysis Results</h2>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Primary Classification</h3>
                        <p style="font-size: 1.2em; font-weight: bold;">{classification['primary_label']}</p>
                        <p>Confidence: {classification['primary_confidence']}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Image Description</h3>
                        <p>{caption['basic_caption']}</p>
                        <p>Construction Terms: {'✅ Yes' if caption['has_construction_terms'] else '❌ No'}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Object Detection</h3>
                        <p>Status: {'✅ Available' if detection.get('available') else '❌ Unavailable'}</p>
                        <p>Total Detections: {detection.get('statistics', {}).get('total_count', 0)}</p>
                    </div>
                </div>
            </div>
            
            {detection_stats if detection.get('available') else '<div class="section"><p><em>Object detection not available for this analysis.</em></p></div>'}
            
            <div class="section">
                <h2>Detailed Analysis</h2>
                <p>{classification['analysis']}</p>
            </div>
            
            <div class="section">
                <h2>Pipeline Information</h2>
                <p><strong>Version:</strong> Enhanced Pipeline v2.0</p>
                <p><strong>Models Used:</strong> CLIP (Classification), BLIP (Captioning), GroundingDINO (Object Detection)</p>
                <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
            </div>
            
            <div class="footer">
                <p>EnviroAudit - AI-Powered Environmental Compliance Monitoring System</p>
                <p>This report was automatically generated by the EnviroAudit AI system.</p>
                <p>For questions or concerns, contact your environmental compliance officer.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _calculate_changes(self, analysis1: Dict, analysis2: Dict, comparison: Dict) -> Dict[str, Any]:
        """Calculate changes between two analyses."""
        changes = {
            "risk_level_change": analysis2["compliance"]["risk_level"] != analysis1["compliance"]["risk_level"],
            "risk_score_change": analysis2["compliance"].get("risk_score", 0) - analysis1["compliance"].get("risk_score", 0),
            "object_count_change": analysis2["object_detection"].get("statistics", {}).get("total_count", 0) - 
                                 analysis1["object_detection"].get("statistics", {}).get("total_count", 0),
            "heavy_machinery_change": analysis2["object_detection"].get("statistics", {}).get("heavy_machinery_count", 0) - 
                                     analysis1["object_detection"].get("statistics", {}).get("heavy_machinery_count", 0),
            "pixel_change_percentage": comparison.get("change_percentage", 0)
        }
        
        # Determine overall change status
        if changes["risk_score_change"] > 2 or changes["heavy_machinery_change"] > 1:
            changes["overall_change"] = "SIGNIFICANT_INCREASE"
        elif changes["risk_score_change"] < -2:
            changes["overall_change"] = "SIGNIFICANT_DECREASE"
        elif abs(changes["risk_score_change"]) > 0.5:
            changes["overall_change"] = "MODERATE_CHANGE"
        else:
            changes["overall_change"] = "MINIMAL_CHANGE"
        
        return changes
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"