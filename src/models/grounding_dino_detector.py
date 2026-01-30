# src/models/grounding_dino_detector.py
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Grounding DINO imports
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from groundingdino.util import box_ops
except ImportError:
    print("Warning: GroundingDINO not installed. Some features may not work.")
    print("Install with: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")

class GroundingDINODetector:
    """Advanced object detection using GroundingDINO."""
    
    def __init__(self, 
                 model_config_path: str = None,
                 model_checkpoint_path: str = None):
        """
        Initialize GroundingDINO detector.
        
        Args:
            model_config_path: Path to model config file
            model_checkpoint_path: Path to model checkpoint
        """
        print("Initializing GroundingDINO detector...")
        
        # Default paths
        if model_config_path is None:
            model_config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
            
        if model_checkpoint_path is None:
            model_checkpoint_path = "weights/groundingdino_swint_ogc.pth"
        
        try:
            # Load model
            self.model = load_model(
                model_config_path,
                model_checkpoint_path
            )
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ GroundingDINO loaded on: {self.device}")
            
            # Construction-specific prompts
            self.construction_prompts = {
                "heavy_machinery": [
                    "excavator", "bulldozer", "backhoe", "crane", "loader",
                    "dump truck", "cement mixer", "forklift", "grader", "compactor",
                    "haul truck", "drill rig", "pile driver"
                ],
                "vehicles": [
                    "construction vehicle", "heavy equipment", "mining truck",
                    "transport truck", "service vehicle"
                ],
                "structures": [
                    "construction site", "building foundation", "scaffolding",
                    "temporary structure", "storage container", "site office"
                ],
                "materials": [
                    "pile of dirt", "pile of gravel", "pile of sand",
                    "construction materials", "debris pile", "soil stockpile"
                ],
                "people": [
                    "construction worker", "worker with hard hat",
                    "safety vest", "construction personnel"
                ]
            }
            
        except Exception as e:
            print(f"✗ Failed to load GroundingDINO: {e}")
            print("Make sure to download weights: python -c 'from groundingdino.util.tools import download_weights; download_weights()'")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if GroundingDINO is available."""
        return self.model is not None
    
    def detect_construction(
        self,
        image: Image.Image,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_detections: int = 20
    ) -> Dict[str, Any]:
        """
        Detect construction-related objects in image.
        
        Args:
            image: PIL Image
            box_threshold: Confidence threshold for boxes
            text_threshold: Confidence threshold for text
            max_detections: Maximum number of detections to return
            
        Returns:
            Dictionary with detections and metadata
        """
        if not self.is_available():
            return {
                "available": False,
                "error": "GroundingDINO not loaded",
                "detections": []
            }
        
        try:
            # Convert PIL to OpenCV format for GroundingDINO
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create text prompt from all construction categories
            all_prompts = []
            for category, prompts in self.construction_prompts.items():
                all_prompts.extend(prompts)
            
            # Join with periods (GroundingDINO works better with periods)
            text_prompt = ". ".join(all_prompts) + "."
            
            # Load image for GroundingDINO
            image_source, image_tensor = load_image(image_cv)
            
            # Run detection
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # Convert to list of detections
            detections = []
            if len(boxes) > 0:
                for box, score, phrase in zip(boxes, logits, phrases):
                    # Convert box from [x_center, y_center, width, height] to [x1, y1, x2, y2]
                    h, w, _ = image_cv.shape
                    box = box * torch.Tensor([w, h, w, h])
                    box[:2] -= box[2:] / 2
                    box[2:] += box[:2]
                    
                    # Convert to list
                    box = box.cpu().numpy().astype(int).tolist()
                    
                    # Determine category
                    category = self._categorize_phrase(phrase)
                    
                    detections.append({
                        "bbox": box,
                        "score": float(score),
                        "label": phrase,
                        "category": category,
                        "confidence": f"{float(score):.1%}"
                    })
            
            # Sort by score and limit
            detections.sort(key=lambda x: x["score"], reverse=True)
            detections = detections[:max_detections]
            
            # Generate statistics
            stats = self._generate_statistics(detections)
            
            # Create annotated image
            annotated_image = self._annotate_image(image.copy(), detections)
            
            return {
                "available": True,
                "detections": detections,
                "statistics": stats,
                "annotated_image": annotated_image,
                "total_detections": len(detections),
                "text_prompt_used": text_prompt
            }
            
        except Exception as e:
            print(f"GroundingDINO detection error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "available": False,
                "error": str(e),
                "detections": []
            }
    
    def _categorize_phrase(self, phrase: str) -> str:
        """Categorize detected phrase."""
        phrase_lower = phrase.lower()
        
        for category, prompts in self.construction_prompts.items():
            for prompt in prompts:
                if prompt in phrase_lower:
                    return category
        
        # Check for specific keywords
        if any(word in phrase_lower for word in ["excavator", "bulldozer", "crane", "loader"]):
            return "heavy_machinery"
        elif any(word in phrase_lower for word in ["truck", "vehicle", "equipment"]):
            return "vehicles"
        elif any(word in phrase_lower for word in ["pile", "dirt", "gravel", "sand"]):
            return "materials"
        elif any(word in phrase_lower for word in ["worker", "person", "people"]):
            return "people"
        elif any(word in phrase_lower for word in ["site", "structure", "building"]):
            return "structures"
        
        return "other"
    
    def _generate_statistics(self, detections: List[Dict]) -> Dict[str, Any]:
        """Generate statistics from detections."""
        if not detections:
            return {
                "heavy_machinery_count": 0,
                "vehicle_count": 0,
                "material_count": 0,
                "people_count": 0,
                "structure_count": 0,
                "total_count": 0,
                "avg_confidence": 0.0
            }
        
        category_counts = {
            "heavy_machinery": 0,
            "vehicles": 0,
            "materials": 0,
            "people": 0,
            "structures": 0,
            "other": 0
        }
        
        total_score = 0.0
        
        for det in detections:
            category = det.get("category", "other")
            if category in category_counts:
                category_counts[category] += 1
            total_score += det["score"]
        
        return {
            "heavy_machinery_count": category_counts["heavy_machinery"],
            "vehicle_count": category_counts["vehicles"],
            "material_count": category_counts["materials"],
            "people_count": category_counts["people"],
            "structure_count": category_counts["structures"],
            "other_count": category_counts["other"],
            "total_count": len(detections),
            "avg_confidence": total_score / len(detections) if detections else 0.0
        }
    
    def _annotate_image(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """Create annotated image with bounding boxes."""
        if not detections:
            return image
        
        draw = ImageDraw.Draw(image)
        
        # Colors for different categories
        category_colors = {
            "heavy_machinery": (255, 0, 0),      # Red
            "vehicles": (0, 0, 255),            # Blue
            "materials": (0, 255, 0),           # Green
            "people": (255, 255, 0),            # Yellow
            "structures": (255, 0, 255),        # Magenta
            "other": (128, 128, 128)            # Gray
        }
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            bbox = det["bbox"]
            label = det["label"]
            score = det["score"]
            category = det.get("category", "other")
            
            color = category_colors.get(category, (128, 128, 128))
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            # Draw label background
            label_text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label_text, font=font)
            draw.rectangle(text_bbox, fill=color)
            
            # Draw label text
            draw.text((bbox[0], bbox[1] - 20), label_text, fill=(255, 255, 255), font=font)
        
        return image
    
    def detect_with_custom_prompt(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25
    ) -> List[Dict]:
        """Detect objects with custom text prompt."""
        if not self.is_available():
            return []
        
        try:
            # Convert PIL to OpenCV
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image_source, image_tensor = load_image(image_cv)
            
            # Run detection
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # Format results
            results = []
            if len(boxes) > 0:
                h, w, _ = image_cv.shape
                for box, score, phrase in zip(boxes, logits, phrases):
                    # Convert box format
                    box = box * torch.Tensor([w, h, w, h])
                    box[:2] -= box[2:] / 2
                    box[2:] += box[:2]
                    
                    results.append({
                        "bbox": box.cpu().numpy().astype(int).tolist(),
                        "score": float(score),
                        "label": phrase,
                        "confidence": f"{float(score):.1%}"
                    })
            
            return results
            
        except Exception as e:
            print(f"Custom prompt detection error: {e}")
            return []