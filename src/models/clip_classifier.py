# src/models/clip_classifier.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict, Any  # Ensure all imports are here
import numpy as np

class CLIPZeroShotClassifier:
    """Zero-shot image classification using CLIP."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"CLIP loaded on: {self.device}")
    
    def classify(
        self,
        image: Image.Image,
        candidate_labels: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify image using zero-shot learning.
        
        Args:
            image: PIL Image
            candidate_labels: List of possible classes
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with scores
        """
        # Preprocess
        inputs = self.processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Calculate probabilities
        logits_per_image = outputs.logits_per_image  # [1, num_labels]
        probs = logits_per_image.softmax(dim=1)[0]  # [num_labels]
        
        # Get top predictions
        values, indices = torch.topk(probs, min(top_k, len(candidate_labels)))
        
        # Format results
        results = []
        for score, idx in zip(values, indices):
            results.append({
                "label": candidate_labels[idx],
                "score": score.item(),
                "confidence": f"{score.item():.1%}"
            })
        
        return results
    
    def classify_construction(self, image: Image.Image) -> Dict[str, Any]:
        """Specialized classification for construction/environmental monitoring."""
        
        construction_labels = [
            "construction site with heavy machinery",
            "construction site with vehicles",
            "mining or quarry operation",
            "land clearing or deforestation",
            "building construction",
            "road construction",
            "natural landscape with no construction",
            "agricultural field",
            "forest or wooded area",
            "urban area with buildings",
            "water body or river",
            "bare earth or excavation"
        ]
        
        results = self.classify(image, construction_labels, top_k=3)
        
        # Determine if construction is present
        construction_keywords = ["construction", "mining", "deforestation", "excavation"]
        is_construction = any(
            any(keyword in result["label"].lower() for keyword in construction_keywords)
            for result in results[:2]  # Check top 2 results
        )
        
        # Calculate overall confidence
        avg_confidence = sum(r["score"] for r in results[:2]) / 2 if len(results) >= 2 else results[0]["score"]
        
        return {
            "predictions": results,
            "is_construction_activity": is_construction,
            "primary_label": results[0]["label"],
            "primary_confidence": results[0]["confidence"],
            "average_confidence": f"{avg_confidence:.1%}",
            "analysis": self._generate_analysis(results, is_construction)
        }
    
    def _generate_analysis(self, predictions: List[Dict], is_construction: bool) -> str:
        """Generate analysis text based on predictions."""
        top_pred = predictions[0]
        
        if is_construction:
            if "heavy machinery" in top_pred["label"]:
                return "Heavy construction activity detected. High likelihood of environmental impact."
            elif "vehicles" in top_pred["label"]:
                return "Construction vehicles detected. Moderate environmental impact possible."
            elif "mining" in top_pred["label"]:
                return "Mining operation detected. Significant environmental monitoring required."
            else:
                return "Construction activity detected. Further inspection recommended."
        else:
            if "natural" in top_pred["label"]:
                return "Natural landscape detected. No immediate environmental concerns."
            elif "agricultural" in top_pred["label"]:
                return "Agricultural activity detected. Standard monitoring applies."
            else:
                return "No construction activity detected. Area appears undisturbed."