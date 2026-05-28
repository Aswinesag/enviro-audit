# src/models/improved_captioner.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from typing import Dict, Optional

class ImprovedCaptioner:
    """Improved image captioning with better prompts."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large"):
        print(f"Loading improved BLIP model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Improved captioner loaded on: {self.device}")
    
    def caption_with_prompt(self, image: Image.Image, prompt: str = None) -> Dict:
        """Generate caption with optional prompt."""
        
        if prompt:
            # Use conditional generation with prompt
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        else:
            # Try different prompts for different image types
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate with different parameters
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                temperature=0.9,
                do_sample=True
            )
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Post-process caption
        caption = self._post_process_caption(caption)
        
        return {
            "caption": caption,
            "has_construction_terms": self._check_construction_terms(caption),
            "confidence_score": self._estimate_confidence(caption)
        }
    
    def _post_process_caption(self, caption: str) -> str:
        """Post-process caption to improve quality."""
        # Remove common artifacts
        if caption.startswith("a "):
            caption = caption[2:]
        elif caption.startswith("an "):
            caption = caption[3:]
        
        # Capitalize first letter
        caption = caption[0].upper() + caption[1:] if caption else caption
        
        return caption
    
    def _check_construction_terms(self, caption: str) -> bool:
        """Check if caption contains construction-related terms."""
        construction_terms = [
            "construction", "excavator", "bulldozer", "crane", "truck",
            "vehicle", "machine", "building", "site", "work", "worker",
            "industrial", "equipment", "machinery", "hard hat", "safety",
            "mining", "quarry", "digging", "drilling", "earth", "dirt",
            "gravel", "sand", "rock", "heavy", "large", "yellow", "equipment"
        ]
        
        caption_lower = caption.lower()
        return any(term in caption_lower for term in construction_terms)
    
    def _estimate_confidence(self, caption: str) -> float:
        """Estimate confidence based on caption characteristics."""
        # Simple heuristic: longer, more specific captions are better
        words = caption.split()
        if len(words) < 3:
            return 0.3  # Low confidence for very short captions
        elif len(words) > 8:
            return 0.8  # High confidence for detailed captions
        else:
            return 0.6  # Medium confidence