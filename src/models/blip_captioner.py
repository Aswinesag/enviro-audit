# src/models/blip_captioner.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from typing import Dict  # ADD THIS IMPORT

class BLIPCaptioner:
    """Image captioning using BLIP."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        print(f"Loading BLIP model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"BLIP loaded on: {self.device}")
    
    def caption(self, image: Image.Image, max_length: int = 50) -> str:
        """Generate caption for image."""
        # Preprocess
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=max_length)
        
        # Decode
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def caption_with_context(self, image: Image.Image, context: str = None) -> Dict:
        """Generate caption with optional context."""
        caption = self.caption(image)
        
        # Enhance caption based on context
        enhanced_caption = self._enhance_caption(caption, context)
        
        return {
            "basic_caption": caption,
            "enhanced_caption": enhanced_caption,
            "has_construction_terms": self._check_construction_terms(caption)
        }
    
    def _enhance_caption(self, caption: str, context: str = None) -> str:
        """Enhance caption with additional details."""
        if "construction" in caption.lower() or "excavator" in caption.lower() or "truck" in caption.lower():
            enhanced = f"{caption} This appears to be construction or industrial activity."
        elif "building" in caption.lower() or "structure" in caption.lower():
            enhanced = f"{caption} Man-made structures are visible."
        elif "tree" in caption.lower() or "forest" in caption.lower() or "field" in caption.lower():
            enhanced = f"{caption} Natural landscape is visible."
        else:
            enhanced = caption
        
        if context:
            enhanced = f"{enhanced} Context: {context}"
        
        return enhanced
    
    def _check_construction_terms(self, caption: str) -> bool:
        """Check if caption contains construction-related terms."""
        construction_terms = [
            "construction", "excavator", "bulldozer", "crane", "truck",
            "vehicle", "machine", "building", "site", "work",
            "industrial", "equipment", "machinery", "hard hat"
        ]
        
        caption_lower = caption.lower()
        return any(term in caption_lower for term in construction_terms)