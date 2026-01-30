from .base import SatelliteProvider
from PIL import Image
import numpy as np
from typing import Dict, Any, Tuple, Optional

class SyntheticProvider(SatelliteProvider):
    """Generates synthetic satellite-like imagery as a fallback."""
    
    def __init__(self):
        super().__init__("Synthetic Generation")

    def fetch_image(
        self, 
        bbox: Tuple[float, float, float, float], 
        width: int, 
        height: int,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        
        try:
            image = self._generate_synthetic_image(width, height)
            
            return {
                "success": True,
                "image": image,
                "metadata": {
                    "provider": self.name,
                    "date": date or "Generated",
                    "source": "Procedural Generation",
                    "note": "This is AI-generated fallback imagery."
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_synthetic_image(self, width: int, height: int) -> Image.Image:
        # Create gradient background
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Create terrain-like patterns
        terrain = (
            np.sin(X * 10) * np.cos(Y * 10) * 0.3 +
            np.sin(X * 20) * np.cos(Y * 20) * 0.2 +
            np.random.randn(height, width) * 0.1
        )
        
        # Add some "construction" features randomly
        construction_mask = np.random.rand(height, width) < 0.05
        
        # Create RGB image
        r = (0.3 + terrain * 0.7) * 255  # Red channel - vegetation/soil
        g = (0.4 + terrain * 0.6) * 255  # Green channel - vegetation
        b = (0.5 + terrain * 0.5) * 255  # Blue channel - water/urban
        
        # Add construction areas (yellow/brown)
        r[construction_mask] = 200 + np.random.randn(*r[construction_mask].shape) * 20
        g[construction_mask] = 180 + np.random.randn(*g[construction_mask].shape) * 20
        b[construction_mask] = 100 + np.random.randn(*b[construction_mask].shape) * 20
        
        # Clip values
        r = np.clip(r, 0, 255)
        g = np.clip(g, 0, 255)
        b = np.clip(b, 0, 255)
        
        # Create PIL Image
        rgb_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return Image.fromarray(rgb_array)
