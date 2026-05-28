from .base import SatelliteProvider
from ..utils import latlon_to_tile
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Tuple, Optional

class OSMProvider(SatelliteProvider):
    """Provides static maps using OpenStreetMap tiles."""
    
    def __init__(self):
        super().__init__("OpenStreetMap")
        self.base_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

    def fetch_image(
        self, 
        bbox: Tuple[float, float, float, float], 
        width: int, 
        height: int,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        
        try:
            # Calculate center tile
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            zoom = 16  # High detail for OSM
            
            xtile, ytile = latlon_to_tile(center_lat, center_lon, zoom)
            
            url = self.base_url.format(z=zoom, x=xtile, y=ytile)
            # Add User-Agent as required by OSM usage policy
            headers = {"User-Agent": "EnviroAudit/2.0"}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = image.resize((width, height), Image.LANCZOS)
                
                return {
                    "success": True,
                    "image": image,
                    "metadata": {
                        "provider": self.name,
                        "date": date or "Static",
                        "source": "OpenStreetMap",
                        "zoom": zoom
                    }
                }
            else:
                 return {"success": False, "error": f"HTTP {response.status_code}"}
                 
        except Exception as e:
            return {"success": False, "error": str(e)}
