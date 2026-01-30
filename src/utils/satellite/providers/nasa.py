from .base import SatelliteProvider
from ..utils import latlon_to_tile
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

class NASAProvider(SatelliteProvider):
    """Provides satellite imagery using NASA GIBS WMTS API."""
    
    def __init__(self):
        super().__init__("NASA GIBS")
        # MODIS Terra SurfaceReflectance (TrueColor)
        self.base_url = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"

    def fetch_image(
        self, 
        bbox: Tuple[float, float, float, float], 
        width: int, 
        height: int,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        
        if not date:
            # Default to yesterday to assume availability
            date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
            
        try:
            # Calculate center tile
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            zoom = 9  # NASA GIBS resolution limit for this layer usually stops around 9-10
            
            xtile, ytile = latlon_to_tile(center_lat, center_lon, zoom)
            
            url = self.base_url.format(date=date, z=zoom, y=ytile, x=xtile)
            print(f"DEBUG: Fetching NASA tile: {url}")
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = image.resize((width, height), Image.LANCZOS)
                
                return {
                    "success": True,
                    "image": image,
                    "metadata": {
                        "provider": self.name,
                        "date": date,
                        "source": "NASA MODIS Terra",
                        "zoom": zoom,
                        "url": url
                    }
                }
            else:
                 return {"success": False, "error": f"HTTP {response.status_code}"}
                 
        except Exception as e:
            return {"success": False, "error": str(e)}
