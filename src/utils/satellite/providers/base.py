from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from PIL import Image

class SatelliteProvider(ABC):
    """Abstract base class for satellite imagery providers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def fetch_image(
        self, 
        bbox: Tuple[float, float, float, float], 
        width: int, 
        height: int,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch satellite imagery for the given bounding box.
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            width: Target image width
            height: Target image height
            date: Optional date string (YYYY-MM-DD)
            
        Returns:
            Dictionary containing:
            - success: bool
            - image: PIL.Image (if success)
            - metadata: Dict (provider info, date, resolution, etc.)
            - error: str (if failed)
        """
        pass
