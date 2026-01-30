import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from PIL import Image

# Import providers
from .providers.nasa import NASAProvider
from .providers.osm import OSMProvider
from .providers.synthetic import SyntheticProvider
from .utils import calculate_bbox

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

class SatelliteImageryOrchestrator:
    """
    Professional satellite imagery orchestrator that intelligently selects
    from multiple free data sources based on location, availability, and quality.
    """
    
    def __init__(self):
        print("Initializing Zero-Key Satellite Orchestrator...")
        
        # Initialize providers
        self.providers = [
            NASAProvider(),      # Priority 1: Real Satellite Data (Free)
            OSMProvider(),       # Priority 2: High Quality Maps (Free)
            SyntheticProvider()  # Priority 3: Fallback (Generated)
        ]
        
        self.cache_dir = "data/satellite_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"✓ Satellite Orchestrator Online ({len(self.providers)} providers active)")

    def get_satellite_image(
        self,
        latitude: float,
        longitude: float,
        bbox_size: float = 0.01,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get satellite imagery from the best available source.
        """
        bbox = calculate_bbox(latitude, longitude, bbox_size)
        
        # Standard image size for analysis
        width, height = 640, 640
        
        print(f"Requesting satellite data for {latitude}, {longitude}...")
        
        # Try providers in order
        for provider in self.providers:
            print(f"  Attempts: {provider.name}...")
            result = provider.fetch_image(bbox, width, height, date)
            
            if result.get("success"):
                print(f"  ✓ Success: {provider.name}")
                
                # Add common metadata
                result["provider"] = provider.name
                result["metadata"]["bbox"] = bbox
                result["coordinates"] = {
                    "center": [latitude, longitude],
                    "bbox": bbox
                }
                return result
            else:
                print(f"  ✗ Failed: {result.get('error')}")
        
        return {
            "success": False, 
            "error": "All providers failed"
        }

    def create_map_visualization(
        self,
        latitude: float,
        longitude: float,
        detections: List[Dict] = None,
        bbox_size: float = 0.01
    ) -> str:
        """Create interactive map using Folium."""
        if not HAS_FOLIUM:
            return "<div>Interactive map unavailable (folium missing)</div>"
            
        try:
            m = folium.Map(location=[latitude, longitude], zoom_start=15)
            
            # Center marker
            folium.Marker(
                [latitude, longitude],
                popup="Analysis Center",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
            # Bounding box
            bbox = calculate_bbox(latitude, longitude, bbox_size)
            folium.Rectangle(
                bounds=[(bbox[1], bbox[0]), (bbox[3], bbox[2])],
                color='#ff7800',
                fill=True,
                fill_color='#ffff00',
                fill_opacity=0.2,
                popup="Analysis Area"
            ).add_to(m)
            
            # Detections
            if detections:
                for det in detections[:10]:
                    # Random jitter within bbox for demo visualization
                    # since we only have global lat/lon for the image center
                    det_lat = latitude + (np.random.rand() - 0.5) * bbox_size * 0.8
                    det_lon = longitude + (np.random.rand() - 0.5) * bbox_size * 0.8
                    
                    folium.CircleMarker(
                        location=[det_lat, det_lon],
                        radius=6,
                        popup=f"{det.get('label', 'Object')}",
                        color="blue",
                        fill=True
                    ).add_to(m)
            
            # Save
            filename = f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = os.path.join(self.cache_dir, filename)
            m.save(filepath)
            
            return m._repr_html_()
            
        except Exception as e:
            return f"<div>Map generation failed: {str(e)}</div>"
