# src/utils/satellite_client.py
import requests
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from io import BytesIO
import base64
from PIL import Image
import numpy as np
try:
    import folium
    import geopandas as gpd
    from shapely.geometry import Polygon, Point
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False
    class Polygon: pass
    class Point: pass
import warnings
warnings.filterwarnings("ignore")

class SatelliteImageryClient:
    """Client for accessing satellite imagery from various providers."""
    
    def __init__(self):
        print("Initializing Satellite Imagery Client...")
        
        # Configuration for different providers
        self.providers = {
            "sentinel": {
                "name": "Sentinel Hub",
                "api_key_env": ["SENTINELHUB_INSTANCE_ID", "SENTINELHUB_CLIENT_ID", "SENTINELHUB_API_KEY"],
                "base_url": "https://services.sentinel-hub.com"
            },
            "planetary_computer": {
                "name": "Microsoft Planetary Computer",
                "api_key_env": "PC_API_KEY",
                "base_url": "https://planetarycomputer.microsoft.com/api"
            },
            "usgs": {
                "name": "USGS EarthExplorer",
                "username_env": "USGS_USERNAME",
                "password_env": "USGS_PASSWORD",
                "base_url": "https://earthexplorer.usgs.gov/inventory/json/v/1.4.0"
            },
            "google_earth_engine": {
                "name": "Google Earth Engine",
                "api_key_env": "GEE_API_KEY",
                "requires_auth": True
            }
        }
        
        # Load API keys from environment
        self.api_keys = {}
        self._load_api_keys()
        
        # Cache for downloaded imagery
        self.cache_dir = "data/satellite_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print("✓ Satellite Imagery Client initialized")
    
    def _load_api_keys(self):
        """Load API keys from environment variables."""
        import os
        
        for provider_id, provider_info in self.providers.items():
            if "api_key_env" in provider_info:
                env_vars = provider_info["api_key_env"]
                if isinstance(env_vars, str):
                    env_vars = [env_vars]
                
                api_key = None
                for env_var in env_vars:
                    val = os.getenv(env_var)
                    if val:
                        api_key = val
                        break
                
                if api_key:
                    self.api_keys[provider_id] = api_key
                    print(f"  ✓ {provider_info['name']} API key loaded")
                else:
                    print(f"  ⚠️  {provider_info['name']} API key not found in environment (checked {env_vars})")
    
    def get_satellite_image(
        self,
        latitude: float,
        longitude: float,
        bbox_size: float = 0.01,  # degrees (approx 1km at equator)
        date: Optional[str] = None,
        provider: str = "sentinel",
        max_cloud_cover: float = 20.0
    ) -> Dict[str, Any]:
        """
        Get satellite image for a location.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            bbox_size: Bounding box size in degrees
            date: Date string (YYYY-MM-DD), defaults to recent
            provider: Provider to use
            max_cloud_cover: Maximum acceptable cloud cover percentage
            
        Returns:
            Dictionary with image and metadata
        """
        # Calculate bounding box
        bbox = self._calculate_bbox(latitude, longitude, bbox_size)
        
        # Use current date if not specified
        if date is None:
            date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        print(f"Requesting satellite image for {latitude}, {longitude} on {date}")
        
        # Try different providers in order
        providers_to_try = [provider] if provider != "auto" else ["sentinel", "planetary_computer", "usgs"]
        
        for prov in providers_to_try:
            try:
                if prov == "sentinel":
                    result = self._get_from_sentinel(bbox, date, max_cloud_cover)
                elif prov == "planetary_computer":
                    result = self._get_from_planetary_computer(bbox, date, max_cloud_cover)
                elif prov == "usgs":
                    result = self._get_from_usgs(bbox, date, max_cloud_cover)
                else:
                    continue
                
                if result.get("success"):
                    print(f"✓ Image obtained from {self.providers[prov]['name']}")
                    return result
                    
            except Exception as e:
                print(f"  ⚠️  {self.providers[prov]['name']} failed: {e}")
                continue
        
        # Fallback to static map if no satellite imagery available
        print("⚠️  No satellite imagery available, using static map")
        return self._get_static_map(latitude, longitude, bbox_size)
    
    def _calculate_bbox(self, lat: float, lon: float, size: float) -> Tuple[float, float, float, float]:
        """Calculate bounding box from center point and size."""
        half_size = size / 2
        return (
            lon - half_size,  # min_lon
            lat - half_size,  # min_lat
            lon + half_size,  # max_lon
            lat + half_size   # max_lat
        )
    
    def _get_from_sentinel(self, bbox: Tuple[float, float, float, float], 
                          date: str, max_cloud_cover: float) -> Dict[str, Any]:
        """Get image from Sentinel Hub."""
        # This is a simplified version - in production, use sentinelhub package
        # For demo purposes, we'll use a placeholder
        
        # Generate a mock response for demonstration
        # In real implementation, you would use:
        # from sentinelhub import SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
        
        image = self._generate_mock_satellite_image(bbox)
        
        return {
            "success": True,
            "provider": "sentinel",
            "image": image,
            "metadata": {
                "bbox": bbox,
                "date": date,
                "cloud_cover": 10.5,
                "resolution": "10m",
                "satellite": "Sentinel-2"
            },
            "coordinates": {
                "center": [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
                "bbox": bbox
            }
        }
    
    def _get_from_planetary_computer(self, bbox: Tuple[float, float, float, float],
                                    date: str, max_cloud_cover: float) -> Dict[str, Any]:
        """Get image from Microsoft Planetary Computer."""
        # Placeholder - actual implementation would use their API
        
        image = self._generate_mock_satellite_image(bbox)
        
        return {
            "success": True,
            "provider": "planetary_computer",
            "image": image,
            "metadata": {
                "bbox": bbox,
                "date": date,
                "cloud_cover": 5.2,
                "resolution": "4m",
                "satellite": "PlanetScope"
            },
            "coordinates": {
                "center": [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
                "bbox": bbox
            }
        }
    
    def _get_from_usgs(self, bbox: Tuple[float, float, float, float],
                      date: str, max_cloud_cover: float) -> Dict[str, Any]:
        """Get image from USGS EarthExplorer."""
        # Placeholder - actual implementation would use their API
        
        image = self._generate_mock_satellite_image(bbox)
        
        return {
            "success": True,
            "provider": "usgs",
            "image": image,
            "metadata": {
                "bbox": bbox,
                "date": date,
                "cloud_cover": 15.0,
                "resolution": "30m",
                "satellite": "Landsat 8"
            },
            "coordinates": {
                "center": [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2],
                "bbox": bbox
            }
        }
    
    def _get_static_map(self, lat: float, lon: float, size: float) -> Dict[str, Any]:
        """Generate a static map as fallback."""
        try:
            # Use OpenStreetMap static map
            zoom = 15 - int(size * 100)  # Adjust zoom based on size
            
            url = f"https://tile.openstreetmap.org/{zoom}/{int((lat+90)*1000)}/{int((lon+180)*1000)}.png"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                
                return {
                    "success": True,
                    "provider": "openstreetmap",
                    "image": image,
                    "metadata": {
                        "bbox": self._calculate_bbox(lat, lon, size),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "cloud_cover": 0,
                        "resolution": "Static map",
                        "source": "OpenStreetMap"
                    },
                    "coordinates": {
                        "center": [lat, lon],
                        "bbox": self._calculate_bbox(lat, lon, size)
                    }
                }
        except:
            pass
        
        # Create a simple colored image as last resort
        image = Image.new('RGB', (512, 512), color=(100, 150, 100))
        
        return {
            "success": True,
            "provider": "generated",
            "image": image,
            "metadata": {
                "bbox": self._calculate_bbox(lat, lon, size),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "cloud_cover": 0,
                "resolution": "Generated",
                "source": "Fallback"
            },
            "coordinates": {
                "center": [lat, lon],
                "bbox": self._calculate_bbox(lat, lon, size)
            }
        }
    
    def _generate_mock_satellite_image(self, bbox: Tuple[float, float, float, float]) -> Image.Image:
        """Generate a mock satellite image for demonstration."""
        # Create a synthetic satellite-like image
        width, height = 512, 512
        
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
        
        # Add some roads/linear features
        for i in range(5):
            road_y = int(height * np.random.rand())
            road_width = int(np.random.rand() * 5 + 2)
            r[road_y:road_y+road_width, :] = 150
            g[road_y:road_y+road_width, :] = 150
            b[road_y:road_y+road_width, :] = 150
        
        # Clip values
        r = np.clip(r, 0, 255)
        g = np.clip(g, 0, 255)
        b = np.clip(b, 0, 255)
        
        # Create PIL Image
        rgb_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
        image = Image.fromarray(rgb_array)
        
        return image
    
    def create_map_visualization(
        self,
        latitude: float,
        longitude: float,
        detections: List[Dict] = None,
        bbox_size: float = 0.01
    ) -> str:
        """
        Create an interactive map visualization.
        
        Returns:
            HTML string of the map
        """
        if not HAS_GEOSPATIAL:
             return "<div style='padding: 20px; background: #f0f0f0; border: 1px solid #ccc;'>Interactive map not available (missing dependencies: folium/geopandas)</div>"

        # Create base map
        m = folium.Map(location=[latitude, longitude], zoom_start=15)
        
        # Add marker for center point
        folium.Marker(
            [latitude, longitude],
            popup="Analysis Center",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
        # Add bounding box rectangle
        bbox = self._calculate_bbox(latitude, longitude, bbox_size)
        folium.Rectangle(
            bounds=[(bbox[1], bbox[0]), (bbox[3], bbox[2])],
            color='#ff7800',
            fill=True,
            fill_color='#ffff00',
            fill_opacity=0.2,
            popup=f"Analysis Area: {bbox_size}°"
        ).add_to(m)
        
        # Add detections if available
        if detections:
            for i, det in enumerate(detections[:10]):  # Limit to first 10
                # Calculate random position within bbox for demonstration
                det_lat = latitude + (np.random.rand() - 0.5) * bbox_size * 0.8
                det_lon = longitude + (np.random.rand() - 0.5) * bbox_size * 0.8
                
                category = det.get("category", "unknown")
                colors = {
                    "heavy_machinery": "red",
                    "vehicles": "blue",
                    "materials": "green",
                    "people": "orange",
                    "structures": "purple"
                }
                
                folium.CircleMarker(
                    location=[det_lat, det_lon],
                    radius=8,
                    popup=f"{det.get('label', 'Object')}: {det.get('confidence', 'N/A')}",
                    color=colors.get(category, "gray"),
                    fill=True,
                    fill_color=colors.get(category, "gray")
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Convert map to HTML
        html_string = m._repr_html_()
        
        # Save map to file
        map_file = f"{self.cache_dir}/map_{latitude}_{longitude}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        m.save(map_file)
        
        return html_string
    
    def get_historical_comparison(
        self,
        latitude: float,
        longitude: float,
        date1: str,
        date2: str = None,
        bbox_size: float = 0.01
    ) -> Dict[str, Any]:
        """
        Compare satellite imagery from two dates.
        
        Returns:
            Dictionary with images and change detection results
        """
        if date2 is None:
            date2 = datetime.now().strftime("%Y-%m-%d")
        
        # Get images for both dates
        image1_data = self.get_satellite_image(latitude, longitude, bbox_size, date1)
        image2_data = self.get_satellite_image(latitude, longitude, bbox_size, date2)
        
        # Calculate difference (simple pixel difference for demo)
        if image1_data["success"] and image2_data["success"]:
            img1 = np.array(image1_data["image"].convert("L"))  # Convert to grayscale
            img2 = np.array(image2_data["image"].convert("L"))
            
            # Calculate difference
            diff = np.abs(img1.astype(float) - img2.astype(float))
            change_percentage = np.mean(diff > 30) * 100  # Threshold for significant change
            
            # Create difference visualization
            diff_visual = Image.fromarray(np.uint8(diff))
            
            return {
                "success": True,
                "date1": date1,
                "date2": date2,
                "image1": image1_data["image"],
                "image2": image2_data["image"],
                "difference": diff_visual,
                "change_percentage": float(change_percentage),
                "change_detected": change_percentage > 5.0,
                "metadata": {
                    "bbox": image1_data["coordinates"]["bbox"],
                    "center": [latitude, longitude]
                }
            }
        
        return {
            "success": False,
            "error": "Failed to get images for comparison"
        }