import math
from typing import Tuple

def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert latitude/longitude to Web Mercator tile coordinates."""
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def tile_to_latlon(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert Web Mercator tile coordinates to latitude/longitude (NW corner)."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def calculate_bbox(lat: float, lon: float, size_degrees: float) -> Tuple[float, float, float, float]:
    """Calculate bounding box (min_lon, min_lat, max_lon, max_lat)."""
    half_size = size_degrees / 2
    return (
        lon - half_size,  # min_lon
        lat - half_size,  # min_lat
        lon + half_size,  # max_lon
        lat + half_size   # max_lat
    )
