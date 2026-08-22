# ============================================================
# NAKSHA - CRS NORMALIZATION UTILITIES (Day 14)
# ============================================================

import rasterio
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry


def get_raster_crs(raster_path: str) -> str:
    """
    Get CRS string of a raster file.
    """
    try:
        with rasterio.open(raster_path) as src:
            return str(src.crs) if src.crs else "EPSG:4326"
    except Exception:
        return "EPSG:4326"


def transform_geometry_to_epsg4326(geometry: BaseGeometry, src_crs: str) -> BaseGeometry:
    """
    Reproject a Shapely geometry from src_crs to EPSG:4326 (WGS84).
    """
    if not src_crs or src_crs.upper() in ("EPSG:4326", "WGS84", "NONE"):
        return geometry

    try:
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        reprojected = transform(transformer.transform, geometry)
        return reprojected
    except Exception:
        return geometry
