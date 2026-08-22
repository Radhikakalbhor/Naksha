# ============================================================
# NAKSHA - CRS NORMALIZATION UTILITIES (Task 1)
# ============================================================

from pathlib import Path
from typing import Optional, Dict, Any
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry


def get_raster_crs(raster_path: str | Path) -> Optional[str]:
    """
    Detect the source CRS from raster metadata.
    Returns CRS string representation if present, or None if CRS is missing.
    """
    try:
        with rasterio.open(str(raster_path)) as src:
            if src.crs is None:
                return None
            return str(src.crs)
    except Exception:
        return None


def inspect_raster_crs(raster_path: str | Path) -> Dict[str, Any]:
    """
    Inspect raster CRS metadata.
    Returns dictionary with CRS details. Raises ValueError if CRS is missing.
    """
    crs_str = get_raster_crs(raster_path)
    if crs_str is None:
        raise ValueError(f"Raster '{raster_path}' has no CRS defined. Georeferencing is missing.")

    with rasterio.open(str(raster_path)) as src:
        epsg = src.crs.to_epsg()
        return {
            "crs": str(src.crs),
            "is_valid": True,
            "epsg": epsg,
            "units": src.crs.linear_units if hasattr(src.crs, "linear_units") else "degrees",
        }


def normalize_raster_crs(
    input_path: str | Path,
    output_path: str | Path,
    target_crs: str = "EPSG:4326",
) -> Path:
    """
    Normalize raster CRS to target_crs (default EPSG:4326).
    Preserves georeferencing, bounds, and resolution while reprojecting pixel data.
    Raises ValueError if input raster has no CRS defined.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(input_path)) as src:
        if src.crs is None:
            raise ValueError(f"Raster '{input_path}' has no CRS defined. Georeferencing is missing.")

        dst_crs = CRS.from_string(target_crs)
        if src.crs == dst_crs:
            if input_path.resolve() != output_path.resolve():
                import shutil
                shutil.copy(input_path, output_path)
            return output_path

        transform_dst, width_dst, height_dst = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform_dst,
            width=width_dst,
            height=height_dst,
        )

        with rasterio.open(str(output_path), "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform_dst,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

    return output_path


def transform_geometry_to_epsg4326(geometry: BaseGeometry, src_crs: str) -> BaseGeometry:
    """
    Reproject a Shapely geometry from src_crs to EPSG:4326 (WGS84).
    """
    if not src_crs or str(src_crs).upper() in ("EPSG:4326", "WGS84", "NONE"):
        return geometry

    try:
        geom_dict = mapping(geometry)
        transformed_dict = transform_geom(src_crs, "EPSG:4326", geom_dict)
        return shape(transformed_dict)
    except Exception:
        return geometry
