# ============================================================
# NAKSHA - COG PREPROCESSING UTILITIES
# ============================================================

from pathlib import Path
import rasterio
from rasterio.enums import Resampling


def is_cog(raster_path: str | Path) -> bool:
    """
    Check if a GeoTIFF file is already formatted as a Cloud Optimized GeoTIFF (COG).
    """
    try:
        with rasterio.open(str(raster_path)) as src:
            if not src.is_tiled:
                return False
            has_overviews = any(len(src.overviews(b)) > 0 for b in range(1, src.count + 1))
            return has_overviews
    except Exception:
        return False


def convert_to_cog(input_path: str | Path, output_path: str | Path) -> Path:
    """
    Convert a GeoTIFF raster to Cloud Optimized GeoTIFF (COG) format.
    Preserves CRS, transform, dimensions, and bands while building overviews and tiled layout.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_cog(input_path):
        if input_path.resolve() != output_path.resolve():
            import shutil
            shutil.copy(input_path, output_path)
        return output_path

    try:
        with rasterio.open(str(input_path)) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
                interleave="pixel"
            )

            with rasterio.open(str(output_path), "w", **profile) as dst:
                for b in range(1, src.count + 1):
                    dst.write(src.read(b), b)

                factors = [2, 4, 8, 16]
                dst.build_overviews(factors, Resampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")
        return output_path
    except Exception as exc:
        raise RuntimeError(f"COG conversion failed for {input_path}: {exc}") from exc
