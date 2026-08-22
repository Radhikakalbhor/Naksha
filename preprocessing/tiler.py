# ============================================================
# NAKSHA - RASTER TILING UTILITIES (Task 2)
# ============================================================

from pathlib import Path
from typing import List
import rasterio
from rasterio.windows import Window


def tile_raster(
    input_path: str | Path,
    output_dir: str | Path,
    tile_size: int = 1024,
    overlap: int = 0,
) -> List[Path]:
    """
    Split a raster into smaller georeferenced tiles using windowed reading/writing.
    Preserves CRS, affine transform, and metadata for every tile including edge tiles.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = []
    with rasterio.open(input_path) as src:
        height = src.height
        width = src.width

        # If raster is smaller than or equal to tile size, don't tile unnecessarily
        if width <= tile_size and height <= tile_size:
            out_file = output_dir / f"tile_0_0_{input_path.name}"
            if input_path.resolve() != out_file.resolve():
                import shutil
                shutil.copy(input_path, out_file)
            return [out_file]

        step = tile_size - overlap if overlap < tile_size else tile_size

        for y in range(0, height, step):
            for x in range(0, width, step):
                w_width = min(tile_size, width - x)
                w_height = min(tile_size, height - y)

                window = Window(col_off=x, row_off=y, width=w_width, height=w_height)
                transform = src.window_transform(window)

                profile = src.profile.copy()
                profile.update(
                    width=w_width,
                    height=w_height,
                    transform=transform,
                )

                tile_filename = f"tile_{y}_{x}_{input_path.stem}.tif"
                tile_path = output_dir / tile_filename

                with rasterio.open(tile_path, "w", **profile) as dst:
                    for b in range(1, src.count + 1):
                        dst.write(src.read(b, window=window), b)

                tile_paths.append(tile_path)

    return tile_paths
