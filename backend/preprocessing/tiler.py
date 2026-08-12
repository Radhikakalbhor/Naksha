import os
import json
import rasterio
from rasterio.windows import Window


def tile_raster(
    input_path,
    output_dir,
    tile_size=512,
    overlap=64
):
    os.makedirs(output_dir, exist_ok=True)

    tile_index = []

    with rasterio.open(input_path) as src:

        width = src.width
        height = src.height

        step = tile_size - overlap

        tile_id = 0

        for y in range(0, height, step):
            for x in range(0, width, step):

                window_width = min(tile_size, width - x)
                window_height = min(tile_size, height - y)

                window = Window(
                    x,
                    y,
                    window_width,
                    window_height
                )

                data = src.read(window=window)
                transform = src.window_transform(window)

                filename = f"tile_{tile_id:04d}.tif"
                output_path = os.path.join(
                    output_dir,
                    filename
                )

                profile = src.profile.copy()

                profile.update({
                    "driver": "GTiff",
                    "height": window_height,
                    "width": window_width,
                    "transform": transform
                })

                with rasterio.open(
                    output_path,
                    "w",
                    **profile
                ) as dst:
                    dst.write(data)

                bounds = rasterio.windows.bounds(
                    window,
                    src.transform
                )

                tile_index.append({
                    "tile_id": tile_id,
                    "filename": filename,
                    "x": x,
                    "y": y,
                    "width": window_width,
                    "height": window_height,
                    "bounds": {
                        "left": bounds[0],
                        "bottom": bounds[1],
                        "right": bounds[2],
                        "top": bounds[3]
                    },
                    "crs": str(src.crs)
                })

                tile_id += 1

    index_path = os.path.join(
        output_dir,
        "tile_index.json"
    )

    with open(index_path, "w") as f:
        json.dump(tile_index, f, indent=2)

    print(f"Created {tile_id} tiles.")
    print(f"Tile index saved to: {index_path}")


if __name__ == "__main__":

    INPUT = "/data/raw/demo_aoi/demo_aoi_cog.tif"

    OUTPUT = "/data/processed/tiles"

    tile_raster(
        INPUT,
        OUTPUT,
        tile_size=512,
        overlap=64
    )