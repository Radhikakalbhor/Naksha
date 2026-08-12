import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling


def normalize_crs(input_path, output_path, target_crs="EPSG:4326"):
    with rasterio.open(input_path) as src:

        if str(src.crs) == target_crs:
            print(f"CRS already normalized: {target_crs}")

            profile = src.profile.copy()

            with rasterio.open(output_path, "w", **profile) as dst:
                for band in range(1, src.count + 1):
                    dst.write(src.read(band), band)

            return

        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds
        )

        profile = src.profile.copy()

        profile.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(output_path, "w", **profile) as dst:

            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )

    print(f"CRS normalized to {target_crs}")