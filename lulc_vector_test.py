import sys
import rasterio
import numpy as np

sys.path.insert(0, "/app")

from gis_engine.vectorization.vectorize import polygonize_mask
from gis_engine.topology.geometry import validate_geometry

INPUT = "/app/gis_engine/postgis/lulc_v1.tif"

CLASSES = {
    0: "urban_land",
    1: "agriculture_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
    6: "unknown",
}

# Filter extremely small fragments.
MIN_AREA = 1e-8

with rasterio.open(INPUT) as src:

    lulc = src.read(1)
    transform = src.transform

    print("RASTER:", INPUT)
    print("SIZE:", src.width, "x", src.height)
    print("CRS:", src.crs)

    for class_id, class_name in CLASSES.items():

        pixel_count = int(
            np.sum(lulc == class_id)
        )

        if pixel_count == 0:
            print()
            print(
                class_id,
                class_name,
                "? 0 pixels, SKIPPING"
            )
            continue

        print()
        print(
            "PROCESSING:",
            class_id,
            class_name
        )

        mask = (
            lulc == class_id
        ).astype(np.uint8)

        geometries = polygonize_mask(
            mask,
            transform,
            min_area=MIN_AREA,
        )

        valid_count = 0
        total_area = 0.0

        for geometry in geometries:

            geometry = validate_geometry(
                geometry
            )

            if geometry is None:
                continue

            valid_count += 1
            total_area += geometry.area

        print(
            "PIXELS:",
            pixel_count
        )

        print(
            "POLYGONS:",
            len(geometries)
        )

        print(
            "VALID:",
            valid_count
        )

        print(
            "TOTAL AREA (DEG2):",
            total_area
        )
