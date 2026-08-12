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
    3: "forest_land",
    4: "water",
    5: "barren_land",
}

MIN_PIXELS = 100

with rasterio.open(INPUT) as src:

    lulc = src.read(1)
    transform = src.transform

    pixel_area = abs(
        transform.a * transform.e
    )

    min_area = MIN_PIXELS * pixel_area

    print("PIXEL AREA (DEG2):", pixel_area)
    print("MIN AREA (DEG2):", min_area)

    for class_id, class_name in CLASSES.items():

        mask = (
            lulc == class_id
        ).astype(np.uint8)

        pixel_count = int(mask.sum())

        if pixel_count == 0:
            continue

        geometries = polygonize_mask(
            mask,
            transform,
            min_area=min_area,
        )

        valid = []

        for geometry in geometries:

            geometry = validate_geometry(
                geometry
            )

            if geometry is not None:
                valid.append(geometry)

        total_area = sum(
            g.area for g in valid
        )

        print()
        print("CLASS:", class_id, class_name)
        print("PIXELS:", pixel_count)
        print("POLYGONS:", len(geometries))
        print("VALID:", len(valid))
        print("TOTAL AREA (DEG2):", total_area)
