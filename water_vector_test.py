import sys
import rasterio
import numpy as np

sys.path.insert(0, "/app")

from gis_engine.vectorization.vectorize import (
    polygonize_mask,
)
from gis_engine.topology.geometry import (
    validate_geometry,
    simplify_geometry,
)

INPUT = "/app/gis_engine/postgis/water_v1.tif"

# Small geographic-area filter.
# We are measuring first; nothing is written to PostGIS.
MIN_AREA = 1e-8

with rasterio.open(INPUT) as src:

    mask = src.read(1)
    transform = src.transform

    print("RASTER:", INPUT)
    print("SIZE:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("WATER PIXELS:", int(np.count_nonzero(mask)))

    print("POLYGONIZING...")

    geometries = polygonize_mask(
        mask,
        transform,
        min_area=MIN_AREA,
    )

    print("RAW POLYGONS:", len(geometries))

    valid = []

    for geometry in geometries:

        geometry = validate_geometry(
            geometry
        )

        if geometry is None:
            continue

        geometry = simplify_geometry(
            geometry,
            tolerance=0.0,
        )

        if geometry is not None:
            valid.append(geometry)

    print("VALID POLYGONS:", len(valid))

    if valid:

        total_area = sum(
            geometry.area
            for geometry in valid
        )

        largest = sorted(
            [geometry.area for geometry in valid],
            reverse=True
        )[:10]

        print("TOTAL AREA (DEG2):", total_area)
        print("TOP 10 AREAS (DEG2):", largest)

    else:

        print("NO VALID WATER POLYGONS FOUND")
