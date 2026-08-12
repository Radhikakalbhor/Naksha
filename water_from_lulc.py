import rasterio
import numpy as np

INPUT = "/app/gis_engine/postgis/lulc_v1.tif"
OUTPUT = "/app/gis_engine/postgis/water_v1.tif"

WATER_CLASS = 4

with rasterio.open(INPUT) as src:

    lulc = src.read(1)

    water = (
        lulc == WATER_CLASS
    ).astype(np.uint8)

    profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        OUTPUT,
        "w",
        **profile
    ) as dst:

        dst.write(
            water,
            1
        )

    print("INPUT:", INPUT)
    print("OUTPUT:", OUTPUT)
    print("WATER CLASS:", WATER_CLASS)
    print("WATER PIXELS:", int(np.count_nonzero(water)))
    print("TOTAL PIXELS:", int(water.size))
    print(
        "WATER RATIO:",
        float(np.count_nonzero(water) / water.size)
    )
    print("CRS:", src.crs)
    print("SIZE:", src.width, "x", src.height)
