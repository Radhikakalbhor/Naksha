import rasterio
import numpy as np
import cv2

INPUT = "/app/gis_engine/postgis/lulc_v1.tif"

CLASSES = {
    0: "urban_land",
    1: "agriculture_land",
    3: "forest_land",
    4: "water",
    5: "barren_land",
}

with rasterio.open(INPUT) as src:
    lulc = src.read(1)

for class_id, name in CLASSES.items():

    mask = (lulc == class_id).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    areas = stats[1:, cv2.CC_STAT_AREA]

    print()
    print("CLASS:", class_id, name)
    print("PIXELS:", int(mask.sum()))
    print("COMPONENTS:", len(areas))

    if len(areas) > 0:
        print("LARGEST COMPONENT:", int(areas.max()))
        print("TOP 10 COMPONENTS:", sorted(areas.tolist(), reverse=True)[:10])
        print("COMPONENTS >= 10 PIXELS:", int(np.sum(areas >= 10)))
        print("COMPONENTS >= 100 PIXELS:", int(np.sum(areas >= 100)))
        print("COMPONENTS >= 1000 PIXELS:", int(np.sum(areas >= 1000)))
