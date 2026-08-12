import sys
import rasterio

sys.path.insert(0, "/app")

from gis_engine.vectorization.vectorize import polygonize_mask
from gis_engine.topology.geometry import validate_geometry, simplify_geometry
from gis_engine.postgis.export import create_feature_sql, write_sql_file

INPUT = "/app/gis_engine/postgis/lulc_v1.tif"
OUTPUT = "/app/gis_engine/postgis/lulc_v1.sql"

LAYER_VERSION_ID = 6

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

    statements = []
    feature_counts = {}

    for class_id, class_name in CLASSES.items():

        mask = (
            lulc == class_id
        ).astype("uint8")

        if not mask.any():
            feature_counts[class_name] = 0
            continue

        geometries = polygonize_mask(
            mask,
            transform,
            min_area=min_area,
        )

        count = 0

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

            if geometry is None:
                continue

            statements.append(
                create_feature_sql(
                    layer_version_id=LAYER_VERSION_ID,
                    feature_type=class_name,
                    geometry=geometry,
                    confidence=None,
                )
            )

            count += 1

        feature_counts[class_name] = count

path = write_sql_file(
    statements,
    OUTPUT,
)

print("LULC FEATURE COUNTS:")

for class_name, count in feature_counts.items():
    print(class_name, ":", count)

print("TOTAL FEATURES:", len(statements))
print("SQL STATEMENTS:", len(statements))
print("SQL FILE:", path)
