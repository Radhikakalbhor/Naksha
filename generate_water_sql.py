import sys
import rasterio

sys.path.insert(0, "/app")

from gis_engine.vectorization.vectorize import polygonize_mask
from gis_engine.topology.geometry import validate_geometry, simplify_geometry
from gis_engine.postgis.export import create_feature_sql, write_sql_file

INPUT = "/app/gis_engine/postgis/water_v1.tif"
OUTPUT = "/app/gis_engine/postgis/water_v1.sql"

LAYER_VERSION_ID = 5
FEATURE_TYPE = "water"

with rasterio.open(INPUT) as src:

    mask = src.read(1)

    geometries = polygonize_mask(
        mask,
        src.transform,
        min_area=1e-8,
    )

valid = []

for geometry in geometries:

    geometry = validate_geometry(geometry)

    if geometry is None:
        continue

    geometry = simplify_geometry(
        geometry,
        tolerance=0.0,
    )

    if geometry is not None:
        valid.append(geometry)

statements = [
    create_feature_sql(
        layer_version_id=LAYER_VERSION_ID,
        feature_type=FEATURE_TYPE,
        geometry=geometry,
        confidence=None,
    )
    for geometry in valid
]

path = write_sql_file(
    statements,
    OUTPUT,
)

print("WATER FEATURES:", len(valid))
print("SQL STATEMENTS:", len(statements))
print("SQL FILE:", path)
