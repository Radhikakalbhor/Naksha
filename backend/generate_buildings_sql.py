import sys
from pathlib import Path

sys.path.insert(0, "/app")

from main import (
    get_building_model,
    building_sliding_window_inference,
    BUILDING_PATCH_SIZE,
    BUILDING_STRIDE,
)

from gis_engine.vectorization.vectorize import (
    polygonize_mask,
    feature_confidence,
)

from gis_engine.topology.geometry import (
    validate_geometry,
    simplify_geometry,
)

from gis_engine.postgis.export import (
    create_layer_version_sql,
    create_feature_sql,
    write_sql_file,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "/data/processed/demo_aoi_normalized.tif"

OUTPUT = "/app/gis_engine/postgis/buildings_v2.sql"

LAYER_VERSION_ID = 7

LAYER_NAME = "demo_buildings"

FEATURE_TYPE = "buildings"

MIN_AREA = 0.0


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Building U-Net...")

model = get_building_model()

print("Building model loaded.")


# ============================================================
# RUN FULL-IMAGE INFERENCE
# ============================================================

print("Running full-image building inference...")

image, mask, probability = (
    building_sliding_window_inference(
        model=model,
        image_path=INPUT,
        patch_size=BUILDING_PATCH_SIZE,
        stride=BUILDING_STRIDE,
    )
)

print("Building inference finished.")


# ============================================================
# READ SOURCE TRANSFORM
# ============================================================

import rasterio

with rasterio.open(INPUT) as src:
    transform = src.transform


# ============================================================
# RASTER -> POLYGONS
# ============================================================

print("Polygonizing building mask...")

geometries = polygonize_mask(
    mask,
    transform,
    min_area=MIN_AREA,
)

print(
    "Raw building geometries:",
    len(geometries)
)


# ============================================================
# CREATE SQL
# ============================================================

statements = []

statements.append(
    create_layer_version_sql(
        layer_name=LAYER_NAME,
        feature_type=FEATURE_TYPE,
        version=2,
    ).replace(
        "RETURNING id;",
        "",
    )
    .strip()
    + ";"
)


feature_count = 0
confidence_values = []


# ============================================================
# PROCESS EACH BUILDING
# ============================================================

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

    confidence = feature_confidence(
        probability=probability,
        geometry=geometry,
        transform=transform,
    )

    if confidence is None:
        continue

    statements.append(
        create_feature_sql(
            layer_version_id=LAYER_VERSION_ID,
            feature_type=FEATURE_TYPE,
            geometry=geometry,
            confidence=confidence,
        )
    )

    confidence_values.append(
        confidence
    )

    feature_count += 1


# ============================================================
# WRITE SQL
# ============================================================

path = write_sql_file(
    statements,
    OUTPUT,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("BUILDINGS VECTOR EXPORT COMPLETE")
print("========================================")

print(
    "Features:",
    feature_count
)

if confidence_values:

    print(
        "Confidence min:",
        min(confidence_values)
    )

    print(
        "Confidence mean:",
        sum(confidence_values)
        / len(confidence_values)
    )

    print(
        "Confidence max:",
        max(confidence_values)
    )

print(
    "SQL file:",
    path
)