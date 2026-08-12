import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import shapes

sys.path.insert(0, "/app")

from main import (
    get_building_model,
    BUILDING_PATCH_SIZE,
    BUILDING_STRIDE,
    BUILDING_THRESHOLD,
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
    create_feature_sql,
    write_sql_file,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "/data/processed/demo_aoi_normalized.tif"

OUTPUT = "/app/gis_engine/postgis/buildings_v2.sql"

PROBABILITY_OUTPUT = (
    "/app/gis_engine/postgis/buildings_v2_probability.npy"
)

LAYER_VERSION_ID = 7
FEATURE_TYPE = "buildings"

PATCH_SIZE = BUILDING_PATCH_SIZE
STRIDE = BUILDING_STRIDE

# Process multiple 512x512 patches together.
# If RAM becomes a problem, reduce this to 4.
BATCH_SIZE = 8

MIN_AREA = 0.0


# ============================================================
# BATCHED BUILDING INFERENCE
# ============================================================

def run_batched_inference(model, image_path):

    print("Opening source GeoTIFF...")

    with rasterio.open(image_path) as src:

        if src.count < 3:
            raise RuntimeError(
                "Input GeoTIFF must contain at least 3 bands."
            )

        image = src.read(
            [1, 2, 3]
        ).transpose(1, 2, 0)

        transform = src.transform
        width = src.width
        height = src.height

    print(
        f"Source size: {width} x {height}"
    )

    # --------------------------------------------------------
    # Same normalization as existing Building API
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    )

    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:

        image = (
            image - image_min
        ) / (
            image_max - image_min
        )

    else:

        image = np.zeros_like(
            image
        )

    # --------------------------------------------------------
    # Prediction arrays
    # --------------------------------------------------------

    full_prediction = np.zeros(
        (height, width),
        dtype=np.float32
    )

    count_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Patch positions
    # --------------------------------------------------------

    x_positions = list(
        range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        )
    )

    y_positions = list(
        range(
            0,
            height - PATCH_SIZE + 1,
            STRIDE
        )
    )

    # Right edge
    if (
        not x_positions
        or x_positions[-1]
        != width - PATCH_SIZE
    ):

        x_positions.append(
            width - PATCH_SIZE
        )

    # Bottom edge
    if (
        not y_positions
        or y_positions[-1]
        != height - PATCH_SIZE
    ):

        y_positions.append(
            height - PATCH_SIZE
        )

    positions = [
        (x, y)
        for y in y_positions
        for x in x_positions
    ]

    total_patches = len(
        positions
    )

    print(
        f"Total patches: {total_patches}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        "Starting batched Building inference..."
    )

    model.eval()

    # --------------------------------------------------------
    # Batched inference
    # --------------------------------------------------------

    with torch.inference_mode():

        for batch_start in range(
            0,
            total_patches,
            BATCH_SIZE
        ):

            batch_positions = positions[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            batch = np.stack(
                [
                    image[
                        y:y + PATCH_SIZE,
                        x:x + PATCH_SIZE
                    ]
                    for x, y in batch_positions
                ],
                axis=0
            )

            # N,H,W,C -> N,C,H,W
            batch = np.transpose(
                batch,
                (0, 3, 1, 2)
            )

            tensor = torch.from_numpy(
                batch
            ).float().to(
                "cpu"
            )

            prediction = model(
                tensor
            )

            prediction = torch.sigmoid(
                prediction
            )

            prediction = (
                prediction
                .squeeze(1)
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Merge predictions
            # ------------------------------------------------

            for index, (
                x,
                y
            ) in enumerate(
                batch_positions
            ):

                pred = prediction[
                    index
                ]

                full_prediction[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ] += pred

                count_map[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ] += 1

            processed = min(
                batch_start + BATCH_SIZE,
                total_patches
            )

            print(
                f"Processed "
                f"{processed}/{total_patches} "
                f"patches"
            )

    # --------------------------------------------------------
    # Average overlapping predictions
    # --------------------------------------------------------

    count_map[
        count_map == 0
    ] = 1

    full_prediction /= count_map

    # --------------------------------------------------------
    # Save probability map
    # --------------------------------------------------------

    np.save(
        PROBABILITY_OUTPUT,
        full_prediction
    )

    print(
        "Probability map saved:"
    )

    print(
        PROBABILITY_OUTPUT
    )

    return (
        full_prediction,
        transform
    )


# ============================================================
# MAIN
# ============================================================

print()
print("========================================")
print("NAKSHA BUILDINGS V2 EXPORT")
print("========================================")

print(
    "Input:",
    INPUT
)

print(
    "Layer version ID:",
    LAYER_VERSION_ID
)

print(
    "Patch size:",
    PATCH_SIZE
)

print(
    "Stride:",
    STRIDE
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print()
print("Loading Building U-Net...")

model = get_building_model()

print(
    "Building model loaded."
)

# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

probability, transform = (
    run_batched_inference(
        model,
        INPUT
    )
)

print()
print(
    "Building inference finished."
)

# ------------------------------------------------------------
# Create binary mask
# ------------------------------------------------------------

print(
    "Creating building mask..."
)

mask = (
    probability >
    BUILDING_THRESHOLD
).astype(
    np.uint8
)

print(
    "Building pixels:",
    int(mask.sum())
)

# ------------------------------------------------------------
# Polygonization
# ------------------------------------------------------------

print()
print(
    "Polygonizing building mask..."
)

geometries = polygonize_mask(
    mask,
    transform,
    min_area=MIN_AREA,
)

print(
    "Raw building geometries:",
    len(geometries)
)

# ------------------------------------------------------------
# Generate feature SQL
# ------------------------------------------------------------

statements = []

feature_count = 0
confidence_values = []

print()
print(
    "Calculating feature confidence..."
)

for index, geometry in enumerate(
    geometries,
    start=1
):

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
            layer_version_id=
                LAYER_VERSION_ID,

            feature_type=
                FEATURE_TYPE,

            geometry=
                geometry,

            confidence=
                confidence,
        )
    )

    confidence_values.append(
        confidence
    )

    feature_count += 1

    if (
        index % 100 == 0
    ):

        print(
            f"Processed "
            f"{index}/{len(geometries)} "
            f"geometries"
        )

# ------------------------------------------------------------
# IMPORTANT:
# We DO NOT create layer_versions here.
#
# ID 7 already exists in PostGIS.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Write SQL
# ------------------------------------------------------------

print()
print(
    "Writing SQL file..."
)

path = write_sql_file(
    statements,
    OUTPUT,
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

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
        /
        len(confidence_values)
    )

    print(
        "Confidence max:",
        max(confidence_values)
    )

print(
    "SQL file:",
    path
)

print(
    "Probability file:",
    PROBABILITY_OUTPUT
)