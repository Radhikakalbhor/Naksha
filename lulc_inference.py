import numpy as np
import rasterio
import torch
import cv2
from pathlib import Path
from transformers import SegformerForSemanticSegmentation

INPUT = "/data/raw/demo_aoi/demo_aoi_cog.tif"
OUTPUT = "/app/gis_engine/postgis/lulc_v1.tif"
MODEL = "florian-morel22/segformer-b0-deepglobe-land-cover"

CLASS_NAMES = {
    0: "urban_land",
    1: "agriculture_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
    6: "unknown",
}

TILE_SIZE = 224

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

print("LOADING LULC MODEL...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL
)

model.eval()
model.to("cpu")

print("MODEL READY")
print(
    "PARAMETERS:",
    sum(p.numel() for p in model.parameters())
)
print(
    "CLASSES:",
    model.config.id2label
)

with rasterio.open(INPUT) as src:

    rgb = src.read([1, 2, 3])

    height = src.height
    width = src.width

    profile = src.profile.copy()

    prediction_map = np.full(
        (height, width),
        6,
        dtype=np.uint8
    )

    class_counts = np.zeros(
        7,
        dtype=np.int64
    )

    total_tiles = 0

    for y in range(0, height, TILE_SIZE):

        for x in range(0, width, TILE_SIZE):

            y2 = min(
                y + TILE_SIZE,
                height
            )

            x2 = min(
                x + TILE_SIZE,
                width
            )

            tile_h = y2 - y
            tile_w = x2 - x

            tile = np.zeros(
                (
                    TILE_SIZE,
                    TILE_SIZE,
                    3
                ),
                dtype=np.uint8
            )

            tile[
                :tile_h,
                :tile_w,
                0
            ] = rgb[
                0,
                y:y2,
                x:x2
            ]

            tile[
                :tile_h,
                :tile_w,
                1
            ] = rgb[
                1,
                y:y2,
                x:x2
            ]

            tile[
                :tile_h,
                :tile_w,
                2
            ] = rgb[
                2,
                y:y2,
                x:x2
            ]

            # RGB -> float [0,1]
            image = (
                tile.astype(np.float32)
                / 255.0
            )

            # ImageNet normalization
            image = (
                image - MEAN
            ) / STD

            # HWC -> CHW
            image = np.transpose(
                image,
                (2, 0, 1)
            )

            pixel_values = torch.from_numpy(
                image
            ).unsqueeze(0).float()

            with torch.no_grad():

                outputs = model(
                    pixel_values=pixel_values
                )

            logits = outputs.logits

            logits = torch.nn.functional.interpolate(
                logits,
                size=(
                    TILE_SIZE,
                    TILE_SIZE
                ),
                mode="bilinear",
                align_corners=False
            )

            pred = torch.argmax(
                logits,
                dim=1
            )[0].cpu().numpy().astype(
                np.uint8
            )

            pred = pred[
                :tile_h,
                :tile_w
            ]

            prediction_map[
                y:y2,
                x:x2
            ] = pred

            unique, counts = np.unique(
                pred,
                return_counts=True
            )

            for cls, count in zip(
                unique,
                counts
            ):
                class_counts[cls] += count

            total_tiles += 1

            if total_tiles % 100 == 0:

                print(
                    "PROCESSED TILES:",
                    total_tiles
                )

profile.update(
    count=1,
    dtype="uint8",
    nodata=6,
    compress="lzw"
)

Path(
    OUTPUT
).parent.mkdir(
    parents=True,
    exist_ok=True
)

with rasterio.open(
    OUTPUT,
    "w",
    **profile
) as dst:

    dst.write(
        prediction_map,
        1
    )

print()
print(
    "TOTAL TILES:",
    total_tiles
)

print(
    "TOTAL PIXELS:",
    int(class_counts.sum())
)

for cls in range(7):

    print(
        f"{cls} ({CLASS_NAMES[cls]}):",
        int(class_counts[cls])
    )

print()
print(
    "OUTPUT:",
    OUTPUT
)

print(
    "CRS:",
    profile.get("crs")
)
