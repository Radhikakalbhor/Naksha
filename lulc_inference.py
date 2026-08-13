from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from transformers import SegformerForSemanticSegmentation


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
    dtype=np.float32,
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


@lru_cache(maxsize=1)
def load_lulc_model():
    """
    Load the existing DeepGlobe SegFormer-B0 LULC model once
    and reuse it across inference requests.
    """

    print("LOADING LULC MODEL...")

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL
    )

    model.eval()
    model.to("cpu")

    print("MODEL READY")
    print(
        "PARAMETERS:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    print(
        "CLASSES:",
        model.config.id2label,
    )

    return model


def run_lulc_inference(
    input_tiff_path,
    output_tiff_path=None,
):
    """
    Run multi-class LULC inference on a GeoTIFF.

    Parameters
    ----------
    input_tiff_path : str or Path
        Path to the input GeoTIFF.

    output_tiff_path : str or Path, optional
        Path where the classified LULC raster should be written.

    Returns
    -------
    prediction_map : np.ndarray
        Multi-class uint8 classification raster.

    probability_map : np.ndarray
        Per-pixel confidence/probability corresponding to
        the predicted class.

    transform : rasterio.Affine
        Spatial transform of the input raster.
    """

    input_tiff_path = Path(
        input_tiff_path
    )

    if output_tiff_path is not None:

        output_tiff_path = Path(
            output_tiff_path
        )

    model = load_lulc_model()

    with rasterio.open(
        input_tiff_path
    ) as src:

        rgb = src.read(
            [1, 2, 3]
        )

        height = src.height
        width = src.width

        profile = src.profile.copy()

        transform = src.transform

        prediction_map = np.full(
            (height, width),
            6,
            dtype=np.uint8,
        )

        probability_map = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        class_counts = np.zeros(
            7,
            dtype=np.int64,
        )

        total_tiles = 0

        for y in range(
            0,
            height,
            TILE_SIZE,
        ):

            for x in range(
                0,
                width,
                TILE_SIZE,
            ):

                y2 = min(
                    y + TILE_SIZE,
                    height,
                )

                x2 = min(
                    x + TILE_SIZE,
                    width,
                )

                tile_h = y2 - y
                tile_w = x2 - x

                tile = np.zeros(
                    (
                        TILE_SIZE,
                        TILE_SIZE,
                        3,
                    ),
                    dtype=np.uint8,
                )

                tile[
                    :tile_h,
                    :tile_w,
                    0,
                ] = rgb[
                    0,
                    y:y2,
                    x:x2,
                ]

                tile[
                    :tile_h,
                    :tile_w,
                    1,
                ] = rgb[
                    1,
                    y:y2,
                    x:x2,
                ]

                tile[
                    :tile_h,
                    :tile_w,
                    2,
                ] = rgb[
                    2,
                    y:y2,
                    x:x2,
                ]

                image = (
                    tile.astype(
                        np.float32
                    )
                    / 255.0
                )

                image = (
                    image - MEAN
                ) / STD

                image = np.transpose(
                    image,
                    (2, 0, 1),
                )

                pixel_values = (
                    torch.from_numpy(
                        image
                    )
                    .unsqueeze(0)
                    .float()
                )

                with torch.no_grad():

                    outputs = model(
                        pixel_values=pixel_values
                    )

                logits = outputs.logits

                logits = (
                    torch.nn.functional.interpolate(
                        logits,
                        size=(
                            TILE_SIZE,
                            TILE_SIZE,
                        ),
                        mode="bilinear",
                        align_corners=False,
                    )
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                tile_probability, tile_prediction = (
                    torch.max(
                        probabilities,
                        dim=1,
                    )
                )

                pred = (
                    tile_prediction[0]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                confidence = (
                    tile_probability[0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                pred = pred[
                    :tile_h,
                    :tile_w,
                ]

                confidence = confidence[
                    :tile_h,
                    :tile_w,
                ]

                prediction_map[
                    y:y2,
                    x:x2,
                ] = pred

                probability_map[
                    y:y2,
                    x:x2,
                ] = confidence

                unique, counts = np.unique(
                    pred,
                    return_counts=True,
                )

                for cls, count in zip(
                    unique,
                    counts,
                ):

                    class_counts[
                        cls
                    ] += count

                total_tiles += 1

                if total_tiles % 100 == 0:

                    print(
                        "PROCESSED TILES:",
                        total_tiles,
                    )

        if output_tiff_path is not None:

            profile.update(
                count=1,
                dtype="uint8",
                nodata=6,
                compress="lzw",
            )

            output_tiff_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with rasterio.open(
                output_tiff_path,
                "w",
                **profile,
            ) as dst:

                dst.write(
                    prediction_map,
                    1,
                )

    print()
    print(
        "TOTAL TILES:",
        total_tiles,
    )

    print(
        "TOTAL PIXELS:",
        int(
            class_counts.sum()
        ),
    )

    for cls in range(7):

        print(
            f"{cls} ({CLASS_NAMES[cls]}):",
            int(
                class_counts[cls]
            ),
        )

    print(
        "PROBABILITY MIN:",
        float(
            probability_map.min()
        ),
    )

    print(
        "PROBABILITY MAX:",
        float(
            probability_map.max()
        ),
    )

    print(
        "PROBABILITY MEAN:",
        float(
            probability_map.mean()
        ),
    )

    if output_tiff_path is not None:

        print(
            "OUTPUT:",
            str(output_tiff_path),
        )

    return (
        prediction_map,
        probability_map,
        transform,
    )


def main():

    input_path = (
        "/data/raw/demo_aoi/demo_aoi_cog.tif"
    )

    output_path = (
        "/app/gis_engine/postgis/lulc_v1.tif"
    )

    run_lulc_inference(
        input_tiff_path=input_path,
        output_tiff_path=output_path,
    )


if __name__ == "__main__":
    main()