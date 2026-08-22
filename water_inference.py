import os
from pathlib import Path

import cv2
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None
import rasterio


INPUT = "/data/raw/demo_aoi/demo_aoi_cog.tif"
OUTPUT = "/app/gis_engine/postgis/water_v1.tif"

TILE_SIZE = 384
WATER_CLASS = 2

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def find_model():
    model_dir = os.getenv(
        "SKYWATER_MODEL_DIR",
        "/app/models/skywater"
    )
    model_file = os.getenv(
        "SKYWATER_MODEL_FILE",
        "skywater_segformer_b2_fp32.onnx"
    )
    model_path = Path(model_dir) / model_file

    if model_path.exists():
        return str(model_path)

    candidates = list(
        Path("/root/.cache/huggingface/hub").glob(
            "models--Realcat--skywater_seg/snapshots/*/skywater_segformer_b2_fp32.onnx"
        )
    )

    if candidates:
        return str(candidates[-1])

    print(f"SkyWater model not found at {model_path}. Downloading from Hugging Face...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="Realcat/skywater_seg",
        filename="skywater_segformer_b2_fp32.onnx",
        local_dir=str(model_path.parent),
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to {model_path}")
    return str(model_path)


def preprocess(image):
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = (
        image - MEAN
    ) / STD

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    return image.astype(
        np.float32
    )[None, ...]


def run_skywater_inference(
    image_path,
    output_path=None,
):
    """
    Run SkyWater segmentation on an arbitrary RGB GeoTIFF.

    Returns:
        water_mask:
            Binary uint8 raster. 1 = water, 0 = background.

        water_probability:
            Float32 raster containing the model's probability
            for the water class.

        transform:
            Rasterio affine transform for the source image.
    """

    image_path = Path(image_path)

    if output_path is not None:
        output_path = Path(output_path)

    model_path = find_model()

    print("MODEL:", model_path)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("INPUT:", input_name)
    print("OUTPUT:", output_name)

    with rasterio.open(image_path) as src:

        rgb = src.read(
            [1, 2, 3]
        )

        height = src.height
        width = src.width

        transform = src.transform

        profile = src.profile.copy()

        water_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        water_probability = np.zeros(
            (height, width),
            dtype=np.float32
        )

        total_tiles = 0

        for y in range(
            0,
            height,
            TILE_SIZE
        ):

            for x in range(
                0,
                width,
                TILE_SIZE
            ):

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

                tensor = preprocess(
                    tile
                )

                prediction = session.run(
                    [output_name],
                    {
                        input_name: tensor
                    }
                )[0]

                prediction = np.asarray(
                    prediction
                )

                if prediction.ndim == 4:
                    prediction = prediction[0]

                # Normalize model output to [classes, height, width]
                if prediction.shape[0] < prediction.shape[-1]:
                    logits = np.moveaxis(
                        prediction,
                        -1,
                        0
                    )
                else:
                    logits = prediction

                # Convert logits to class probabilities.
                logits = (
                    logits -
                    np.max(
                        logits,
                        axis=0,
                        keepdims=True
                    )
                )

                exp_logits = np.exp(
                    logits
                )

                probabilities = (
                    exp_logits /
                    np.sum(
                        exp_logits,
                        axis=0,
                        keepdims=True
                    )
                )

                classes = np.argmax(
                    probabilities,
                    axis=0
                )

                classes = cv2.resize(
                    classes.astype(np.uint8),
                    (
                        TILE_SIZE,
                        TILE_SIZE
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

                water_prob = cv2.resize(
                    probabilities[WATER_CLASS].astype(
                        np.float32
                    ),
                    (
                        TILE_SIZE,
                        TILE_SIZE
                    ),
                    interpolation=cv2.INTER_LINEAR
                )

                tile_water = (
                    classes[
                        :tile_h,
                        :tile_w
                    ] == WATER_CLASS
                )

                water_mask[
                    y:y2,
                    x:x2
                ] = tile_water.astype(
                    np.uint8
                )

                water_probability[
                    y:y2,
                    x:x2
                ] = water_prob[
                    :tile_h,
                    :tile_w
                ]

                total_tiles += 1

                if total_tiles % 50 == 0:
                    print(
                        "PROCESSED TILES:",
                        total_tiles
                    )

        if output_path is not None:

            profile.update(
                count=1,
                dtype="uint8",
                nodata=0,
                compress="lzw"
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as dst:

                dst.write(
                    water_mask,
                    1
                )

            print(
                "OUTPUT:",
                output_path
            )

        print(
            "TOTAL TILES:",
            total_tiles
        )

        print(
            "WATER PIXELS:",
            int(
                np.count_nonzero(
                    water_mask
                )
            )
        )

        print(
            "WATER RATIO:",
            float(
                np.count_nonzero(
                    water_mask
                ) /
                water_mask.size
            )
        )

        print(
            "CRS:",
            src.crs
        )

        return (
            water_mask,
            water_probability,
            transform,
        )


def main():

    run_skywater_inference(
        image_path=INPUT,
        output_path=OUTPUT,
    )


if __name__ == "__main__":
    main()