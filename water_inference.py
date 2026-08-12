import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rasterio


INPUT = "/data/raw/demo_aoi/demo_aoi_cog.tif"
MODEL = "/root/.cache/huggingface/hub/models--Realcat--skywater_seg/snapshots"
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
    candidates = list(
        Path("/root/.cache/huggingface/hub").glob(
            "models--Realcat--skywater_seg/snapshots/*/skywater_segformer_b2_fp32.onnx"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "SkyWater FP32 ONNX model was not found in the Hugging Face cache."
        )

    return str(candidates[-1])


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


def main():

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

    with rasterio.open(INPUT) as src:

        rgb = src.read(
            [1, 2, 3]
        )

        height = src.height
        width = src.width

        profile = src.profile.copy()

        water_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        total_tiles = 0
        water_pixels = 0

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

                    prediction = prediction[
                        0
                    ]

                if prediction.shape[0] < prediction.shape[-1]:

                    classes = np.argmax(
                        prediction,
                        axis=0
                    )

                else:

                    classes = np.argmax(
                        prediction,
                        axis=-1
                    )

                classes = cv2.resize(
                    classes.astype(np.uint8),
                    (
                        TILE_SIZE,
                        TILE_SIZE
                    ),
                    interpolation=cv2.INTER_NEAREST
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

                water_pixels += int(
                    tile_water.sum()
                )

                total_tiles += 1

                if total_tiles % 50 == 0:
                    print(
                        "PROCESSED TILES:",
                        total_tiles
                    )

        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
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
                water_mask,
                1
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
            "OUTPUT:",
            OUTPUT
        )

        print(
            "CRS:",
            src.crs
        )


if __name__ == "__main__":
    main()
