import rasterio
import cv2
import numpy as np
import onnxruntime as ort

INPUT = "/data/raw/demo_aoi/demo_aoi_cog.tif"
MODEL = "/root/.cache/huggingface/hub/models--Realcat--skywater_seg/snapshots/dac255883ec5faf508561a47172096bfd8708db0/skywater_segformer_b2_fp32.onnx"

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

with rasterio.open(INPUT) as src:
    rgb = src.read([1, 2, 3])
    height = src.height
    width = src.width

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

counts = np.zeros(4, dtype=np.int64)
tiles = 0

for y in range(0, height, 384):
    for x in range(0, width, 384):

        y2 = min(y + 384, height)
        x2 = min(x + 384, width)

        tile = np.zeros(
            (384, 384, 3),
            dtype=np.uint8
        )

        tile[:y2-y, :x2-x, 0] = rgb[0, y:y2, x:x2]
        tile[:y2-y, :x2-x, 1] = rgb[1, y:y2, x:x2]
        tile[:y2-y, :x2-x, 2] = rgb[2, y:y2, x:x2]

        tile = tile.astype(np.float32) / 255.0
        tile = (tile - mean) / std
        tile = np.transpose(tile, (2, 0, 1))
        tile = tile[None].astype(np.float32)

        output = session.run(
            None,
            {"input": tile}
        )[0]

        prediction = np.argmax(
            output[0],
            axis=0
        )

        prediction = prediction[:y2-y, :x2-x]

        unique, tile_counts = np.unique(
            prediction,
            return_counts=True
        )

        for cls, count in zip(unique, tile_counts):
            counts[cls] += count

        tiles += 1

        if 2 in unique:
            print(
                "WATER TILE:",
                tiles,
                "water pixels:",
                int(np.sum(prediction == 2))
            )

print()
print("TOTAL TILES:", tiles)
print("BACKGROUND:", int(counts[0]))
print("SKY:", int(counts[1]))
print("WATER:", int(counts[2]))
print("PERSON:", int(counts[3]))
print("TOTAL PIXELS:", int(counts.sum()))
print(
    "WATER RATIO:",
    float(counts[2] / counts.sum())
)
