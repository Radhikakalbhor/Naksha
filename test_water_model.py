import cv2
import numpy as np
import onnxruntime as ort

IMAGE = "/data/raw/water_bodies/water_body_1.jpg"
MODEL = "/root/.cache/huggingface/hub/models--Realcat--skywater_seg/snapshots/dac255883ec5faf508561a47172096bfd8708db0/skywater_segformer_b2_fp32.onnx"

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

image = cv2.cvtColor(
    cv2.imread(IMAGE),
    cv2.COLOR_BGR2RGB
)

height, width = image.shape[:2]

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

counts = np.zeros(4, dtype=np.int64)

for y in range(0, height, 384):
    for x in range(0, width, 384):

        y2 = min(y + 384, height)
        x2 = min(x + 384, width)

        tile = np.zeros(
            (384, 384, 3),
            dtype=np.uint8
        )

        tile[:y2-y, :x2-x] = image[y:y2, x:x2]

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

print("IMAGE SIZE:", width, "x", height)
print("BACKGROUND:", int(counts[0]))
print("SKY:", int(counts[1]))
print("WATER:", int(counts[2]))
print("PERSON:", int(counts[3]))
print("TOTAL:", int(counts.sum()))
print("WATER RATIO:", float(counts[2] / counts.sum()))
