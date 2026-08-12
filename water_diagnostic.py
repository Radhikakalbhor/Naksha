import rasterio
import numpy as np
import onnxruntime as ort

IMAGE = "/data/raw/demo_aoi/demo_aoi_cog.tif"
MODEL = "/root/.cache/huggingface/hub/models--Realcat--skywater_seg/snapshots/dac255883ec5faf508561a47172096bfd8708db0/skywater_segformer_b2_fp32.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

with rasterio.open(IMAGE) as src:
    rgb = src.read([1, 2, 3])
    print("IMAGE:", rgb.shape)

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("MODEL INPUT:", session.get_inputs()[0].shape)
print("MODEL OUTPUT:", session.get_outputs()[0].shape)

coords = [
    (0, 0),
    (1000, 1000),
    (2000, 4000),
    (3000, 6000),
    (5000, 9000),
]

for i, (y, x) in enumerate(coords, 1):

    tile = np.transpose(
        rgb[:, y:y+384, x:x+384],
        (1, 2, 0)
    )

    image = tile.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    image = image[None].astype(np.float32)

    result = session.run(
        [output_name],
        {input_name: image}
    )[0]

    logits = result[0]

    classes = np.argmax(
        logits,
        axis=0
    )

    unique, counts = np.unique(
        classes,
        return_counts=True
    )

    print()
    print("TEST", i)
    print("POSITION:", y, x)
    print("RGB MEAN:", tile.mean(axis=(0, 1)).round(2).tolist())
    print("ARGMAX:", dict(zip(unique.tolist(), counts.tolist())))
    print("WATER LOGIT MIN:", float(logits[2].min()))
    print("WATER LOGIT MAX:", float(logits[2].max()))
    print("WATER LOGIT MEAN:", float(logits[2].mean()))
