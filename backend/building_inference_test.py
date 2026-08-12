import os
import cv2
import numpy as np
import torch

from models.building_unet import UNet


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = "/data/raw/spacenet/SN2_buildings_train_AOI_3_Paris_PS-RGB_img10.tif"
OUTPUT_DIR = "/data/predictions/buildings"

THRESHOLD = 0.5
MODEL_PATH = "/app/models/building_best_unet.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Load model
# ============================================================

print("Loading Building U-Net...")

model = UNet().to(device)

state = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state)
model.eval()

print("Building model loaded successfully.")


# ============================================================
# Read SpaceNet reference image
# ============================================================

image = cv2.imread(
    INPUT_PATH,
    cv2.IMREAD_COLOR
)

if image is None:
    raise RuntimeError(
        f"Could not read image: {INPUT_PATH}"
    )

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

height, width = image.shape[:2]

print(
    f"Original image size: {width} x {height}"
)


# ============================================================
# Take a genuine 512x512 crop
# ============================================================

if height < 512 or width < 512:
    raise RuntimeError(
        "Reference image is smaller than 512x512."
    )

image = image[:512, :512]

print("Using reference crop: 512 x 512")


# ============================================================
# Preprocessing
# Match author's training pipeline
# ============================================================

image_float = image.astype(
    np.float32
)

img_min = image_float.min()
img_max = image_float.max()

if img_max > img_min:

    image_normalized = (
        image_float - img_min
    ) / (
        img_max - img_min
    )

else:

    image_normalized = np.zeros_like(
        image_float
    )


# HWC -> CHW
input_array = np.transpose(
    image_normalized,
    (2, 0, 1)
)

# NumPy -> PyTorch
input_tensor = torch.from_numpy(
    input_array
).float()

# Add batch dimension
input_tensor = input_tensor.unsqueeze(
    0
).to(device)


# ============================================================
# Building segmentation
# ============================================================

print("Running building segmentation...")

with torch.no_grad():

    output = model(
        input_tensor
    )


# Logits -> probability
prediction = torch.sigmoid(
    output
)

prediction = (
    prediction
    .squeeze()
    .cpu()
    .numpy()
)


# ============================================================
# Prediction statistics
# ============================================================

print()
print("Prediction statistics:")

print(
    "Min:",
    float(prediction.min())
)

print(
    "Max:",
    float(prediction.max())
)

print(
    "Mean:",
    float(prediction.mean())
)


# ============================================================
# Binary building mask
# ============================================================

mask = (
    prediction >= THRESHOLD
).astype(
    np.uint8
)

building_pixels = int(
    mask.sum()
)

print(
    "Pixels >= 0.5:",
    building_pixels
)


# ============================================================
# Building coverage
# ============================================================

total_pixels = 512 * 512

coverage = (
    building_pixels /
    total_pixels
) * 100


print()
print("Building inference completed.")

print(
    "Building pixels:",
    building_pixels
)

print(
    f"Building coverage: {coverage:.2f}%"
)


# ============================================================
# Save prediction
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "spacenet_reference_buildings.png"
)

cv2.imwrite(
    output_path,
    mask * 255
)

print(
    "Prediction saved to:",
    output_path
)