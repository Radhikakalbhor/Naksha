import os
import sys
import cv2
import numpy as np
import torch

# Add the road model folder to Python's import path
ROAD_MODEL_DIR = "/app/models/road"
sys.path.insert(0, ROAD_MODEL_DIR)

from dlinknet3 import DLinkNet34


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/app/models/road/model_best.pth"
INPUT_PATH = "/data/processed/tiles/tile_0000.tif"
OUTPUT_DIR = "/data/predictions/roads"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_road_inference():

    print("Device:", DEVICE)
    print("Loading D-LinkNet34...")

    model = DLinkNet34(num_classes=1)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("Road model loaded successfully.")

    # Read the original tile
    image_bgr = cv2.imread(
        INPUT_PATH,
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise RuntimeError(
            f"Could not read image: {INPUT_PATH}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    original_height, original_width = image_rgb.shape[:2]

    print(
        f"Original tile size: "
        f"{original_width} x {original_height}"
    )

    # Reference model expects 1024 x 1024
    image_resized = cv2.resize(
        image_rgb,
        (1024, 1024),
        interpolation=cv2.INTER_LINEAR
    )

    # Exact normalization from the reference inference code
    input_tensor = (
        image_resized.astype(np.float32) / 255.0
    )

    input_tensor = input_tensor * 3.2 - 1.6

    input_tensor = np.transpose(
        input_tensor,
        (2, 0, 1)
    )

    input_tensor = torch.tensor(
        input_tensor,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    print("Running road segmentation...")

    with torch.no_grad():

        output = model(input_tensor)

        # Handle models that return [B,1,H,W]
        prediction = output.squeeze().cpu().numpy()

        print()
        print("Prediction statistics:")
        print("Min:", float(prediction.min()))
        print("Max:", float(prediction.max()))
        print("Mean:", float(prediction.mean()))
        print("Pixels >= 0.5:", int(np.sum(prediction >= 0.5)))

        prediction_mask = (prediction >= 0.5).astype(np.uint8)

        # Resize mask back to original tile size
        prediction_mask = cv2.resize(
            prediction_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        input_name = os.path.splitext(
            os.path.basename(INPUT_PATH)
        )[0]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{input_name}_roads.png"
        )

        cv2.imwrite(
            output_path,
            prediction_mask * 255
        )

        road_pixels = int(
            np.count_nonzero(prediction_mask)
        )

        total_pixels = prediction_mask.size

        percentage = (
            road_pixels / total_pixels
        ) * 100

        print()
        print("Road inference completed.")
        print("Road pixels:", road_pixels)
        print(
            f"Road coverage: {percentage:.2f}%"
        )
        print(
            "Prediction saved to:",
            output_path
        )


if __name__ == "__main__":
    run_road_inference()