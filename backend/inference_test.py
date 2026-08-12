from deepforest import main
import os


def run_tree_detection():

    model = main.deepforest()

    model.load_model(
        model_name="weecology/deepforest-tree",
        revision="main"
    )

    image_path = "/data/processed/tiles/tile_0250.tif"

    print("Running DeepForest on:", image_path)

    predictions = model.predict_tile(
        path=image_path,
        patch_size=400,
        patch_overlap=0.05
    )

    if predictions is None or len(predictions) == 0:
        print("No tree detections were produced.")
        return

    print(f"Tree detections: {len(predictions)}")

    output_path = "/data/predictions/trees/tile_0250_trees.csv"

    predictions.to_csv(output_path, index=False)

    print("Tree predictions saved to:", output_path)


if __name__ == "__main__":
    run_tree_detection()