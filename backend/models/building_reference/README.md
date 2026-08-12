# Deep Learning-Based Building Footprint Extraction from Satellite Imagery

An end-to-end deep learning pipeline for automated building footprint extraction from high-resolution satellite imagery using semantic segmentation.

This project implements and compares three state-of-the-art semantic segmentation architectures—**U-Net**, **DeepLabV3+**, and **SegFormer**—for extracting building footprints from aerial imagery. The complete workflow includes data preprocessing, GeoJSON-to-mask conversion, patch-based training, sliding-window inference, and object-level building count estimation. The project demonstrates the strengths of both CNN-based and transformer-based models for remote sensing applications. :contentReference[oaicite:0]{index=0}

---

## Project Pipeline

<p align="center">
  <img src="Images/Pipeline.png" alt="Project Pipeline" width="100%">
</p>

---

## Features

- End-to-end semantic segmentation pipeline
- GeoJSON to binary mask generation
- Random 512 × 512 patch extraction
- Online data augmentation
- Training and comparison of three segmentation models
- Sliding-window inference for full-resolution satellite images
- Building count estimation using Connected Component Analysis
- Pixel-level and object-level performance evaluation

---

## Models Implemented

- **U-Net**
- **DeepLabV3+**
- **SegFormer**

All models were trained using the same preprocessing pipeline and evaluated using identical metrics for a fair comparison.

---

## Dataset

The project uses the **SpaceNet 2 Paris Buildings** dataset.

**Kaggle Dataset:**

https://www.kaggle.com/datasets/ugorjiir/spacenet-2-paris-buildings

The dataset contains:

- High-resolution RGB satellite images (`.tif`)
- Building footprint annotations (`.geojson`)

Binary segmentation masks are generated during preprocessing. :contentReference[oaicite:1]{index=1}

---

## Pretrained Model Weights

The trained model checkpoints are hosted on Hugging Face.

**Hugging Face Repository:**

https://huggingface.co/Abhatta7/building-footprint-segmentation

Available models:

- `best_unet.pth`
- `best_deeplab.pth`
- `best_segformer.pth`

---

## Methodology

The overall workflow consists of:

1. Dataset preparation
2. GeoJSON to binary mask generation
3. Random patch extraction (512 × 512)
4. Data augmentation
5. Model training
6. Sliding-window inference
7. Building count estimation
8. Performance evaluation and comparison

---

## Results

### Segmentation Performance

| Model | IoU | Dice Score |
|-------|----:|-----------:|
| U-Net | 0.6256 | 0.7438 |
| **DeepLabV3+** | **0.7254** | **0.8289** |
| SegFormer | 0.7063 | 0.8165 |

### Building Count Estimation

| Model | Mean Count Error |
|-------|-----------------:|
| U-Net | 2.2087 |
| DeepLabV3+ | 3.2043 |
| **SegFormer** | **2.0174** |

DeepLabV3+ achieved the highest pixel-level segmentation performance, while SegFormer achieved the lowest building count error, demonstrating stronger object-level consistency. :contentReference[oaicite:2]{index=2}

---

## Qualitative Results

### U-Net

<p align="center">
  <img src="Images/UNet_prediction.png" width="800">
</p>

---

### DeepLabV3+

<p align="center">
  <img src="Images/DeeplabV3+_prediction.png" width="800">
</p>

---

### SegFormer

<p align="center">
  <img src="Images/SegFormer_prediction.png" width="800">
</p>

---

## Project Structure

```text
Building-Footprint-Extraction/
│
├── images/
│   ├── pipeline.png
│   ├── unet_results.png
│   ├── deeplab_results.png
│   └── segformer_results.png
│
├── report/
│   └── Building_Footprint_Extraction_Final_Report.pdf
│
├── Building_Footprint.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Technologies Used

- Python
- PyTorch
- Transformers
- Segmentation Models PyTorch
- Rasterio
- GeoPandas
- OpenCV
- NumPy
- Albumentations
- Matplotlib
- Scikit-learn

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AryakBhattacharya/building-footprint-extraction.git

cd building-footprint-extraction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and organize it according to the notebook before training or inference.

---

## Future Work

- Train on multi-city datasets for improved generalization
- Explore larger transformer-based segmentation models
- Incorporate instance segmentation to separate adjacent buildings
- Optimize hyperparameters for improved accuracy
- Develop an interactive web application for deployment :contentReference[oaicite:3]{index=3}

---

## Report

The complete project report is available in the `report/` directory.

---

## Author

**Aryak Bhattacharya**

MS in Artificial Intelligence  
DePaul University

---

## License

This project is released under the MIT License.
