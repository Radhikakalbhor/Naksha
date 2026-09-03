# ============================================================
# NAKSHA - AI POWERED ORTHOPHOTO DIGITIZATION API
# ============================================================

import os
import sys
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import rasterio
try:
    import torch
    import torch.nn as nn
    nn_Module = nn.Module
except ImportError:
    torch = None
    nn = None
    nn_Module = object

import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import psycopg
from export_engine import export_layer
from gis_engine.vectorization.vectorize import (
    skeletonize_road_mask,
    skeleton_to_lines,
    polygonize_mask,
    feature_confidence,
)
from shapely.geometry import mapping, shape
from gis_engine.topology.geometry import validate_geometry
from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql

try:
    from deepforest import main as deepforest_main
except ImportError:
    deepforest_main = None

from water_inference import run_skywater_inference
from lulc_inference import run_lulc_inference
from lulc_endpoint import router as lulc_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("naksha")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Naksha API",
    version="0.6.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(lulc_router)

# ============================================================
# POSTGIS
# ============================================================

POSTGIS_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "dbname": os.getenv("POSTGRES_DB", "naksha"),
    "user": os.getenv("POSTGRES_USER", "naksha"),
    "password": os.getenv("POSTGRES_PASSWORD", "naksha_dev"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

ALLOWED_LAYERS = {
    "demo_buildings",
    "demo_roads",
    "demo_farms",
    "demo_trees",
    "demo_water",
    "demo_lulc",
    "uploaded_buildings",
    "uploaded_roads",
    "uploaded_trees",
    "uploaded_farms",
    "uploaded_water",
    "uploaded_lulc",
}



def get_postgis_connection():
    conn = psycopg.connect(
        host=POSTGIS_CONFIG["host"],
        dbname=POSTGIS_CONFIG["dbname"],
        user=POSTGIS_CONFIG["user"],
        password=POSTGIS_CONFIG["password"],
        port=POSTGIS_CONFIG["port"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE vector_features 
                ADD COLUMN IF NOT EXISTS qc_status TEXT DEFAULT 'pending';
            """)
            cur.execute("""
                ALTER TABLE vector_features 
                ADD COLUMN IF NOT EXISTS source_model TEXT,
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    celery_task_id TEXT,
                    feature_type TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    input_file TEXT,
                    result JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    except Exception:
        conn.rollback()
    return conn


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch is not None
    else "cpu"
)

print("Naksha device:", DEVICE)


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = Path("/data") if Path("/data").exists() else BASE_DIR / "data"


# ============================================================
# FIELD MODEL / DELINEATE ANYTHING
# ============================================================

FIELD_MODEL_DIR = (
    MODELS_DIR / "delineate_anything"
)

FIELD_INPUT_DIR = (
    FIELD_MODEL_DIR / "data" / "images"
)

FIELD_OUTPUT_DIR = (
    FIELD_MODEL_DIR / "data" / "delineated"
)

FIELD_BATCH_CONFIG = (
    FIELD_MODEL_DIR / "batch_naksha_api.yaml"
)

FIELD_INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIELD_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BUILDING MODEL
# ============================================================

BUILDING_MODEL_PATH = (
    MODELS_DIR / "building_best_unet.pth"
)

BUILDING_OUTPUT_DIR = (
    DATA_DIR / "predictions" / "buildings"
)

BUILDING_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BUILDING_PATCH_SIZE = 512
BUILDING_STRIDE = 256

# Exact threshold used by the reference
# sliding-window inference code
BUILDING_THRESHOLD = 0.3


# ============================================================
# ROAD MODEL
# ============================================================

ROAD_MODEL_PATH = (
    MODELS_DIR / "road" / "model_best.pth"
)

ROAD_OUTPUT_DIR = (
    DATA_DIR / "predictions" / "roads"
)

ROAD_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ROAD_THRESHOLD = 0.5
ROAD_TARGET_SIZE = (1024, 1024)

ROAD_MIN_SIZE = 150
ROAD_KERNEL_SIZE = 3
ROAD_DILATE_ITER = 1


# ============================================================

# ============================================================

# TREE MODEL

# ============================================================

TREE_OUTPUT_DIR = (
    DATA_DIR / "predictions" / "trees"
)

TREE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WATER_OUTPUT_DIR = (
    DATA_DIR / "predictions" / "water"
)

WATER_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
LULC_OUTPUT_DIR = (
    DATA_DIR / "predictions" / "lulc"
)

LULC_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TREE_MODEL_NAME = "weecology/deepforest-tree"
TREE_PATCH_SIZE = 400
TREE_PATCH_OVERLAP = 0.05

_tree_model = None

def get_tree_model():

    global _tree_model

    if _tree_model is None:

        print(
            "Loading DeepForest tree model..."
        )

        model = deepforest_main.deepforest()

        model.load_model(
            model_name=TREE_MODEL_NAME,
            revision="main"
        )

        _tree_model = model

        print(
            "DeepForest tree model loaded successfully."
        )

    return _tree_model


# BUILDING U-NET
# ============================================================

class DoubleConv(nn_Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(self, x):

        return self.conv(x)


class UNet(nn_Module):

    def __init__(self):

        super().__init__()

        self.down1 = DoubleConv(
            3,
            64
        )

        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(
            64,
            128
        )

        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(
            128,
            256
        )

        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(
            256,
            512
        )

        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.conv4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.conv3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.conv2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.conv1 = DoubleConv(
            128,
            64
        )

        self.out = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        d1 = self.down1(x)

        d2 = self.down2(
            self.pool1(d1)
        )

        d3 = self.down3(
            self.pool2(d2)
        )

        d4 = self.down4(
            self.pool3(d3)
        )

        b = self.bottleneck(
            self.pool4(d4)
        )

        u4 = self.up4(b)

        u4 = torch.cat(
            [u4, d4],
            dim=1
        )

        u4 = self.conv4(u4)

        u3 = self.up3(u4)

        u3 = torch.cat(
            [u3, d3],
            dim=1
        )

        u3 = self.conv3(u3)

        u2 = self.up2(u3)

        u2 = torch.cat(
            [u2, d2],
            dim=1
        )

        u2 = self.conv2(u2)

        u1 = self.up1(u2)

        u1 = torch.cat(
            [u1, d1],
            dim=1
        )

        u1 = self.conv1(u1)

        return self.out(u1)


# ============================================================
# BUILDING MODEL LOADER
# ============================================================

_building_model = None


def get_building_model():

    global _building_model

    if _building_model is None:

        print(
            "Loading Building U-Net..."
        )

        if not BUILDING_MODEL_PATH.exists():

            raise RuntimeError(
                f"Building model not found: "
                f"{BUILDING_MODEL_PATH}"
            )

        model = UNet()

        checkpoint = torch.load(
            str(BUILDING_MODEL_PATH),
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint
        )

        model.to(DEVICE)

        model.eval()

        _building_model = model

        print(
            "Building U-Net loaded successfully."
        )

    return _building_model


# ============================================================
# BUILDING SLIDING WINDOW INFERENCE
# ============================================================

def building_sliding_window_inference(
    model,
    image_path,
    patch_size=512,
    stride=256
):

    # --------------------------------------------------------
    # Load RGB bands
    # --------------------------------------------------------

    with rasterio.open(
        str(image_path)
    ) as src:

        if src.count < 3:
            raise RuntimeError(
                "Input GeoTIFF must contain "
                "at least 3 bands."
            )

        img = src.read(
            [1, 2, 3]
        ).transpose(1, 2, 0)

    # --------------------------------------------------------
    # Exact reference normalization
    # --------------------------------------------------------

    img = img.astype(
        np.float32
    )

    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (
            img - img_min
        ) / (
            img_max - img_min
        )
    else:
        img = np.zeros_like(img)

    h, w, _ = img.shape

    # --------------------------------------------------------
    # Prediction containers
    # --------------------------------------------------------

    full_pred = np.zeros(
        (h, w),
        dtype=np.float32
    )

    count_map = np.zeros(
        (h, w),
        dtype=np.float32
    )

    model.eval()

    # --------------------------------------------------------
    # Sliding window
    # --------------------------------------------------------

    BATCH_SIZE = 2

    with torch.inference_mode():

        x_positions = list(
            range(
                0,
                w - patch_size + 1,
                stride
            )
        )

        y_positions = list(
            range(
                0,
                h - patch_size + 1,
                stride
            )
        )

        # Handle right/bottom edges

        if (
            not x_positions
            or
            x_positions[-1]
            != w - patch_size
        ):
            x_positions.append(
                w - patch_size
            )

        if (
            not y_positions
            or
            y_positions[-1]
            != h - patch_size
        ):
            y_positions.append(
                h - patch_size
            )

        # ----------------------------------------------------
        # Build patch locations
        # ----------------------------------------------------

        patch_locations = [
            (y, x)
            for y in y_positions
            for x in x_positions
        ]

        total_patches = len(
            patch_locations
        )

        total_batches = (
            total_patches + BATCH_SIZE - 1
        ) // BATCH_SIZE

        # ----------------------------------------------------
        # Batched inference
        # ----------------------------------------------------

        for batch_start in range(
            0,
            total_patches,
            BATCH_SIZE
        ):

            batch_locations = patch_locations[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            patches = []

            for y, x in batch_locations:

                patch = img[
                    y:y + patch_size,
                    x:x + patch_size
                ]

                patches.append(
                    patch
                )

            patch_array = np.stack(
                patches,
                axis=0
            )

            patch_tensor = torch.from_numpy(
                patch_array
            ).permute(
                0,
                3,
                1,
                2
            ).to(
                DEVICE
            )

            pred = model(
                patch_tensor
            )

            pred = torch.sigmoid(
                pred
            )

            pred = (
                pred
                .squeeze(1)
                .cpu()
                .numpy()
            )

            for i, (y, x) in enumerate(
                batch_locations
            ):

                full_pred[
                    y:y + patch_size,
                    x:x + patch_size
                ] += pred[i]

                count_map[
                    y:y + patch_size,
                    x:x + patch_size
                ] += 1

            current_batch = (
                batch_start // BATCH_SIZE
            ) + 1

            print(
                f"Building batch "
                f"{current_batch}/{total_batches} "
                f"({min(batch_start + BATCH_SIZE, total_patches)}/"
                f"{total_patches} patches)"
            )

    # --------------------------------------------------------
    # Blend overlapping predictions
    # --------------------------------------------------------

    count_map[
        count_map == 0
    ] = 1

    full_pred = (
        full_pred /
        count_map
    )

    # --------------------------------------------------------
    # Exact reference threshold
    # --------------------------------------------------------

    final_mask = (
        full_pred >
        BUILDING_THRESHOLD
    ).astype(
        np.uint8
    )

    return (
        img,
        final_mask,
        full_pred
    )

# ============================================================
# ROAD MODEL IMPORTS
# ============================================================

try:
    from models.road.dlinknet3 import (
        DLinkNet34
    )
    from models.road.postprocess import (
        postprocess_mask
    )
except ImportError:
    DLinkNet34 = None
    postprocess_mask = None


# ============================================================
# ROAD MODEL LOADER
# ============================================================

_road_model = None


def get_road_model():

    global _road_model

    if _road_model is None:

        print(
            "Loading Road DLinkNet34..."
        )

        if not ROAD_MODEL_PATH.exists():

            raise RuntimeError(
                f"Road model not found: "
                f"{ROAD_MODEL_PATH}"
            )

        # IMPORTANT:
        # pretrained=False prevents downloading
        # ResNet weights from the internet.

        model = DLinkNet34(
            num_classes=1,
            pretrained=False
        )

        checkpoint = torch.load(
            str(ROAD_MODEL_PATH),
            map_location=DEVICE
        )

        if (
            isinstance(checkpoint, dict)
            and
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            state_dict = checkpoint

        model.load_state_dict(
            state_dict
        )

        model.to(DEVICE)

        model.eval()

        _road_model = model

        print(
            "Road DLinkNet34 loaded successfully."
        )

    return _road_model


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "Naksha API",
        "version": app.version,
        "status": "running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "version": app.version
    }


# ============================================================
# FIELD INFERENCE
# ============================================================
# ============================================================
# POSTGIS LAYERS
# ============================================================

@app.get("/layers")
def list_layers():

    try:
        with get_postgis_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        lv.id,
                        lv.layer_name,
                        lv.feature_type,
                        lv.version,
                        COUNT(vf.id) AS feature_count
                    FROM layer_versions lv
                    LEFT JOIN vector_features vf
                        ON vf.layer_version_id = lv.id
                    WHERE lv.layer_name = ANY(%s)
                    GROUP BY
                        lv.id,
                        lv.layer_name,
                        lv.feature_type,
                        lv.version
                    ORDER BY lv.id;
                    """,
                    (list(ALLOWED_LAYERS),),
                )

                rows = cur.fetchall()

        return {
            "layers": [
                {
                    "id": row[0],
                    "layer_name": row[1],
                    "feature_type": row[2],
                    "version": row[3],
                    "feature_count": row[4],
                }
                for row in rows
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to retrieve layers from PostGIS.",
                "error": str(e),
            },
        )


@app.get("/layers/{layer_name}")
def get_layer(layer_name: str):

    if layer_name not in ALLOWED_LAYERS:
        raise HTTPException(
            status_code=404,
            detail="Layer not found.",
        )

    try:
        with get_postgis_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT
                        vf.id,
                        vf.feature_type,
                        vf.confidence,
                        ST_AsGeoJSON(vf.geometry)
                    FROM vector_features vf
                    JOIN layer_versions lv
                        ON lv.id = vf.layer_version_id
                    WHERE lv.layer_name = %s
                    ORDER BY vf.id;
                    """,
                    (layer_name,),
                )

                rows = cur.fetchall()

        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": row[0],
                    "geometry": json.loads(row[3]),
                    "properties": {
                        "feature_type": row[1],
                        "confidence": row[2],
                    },
                }
                for row in rows
            ],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to retrieve layer features from PostGIS.",
                "error": str(e),
            },
        )
# ============================================================
# DAY 6 - VECTOR LAYER EXPORT
# ============================================================

@app.get("/layers/{layer_name}/export")
def export_vector_layer_query(
    layer_name: str,
    format: Optional[str] = Query(None, alias="format"),
    export_format: Optional[str] = Query(None),
    qc_only: bool = Query(False),
):
    fmt = format or export_format
    if not fmt:
        raise HTTPException(
            status_code=400,
            detail="Export format required. Provide path parameter '/export/{format}' or query parameter '?format={format}'."
        )
    return export_vector_layer(
        layer_name=layer_name,
        export_format=fmt,
        qc_only=qc_only,
    )


@app.get("/layers/{layer_name}/export/{export_format}")
def export_vector_layer(
    layer_name: str,
    export_format: str,
    qc_only: bool = Query(False),
):
    """
    Export a PostGIS vector layer.

    Supported formats:
    - geojson
    - shapefile
    - geopackage
    - filegdb
    """

    if layer_name not in ALLOWED_LAYERS:
        raise HTTPException(
            status_code=404,
            detail="Layer not found.",
        )

    export_format = (
        export_format.lower().strip()
    )

    allowed_formats = {
        "geojson",
        "shapefile",
        "geopackage",
        "filegdb",
    }

    if export_format not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported export format.",
                "supported_formats": sorted(
                    allowed_formats
                ),
            },
        )

    export_dir = (
        DATA_DIR /
        "exports"
    )

    export_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        with get_postgis_connection() as conn:

            output_path, media_type = export_layer(
                conn=conn,
                layer_name=layer_name,
                export_format=export_format,
                output_dir=export_dir,
                qc_only=qc_only,
            )

        return FileResponse(
            path=str(output_path),
            media_type=media_type,
            filename=output_path.name,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        print(
            "Layer export error:",
            str(e),
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Layer export failed.",
                "error": str(e),
            },
        )
@app.post("/inference/fields")
def field_inference(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    allowed_extensions = {
        ".tif",
        ".tiff"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only .tif and .tiff files "
                "are supported."
            )
        )

    job_id = uuid.uuid4().hex[:8]

    job_name = (
        f"Naksha_{job_id}"
    )

    input_folder = (
        FIELD_INPUT_DIR /
        job_name
    )

    input_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    input_path = (
        input_folder /
        file.filename
    )

    try:

        # ----------------------------------------------------
        # Save upload
        # ----------------------------------------------------

        with input_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"Starting field inference: "
            f"{input_path}"
        )

        # ----------------------------------------------------
        # Batch configuration
        # ----------------------------------------------------

        batch_config = f"""
base_config: conf_sample.yaml

data_root: data/images
output_root: data/delineated
temp_root: data/temp_{job_id}
keep_temp: false
mask_root: data/masks

include:

- {job_name}

exclude: null
override: null
"""

        FIELD_BATCH_CONFIG.write_text(
            batch_config,
            encoding="utf-8"
        )

        # ----------------------------------------------------
        # Run Delineate Anything
        # ----------------------------------------------------

        result = subprocess.run(
            [
                sys.executable,
                "delineate.py",
                "-b",
                str(FIELD_BATCH_CONFIG)
            ],
            cwd=str(FIELD_MODEL_DIR),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            print(result.stdout)
            print(result.stderr)

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "Field delineation failed.",
                    "stdout":
                        result.stdout[-4000:],
                    "stderr":
                        result.stderr[-4000:]
                }
            )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        gpkg_path = (
            FIELD_OUTPUT_DIR /
            f"{job_name}.gpkg"
        )

        simplified_gpkg_path = (
            FIELD_OUTPUT_DIR /
            f"{job_name}.simp.gpkg"
        )

        if not gpkg_path.exists():

            raise HTTPException(
                status_code=500,
                detail=(
                    "Delineation completed "
                    "but no GeoPackage was produced."
                )
            )

        features_stored = 0
        layer_version_id = None
        postgis_status = "skipped"
        postgis_error = None

        gpkg_to_read = (
            simplified_gpkg_path
            if simplified_gpkg_path.exists()
            else gpkg_path
        )

        try:
            import geopandas as gpd

            print(f"Reading field geometries from: {gpkg_to_read}")
            gdf = gpd.read_file(gpkg_to_read)

            if len(gdf) > 0:
                print("Storing field features in PostGIS...")

                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:

                        cur.execute(
                            """
                            SELECT COALESCE(MAX(version), 0) + 1
                            FROM layer_versions
                            WHERE layer_name = 'uploaded_farms'
                            """
                        )
                        next_version = cur.fetchone()[0]

                        cur.execute(
                            create_layer_version_sql(
                                layer_name="uploaded_farms",
                                feature_type="farms",
                                version=next_version,
                            )
                        )
                        layer_version_id = cur.fetchone()[0]

                        print(f"Created layer version: {layer_version_id}")

                        for _, row in gdf.iterrows():
                            geometry = row.geometry

                            if geometry is None or geometry.is_empty:
                                continue

                            validated_geometry = validate_geometry(geometry)

                            if validated_geometry is None:
                                continue

                            conf_val = None
                            if "confidence" in row and row["confidence"] is not None:
                                try:
                                    conf_float = float(row["confidence"])
                                    if not math.isnan(conf_float):
                                        conf_val = conf_float
                                except Exception:
                                    conf_val = None

                            cur.execute(
                                create_feature_sql(
                                    layer_version_id=layer_version_id,
                                    feature_type="farms",
                                    geometry=validated_geometry,
                                    confidence=conf_val,
                                )
                            )
                            features_stored += 1

                    conn.commit()

                postgis_status = "success"
                print(f"Stored {features_stored} field features in PostGIS")

        except ImportError:
            postgis_status = "skipped"
            postgis_error = "geopandas not available"
            print("geopandas not available, skipping PostGIS storage")

        except Exception as e:
            postgis_status = "failed"
            postgis_error = str(e)
            print(f"PostGIS storage failed: {e}")

        return {

            "status": "success",

            "job_id": job_id,

            "input_file":
                file.filename,

            "output": {

                "gpkg":
                    str(gpkg_path),

                "simplified_gpkg":
                    (
                        str(
                            simplified_gpkg_path
                        )
                        if simplified_gpkg_path.exists()
                        else None
                    )
            },
            "features_stored": features_stored,
            "layer_version_id": layer_version_id,
            "postgis_status": postgis_status,
            "confidence_available": False,
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# FIELD RESULT -> GEOJSON
# ============================================================

@app.get(
    "/inference/fields/{job_id}/result"
)
def field_result(
    job_id: str
):

    job_name = (
        f"Naksha_{job_id}"
    )

    gpkg_path = (
        FIELD_OUTPUT_DIR /
        f"{job_name}.simp.gpkg"
    )

    if not gpkg_path.exists():
        gpkg_path = (
            FIELD_OUTPUT_DIR /
            f"{job_name}.gpkg"
        )

    if not gpkg_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Field result not found."
        )

    try:

        # ----------------------------------------------------
        # Convert GeoPackage to GeoJSON using GDAL ogr2ogr
        # ----------------------------------------------------

        result = subprocess.run(
            [
                "ogr2ogr",
                "-f",
                "GeoJSON",
                "/vsistdout/",
                str(gpkg_path),
                "-nln",
                "fields"
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "Could not convert Field GeoPackage to GeoJSON.",
                    "stderr":
                        result.stderr[-4000:]
                }
            )

        if not result.stdout.strip():

            raise HTTPException(
                status_code=500,
                detail="GeoPackage conversion returned no GeoJSON."
            )

        geojson = json.loads(
            result.stdout
        )

        return geojson

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ============================================================

# ============================================================

# TREE INFERENCE

# ============================================================

@app.post("/inference/trees")
def tree_inference(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    allowed_extensions = {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Supported files: "
                ".tif, .tiff, .png, .jpg, .jpeg"
            )
        )

    job_id = uuid.uuid4().hex[:8]

    input_path = (
        TREE_OUTPUT_DIR /
        f"tree_input_{job_id}"
        f"{extension}"
    )

    output_path = (
        TREE_OUTPUT_DIR /
        f"Naksha_{job_id}_trees.geojson"
    )

    try:

        # ----------------------------------------------------
        # Save upload
        # ----------------------------------------------------

        with input_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"Starting tree inference: "
            f"{input_path}"
        )

        # ----------------------------------------------------
        # Load DeepForest model
        # ----------------------------------------------------

        model = get_tree_model()

        print(
            "Running DeepForest tree detection..."
        )

        # ----------------------------------------------------
        # Run tiled inference
        # ----------------------------------------------------

        predictions = model.predict_tile(
            path=str(input_path),
            patch_size=TREE_PATCH_SIZE,
            patch_overlap=TREE_PATCH_OVERLAP
        )

        # ----------------------------------------------------
        # Handle empty prediction
        # ----------------------------------------------------

        if (
            predictions is None
            or len(predictions) == 0
        ):

            geojson = {
                "type": "FeatureCollection",
                "features": []
            }

            with output_path.open(
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    geojson,
                    f
                )

            return {
                "status": "success",
                "job_id": job_id,
                "model": TREE_MODEL_NAME,
                "tree_count": 0,
                "output": {
                    "result": str(output_path)
                }
            }

        # ----------------------------------------------------
        # Convert detections to GeoJSON
        # ----------------------------------------------------

        features = []

        with rasterio.open(
            str(input_path)
        ) as src:

            transform = src.transform
            crs = src.crs

        for _, row in predictions.iterrows():

            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            # Pixel coordinates ? geographic coordinates

            left, top = rasterio.transform.xy(
                transform,
                ymin,
                xmin,
                offset="ul"
            )

            right, bottom = rasterio.transform.xy(
                transform,
                ymax,
                xmax,
                offset="ul"
            )

            coordinates = [
                [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                    [left, top]
                ]
            ]

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                },
                "properties": {
                    "feature_type": "trees",
                    "label": str(
                        row.get(
                            "label",
                            "Tree"
                        )
                    ),
                    "confidence": float(
                        row["score"]
                    )
                }
            })

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        if crs is not None:

            try:

                epsg = crs.to_epsg()

                if epsg is not None:

                    geojson["crs"] = {
                        "type": "name",
                        "properties": {
                            "name":
                                f"EPSG:{epsg}"
                        }
                    }

            except Exception:

                pass

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                geojson,
                f
            )

        tree_count = len(features)

        confidence_values = [
            feature["properties"]["confidence"]
            for feature in features
        ]

        features_stored = 0
        layer_version_id = None
        postgis_status = "skipped"
        postgis_error = None

        if features:
            try:
                print("Storing tree features in PostGIS...")

                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:

                        cur.execute(
                            """
                            SELECT COALESCE(MAX(version), 0) + 1
                            FROM layer_versions
                            WHERE layer_name = 'uploaded_trees'
                            """
                        )
                        next_version = cur.fetchone()[0]

                        cur.execute(
                            create_layer_version_sql(
                                layer_name="uploaded_trees",
                                feature_type="trees",
                                version=next_version,
                            )
                        )
                        layer_version_id = cur.fetchone()[0]

                        print(f"Created layer version: {layer_version_id}")

                        for feature in features:
                            geometry = shape(feature["geometry"])
                            validated_geometry = validate_geometry(geometry)

                            if validated_geometry is None:
                                continue

                            confidence = feature["properties"]["confidence"]

                            cur.execute(
                                create_feature_sql(
                                    layer_version_id=layer_version_id,
                                    feature_type="trees",
                                    geometry=validated_geometry,
                                    confidence=confidence,
                                )
                            )
                            features_stored += 1

                    conn.commit()

                postgis_status = "success"
                print(f"Stored {features_stored} tree features in PostGIS")

            except Exception as e:
                postgis_status = "failed"
                postgis_error = str(e)
                print(f"PostGIS storage failed: {e}")

        response = {
            "status": "success",
            "job_id": job_id,
            "input_file": file.filename,
            "model": TREE_MODEL_NAME,
            "output": {
                "result": str(output_path)
            },
            "statistics": {
                "tree_count": tree_count,
                "confidence_min": min(
                    confidence_values
                ) if confidence_values else 0.0,
                "confidence_mean": (
                    sum(confidence_values)
                    / len(confidence_values)
                ) if confidence_values else 0.0,
                "confidence_max": max(
                    confidence_values
                ) if confidence_values else 0.0,
            },
            "features_stored": features_stored,
            "layer_version_id": layer_version_id,
            "postgis_status": postgis_status,
        }

        if postgis_error:
            response["postgis_error"] = postgis_error

        return response

    except HTTPException:

        raise

    except Exception as e:

        print(
            "Tree inference error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Tree inference failed.",
                "error":
                    str(e)
            }
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary input
        # ----------------------------------------------------

        try:

            if input_path.exists():

                input_path.unlink()

        except Exception:

            pass


# ============================================================
# WATER INFERENCE
# ============================================================

@app.post("/inference/water")
def water_inference(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    allowed_extensions = {
        ".tif",
        ".tiff"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only .tif and .tiff files "
                "are supported."
            )
        )

    job_id = uuid.uuid4().hex[:8]

    input_path = (
        WATER_OUTPUT_DIR /
        f"water_input_{job_id}"
        f"{extension}"
    )

    mask_output_path = (
        WATER_OUTPUT_DIR /
        f"Naksha_{job_id}_water_mask.tif"
    )

    prob_output_path = (
        WATER_OUTPUT_DIR /
        f"Naksha_{job_id}_water_prob.tif"
    )

    try:


        # ----------------------------------------------------
        # Save upload
        # ----------------------------------------------------

        with input_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"Starting water inference: "
            f"{input_path}"
        )

        # ----------------------------------------------------
        # Run SkyWater inference
        # ----------------------------------------------------

        water_mask, water_probability, transform = run_skywater_inference(
            image_path=str(input_path),
            output_path=str(mask_output_path),
        )

        # ----------------------------------------------------
        # Save probability raster
        # ----------------------------------------------------

        with rasterio.open(str(input_path)) as src:
            profile = src.profile.copy()
            profile.update(
                count=1,
                dtype="float32",
                nodata=0.0,
                compress="lzw"
            )

        prob_output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with rasterio.open(
            str(prob_output_path),
            "w",
            **profile
        ) as dst:
            dst.write(water_probability, 1)

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        height, width = water_mask.shape
        water_pixels = int(np.count_nonzero(water_mask))
        total_pixels = water_mask.size
        water_coverage = (water_pixels / total_pixels) * 100

        prob_valid = water_probability[water_probability > 0]
        prob_min = float(prob_valid.min()) if prob_valid.size > 0 else 0.0
        prob_max = float(prob_valid.max()) if prob_valid.size > 0 else 0.0
        prob_mean = float(prob_valid.mean()) if prob_valid.size > 0 else 0.0

        print(
            "Water inference completed."
        )
        print(
            f"Image dimensions: {width}x{height}"
        )
        print(
            f"Water pixels: {water_pixels}"
        )
        print(
            f"Water coverage: {water_coverage:.2f}%"
        )
        print(
            f"Probability stats - min: {prob_min:.4f}, max: {prob_max:.4f}, mean: {prob_mean:.4f}"
        )
        print(
            f"Mask saved to: {mask_output_path}"
        )
        print(
            f"Probability saved to: {prob_output_path}"
        )

        # ----------------------------------------------------
        # PostGIS: Polygonize mask and store water features
        # ----------------------------------------------------

        features_stored = 0
        layer_version_id = None
        postgis_status = "skipped"
        postgis_error = None

        try:
            print("Polygonizing water mask...")

            geometries = polygonize_mask(
                water_mask,
                transform,
                min_area=0.0,
            )

            print(f"Raw water geometries: {len(geometries)}")

            if geometries:
                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:

                        cur.execute(
                            """
                            SELECT COALESCE(MAX(version), 0) + 1
                            FROM layer_versions
                            WHERE layer_name = 'uploaded_water'
                            """
                        )
                        next_version = cur.fetchone()[0]

                        cur.execute(
                            create_layer_version_sql(
                                layer_name="uploaded_water",
                                feature_type="water",
                                version=next_version,
                            )
                        )
                        layer_version_id = cur.fetchone()[0]

                        print(f"Created layer version: {layer_version_id}")

                        for geometry in geometries:
                            geometry = validate_geometry(geometry)

                            if geometry is None:
                                continue

                            confidence = feature_confidence(
                                probability=water_probability,
                                geometry=geometry,
                                transform=transform,
                            )

                            if confidence is None:
                                continue

                            cur.execute(
                                create_feature_sql(
                                    layer_version_id=layer_version_id,
                                    feature_type="water",
                                    geometry=geometry,
                                    confidence=confidence,
                                    source_model="skywater_segmentation",
                                )
                            )
                            features_stored += 1

                    conn.commit()

                postgis_status = "success"
                print(f"Stored {features_stored} water features in PostGIS")

        except Exception as e:
            postgis_status = "failed"
            postgis_error = str(e)
            print(f"PostGIS storage failed: {e}")

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        response = {

            "status":
                "success",

            "job_id":
                job_id,

            "input_file":
                file.filename,

            "model":
                "SkyWater SegFormer-B2",

            "preprocessing":
                "RGB bands 1,2,3 + ImageNet normalization + 384x384 tiling",

            "output": {

                "mask":
                    str(mask_output_path),

                "probability":
                    str(prob_output_path),
            },

            "statistics": {

                "image_width":
                    width,

                "image_height":
                    height,

                "water_pixels":
                    water_pixels,

                "water_coverage_percent":
                    round(
                        water_coverage,
                        2
                    ),

                "probability_min":
                    prob_min,

                "probability_max":
                    prob_max,

                "probability_mean":
                    prob_mean,
            },

            "features_stored":
                features_stored,

            "layer_version_id":
                layer_version_id,

            "postgis_status":
                postgis_status,
        }

        if postgis_error:
            response["postgis_error"] = postgis_error

        return response

    except HTTPException:

        raise

    except Exception as e:

        print(
            "Water inference error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Water inference failed.",
                "error":
                    str(e)
            }
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary input
        # ----------------------------------------------------

        try:

            if input_path.exists():

                input_path.unlink()

        except Exception:

            pass


# ============================================================

# BUILDING INFERENCE
# ============================================================

@app.post("/inference/buildings")
def building_inference(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    allowed_extensions = {
        ".tif",
        ".tiff"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only .tif and .tiff files "
                "are supported."
            )
        )

    job_id = uuid.uuid4().hex[:8]

    input_path = (
        BUILDING_OUTPUT_DIR /
        f"building_input_{job_id}"
        f"{extension}"
    )

    output_path = (
        BUILDING_OUTPUT_DIR /
        f"Naksha_{job_id}_buildings.png"
    )

    try:

        # ----------------------------------------------------
        # Save upload
        # ----------------------------------------------------

        with input_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"Starting building inference: "
            f"{input_path}"
        )

        # ----------------------------------------------------
        # Read original image dimensions and transform
        # ----------------------------------------------------

        with rasterio.open(
            str(input_path)
        ) as src:

            if src.count < 3:

                raise RuntimeError(
                    "Input GeoTIFF must contain "
                    "at least 3 bands."
                )

            original_width = src.width
            original_height = src.height
            src_transform = src.transform

        print(
            "Original image size:",
            original_width,
            "x",
            original_height
        )

        if (
            original_width <
            BUILDING_PATCH_SIZE
            or
            original_height <
            BUILDING_PATCH_SIZE
        ):

            raise RuntimeError(
                "Input image must be at least "
                "512 x 512 pixels."
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = get_building_model()

        print(
            "Running building sliding-window "
            "segmentation..."
        )

        # ----------------------------------------------------
        # Full-image inference
        # ----------------------------------------------------

        image, mask, prediction = (
            building_sliding_window_inference(
                model=model,
                image_path=input_path,
                patch_size=BUILDING_PATCH_SIZE,
                stride=BUILDING_STRIDE
            )
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        prediction_min = float(
            prediction.min()
        )

        prediction_max = float(
            prediction.max()
        )

        prediction_mean = float(
            prediction.mean()
        )

        building_pixels = int(
            mask.sum()
        )

        coverage = (
            building_pixels /
            mask.size
        ) * 100

        print(
            "Building prediction statistics:"
        )

        print(
            "Min:",
            prediction_min
        )

        print(
            "Max:",
            prediction_max
        )

        print(
            "Mean:",
            prediction_mean
        )

        print(
            "Building pixels:",
            building_pixels
        )

        print(
            f"Building coverage: "
            f"{coverage:.2f}%"
        )

        # ----------------------------------------------------
        # Save full-size mask
        # ----------------------------------------------------

        cv2.imwrite(
            str(output_path),
            mask * 255
        )

        if not output_path.exists():

            raise RuntimeError(
                "Building inference completed "
                "but output mask was not created."
            )

        print(
            "Building inference completed."
        )

        print(
            "Prediction saved to:",
            output_path
        )

        # ----------------------------------------------------
        # PostGIS: Polygonize mask and store building features
        # ----------------------------------------------------

        features_stored = 0
        layer_version_id = None
        postgis_status = "skipped"
        postgis_error = None

        try:
            print("Polygonizing building mask...")

            geometries = polygonize_mask(
                mask,
                src_transform,
                min_area=0.0,
            )

            print(f"Raw building geometries: {len(geometries)}")

            if geometries:
                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT COALESCE(MAX(version), 0) + 1
                            FROM layer_versions
                            WHERE layer_name = 'uploaded_buildings'
                            """
                        )
                        next_version = cur.fetchone()[0]

                        cur.execute(
                            create_layer_version_sql(
                                layer_name="uploaded_buildings",
                                feature_type="buildings",
                                version=next_version,
                            )
                        )
                        layer_version_id = cur.fetchone()[0]

                        print(f"Created layer version: {layer_version_id}")

                        for geometry in geometries:
                            geometry = validate_geometry(geometry)

                            if geometry is None:
                                continue

                            confidence = feature_confidence(
                                probability=prediction,
                                geometry=geometry,
                                transform=src_transform,
                            )

                            if confidence is None:
                                continue

                            cur.execute(
                                create_feature_sql(
                                    layer_version_id=layer_version_id,
                                    feature_type="buildings",
                                    geometry=geometry,
                                    confidence=confidence,
                                )
                            )
                            features_stored += 1

                    conn.commit()

                postgis_status = "success"
                print(f"Stored {features_stored} building features in PostGIS")

        except Exception as e:
            postgis_status = "failed"
            postgis_error = str(e)
            print(f"PostGIS storage failed: {e}")

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        response = {

            "status":
                "success",

            "job_id":
                job_id,

            "input_file":
                file.filename,

            "model":
                "Building U-Net",

            "preprocessing":
                "Rasterio bands 1,2,3 + "
                "global min-max normalization "
                "+ 512x512 sliding window "
                "+ 256 stride",

            "output": {

                "mask":
                    str(output_path)
            },

            "statistics": {

                "image_width":
                    original_width,

                "image_height":
                    original_height,

                "processed_width":
                    original_width,

                "processed_height":
                    original_height,

                "patch_size":
                    BUILDING_PATCH_SIZE,

                "stride":
                    BUILDING_STRIDE,

                "threshold":
                    BUILDING_THRESHOLD,

                "building_pixels":
                    building_pixels,

                "building_coverage_percent":
                    round(
                        coverage,
                        2
                    ),

                "prediction_min":
                    prediction_min,

                "prediction_max":
                    prediction_max,

                "prediction_mean":
                    prediction_mean,


            },

            "features_stored":
                features_stored,

            "layer_version_id":
                layer_version_id,

            "postgis_status":
                postgis_status,
        }

        if postgis_error:
            response["postgis_error"] = postgis_error

        return response

    except HTTPException:

        raise

    except Exception as e:

        print(
            "Building inference error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Building inference failed.",
                "error":
                    str(e)
            }
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary TIFF
        # ----------------------------------------------------

        try:

            if input_path.exists():

                input_path.unlink()

        except Exception:

            pass


# ============================================================
# ROAD INFERENCE
# ============================================================

@app.post("/inference/roads")
def road_inference(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    allowed_extensions = {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Supported files: "
                ".tif, .tiff, .png, .jpg, .jpeg"
            )
        )

    job_id = uuid.uuid4().hex[:8]

    input_path = (
        ROAD_OUTPUT_DIR /
        f"road_input_{job_id}"
        f"{extension}"
    )

    output_path = (
        ROAD_OUTPUT_DIR /
        f"Naksha_{job_id}_roads.png"
    )

    try:

        # ----------------------------------------------------
        # Save upload
        # ----------------------------------------------------

        with input_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"Starting road inference: "
            f"{input_path}"
        )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bgr = cv2.imread(
            str(input_path)
        )

        if image_bgr is None:

            raise RuntimeError(
                "Could not read input image."
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        original_height, original_width = (
            image_rgb.shape[:2]
        )

        print(
            "Original image size:",
            original_width,
            "x",
            original_height
        )

        # ----------------------------------------------------
        # Resize to 1024x1024
        # ----------------------------------------------------

        image_resized = cv2.resize(
            image_rgb,
            ROAD_TARGET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Reference road normalization
        # ----------------------------------------------------

        input_array = (
            image_resized.astype(
                np.float32
            ) / 255.0
        )

        input_array = (
            input_array * 3.2
            - 1.6
        )

        input_array = np.transpose(
            input_array,
            (2, 0, 1)
        )

        input_tensor = torch.tensor(
            input_array,
            dtype=torch.float32
        ).unsqueeze(
            0
        ).to(DEVICE)

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = get_road_model()

        print(
            "Running road segmentation..."
        )

        with torch.no_grad():

            output = model(
                input_tensor
            )

        pred = (
            output
            .squeeze()
            .cpu()
            .numpy()
        )

        prediction_min = float(
            pred.min()
        )

        prediction_max = float(
            pred.max()
        )

        prediction_mean = float(
            pred.mean()
        )

        print(
            "Road prediction statistics:"
        )

        print(
            "Min:",
            prediction_min
        )

        print(
            "Max:",
            prediction_max
        )

        print(
            "Mean:",
            prediction_mean
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        pred_mask = (
            pred >
            ROAD_THRESHOLD
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Post-processing
        # ----------------------------------------------------

        pred_mask = postprocess_mask(
            pred_mask,
            min_size=ROAD_MIN_SIZE,
            kernel_size=ROAD_KERNEL_SIZE,
            dilate_iter=ROAD_DILATE_ITER
        )

        # ----------------------------------------------------
        # Resize back to original size
        # ----------------------------------------------------

        pred_mask_resized = cv2.resize(
            pred_mask.astype(
                np.uint8
            ),
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_NEAREST
        )

        pred_resized = cv2.resize(
            pred,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

        # ROAD MASK -> CENTERLINE VECTORIZATION
        # ----------------------------------------------------

        road_geojson_path = (
            ROAD_OUTPUT_DIR /
            f"Naksha_{job_id}_roads.geojson"
        )

        try:

            road_skeleton = skeletonize_road_mask(
                pred_mask_resized
            )

            with rasterio.open(
                str(input_path)
            ) as road_src:

                road_lines = skeleton_to_lines(
                    road_skeleton,
                    road_src.transform
                )

                road_crs = road_src.crs

            road_features = []

            for line in road_lines:

                if line is None or line.is_empty:
                    continue

                validated_line = validate_geometry(line)
                if validated_line is None:
                    continue

                road_confidence = feature_confidence(
                    probability=pred_resized,
                    geometry=validated_line,
                    transform=road_src.transform,
                )

                if road_confidence is None:
                    continue

                road_features.append({
                    "type": "Feature",
                    "geometry": mapping(validated_line),
                    "properties": {
                        "feature_type": "roads",
                        "confidence": road_confidence
                    }
                })

            road_geojson = {
                "type": "FeatureCollection",
                "features": road_features
            }

            if road_crs is not None:

                try:

                    epsg = road_crs.to_epsg()

                    if epsg is not None:

                        road_geojson["crs"] = {
                            "type": "name",
                            "properties": {
                                "name": f"EPSG:{epsg}"
                            }
                        }

                except Exception:
                    pass

            with road_geojson_path.open(
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    road_geojson,
                    f
                )

            print(
                "Road vectorization completed."
            )

            print(
                "Road centerlines:",
                len(road_features)
            )

            print(
                "Road GeoJSON saved to:",
                road_geojson_path
            )

            features_stored = 0
            layer_version_id = None
            postgis_status = "skipped"
            postgis_error = None

            if road_features:
                try:
                    print("Storing road features in PostGIS...")

                    with get_postgis_connection() as conn:
                        with conn.cursor() as cur:

                            cur.execute(
                                """
                                SELECT COALESCE(MAX(version), 0) + 1
                                FROM layer_versions
                                WHERE layer_name = 'uploaded_roads'
                                """
                            )
                            next_version = cur.fetchone()[0]

                            cur.execute(
                                create_layer_version_sql(
                                    layer_name="uploaded_roads",
                                    feature_type="roads",
                                    version=next_version,
                                )
                            )
                            layer_version_id = cur.fetchone()[0]

                            print(f"Created layer version: {layer_version_id}")

                            for feature in road_features:
                                geometry = shape(feature["geometry"])
                                confidence = feature["properties"]["confidence"]

                                cur.execute(
                                    create_feature_sql(
                                        layer_version_id=layer_version_id,
                                        feature_type="roads",
                                        geometry=geometry,
                                        confidence=confidence,
                                    )
                                )
                                features_stored += 1

                        conn.commit()

                    postgis_status = "success"
                    print(f"Stored {features_stored} road features in PostGIS")

                except Exception as e:
                    postgis_status = "failed"
                    postgis_error = str(e)
                    print(f"PostGIS storage failed: {e}")

        except Exception as vector_error:

            print(
                "Road vectorization warning:",
                str(vector_error)
            )

            road_geojson_path = None

        road_pixels = int(
            pred_mask_resized.sum()
        )

        total_pixels = (
            pred_mask_resized.size
        )

        road_coverage = (
            road_pixels /
            total_pixels
        ) * 100

        # Save
        # ----------------------------------------------------

        cv2.imwrite(
            str(output_path),
            pred_mask_resized * 255
        )

        if not output_path.exists():

            raise RuntimeError(
                "Road inference completed "
                "but output mask was not created."
            )

        print(
            "Road inference completed."
        )

        print(
            "Road pixels:",
            road_pixels
        )

        print(
            f"Road coverage: "
            f"{road_coverage:.2f}%"
        )

        print(
            "Prediction saved to:",
            output_path
        )

        response = {

            "status":
                "success",

            "job_id":
                job_id,

            "input_file":
                file.filename,

            "model":
                "DLinkNet34",

            "preprocessing":
                "RGB + resize 1024x1024 + "
                "reference road normalization",

            "output": {

                "mask":
                    str(output_path)
            },

            "statistics": {

                "image_width":
                    original_width,

                "image_height":
                    original_height,

                "processed_width":
                    1024,

                "processed_height":
                    1024,

                "road_pixels":
                    road_pixels,

                "road_coverage_percent":
                    round(
                        road_coverage,
                        2
                    ),

                "prediction_min":
                    prediction_min,

                "prediction_max":
                    prediction_max,

                "prediction_mean":
                    prediction_mean,


            },

            "features_stored":
                features_stored,

            "layer_version_id":
                layer_version_id,

            "postgis_status":
                postgis_status,
        }

        if postgis_error:
            response["postgis_error"] = postgis_error

        return response


    except HTTPException:

        raise

    except Exception as e:

        print(
            "Road inference error:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Road inference failed.",
                "error":
                    str(e)
            }
        )

    finally:

        # ----------------------------------------------------
        # Remove temporary input
        # ----------------------------------------------------

        try:

            if input_path.exists():

                input_path.unlink()

        except Exception:

            pass


# ============================================================

# ============================================================

# TREE RESULT DOWNLOAD

# ============================================================

@app.get(
    "/inference/trees/{job_id}/result"
)
def tree_result(
    job_id: str
):

    result_path = (
        TREE_OUTPUT_DIR /
        f"Naksha_{job_id}_trees.geojson"
    )

    if not result_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Tree result not found."
        )

    return FileResponse(
        path=str(result_path),
        media_type="application/geo+json",
        filename=result_path.name
    )


# ============================================================

# ============================================================

# ROAD RESULT DOWNLOAD

# ============================================================

@app.get(
"/inference/roads/{job_id}/result"
)
def road_result(
job_id: str
):

    result_path = (
        ROAD_OUTPUT_DIR /
        f"Naksha_{job_id}_roads.geojson"
    )

    if not result_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Road result not found."
        )

    return FileResponse(
        path=str(result_path),
        media_type="application/geo+json",
        filename=result_path.name
    )


# BUILDING MASK DOWNLOAD
# ============================================================

@app.get(
    "/inference/buildings/{job_id}/mask"
)
def building_mask(
    job_id: str
):

    mask_path = (
        BUILDING_OUTPUT_DIR /
        f"Naksha_{job_id}_buildings.png"
    )

    if not mask_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Building mask not found."
        )

    return FileResponse(
        path=str(mask_path),
        media_type="image/png",
        filename=mask_path.name
    )


# ============================================================
# ROAD MASK DOWNLOAD
# ============================================================

@app.get(
    "/inference/roads/{job_id}/mask"
)
def road_mask(
    job_id: str
):

    mask_path = (
        ROAD_OUTPUT_DIR /
        f"Naksha_{job_id}_roads.png"
    )

    if not mask_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Road mask not found."
        )

    return FileResponse(
        path=str(mask_path),
        media_type="image/png",
        filename=mask_path.name
    )


# ============================================================
# DAY 8 - QC WORKFLOW ENDPOINTS
# ============================================================

@app.get("/qc/queue")
def get_qc_queue(
    layer_name: Optional[str] = Query(None),
    status: Optional[str] = Query("pending"),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
):
    if layer_name and layer_name not in ALLOWED_LAYERS:
        raise HTTPException(status_code=404, detail="Layer not found.")

    effective_limit = min(limit, 1000)

    try:
        query_parts = [
            """
            SELECT
                vf.id,
                vf.feature_type,
                vf.confidence,
                ST_AsGeoJSON(vf.geometry),
                vf.qc_status,
                lv.id AS layer_version_id,
                lv.layer_name,
                vf.source_model
            FROM vector_features vf
            JOIN layer_versions lv ON lv.id = vf.layer_version_id
            """
        ]
        where_clauses = []
        params = []

        if layer_name:
            where_clauses.append("lv.layer_name = %s")
            params.append(layer_name)

        if status:
            if status == "pending":
                where_clauses.append("(vf.qc_status = 'pending' OR vf.qc_status IS NULL)")
            else:
                where_clauses.append("vf.qc_status = %s")
                params.append(status)

        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        query_parts.append("ORDER BY vf.confidence ASC NULLS FIRST, vf.id ASC")
        query_parts.append("LIMIT %s OFFSET %s;")
        params.extend([effective_limit, offset])

        sql = " ".join(query_parts)

        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

        return {
            "type": "FeatureCollection",
            "layer_name": layer_name,
            "status": status,
            "limit": effective_limit,
            "offset": offset,
            "features": [
                {
                    "type": "Feature",
                    "id": row[0],
                    "geometry": json.loads(row[3]) if row[3] else None,
                    "properties": {
                        "feature_id": row[0],
                        "layer_version_id": row[5],
                        "layer_name": row[6],
                        "feature_type": row[1],
                        "confidence": row[2],
                        "qc_status": row[4] or "pending",
                        "source_model": row[7],
                    },
                }
                for row in rows
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "Failed to fetch QC queue.", "error": str(e)})


@app.get("/qc/{feature_id}")
def get_qc_feature(feature_id: int):
    try:
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        vf.id,
                        vf.feature_type,
                        vf.confidence,
                        ST_AsGeoJSON(vf.geometry),
                        vf.qc_status,
                        lv.id AS layer_version_id,
                        lv.layer_name,
                        vf.source_model
                    FROM vector_features vf
                    JOIN layer_versions lv ON lv.id = vf.layer_version_id
                    WHERE vf.id = %s;
                    """,
                    (feature_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Feature not found.")
        return {
            "feature_id": row[0],
            "layer_version_id": row[5],
            "layer_name": row[6],
            "feature_type": row[1],
            "confidence": row[2],
            "qc_status": row[4] or "pending",
            "source_model": row[7],
            "geometry": json.loads(row[3]) if row[3] else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "Failed to fetch feature.", "error": str(e)})


@app.post("/qc/{feature_id}/accept")
def accept_qc_feature(feature_id: int):
    try:
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE vector_features SET qc_status = 'accepted', updated_at = CURRENT_TIMESTAMP WHERE id = %s RETURNING id;",
                    (feature_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Feature not found.")
            conn.commit()
        return {"status": "success", "feature_id": feature_id, "qc_status": "accepted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "Failed to accept feature.", "error": str(e)})


@app.post("/qc/{feature_id}/reject")
def reject_qc_feature(feature_id: int):
    try:
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE vector_features SET qc_status = 'rejected', updated_at = CURRENT_TIMESTAMP WHERE id = %s RETURNING id;",
                    (feature_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Feature not found.")
            conn.commit()
        return {"status": "success", "feature_id": feature_id, "qc_status": "rejected"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "Failed to reject feature.", "error": str(e)})


@app.put("/qc/{feature_id}")
def update_qc_feature(feature_id: int, payload: dict = Body(...)):
    geom_data = payload.get("geometry")
    if not geom_data:
        raise HTTPException(status_code=400, detail="Missing geometry in request payload.")
    try:
        geom_obj = shape(geom_data)
        validated_geom = validate_geometry(geom_obj)
        if validated_geom is None or validated_geom.is_empty:
            raise HTTPException(status_code=400, detail="Invalid geometry provided.")
        wkt = validated_geom.wkt.replace("'", "''")
        confidence = payload.get("confidence")
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                if confidence is not None:
                    cur.execute(
                        """
                        UPDATE vector_features
                        SET geometry = ST_GeomFromText(%s, 4326),
                            confidence = %s,
                            qc_status = 'edited',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        RETURNING id;
                        """,
                        (wkt, confidence, feature_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE vector_features
                        SET geometry = ST_GeomFromText(%s, 4326),
                            qc_status = 'edited',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        RETURNING id;
                        """,
                        (wkt, feature_id),
                    )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Feature not found.")
            conn.commit()
        return {"status": "success", "feature_id": feature_id, "qc_status": "edited"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "Failed to update feature.", "error": str(e)})


# ============================================================
# DAY 9 - ASYNC JOBS (CELERY + REDIS)
# ============================================================

IN_MEMORY_JOBS = {}


def save_job_to_db(job_id: str, feature_type: str, status: str, input_file: str, celery_task_id: str = None, result: dict = None):
    try:
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO jobs (job_id, celery_task_id, feature_type, status, input_file, result, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (job_id) DO UPDATE SET
                        celery_task_id = COALESCE(EXCLUDED.celery_task_id, jobs.celery_task_id),
                        feature_type = EXCLUDED.feature_type,
                        status = EXCLUDED.status,
                        input_file = EXCLUDED.input_file,
                        result = EXCLUDED.result,
                        updated_at = CURRENT_TIMESTAMP;
                """, (job_id, celery_task_id, feature_type, status, input_file, json.dumps(result or {})))
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to persist job {job_id} to DB: {e}")


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    feature_type: str = Query("buildings"),
    file: UploadFile = File(...)
):
    job_id = uuid.uuid4().hex[:8]
    ext = Path(file.filename).suffix.lower() if file.filename else ".tif"
    temp_path = DATA_DIR / "temp" / f"job_input_{job_id}{ext}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # COG conversion
    cog_path = DATA_DIR / "temp" / f"job_cog_{job_id}{ext}"
    try:
        from preprocessing.cog import convert_to_cog
        convert_to_cog(temp_path, cog_path)
    except Exception as exc:
        logger.warning(f"COG conversion warning ({exc}), using raw input.")
        cog_path = temp_path

    # MinIO storage upload
    minio_status = "local_fallback"
    try:
        from gis_engine.storage import upload_raster_to_minio
        upload_raster_to_minio(temp_path, job_id, category="raw")
        _, minio_status = upload_raster_to_minio(cog_path, job_id, category="cogs")
    except Exception as exc:
        logger.warning(f"MinIO storage warning ({exc}), retaining local path.")

    job_info = {
        "job_id": job_id,
        "feature_type": feature_type,
        "status": "pending",
        "input_file": file.filename,
        "cog_path": str(cog_path),
        "minio_storage": minio_status,
    }
    IN_MEMORY_JOBS[job_id] = job_info
    save_job_to_db(job_id, feature_type, "pending", file.filename, result=job_info)

    celery_dispatched = False
    try:
        from tasks import dispatch_task
        result = dispatch_task(job_id, feature_type, str(cog_path))
        if result and hasattr(result, "id"):
            job_info["celery_task_id"] = result.id
            job_info["status"] = "queued"
            save_job_to_db(job_id, feature_type, "queued", file.filename, celery_task_id=result.id, result=job_info)
            celery_dispatched = True
    except Exception as exc:
        logger.warning(f"Celery dispatch unavailable ({exc}), using background task fallback.")

    if not celery_dispatched:
        async def run_fallback():
            job_info["status"] = "processing"
            save_job_to_db(job_id, feature_type, "processing", file.filename, result=job_info)
            try:
                from tasks import run_inference_job
                res = run_inference_job(job_id, feature_type, str(cog_path))
                job_info["status"] = "completed"
                job_info["result"] = res
                save_job_to_db(job_id, feature_type, "completed", file.filename, result=job_info)
            except Exception as e:
                job_info["status"] = "failed"
                job_info["error"] = str(e)
                save_job_to_db(job_id, feature_type, "failed", file.filename, result=job_info)

        background_tasks.add_task(run_fallback)

    return {"status": "success", "job_id": job_id, "message": "Inference job submitted successfully."}


@app.get("/jobs")
def list_jobs(task_type: Optional[str] = Query(None), limit: Optional[int] = Query(None)):
    jobs_list = []
    try:
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                where_clause = "WHERE feature_type = %s" if task_type else ""
                params = (task_type,) if task_type else ()
                limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""

                cur.execute(f"""
                    SELECT job_id, celery_task_id, feature_type, status, input_file, result, created_at, updated_at
                    FROM jobs
                    {where_clause}
                    ORDER BY created_at DESC
                    {limit_clause};
                """, params)
                rows = cur.fetchall()
                for row in rows:
                    jid, c_tid, ftype, status, inp, res, cat, uat = row
                    res_dict = res if isinstance(res, dict) else (json.loads(res) if res else {})
                    info = {
                        "job_id": jid,
                        "celery_task_id": c_tid,
                        "feature_type": ftype,
                        "status": status,
                        "input_file": inp,
                        "created_at": cat.isoformat() if cat else None,
                        "updated_at": uat.isoformat() if uat else None,
                    }
                    if isinstance(res_dict, dict):
                        info.update(res_dict)

                    if c_tid:
                        try:
                            from tasks import celery_app
                            if celery_app is not None:
                                res_celery = celery_app.AsyncResult(c_tid)
                                state = res_celery.state
                                if state in ("PENDING", "RECEIVED"):
                                    info["status"] = "queued"
                                elif state == "STARTED":
                                    info["status"] = "processing"
                                elif state == "SUCCESS":
                                    info["status"] = "completed"
                                elif state == "FAILURE":
                                    info["status"] = "failed"
                        except Exception:
                            pass
                    jobs_list.append(info)
    except Exception as e:
        logger.warning(f"Error querying jobs from DB ({e}), fallback to memory.")
        for jid, info in IN_MEMORY_JOBS.items():
            if task_type and info.get("feature_type") != task_type:
                continue
            jobs_list.append(dict(info))

    return {"jobs": jobs_list, "total": len(jobs_list)}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    try:
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT job_id, celery_task_id, feature_type, status, input_file, result, created_at, updated_at
                    FROM jobs
                    WHERE job_id = %s;
                """, (job_id,))
                row = cur.fetchone()
                if row:
                    jid, c_tid, ftype, status, inp, res, cat, uat = row
                    res_dict = res if isinstance(res, dict) else (json.loads(res) if res else {})
                    job_info = {
                        "job_id": jid,
                        "celery_task_id": c_tid,
                        "feature_type": ftype,
                        "status": status,
                        "input_file": inp,
                        "created_at": cat.isoformat() if cat else None,
                        "updated_at": uat.isoformat() if uat else None,
                    }
                    if isinstance(res_dict, dict):
                        job_info.update(res_dict)

                    if c_tid:
                        try:
                            from tasks import celery_app
                            if celery_app is not None:
                                res_celery = celery_app.AsyncResult(c_tid)
                                state = res_celery.state
                                if state in ("PENDING", "RECEIVED"):
                                    job_info["status"] = "queued"
                                elif state == "STARTED":
                                    job_info["status"] = "processing"
                                elif state == "SUCCESS":
                                    job_info["status"] = "completed"
                                    if res_celery.result:
                                        job_info["result"] = res_celery.result
                                elif state == "FAILURE":
                                    job_info["status"] = "failed"
                                    if res_celery.result:
                                        job_info["error"] = str(res_celery.result)
                        except Exception:
                            pass
                    return job_info
    except Exception as e:
        logger.warning(f"Error fetching job {job_id} from DB: {e}")

    if job_id in IN_MEMORY_JOBS:
        return IN_MEMORY_JOBS[job_id]

    raise HTTPException(status_code=404, detail="Job not found.")


# ============================================================
# DAY 10 - INGESTION & ACCURACY REPORT
# ============================================================

@app.post("/ingest")
def ingest_imagery(
    file: UploadFile = File(...),
    target_crs: str = Query("EPSG:4326"),
    tile_size: int = Query(512),
    overlap: int = Query(64)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    job_id = uuid.uuid4().hex[:8]
    raw_dir = DATA_DIR / "raw" / f"ingest_{job_id}"
    processed_dir = DATA_DIR / "processed" / f"ingest_{job_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / file.filename
    with raw_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # 1. CRS Inspection & Normalization
    normalized_path = processed_dir / f"normalized_{file.filename}"
    detected_crs = "EPSG:4326"
    try:
        from gis_engine.crs.crs_utils import get_raster_crs, normalize_raster_crs
        detected_crs = get_raster_crs(raw_path) or "EPSG:4326"
        normalize_raster_crs(raw_path, normalized_path, target_crs=target_crs)
    except Exception as e:
        logger.warning(f"CRS normalization failed ({e}), using raw source file.")
        shutil.copy(raw_path, normalized_path)

    # 2. COG Conversion
    cog_path = processed_dir / f"cog_{file.filename}"
    try:
        from preprocessing.cog import convert_to_cog
        convert_to_cog(normalized_path, cog_path)
    except Exception as e:
        logger.warning(f"COG conversion warning ({e}), using normalized file.")
        shutil.copy(normalized_path, cog_path)

    # 3. Tiling
    tiles_dir = processed_dir / "tiles"
    tile_files = []
    try:
        from preprocessing.tiler import tile_raster
        tile_files = [str(p) for p in tile_raster(cog_path, tiles_dir, tile_size=tile_size, overlap=overlap)]
    except Exception as e:
        logger.warning(f"Tiling warning ({e}).")

    # 4. MinIO Storage Upload
    minio_status = "local_fallback"
    try:
        from gis_engine.storage import upload_raster_to_minio
        upload_raster_to_minio(raw_path, job_id, category="raw")
        _, minio_status = upload_raster_to_minio(cog_path, job_id, category="cogs")
    except Exception as e:
        logger.warning(f"MinIO storage warning ({e}).")

    return {
        "status": "success",
        "job_id": job_id,
        "input_filename": file.filename,
        "crs": target_crs,
        "outputs": {
            "raw": str(raw_path),
            "normalized": str(normalized_path),
            "cog": str(cog_path),
            "tiles_dir": str(tiles_dir),
        },
        "minio_storage": minio_status,
    }


FEATURE_TYPE_REFERENCE_MAP = {
    "trees": [
        BASE_DIR / "gis_engine" / "postgis" / "trees_v1.geojson",
    ],
    "farms": [
        BASE_DIR / "gis_engine" / "postgis" / "farms_v1.geojson",
    ],
    "fields": [
        BASE_DIR / "gis_engine" / "postgis" / "farms_v1.geojson",
    ],
    "buildings": [
        BASE_DIR / "gis_engine" / "postgis" / "buildings_v1.geojson",
        DATA_DIR / "raw" / "spacenet" / "SN2_buildings_train_AOI_3_Paris_geojson_buildings_img10.geojson",
    ],
    "roads": [
        BASE_DIR / "gis_engine" / "postgis" / "roads_v1.geojson",
    ],
    "water": [
        BASE_DIR / "gis_engine" / "postgis" / "water_v1.geojson",
    ],
    "lulc": [
        BASE_DIR / "gis_engine" / "postgis" / "lulc_v1.geojson",
    ],
}


def get_ground_truth_reference_and_metrics(
    feature_type: str,
    pred_geoms: list,
    user_ref_path: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Find matching reference dataset strictly matching feature_type, validate it,
    and calculate ground-truth accuracy metrics if available.

    Returns:
        (ground_truth_accuracy_available: bool, reference_source_path: str|None, metrics_dict: dict|None)
    """
    ftype_norm = (feature_type or "").lower().strip()

    candidates = []
    if user_ref_path:
        p = Path(user_ref_path)
        if p.exists():
            candidates.append(p)

    matching_map_paths = FEATURE_TYPE_REFERENCE_MAP.get(ftype_norm, [])
    candidates.extend(matching_map_paths)

    valid_ref_path = None
    gt_geoms = []

    for cand in candidates:
        if cand.exists():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                features = data.get("features", [])
                parsed_geoms = []
                for feat in features:
                    if feat and feat.get("geometry"):
                        try:
                            g = shape(feat["geometry"])
                            if g and not g.is_empty:
                                parsed_geoms.append(g)
                        except Exception:
                            continue
                if parsed_geoms:
                    valid_ref_path = cand
                    gt_geoms = parsed_geoms
                    break
            except Exception as exc:
                logger.warning(f"Error reading reference dataset candidate {cand}: {exc}")
                continue

    if not valid_ref_path or not gt_geoms:
        return False, None, None

    try:
        from gis_engine.validation import compute_vector_metrics
        metrics = compute_vector_metrics(pred_geoms, gt_geoms, iou_threshold=0.5)
        metrics["predicted_feature_count"] = len(pred_geoms)
        metrics["reference_feature_count"] = len(gt_geoms)
        return True, str(valid_ref_path), metrics
    except Exception as e:
        logger.warning(f"Failed computing vector metrics with reference {valid_ref_path}: {e}")
        return False, str(valid_ref_path), None


def fetch_layer_quality_report(identifier: str | int):
    try:
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                is_numeric = isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit())
                if is_numeric:
                    layer_id_param = int(identifier)
                    layer_name_param = None
                else:
                    layer_id_param = None
                    layer_name_param = str(identifier)

                sql = """
                SELECT
                    lv.id AS layer_id,
                    lv.layer_name,
                    lv.version,
                    lv.feature_type,
                    COUNT(vf.id) AS total_features,
                    COUNT(CASE WHEN vf.qc_status = 'pending' OR vf.qc_status IS NULL THEN 1 END) AS pending_count,
                    COUNT(CASE WHEN vf.qc_status = 'accepted' THEN 1 END) AS accepted_count,
                    COUNT(CASE WHEN vf.qc_status = 'edited' THEN 1 END) AS edited_count,
                    COUNT(CASE WHEN vf.qc_status = 'rejected' THEN 1 END) AS rejected_count,
                    COALESCE(MIN(vf.confidence), 0.0) AS min_confidence,
                    COALESCE(MAX(vf.confidence), 0.0) AS max_confidence,
                    COALESCE(AVG(vf.confidence), 0.0) AS mean_confidence,
                    COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vf.confidence), 0.0) AS median_confidence,
                    COUNT(CASE WHEN vf.confidence >= 0.90 THEN 1 END) AS high_confidence_count,
                    COUNT(CASE WHEN vf.confidence >= 0.75 AND vf.confidence < 0.90 THEN 1 END) AS medium_confidence_count,
                    COUNT(CASE WHEN vf.confidence < 0.50 THEN 1 END) AS low_confidence_count,
                    COUNT(CASE WHEN ST_IsValid(vf.geometry) THEN 1 END) AS valid_geometry_count,
                    COUNT(CASE WHEN NOT ST_IsValid(vf.geometry) THEN 1 END) AS invalid_geometry_count
                FROM layer_versions lv
                LEFT JOIN vector_features vf ON vf.layer_version_id = lv.id
                WHERE (%s::TEXT IS NOT NULL AND lv.layer_name = %s)
                   OR (%s::INT IS NOT NULL AND lv.id = %s)
                GROUP BY lv.id, lv.layer_name, lv.version, lv.feature_type
                ORDER BY lv.version DESC
                LIMIT 1;
                """
                cur.execute(sql, (layer_name_param, layer_name_param, layer_id_param, layer_id_param))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Layer not found.")

                (
                    lid, lname, ver, ftype, total,
                    pend, acc, ed, rej,
                    c_min, c_max, c_mean, c_med,
                    c_high, c_med_thresh, c_low,
                    g_valid, g_invalid
                ) = row

                total_int = int(total or 0)
                pend_pct = round((pend / total_int * 100), 2) if total_int > 0 else 0.0
                acc_pct = round((acc / total_int * 100), 2) if total_int > 0 else 0.0
                ed_pct = round((ed / total_int * 100), 2) if total_int > 0 else 0.0
                rej_pct = round((rej / total_int * 100), 2) if total_int > 0 else 0.0

                cur.execute("""
                    SELECT ST_AsGeoJSON(vf.geometry)
                    FROM vector_features vf
                    WHERE vf.layer_version_id = %s;
                """, (lid,))
                pred_rows = cur.fetchall()
                pred_geoms = [shape(json.loads(r[0])) for r in pred_rows if r[0]]

                gt_avail, gt_ref_src, gt_metrics = get_ground_truth_reference_and_metrics(ftype, pred_geoms)

                res_dict = {
                    "status": "success",
                    "layer_id": lid,
                    "layer_name": lname,
                    "version": ver,
                    "feature_type": ftype,
                    "total_features": total_int,
                    "qc": {
                        "pending": int(pend or 0),
                        "accepted": int(acc or 0),
                        "edited": int(ed or 0),
                        "rejected": int(rej or 0),
                        "pending_pct": pend_pct,
                        "accepted_pct": acc_pct,
                        "edited_pct": ed_pct,
                        "rejected_pct": rej_pct,
                    },
                    "confidence": {
                        "min": round(float(c_min or 0.0), 4),
                        "max": round(float(c_max or 0.0), 4),
                        "mean": round(float(c_mean or 0.0), 4),
                        "median": round(float(c_med or 0.0), 4),
                        "high_confidence": int(c_high or 0),
                        "medium_confidence": int(c_med_thresh or 0),
                        "low_confidence": int(c_low or 0),
                    },
                    "geometry": {
                        "valid": int(g_valid or 0),
                        "invalid": int(g_invalid or 0),
                    },
                    "ground_truth_accuracy_available": gt_avail,
                }
                if gt_avail and gt_metrics:
                    res_dict["reference_source"] = gt_ref_src
                    res_dict["ground_truth_metrics"] = gt_metrics

                return res_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message": "Failed to generate quality report.", "error": str(e)})


@app.get("/accuracy-report/{layer_id}")
def get_accuracy_report_by_id(layer_id: str):
    return fetch_layer_quality_report(layer_id)


@app.get("/accuracy/report")
def get_accuracy_report(layer_name: Optional[str] = Query(None), layer_id: Optional[str] = Query(None)):
    target = layer_id or layer_name or "uploaded_buildings"
    return fetch_layer_quality_report(target)


@app.post("/accuracy/evaluate")
def evaluate_ground_truth_accuracy(
    layer_name: Optional[str] = Query(None),
    layer_id: Optional[str] = Query(None),
    reference_file_path: Optional[str] = Query(None)
):
    target_layer = layer_id or layer_name or "uploaded_buildings"
    try:
        with get_postgis_connection() as conn:
            with conn.cursor() as cur:
                is_numeric = isinstance(target_layer, int) or (isinstance(target_layer, str) and target_layer.isdigit())
                if is_numeric:
                    where_sql = "lv.id = %s"
                    param = int(target_layer)
                else:
                    where_sql = "lv.layer_name = %s"
                    param = str(target_layer)

                cur.execute(f"""
                    SELECT lv.id, lv.layer_name, lv.feature_type
                    FROM layer_versions lv
                    WHERE {where_sql}
                    ORDER BY lv.version DESC
                    LIMIT 1;
                """, (param,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Layer not found.")

                lid, lname, ftype = row

                cur.execute("""
                    SELECT ST_AsGeoJSON(vf.geometry)
                    FROM vector_features vf
                    WHERE vf.layer_version_id = %s;
                """, (lid,))
                pred_rows = cur.fetchall()
                pred_geoms = [shape(json.loads(r[0])) for r in pred_rows if r[0]]

                gt_avail, gt_ref_src, gt_metrics = get_ground_truth_reference_and_metrics(
                    ftype, pred_geoms, user_ref_path=reference_file_path
                )

                res = {
                    "status": "success",
                    "layer_id": lid,
                    "layer_name": lname,
                    "feature_type": ftype,
                    "ground_truth_accuracy_available": gt_avail,
                    "reference_source": gt_ref_src,
                }
                if gt_avail and gt_metrics:
                    res["metrics"] = gt_metrics

                return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ground truth evaluation failed: {e}")
