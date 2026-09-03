import os
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

try:
    from celery import Celery
    celery_app = Celery(
        "naksha",
        broker=REDIS_URL,
        backend=REDIS_URL
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
    )
    task_decorator = celery_app.task
except ImportError:
    celery_app = None
    def task_decorator(*args, **kwargs):
        def decorator(func):
            func.delay = lambda *a, **k: func(*a, **k)
            return func
        return decorator

JOBS_STORE = {}


def run_inference_job(job_id: str, feature_type: str, file_path: str):
    logger.info(f"Starting async inference for job_id={job_id}, type={feature_type}, file={file_path}")
    ftype = (feature_type or "buildings").lower().strip()
    input_path = Path(file_path)

    try:
        if ftype in ("buildings", "building"):
            from main import get_building_model, building_sliding_window_inference, get_postgis_connection, BUILDING_PATCH_SIZE, BUILDING_STRIDE
            from gis_engine.vectorization.vectorize import polygonize_mask, feature_confidence
            from gis_engine.topology.geometry import validate_geometry
            from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql
            import rasterio

            with rasterio.open(str(input_path)) as src:
                src_transform = src.transform

            model = get_building_model()
            image, mask, prediction = building_sliding_window_inference(
                model=model,
                image_path=input_path,
                patch_size=BUILDING_PATCH_SIZE,
                stride=BUILDING_STRIDE
            )

            geometries = polygonize_mask(mask, src_transform, min_area=0.0)
            features_stored = 0
            layer_version_id = None

            if geometries:
                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM layer_versions WHERE layer_name = 'uploaded_buildings'")
                        next_version = cur.fetchone()[0]
                        cur.execute(create_layer_version_sql("uploaded_buildings", "buildings", next_version))
                        layer_version_id = cur.fetchone()[0]

                        for geometry in geometries:
                            geometry = validate_geometry(geometry)
                            if geometry is None:
                                continue
                            confidence = feature_confidence(probability=prediction, geometry=geometry, transform=src_transform)
                            if confidence is None:
                                continue
                            cur.execute(create_feature_sql(layer_version_id, "buildings", geometry, confidence, source_model="building_unet_resnet34"))
                            features_stored += 1
                    conn.commit()

            result = {
                "status": "completed",
                "job_id": job_id,
                "feature_type": "buildings",
                "features_stored": features_stored,
                "layer_version_id": layer_version_id,
                "building_pixels": int(mask.sum()),
            }

        elif ftype in ("roads", "road"):
            import cv2
            import torch
            import numpy as np
            import rasterio
            from main import get_road_model, get_postgis_connection, ROAD_TARGET_SIZE, ROAD_THRESHOLD, DEVICE
            from gis_engine.vectorization.vectorize import skeletonize_road_mask, skeleton_to_lines
            from gis_engine.topology.geometry import validate_geometry
            from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql

            with rasterio.open(str(input_path)) as src:
                src_transform = src.transform

            image_bgr = cv2.imread(str(input_path))
            if image_bgr is None:
                raise RuntimeError(f"Could not read input image {input_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, ROAD_TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            input_array = (image_resized.astype(np.float32) / 255.0) * 3.2 - 1.6
            input_array = np.transpose(input_array, (2, 0, 1))
            input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(DEVICE)

            model = get_road_model()
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()

            mask = (prob >= ROAD_THRESHOLD).astype(np.uint8)
            skel = skeletonize_road_mask(mask)
            lines = skeleton_to_lines(skel, src_transform)

            features_stored = 0
            layer_version_id = None
            if lines:
                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM layer_versions WHERE layer_name = 'uploaded_roads'")
                        next_version = cur.fetchone()[0]
                        cur.execute(create_layer_version_sql("uploaded_roads", "roads", next_version))
                        layer_version_id = cur.fetchone()[0]

                        for line in lines:
                            val_line = validate_geometry(line)
                            if val_line is None or val_line.is_empty:
                                continue
                            cur.execute(create_feature_sql(layer_version_id, "roads", val_line, confidence=0.85, source_model="dlinknet34"))
                            features_stored += 1
                    conn.commit()

            result = {
                "status": "completed",
                "job_id": job_id,
                "feature_type": "roads",
                "features_stored": features_stored,
                "layer_version_id": layer_version_id,
                "road_pixels": int(mask.sum()),
            }

        elif ftype in ("trees", "tree"):
            import rasterio
            from shapely.geometry import Polygon
            from main import get_tree_model, get_postgis_connection, TREE_PATCH_SIZE, TREE_PATCH_OVERLAP
            from gis_engine.topology.geometry import validate_geometry
            from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql

            model = get_tree_model()
            predictions = model.predict_tile(path=str(input_path), patch_size=TREE_PATCH_SIZE, patch_overlap=TREE_PATCH_OVERLAP)

            features_stored = 0
            layer_version_id = None

            if predictions is not None and len(predictions) > 0:
                with rasterio.open(str(input_path)) as src:
                    src_transform = src.transform

                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM layer_versions WHERE layer_name = 'uploaded_trees'")
                        next_version = cur.fetchone()[0]
                        cur.execute(create_layer_version_sql("uploaded_trees", "trees", next_version))
                        layer_version_id = cur.fetchone()[0]

                        for _, row in predictions.iterrows():
                            xmin, ymin, xmax, ymax = float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])
                            left, top = rasterio.transform.xy(src_transform, ymin, xmin, offset="ul")
                            right, bottom = rasterio.transform.xy(src_transform, ymax, xmax, offset="ul")
                            poly = Polygon([[left, top], [right, top], [right, bottom], [left, bottom], [left, top]])
                            val_poly = validate_geometry(poly)
                            if val_poly is None:
                                continue
                            score = float(row["score"]) if "score" in row else 0.8
                            cur.execute(create_feature_sql(layer_version_id, "trees", val_poly, confidence=score, source_model="deepforest"))
                            features_stored += 1
                    conn.commit()

            result = {
                "status": "completed",
                "job_id": job_id,
                "feature_type": "trees",
                "features_stored": features_stored,
                "layer_version_id": layer_version_id,
                "tree_count": features_stored,
            }

        elif ftype in ("water",):
            from water_inference import run_skywater_inference
            from main import get_postgis_connection, WATER_OUTPUT_DIR
            from gis_engine.vectorization.vectorize import polygonize_mask, feature_confidence
            from gis_engine.topology.geometry import validate_geometry
            from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql

            mask_output_path = WATER_OUTPUT_DIR / f"Naksha_{job_id}_water_mask.tif"
            water_mask, water_probability, transform = run_skywater_inference(
                image_path=str(input_path), output_path=str(mask_output_path)
            )

            geometries = polygonize_mask(water_mask, transform, min_area=0.0)
            features_stored = 0
            layer_version_id = None

            if geometries:
                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM layer_versions WHERE layer_name = 'uploaded_water'")
                        next_version = cur.fetchone()[0]
                        cur.execute(create_layer_version_sql("uploaded_water", "water", next_version))
                        layer_version_id = cur.fetchone()[0]

                        for geom in geometries:
                            val_geom = validate_geometry(geom)
                            if val_geom is None:
                                continue
                            conf = feature_confidence(water_probability, val_geom, transform)
                            if conf is None:
                                continue
                            cur.execute(create_feature_sql(layer_version_id, "water", val_geom, confidence=conf, source_model="skywater_segformer_b2"))
                            features_stored += 1
                    conn.commit()

            result = {
                "status": "completed",
                "job_id": job_id,
                "feature_type": "water",
                "features_stored": features_stored,
                "layer_version_id": layer_version_id,
                "water_pixels": int(water_mask.sum()),
            }

        elif ftype in ("lulc",):
            from lulc_inference import run_lulc_inference
            from main import get_postgis_connection, DATA_DIR
            from gis_engine.topology.geometry import validate_geometry
            from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql

            lulc_output_dir = DATA_DIR / "predictions" / "lulc"
            lulc_output_dir.mkdir(parents=True, exist_ok=True)

            raster_output, prob_output, stats, geoms_by_class, transform = run_lulc_inference(
                file_path=str(input_path), output_dir=lulc_output_dir, job_id=job_id
            )

            features_stored = 0
            layer_version_id = None

            if geoms_by_class:
                with get_postgis_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM layer_versions WHERE layer_name = 'uploaded_lulc'")
                        next_version = cur.fetchone()[0]
                        cur.execute(create_layer_version_sql("uploaded_lulc", "lulc", next_version))
                        layer_version_id = cur.fetchone()[0]

                        for class_name, items in geoms_by_class.items():
                            for geom, conf in items:
                                val_geom = validate_geometry(geom)
                                if val_geom is None:
                                    continue
                                cur.execute(create_feature_sql(layer_version_id, "lulc", val_geom, confidence=float(conf), source_model="segformer_b0_lulc"))
                                features_stored += 1
                    conn.commit()

            result = {
                "status": "completed",
                "job_id": job_id,
                "feature_type": "lulc",
                "features_stored": features_stored,
                "layer_version_id": layer_version_id,
            }

        elif ftype in ("fields", "field", "farms", "farm"):
            import geopandas as gpd
            from main import get_postgis_connection, FIELD_MODEL_DIR, FIELD_INPUT_DIR, FIELD_OUTPUT_DIR, FIELD_BATCH_CONFIG
            from gis_engine.topology.geometry import validate_geometry
            from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql

            job_name = f"Naksha_{job_id}"
            input_folder = FIELD_INPUT_DIR / job_name
            input_folder.mkdir(parents=True, exist_ok=True)
            dest_path = input_folder / input_path.name
            if input_path.resolve() != dest_path.resolve():
                shutil.copy2(input_path, dest_path)

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
            FIELD_BATCH_CONFIG.write_text(batch_config, encoding="utf-8")

            res_cmd = subprocess.run(
                [sys.executable, "delineate.py", "-b", str(FIELD_BATCH_CONFIG)],
                cwd=str(FIELD_MODEL_DIR), capture_output=True, text=True
            )

            gpkg_path = FIELD_OUTPUT_DIR / f"{job_name}.gpkg"
            simplified_gpkg_path = FIELD_OUTPUT_DIR / f"{job_name}.simp.gpkg"
            gpkg_to_read = simplified_gpkg_path if simplified_gpkg_path.exists() else gpkg_path

            features_stored = 0
            layer_version_id = None

            if gpkg_to_read.exists():
                gdf = gpd.read_file(gpkg_to_read)
                if len(gdf) > 0:
                    with get_postgis_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM layer_versions WHERE layer_name = 'uploaded_farms'")
                            next_version = cur.fetchone()[0]
                            cur.execute(create_layer_version_sql("uploaded_farms", "farms", next_version))
                            layer_version_id = cur.fetchone()[0]

                            for _, row in gdf.iterrows():
                                geom = row.geometry
                                if geom is None or geom.is_empty:
                                    continue
                                val_geom = validate_geometry(geom)
                                if val_geom is None:
                                    continue
                                cur.execute(create_feature_sql(layer_version_id, "farms", val_geom, confidence=0.85, source_model="delineate_anything"))
                                features_stored += 1
                        conn.commit()

            result = {
                "status": "completed",
                "job_id": job_id,
                "feature_type": "fields",
                "features_stored": features_stored,
                "layer_version_id": layer_version_id,
            }
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

        JOBS_STORE[job_id] = result
        try:
            from main import save_job_to_db
            save_job_to_db(job_id, feature_type, "completed", str(file_path), result=result)
        except Exception:
            pass
        return result

    except Exception as exc:
        logger.error(f"Async inference failed for job_id={job_id}, type={feature_type}: {exc}", exc_info=True)
        error_result = {
            "status": "failed",
            "job_id": job_id,
            "feature_type": feature_type,
            "error": str(exc)
        }
        JOBS_STORE[job_id] = error_result
        try:
            from main import save_job_to_db
            save_job_to_db(job_id, feature_type, "failed", str(file_path), result=error_result)
        except Exception:
            pass
        return error_result


@task_decorator(name="process_building_inference")
def process_building_inference_task(job_id: str, file_path: str):
    return run_inference_job(job_id, "buildings", file_path)


@task_decorator(name="process_road_inference")
def process_road_inference_task(job_id: str, file_path: str):
    return run_inference_job(job_id, "roads", file_path)


@task_decorator(name="process_tree_inference")
def process_tree_inference_task(job_id: str, file_path: str):
    return run_inference_job(job_id, "trees", file_path)


@task_decorator(name="process_water_inference")
def process_water_inference_task(job_id: str, file_path: str):
    return run_inference_job(job_id, "water", file_path)


@task_decorator(name="process_lulc_inference")
def process_lulc_inference_task(job_id: str, file_path: str):
    return run_inference_job(job_id, "lulc", file_path)


@task_decorator(name="process_field_inference")
def process_field_inference_task(job_id: str, file_path: str):
    return run_inference_job(job_id, "fields", file_path)


FEATURE_TASK_MAP = {
    "buildings": process_building_inference_task,
    "building": process_building_inference_task,
    "roads": process_road_inference_task,
    "road": process_road_inference_task,
    "trees": process_tree_inference_task,
    "tree": process_tree_inference_task,
    "water": process_water_inference_task,
    "lulc": process_lulc_inference_task,
    "fields": process_field_inference_task,
    "field": process_field_inference_task,
    "farms": process_field_inference_task,
    "farm": process_field_inference_task,
}


def dispatch_task(job_id: str, feature_type: str, file_path: str):
    ftype = (feature_type or "buildings").lower().strip()
    task_fn = FEATURE_TASK_MAP.get(ftype, process_building_inference_task)
    return task_fn.delay(job_id, file_path)
