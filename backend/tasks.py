import os
import json
import logging
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

@task_decorator(name="process_building_inference")
def process_building_inference_task(job_id: str, file_path: str):
    logger.info(f"Starting async building inference for job_id={job_id}, file={file_path}")
    try:
        from main import get_building_model, building_sliding_window_inference, get_postgis_connection, BUILDING_PATCH_SIZE, BUILDING_STRIDE
        from gis_engine.vectorization.vectorize import polygonize_mask, feature_confidence
        from gis_engine.topology.geometry import validate_geometry
        from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql
        import rasterio
        import numpy as np

        input_path = Path(file_path)

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
                                source_model="building_unet_resnet34",
                            )
                        )
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
        JOBS_STORE[job_id] = result
        return result

    except Exception as exc:
        logger.error(f"Async building inference failed for job_id={job_id}: {exc}", exc_info=True)
        error_result = {
            "status": "failed",
            "job_id": job_id,
            "error": str(exc)
        }
        JOBS_STORE[job_id] = error_result
        return error_result
