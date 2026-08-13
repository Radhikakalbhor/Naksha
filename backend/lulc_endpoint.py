import shutil
import uuid
from pathlib import Path

import numpy as np
import rasterio
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.responses import FileResponse

from lulc_inference import run_lulc_inference
from gis_engine.vectorization.vectorize import (
    polygonize_mask,
    feature_confidence,
)
from gis_engine.topology.geometry import validate_geometry
from gis_engine.postgis.export import (
    create_layer_version_sql,
    create_feature_sql,
)


router = APIRouter()


LULC_OUTPUT_DIR = Path(
    "/data/predictions/lulc"
)

LULC_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LULC_CLASSES = {
    0: "urban_land",
    1: "agriculture_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
    6: "unknown",
}


def get_postgis_connection():
    import psycopg

    return psycopg.connect(
        host="naksha-postgres",
        dbname="naksha",
        user="naksha",
        password="naksha_dev",
        port=5432,
    )


@router.post("/inference/lulc")
async def lulc_inference(
    file: UploadFile = File(...),
):
    """
    Run multi-class LULC inference and store
    polygonized features in PostGIS.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in {
        ".tif",
        ".tiff",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only .tif and .tiff files "
                "are supported."
            ),
        )

    job_id = uuid.uuid4().hex[:8]

    input_path = (
        LULC_OUTPUT_DIR
        / f"lulc_input_{job_id}{extension}"
    )

    raster_output_path = (
        LULC_OUTPUT_DIR
        / f"Naksha_{job_id}_lulc.tif"
    )

    probability_output_path = (
        LULC_OUTPUT_DIR
        / f"Naksha_{job_id}_lulc_probability.tif"
    )

    try:

        # --------------------------------------------------
        # Save input
        # --------------------------------------------------

        with input_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # --------------------------------------------------
        # LULC inference
        # --------------------------------------------------

        (
            prediction_map,
            probability_map,
            transform,
        ) = run_lulc_inference(
            input_tiff_path=input_path,
            output_tiff_path=raster_output_path,
        )

        # --------------------------------------------------
        # Save probability raster
        # --------------------------------------------------

        with rasterio.open(
            input_path
        ) as src:

            probability_profile = (
                src.profile.copy()
            )

        probability_profile.update(
            count=1,
            dtype="float32",
            nodata=0,
            compress="lzw",
        )

        with rasterio.open(
            probability_output_path,
            "w",
            **probability_profile,
        ) as dst:

            dst.write(
                probability_map.astype(
                    np.float32
                ),
                1,
            )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        height, width = (
            prediction_map.shape
        )

        total_pixels = (
            prediction_map.size
        )

        class_statistics = {}

        for class_id, class_name in (
            LULC_CLASSES.items()
        ):

            pixel_count = int(
                np.count_nonzero(
                    prediction_map == class_id
                )
            )

            class_statistics[
                class_name
            ] = {
                "class_id": class_id,
                "pixel_count": pixel_count,
                "coverage_percent": round(
                    (
                        pixel_count
                        / total_pixels
                        * 100
                    ),
                    4,
                ),
            }

        # --------------------------------------------------
        # PostGIS
        # --------------------------------------------------

        features_stored = 0
        layer_version_id = None
        postgis_status = "skipped"
        postgis_error = None

        try:

            geometries_by_class = {}

            for class_id, class_name in (
                LULC_CLASSES.items()
            ):

                class_mask = (
                    prediction_map == class_id
                ).astype(
                    np.uint8
                )

                if not np.any(class_mask):
                    continue

                geometries = polygonize_mask(
                    class_mask,
                    transform,
                    min_area=0.0,
                )

                geometries_by_class[
                    class_name
                ] = geometries

            with get_postgis_connection() as conn:

                with conn.cursor() as cur:

                    # Determine next version safely.
                    cur.execute(
                        """
                        SELECT COALESCE(
                            MAX(version),
                            0
                        ) + 1
                        FROM layer_versions
                        WHERE layer_name = %s
                        """,
                        (
                            "uploaded_lulc",
                        ),
                    )

                    version = cur.fetchone()[0]

                    cur.execute(
                        create_layer_version_sql(
                            layer_name="uploaded_lulc",
                            feature_type="lulc",
                            version=version,
                        )
                    )

                    layer_version_id = (
                        cur.fetchone()[0]
                    )

                    for class_name, geometries in (
                        geometries_by_class.items()
                    ):

                        for geometry in geometries:

                            geometry = (
                                validate_geometry(
                                    geometry
                                )
                            )

                            if geometry is None:
                                continue

                            confidence = (
                                feature_confidence(
                                    probability=probability_map,
                                    geometry=geometry,
                                    transform=transform,
                                )
                            )

                            if confidence is None:
                                continue

                            cur.execute(
                                create_feature_sql(
                                    layer_version_id=(
                                        layer_version_id
                                    ),
                                    feature_type=(
                                        class_name
                                    ),
                                    geometry=geometry,
                                    confidence=(
                                        confidence
                                    ),
                                )
                            )

                            features_stored += 1

                conn.commit()

            postgis_status = "success"

        except Exception as exc:

            postgis_status = "failed"
            postgis_error = str(exc)

        return {
            "status": "success",
            "job_id": job_id,
            "input_file": file.filename,
            "model": (
                "SegFormer-B0 "
                "DeepGlobe Land Cover"
            ),
            "preprocessing": (
                "RGB bands 1,2,3 + "
                "ImageNet normalization + "
                "224x224 tiling"
            ),
            "classes": LULC_CLASSES,
            "output": {
                "raster": str(
                    raster_output_path
                ),
                "probability": str(
                    probability_output_path
                ),
            },
            "statistics": {
                "image_width": width,
                "image_height": height,
                "total_pixels": total_pixels,
                "classes": class_statistics,
                "probability_min": float(
                    probability_map.min()
                ),
                "probability_max": float(
                    probability_map.max()
                ),
                "probability_mean": float(
                    probability_map.mean()
                ),
            },
            "features_stored": features_stored,
            "layer_version_id": layer_version_id,
            "postgis_status": postgis_status,
            "postgis_error": postgis_error,
        }

    except Exception as exc:

        if input_path.exists():
            input_path.unlink(
                missing_ok=True
            )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "LULC inference failed.",
                "error": str(exc),
            },

        )


@router.get("/inference/lulc/{job_id}/raster")
async def download_lulc_raster(
    job_id: str,
):
    raster_path = (
        LULC_OUTPUT_DIR
        / f"Naksha_{job_id}_lulc.tif"
    )

    if not raster_path.exists():
        raise HTTPException(
            status_code=404,
            detail="LULC raster not found.",
        )

    return FileResponse(
        path=str(raster_path),
        media_type="image/tiff",
        filename=raster_path.name,
    )
