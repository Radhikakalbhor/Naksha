import os
import json
import zipfile
import tempfile
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import shapely.geometry
from shapely.geometry import Polygon, LineString, Point


def run_tests():
    print("============================================================")
    print("NAKSHA BACKEND AUDIT & RELIABILITY TEST SUITE")
    print("============================================================")

    # 1. Imports check
    print("\n--- 1. Testing Backend Imports ---")
    import main
    import export_engine
    import water_inference
    import lulc_inference
    import lulc_endpoint
    from gis_engine.vectorization.vectorize import (
        polygonize_mask,
        feature_confidence,
        skeletonize_road_mask,
        skeleton_to_lines,
    )
    from gis_engine.topology.geometry import validate_geometry, simplify_geometry
    from gis_engine.postgis.export import create_layer_version_sql, create_feature_sql

    print("[PASS] All backend modules imported cleanly without errors.")

    # 2. Polygonization & Confidence Test
    print("\n--- 2. Testing Vectorization & Confidence Utilities ---")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 20:50] = 1
    prob = np.full((100, 100), 0.85, dtype=np.float32)

    import rasterio.transform
    transform = rasterio.transform.from_origin(0, 100, 1, 1)

    geoms = polygonize_mask(mask, transform)
    assert len(geoms) == 1, f"Expected 1 polygon, got {len(geoms)}"
    conf = feature_confidence(prob, geoms[0], transform)
    assert conf is not None and abs(conf - 0.85) < 1e-4, f"Expected confidence ~0.85, got {conf}"
    print(f"[PASS] polygonize_mask produced {len(geoms)} geometry, confidence = {conf:.4f}")

    # 3. Geometry Validation Test
    print("\n--- 3. Testing Geometry Validation ---")
    # Self-intersecting polygon (bowtie)
    invalid_poly = Polygon([(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)])
    valid_poly = validate_geometry(invalid_poly)
    assert valid_poly is not None and valid_poly.is_valid, "Failed to repair invalid polygon"
    print("[PASS] validate_geometry repaired self-intersecting polygon successfully.")

    # 4. SQL Generator Test
    print("\n--- 4. Testing PostGIS SQL Generators ---")
    sql_version = create_layer_version_sql("uploaded_buildings", "buildings", 1)
    assert "INSERT INTO layer_versions" in sql_version and "'uploaded_buildings'" in sql_version

    sql_feature = create_feature_sql(1, "buildings", Point(10, 20), 0.92)
    assert "INSERT INTO vector_features" in sql_feature and "0.92" in sql_feature
    print("[PASS] PostGIS SQL generation validated.")

    # 5. Road Skeletonization Test
    print("\n--- 5. Testing Road Skeletonization & Centerline Extraction ---")
    road_mask = np.zeros((100, 100), dtype=np.uint8)
    road_mask[48:52, :] = 1  # Horizontal road strip
    skel = skeletonize_road_mask(road_mask)
    lines = skeleton_to_lines(skel, transform)
    assert len(lines) > 0, "Failed to extract road centerlines"
    print(f"[PASS] Road skeletonization extracted {len(lines)} centerline segment(s).")

    # 6. Export Engine Empty Feature Test
    print("\n--- 6. Testing Export Engine Guard Rails & QC Filtering ---")
    class MockConn:
        def cursor(self):
            class MockCursor:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def execute(self, sql, params=()): pass
                def fetchall(self): return [] # Empty features
            return MockCursor()

    try:
        export_engine.export_layer(MockConn(), "demo_buildings", "geojson", Path(tempfile.gettempdir()))
        assert False, "Should have raised ValueError on empty layer"
    except ValueError as e:
        print(f"[PASS] Cleanly caught empty layer export error: {e}")

    # 7. Celery Tasks & Async Jobs Import Test
    print("\n--- 7. Testing Celery Tasks & Preprocessing Utilities ---")
    import tasks
    assert hasattr(tasks, "process_building_inference_task"), "Missing process_building_inference_task"
    from preprocessing.crs_utils import normalize_crs
    from preprocessing.tiler import tile_raster
    print("[PASS] Celery tasks, crs_utils, and tiler imported cleanly.")

    # 8. QC & Feature SQL Generator Test
    print("\n--- 8. Testing Feature SQL with QC Status ---")
    sql_feature_qc = create_feature_sql(1, "buildings", Point(10, 20), 0.95, qc_status="accepted")
    assert "'accepted'" in sql_feature_qc and "qc_status" in sql_feature_qc
    print("[PASS] Feature SQL with qc_status validated.")

    # 9. COG & MinIO Storage Utility Test
    print("\n--- 9. Testing COG & MinIO Storage Utilities ---")
    from preprocessing.cog import is_cog, convert_to_cog
    from gis_engine.storage import get_minio_client, upload_raster_to_minio
    print("[PASS] COG conversion and MinIO storage utilities imported successfully.")

    print("\n============================================================")
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("============================================================")

if __name__ == "__main__":
    run_tests()
