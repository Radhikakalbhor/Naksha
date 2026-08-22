# ============================================================
# NAKSHA - MINIO OBJECT STORAGE UTILITIES (Task 1)
# ============================================================

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000").replace("http://", "").replace("https://", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "naksha"))
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "naksha_dev_password"))
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "naksha-rasters")


def get_minio_client():
    """
    Initialize and return a Minio client instance, or None if unavailable.
    Uses fast socket checking to select reachable endpoint (minio:9000 in Docker, 127.0.0.1:9000 on host).
    """
    try:
        from minio import Minio
        import socket

        access_key = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "naksha"))
        secret_key = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "naksha_dev_password"))

        endpoints_to_try = [MINIO_ENDPOINT]
        if MINIO_ENDPOINT not in ("127.0.0.1:9000", "localhost:9000"):
            endpoints_to_try.append("127.0.0.1:9000")

        reachable_endpoint = None
        for ep in endpoints_to_try:
            parts = ep.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 9000
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    reachable_endpoint = ep
                    break
            except Exception:
                continue

        if not reachable_endpoint:
            return None

        client = Minio(
            reachable_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        return client
    except Exception as e:
        logger.warning(f"Failed to initialize MinIO client ({e}). Local storage fallback will be used.")
        return None


def ensure_bucket_exists(client, bucket_name: str = MINIO_BUCKET) -> bool:
    """
    Ensure the target bucket exists in MinIO.
    """
    if client is None:
        return False
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
        return True
    except Exception as e:
        logger.warning(f"MinIO bucket check/create failed ({e}).")
        return False


def upload_raster_to_minio(
    file_path: str | Path,
    job_id: str,
    category: str = "raw",
    bucket_name: str = MINIO_BUCKET,
) -> Tuple[Optional[str], str]:
    """
    Upload a raster file to MinIO with key `{category}/{job_id}/{filename}`.
    Returns (minio_key, status) where status is 'uploaded' or 'local_fallback'.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None, "file_not_found"

    object_name = f"{category}/{job_id}/{file_path.name}"

    client = get_minio_client()
    if client and ensure_bucket_exists(client, bucket_name):
        try:
            client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(file_path),
            )
            logger.info(f"Successfully uploaded {file_path.name} to MinIO bucket '{bucket_name}' with key '{object_name}'.")
            return object_name, "uploaded"
        except Exception as e:
            logger.warning(f"MinIO fput_object failed ({e}). Retaining local copy.")
            return object_name, "local_fallback"

    return object_name, "local_fallback"
