# ============================================================
# NAKSHA - CENTRALIZED CONFIGURATION & LOGGING (Day 15)
# ============================================================

import os
import logging
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/data") if Path("/data").exists() else BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Database Configuration (PostGIS)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "naksha")
POSTGRES_USER = os.getenv("POSTGRES_USER", "naksha")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "naksha_dev")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

POSTGIS_CONFIG = {
    "host": POSTGRES_HOST,
    "dbname": POSTGRES_DB,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "port": POSTGRES_PORT,
}

# Redis & Celery Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# MinIO Storage Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000").replace("http://", "").replace("https://", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "naksha"))
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "naksha_dev_password"))
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "naksha-rasters")

# Model Configuration
SKYWATER_MODEL_DIR = Path(os.getenv("SKYWATER_MODEL_DIR", MODELS_DIR / "skywater"))
SKYWATER_MODEL_FILE = os.getenv("SKYWATER_MODEL_FILE", "skywater_segformer_b2_fp32.onnx")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("naksha")
