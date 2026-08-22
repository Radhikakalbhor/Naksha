FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        python3-gdal \
        libgl1 \
        libglib2.0-0 \
        libgthread-2.0-0 \
        libxcb1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    rasterio \
    numpy \
    "psycopg[binary]" \
    GDAL==3.10.3 \
    torch \
    torchvision \
    opencv-python==4.11.0.86 \
    onnxruntime \
    PyYAML==6.0.2 \
    shapely==2.1.1 \
    huggingface-hub==0.32.4 \
    tornado==6.5.1 \
    tqdm==4.67.1 \
    ultralytics==8.3.148 \
    numba==0.62.1 \
    deepforest==2.1.0 \
    scikit-image==0.26.0 \
    transformers \
    geopandas \
    celery \
    redis

WORKDIR /app

ENV SKYWATER_MODEL_DIR=/app/models/skywater
ENV SKYWATER_MODEL_FILE=skywater_segformer_b2_fp32.onnx

RUN mkdir -p ${SKYWATER_MODEL_DIR} && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Realcat/skywater_seg', filename='skywater_segformer_b2_fp32.onnx', local_dir='${SKYWATER_MODEL_DIR}', local_dir_use_symlinks=False)"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]