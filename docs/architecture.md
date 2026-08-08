# Naksha - System Architecture

## Overview

Naksha is an AI-powered Orthophoto Digitization Platform that automatically detects, classifies, vectorizes, validates, and exports geospatial features from orthophotos and related datasets.

---

## High-Level Architecture

Client Layer
- React + TypeScript SPA
- MapLibre GL / Deck.gl Viewer
- QC Console

↓

API Gateway
- FastAPI
- REST API
- Authentication
- Request Validation

↓

Core Services
- Authentication & User Management
- File Upload & Storage
- Job Orchestration
- Metadata Service

↓

Object Storage
- MinIO (Development)
- AWS S3 (Production)

↓

AI Inference Engine
- Preprocessing
- Model Serving
- Building Detection
- Road Detection
- Tree Detection
- Farm Boundary Detection
- Water Detection
- LULC Classification

↓

GIS Engine
- Raster to Vector
- Confidence Scoring
- Topology Cleanup
- CRS Management

↓

Databases
- PostgreSQL
- PostGIS

↓

Export Engine
- Shapefile
- GeoJSON
- GeoPackage
- FileGDB

↓

QC / Validation Module

---

## Technology Stack

Frontend
- React
- TypeScript
- MapLibre GL
- Deck.gl

Backend
- FastAPI
- Pydantic

AI
- PyTorch
- ONNX Runtime

GIS
- GDAL
- Rasterio
- GeoPandas
- Shapely
- Fiona

Storage
- PostgreSQL
- PostGIS
- MinIO

Infrastructure
- Docker
- Docker Compose

---

## Processing Pipeline

1. Upload Orthophoto
2. Store in Object Storage
3. Register Metadata
4. Create Processing Job
5. Tile & Preprocess Imagery
6. AI Model Inference
7. Raster to Vector Conversion
8. Confidence Scoring
9. Store in PostGIS
10. Human QC
11. Export GIS Files

---

## Deliverables

- Building Footprints
- Road Network
- Farm Boundaries
- Tree Inventory
- Water Bodies
- LULC Classification
- Accuracy Report
- GIS Exports