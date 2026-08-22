-- ============================================================
-- NAKSHA - POSTGIS VECTOR LAYER SCHEMA
-- Day 5 - Versioned vector layers
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS layer_versions (
    id BIGSERIAL PRIMARY KEY,
    layer_name TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (layer_name, version)
);

CREATE TABLE IF NOT EXISTS vector_features (
    id BIGSERIAL PRIMARY KEY,

    layer_version_id BIGINT NOT NULL
        REFERENCES layer_versions(id)
        ON DELETE CASCADE,

    feature_type TEXT NOT NULL,

    confidence DOUBLE PRECISION
        CHECK (
            confidence IS NULL
            OR (
                confidence >= 0.0
                AND confidence <= 1.0
            )
        ),

    geometry GEOMETRY(GEOMETRY, 4326) NOT NULL,

    qc_status TEXT DEFAULT 'pending'
        CHECK (
            qc_status IN ('pending', 'accepted', 'edited', 'rejected')
        )
);

CREATE INDEX IF NOT EXISTS idx_vector_features_geometry
ON vector_features
USING GIST (geometry);

CREATE INDEX IF NOT EXISTS idx_vector_features_layer_version
ON vector_features (layer_version_id);

CREATE INDEX IF NOT EXISTS idx_vector_features_feature_type
ON vector_features (feature_type);

CREATE INDEX IF NOT EXISTS idx_vector_features_qc_status
ON vector_features (qc_status);