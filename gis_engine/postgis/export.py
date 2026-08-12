# ============================================================
# NAKSHA - POSTGIS EXPORT UTILITIES
# Day 5 - Versioned vector layer export
# ============================================================

from pathlib import Path

from shapely.geometry.base import BaseGeometry


def create_layer_version_sql(
    layer_name: str,
    feature_type: str,
    version: int,
) -> str:
    """
    Generate SQL for creating a new vector layer version.
    """

    return f"""
INSERT INTO layer_versions (
    layer_name,
    feature_type,
    version
)
VALUES (
    '{layer_name}',
    '{feature_type}',
    {version}
)
RETURNING id;
""".strip()


def create_feature_sql(
    layer_version_id: int,
    feature_type: str,
    geometry: BaseGeometry,
    confidence: float | None = None,
) -> str:
    """
    Generate SQL for inserting one validated vector feature.
    """

    if geometry is None or geometry.is_empty:
        raise ValueError(
            "Cannot export an empty geometry."
        )

    wkt = geometry.wkt.replace(
        "'",
        "''"
    )

    confidence_sql = (
        "NULL"
        if confidence is None
        else str(float(confidence))
    )

    return f"""
INSERT INTO vector_features (
    layer_version_id,
    feature_type,
    confidence,
    geometry
)
VALUES (
    {layer_version_id},
    '{feature_type}',
    {confidence_sql},
    ST_GeomFromText(
        '{wkt}',
        4326
    )
);
""".strip()


def write_sql_file(
    statements: list[str],
    output_path: str | Path,
) -> Path:
    """
    Write generated PostGIS SQL statements to a file.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        "\n\n".join(statements)
        + "\n",
        encoding="utf-8"
    )

    return output_path