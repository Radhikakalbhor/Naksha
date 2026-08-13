# ============================================================
# NAKSHA - DAY 6 EXPORT ENGINE
# PostGIS -> GeoJSON / Shapefile / GeoPackage / FileGDB
# ============================================================

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import psycopg


SUPPORTED_FORMATS = {
    "geojson",
    "shapefile",
    "geopackage",
    "filegdb",
}


FORMAT_CONFIG = {
    "geojson": {
        "driver": "GeoJSON",
        "extension": ".geojson",
        "media_type": "application/geo+json",
    },
    "shapefile": {
        "driver": "ESRI Shapefile",
        "extension": ".shp",
        "media_type": "application/zip",
    },
    "geopackage": {
        "driver": "GPKG",
        "extension": ".gpkg",
        "media_type": "application/geopackage+sqlite3",
    },
    "filegdb": {
        "driver": "OpenFileGDB",
        "extension": ".gdb",
        "media_type": "application/zip",
    },
}


def fetch_layer_geojson(
    conn: psycopg.Connection,
    layer_name: str,
) -> dict:
    """
    Read a PostGIS layer and convert it to GeoJSON.
    """

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                vf.id,
                vf.feature_type,
                vf.confidence,
                ST_AsGeoJSON(vf.geometry)
            FROM vector_features vf
            JOIN layer_versions lv
                ON lv.id = vf.layer_version_id
            WHERE lv.layer_name = %s
            ORDER BY vf.id;
            """,
            (layer_name,),
        )

        rows = cur.fetchall()

    features = []

    for row in rows:

        features.append(
            {
                "type": "Feature",
                "id": row[0],
                "geometry": json.loads(row[3]),
                "properties": {
                    "feature_type": row[1],
                    "confidence": row[2],
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def create_source_geojson(
    geojson: dict,
    path: Path,
) -> None:
    """
    Write temporary GeoJSON source.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            geojson,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_ogr2ogr(
    source_path: Path,
    destination_path: Path,
    driver: str,
) -> None:
    """
    Convert GeoJSON using GDAL/OGR.
    """

    command = [
        "ogr2ogr",
        "-f",
        driver,
        str(destination_path),
        str(source_path),
        "-nln",
        "features",
        "-nlt",
        "PROMOTE_TO_MULTI",
    ]

    if driver in ("GPKG", "GeoJSON"):
        command.extend(["-lco", "GEOMETRY_NAME=geometry"])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "ogr2ogr export failed: "
            + (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown GDAL error."
            )
        )


def package_directory(
    directory: Path,
    output_zip: Path,
) -> Path:
    """
    Create a ZIP archive from a directory.
    """

    shutil.make_archive(
        str(output_zip.with_suffix("")),
        "zip",
        root_dir=directory.parent,
        base_dir=directory.name,
    )

    return output_zip


def export_layer(
    conn: psycopg.Connection,
    layer_name: str,
    export_format: str,
    output_dir: Path,
) -> tuple[Path, str]:
    """
    Export a PostGIS layer.

    Returns:
        (output_path, media_type)
    """

    export_format = export_format.lower().strip()

    if export_format not in SUPPORTED_FORMATS:

        raise ValueError(
            f"Unsupported export format: {export_format}"
        )

    config = FORMAT_CONFIG[export_format]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    geojson = fetch_layer_geojson(
        conn,
        layer_name,
    )

    if not geojson.get("features"):
        raise ValueError(
            f"Layer '{layer_name}' has no features to export."
        )

    with tempfile.TemporaryDirectory(
        dir=str(output_dir)
    ) as temp_dir:

        temp_path = Path(temp_dir)

        source_path = (
            temp_path /
            "source.geojson"
        )

        create_source_geojson(
            geojson,
            source_path,
        )

        if export_format == "geojson":

            output_path = (
                output_dir /
                f"{layer_name}.geojson"
            )

            output_path.write_text(
                json.dumps(
                    geojson,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            return (
                output_path,
                config["media_type"],
            )

        if export_format == "geopackage":

            output_path = (
                output_dir /
                f"{layer_name}.gpkg"
            )

            run_ogr2ogr(
                source_path,
                output_path,
                config["driver"],
            )

            return (
                output_path,
                config["media_type"],
            )

        if export_format == "shapefile":

            shapefile_dir = (
                temp_path /
                f"{layer_name}_shapefile"
            )

            shapefile_dir.mkdir()

            run_ogr2ogr(
                source_path,
                shapefile_dir,
                config["driver"],
            )

            zip_base = (
                output_dir /
                layer_name
            )

            zip_path = (
                output_dir /
                f"{layer_name}.zip"
            )

            shutil.make_archive(
                str(zip_base),
                "zip",
                root_dir=shapefile_dir,
            )

            return (
                zip_path,
                config["media_type"],
            )

        if export_format == "filegdb":

            gdb_dir = (
                temp_path /
                f"{layer_name}.gdb"
            )

            run_ogr2ogr(
                source_path,
                gdb_dir,
                config["driver"],
            )

            zip_base = (
                temp_path /
                f"{layer_name}_filegdb"
            )

            zip_path = (
                output_dir /
                f"{layer_name}_filegdb.zip"
            )

            shutil.make_archive(
                str(zip_base),
                "zip",
                root_dir=temp_path,
                base_dir=gdb_dir.name,
            )

            generated_zip = (
                temp_path /
                f"{layer_name}_filegdb.zip"
            )

            if generated_zip.exists():
                shutil.move(
                    str(generated_zip),
                    str(zip_path),
                )

            return (
                zip_path,
                config["media_type"],
            )

    raise RuntimeError(
        "Export completed without producing a file."
    )