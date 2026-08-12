# ============================================================
# NAKSHA - GIS GEOMETRY UTILITIES
# Day 5 - Geometry validation + simplification
# ============================================================

from shapely.geometry.base import BaseGeometry


def validate_geometry(
    geometry: BaseGeometry
) -> BaseGeometry | None:
    """
    Validate a Shapely geometry.

    Invalid geometries are repaired using make_valid().
    Empty geometries are discarded.
    """

    if geometry is None or geometry.is_empty:
        return None

    if not geometry.is_valid:
        geometry = geometry.make_valid()

    if geometry.is_empty:
        return None

    return geometry


def simplify_geometry(
    geometry: BaseGeometry,
    tolerance: float = 0.0
) -> BaseGeometry | None:
    """
    Simplify a valid geometry while preserving topology.
    """

    geometry = validate_geometry(
        geometry
    )

    if geometry is None:
        return None

    if tolerance <= 0:
        return geometry

    simplified = geometry.simplify(
        tolerance,
        preserve_topology=True
    )

    return validate_geometry(
        simplified
    )