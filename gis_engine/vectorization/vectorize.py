# ============================================================
# NAKSHA - GIS VECTORIZATION UTILITIES
# Day 5 - Raster -> Vector
# ============================================================

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes, geometry_mask
from shapely.geometry import shape, LineString
from skimage.morphology import skeletonize


# ============================================================
# SOURCE RASTER METADATA
# ============================================================

def get_raster_metadata(
    raster_path: str | Path
):
    """
    Read the geospatial metadata from the source raster.

    Prediction PNGs do not contain CRS/geotransform
    information, so the original GeoTIFF is used as
    the georeferencing source.
    """

    raster_path = Path(raster_path)

    with rasterio.open(
        str(raster_path)
    ) as src:

        return {
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
        }


# ============================================================
# RASTER MASK -> POLYGONS
# ============================================================

def polygonize_mask(
    mask: np.ndarray,
    transform,
    min_area: float = 0.0
):
    """
    Convert a binary raster mask into Shapely polygons.

    Parameters
    ----------
    mask:
        Binary NumPy array containing 0/1 values.

    transform:
        Affine transform from the original source raster.

    min_area:
        Optional minimum polygon area.

    Returns
    -------
    list
        List of Shapely geometry objects.
    """

    binary_mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    geometries = []

    for geometry_json, value in shapes(
        binary_mask,
        mask=binary_mask.astype(bool),
        transform=transform
    ):

        if value != 1:
            continue

        geometry = shape(
            geometry_json
        )

        if geometry.is_empty:
            continue

        if (
            min_area > 0
            and geometry.area < min_area
        ):
            continue

        geometries.append(
            geometry
        )

    return geometries
# ============================================================
# FEATURE GEOMETRY -> CONFIDENCE
# ============================================================

def feature_confidence(
    probability: np.ndarray,
    geometry,
    transform,
) -> float | None:
    """
    Calculate a per-feature confidence from a probability raster.

    The confidence is the mean model probability of pixels
    covered by the feature geometry.

    Parameters
    ----------
    probability:
        2D model probability raster with values in [0, 1].

    geometry:
        Shapely geometry representing one extracted feature.

    transform:
        Affine transform for the probability raster.

    Returns
    -------
    float | None
        Mean probability for the feature, or None when no
        probability pixels intersect the geometry.
    """

    if (
        probability is None
        or geometry is None
        or geometry.is_empty
    ):
        return None

    probability = np.asarray(
        probability,
        dtype=np.float32
    )

    if probability.ndim != 2:
        raise ValueError(
            "Probability raster must be 2D."
        )

    mask = geometry_mask(
        [geometry],
        out_shape=probability.shape,
        transform=transform,
        invert=True,
    )

    values = probability[mask]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return None

    confidence = float(
        np.mean(values)
    )

    # --------------------------------------------------
    # Geometric Signals (Area Plausibility & Edge Smoothness)
    # --------------------------------------------------
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type in ("Polygon", "MultiPolygon"):
        area = float(geometry.area)
        perimeter = float(geometry.length)

        # 1. Area Plausibility
        area_factor = 1.0 if area > 0 else 0.5

        # 2. Edge Smoothness (Isoperimetric Quotient / Compactness)
        smoothness_factor = 1.0
        if perimeter > 0 and area > 0:
            compactness = (4.0 * np.pi * area) / (perimeter ** 2)
            smoothness_factor = max(0.5, min(1.0, float(compactness * 2.0)))

        confidence = confidence * area_factor * smoothness_factor

    return max(
        0.0,
        min(
            1.0,
            confidence
        )
    )


# ============================================================
# ROAD MASK -> SKELETON
# ============================================================

def skeletonize_road_mask(
    mask: np.ndarray
):
    """
    Convert a binary road mask into a one-pixel-wide
    skeleton suitable for road centerline extraction.
    """

    binary_mask = (
        mask > 0
    )

    skeleton = skeletonize(
        binary_mask
    )

    return skeleton.astype(
        np.uint8
    )


# ============================================================
# ROAD SKELETON -> LINESTRINGS
# ============================================================

def skeleton_to_lines(
    skeleton: np.ndarray,
    transform
):
    """
    Convert a one-pixel-wide road skeleton into
    georeferenced LineString geometries.

    Each connected skeleton component is converted
    into a LineString using the source raster transform.
    """

    binary = (
        skeleton > 0
    ).astype(
        np.uint8
    )

    lines = []

    height, width = binary.shape

    visited = np.zeros_like(
        binary,
        dtype=bool
    )

    # 8-connected neighbourhood
    neighbours = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for y in range(height):

        for x in range(width):

            if (
                binary[y, x] == 0
                or visited[y, x]
            ):
                continue

            stack = [
                (y, x)
            ]

            component = []

            while stack:

                cy, cx = stack.pop()

                if visited[cy, cx]:
                    continue

                visited[cy, cx] = True

                if binary[cy, cx] == 0:
                    continue

                component.append(
                    (cy, cx)
                )

                for dy, dx in neighbours:

                    ny = cy + dy
                    nx = cx + dx

                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and binary[ny, nx]
                        and not visited[ny, nx]
                    ):

                        stack.append(
                            (ny, nx)
                        )

            if len(component) < 2:
                continue

            # Convert pixel coordinates to
            # geographic coordinates.
            coords = []

            for cy, cx in component:

                px, py = transform * (
                    cx + 0.5,
                    cy + 0.5
                )

                coords.append(
                    (px, py)
                )

            # Remove consecutive duplicate coordinates.
            unique_coords = []

            for coord in coords:

                if (
                    not unique_coords
                    or coord != unique_coords[-1]
                ):
                    unique_coords.append(
                        coord
                    )

            if len(unique_coords) < 2:
                continue

            line = LineString(
                unique_coords
            )

            if (
                not line.is_empty
                and line.is_valid
                and line.length > 0
            ):

                lines.append(
                    line
                )

    return lines