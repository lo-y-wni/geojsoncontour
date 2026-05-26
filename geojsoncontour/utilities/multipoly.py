#!/usr/bin/python3.4
# -*- encoding: utf-8 -*-
"""Helper module for transformation of matplotlib.contour(f) to GeoJSON."""
import enum
import math

from geojson import MultiPolygon
import numpy as np


class Orientation(enum.IntEnum):
    CW = enum.auto()
    CCW = enum.auto()


_AREA_EPSILON = 1e-12


def remove_consecutive_duplicate_vertices(vertices):
    vertices = np.asarray(vertices)
    if len(vertices) == 0:
        return vertices
    keep = [True]
    for index in range(1, len(vertices)):
        keep.append(not np.allclose(vertices[index], vertices[index - 1]))
    return vertices[np.array(keep)]


def _signed_area(vertices):
    if len(vertices) == 0:
        return 0.0
    if np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _clean_ring(vertices):
    vertices = remove_consecutive_duplicate_vertices(vertices)
    if len(vertices) == 0:
        return vertices
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    vertices = remove_consecutive_duplicate_vertices(vertices)
    if len(vertices) == 0:
        return vertices
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    return vertices


def _point_on_segment(point, start, end):
    segment = end - start
    point_vector = point - start
    cross_product = segment[0] * point_vector[1] - segment[1] * point_vector[0]
    if not np.isclose(cross_product, 0.0):
        return False
    dot_product = np.dot(point_vector, segment)
    if dot_product < 0:
        return False
    return dot_product <= np.dot(segment, segment)


def _point_in_ring(point, ring):
    ring = np.asarray(ring)
    inside = False
    x, y = point
    for start, end in zip(ring[:-1], ring[1:]):
        if _point_on_segment(point, start, end):
            return True
        x0, y0 = start
        x1, y1 = end
        if (y0 > y) == (y1 > y):
            continue
        intersection_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
        if x < intersection_x:
            inside = not inside
    return inside


def _ensure_orientation(vertices, desired_orientation):
    if orientation(vertices) == desired_orientation:
        return vertices
    return vertices[::-1, :]


def orientation(vertices) -> Orientation:
    """
    Determines the orientation of a closed polygon using the signed area (shoelace formula).
    Returns Orientation.CCW for counter-clockwise, Orientation.CW for clockwise.
    """
    if _signed_area(vertices) > 0:
        return Orientation.CCW
    else:
        return Orientation.CW

def multi_polygon(path, min_angle_deg, ndigits):
    rings = []
    for linestring in path.to_polygons():
        linestring = np.asarray(linestring)
        if min_angle_deg:
            linestring = keep_high_angle(linestring, min_angle_deg)
        if ndigits is not None:
            linestring = np.around(linestring, ndigits)

        linestring = _clean_ring(linestring)
        area = _signed_area(linestring)
        if abs(area) <= _AREA_EPSILON:
            continue
        rings.append({
            "coordinates": linestring,
            "abs_area": abs(area),
            "depth": 0,
        })

    for ring_index, ring in enumerate(rings):
        point = ring["coordinates"][0]
        ring["depth"] = sum(
            _point_in_ring(point, other["coordinates"])
            for other_index, other in enumerate(rings)
            if other_index != ring_index and other["abs_area"] > ring["abs_area"]
        )

    shell_indexes = [
        ring_index for ring_index, ring in enumerate(rings)
        if ring["depth"] % 2 == 0
    ]
    hole_indexes = [
        ring_index for ring_index, ring in enumerate(rings)
        if ring["depth"] % 2 == 1
    ]

    polygons = []
    shell_position_by_index = {}
    for shell_index in shell_indexes:
        shell = _ensure_orientation(rings[shell_index]["coordinates"], Orientation.CCW)
        shell_position_by_index[shell_index] = len(polygons)
        polygons.append([shell.tolist()])

    for hole_index in hole_indexes:
        hole = rings[hole_index]
        containing_shell_indexes = [
            shell_index for shell_index in shell_indexes
            if rings[shell_index]["depth"] == hole["depth"] - 1
            and _point_in_ring(hole["coordinates"][0], rings[shell_index]["coordinates"])
        ]
        if not containing_shell_indexes:
            containing_shell_indexes = [
                shell_index for shell_index in shell_indexes
                if _point_in_ring(hole["coordinates"][0], rings[shell_index]["coordinates"])
            ]
        if not containing_shell_indexes:
            continue
        containing_shell_index = min(
            containing_shell_indexes,
            key=lambda shell_index: rings[shell_index]["abs_area"]
        )
        hole_coordinates = _ensure_orientation(hole["coordinates"], Orientation.CW)
        polygons[shell_position_by_index[containing_shell_index]].append(
            hole_coordinates.tolist()
        )

    return MultiPolygon(coordinates=polygons)


def unit_vector(vector):
    """Return the unit vector of the vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


def angle(v1, v2):
    """Return the angle in radians between vectors 'v1' and 'v2'."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def keep_high_angle(vertices, min_angle_deg):
    """Keep vertices with angles higher then given minimum."""
    v = remove_consecutive_duplicate_vertices(vertices)
    if len(v) <= 2:
        return v

    is_closed = np.allclose(v[0], v[-1])
    if is_closed:
        ring = v[:-1]
        if len(ring) < 3:
            return np.empty((0, v.shape[1]), dtype=vertices.dtype)
        accepted = []
        for index in range(len(ring)):
            previous_vector = ring[index] - ring[index - 1]
            next_vector = ring[(index + 1) % len(ring)] - ring[index]
            diff_angle = np.fabs(angle(previous_vector, next_vector) * 180.0 / np.pi)
            if diff_angle > min_angle_deg:
                accepted.append(ring[index])
        if len(accepted) < 3:
            return np.empty((0, v.shape[1]), dtype=vertices.dtype)
        accepted.append(accepted[0])
        return np.array(accepted, dtype=vertices.dtype)

    accepted = [v[0]]
    for index in range(1, len(v) - 1):
        previous_vector = v[index] - v[index - 1]
        next_vector = v[index + 1] - v[index]
        diff_angle = np.fabs(angle(previous_vector, next_vector) * 180.0 / np.pi)
        if diff_angle > min_angle_deg:
            accepted.append(v[index])
    accepted.append(v[-1])
    return np.array(accepted, dtype=vertices.dtype)


def safe_json_number(value, digits=6):
    """Coerce a value to a finite float suitable for strict JSON.

    RFC 7946 §3.1.1 / RFC 8259 do not permit NaN, Infinity, or -Infinity.
    Returns None if the value cannot be represented as a finite JSON number.
    """
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return float(f"{as_float:.{digits}f}")


def set_contourf_properties(stroke_width, fcolor, fill_opacity, level, unit,
                             level_index=None, level_lower=None, level_upper=None):
    """Set property values for Polygon."""
    properties = {
        "stroke": fcolor,
        "stroke-width": stroke_width,
        "stroke-opacity": 1,
        "fill": fcolor,
        "fill-opacity": fill_opacity,
        "title": "{} {}".format(level, unit),
    }
    if level_index is not None:
        properties["level-index"] = int(level_index)
    safe_lower = safe_json_number(level_lower)
    safe_upper = safe_json_number(level_upper)
    if safe_lower is not None:
        properties["level-lower"] = safe_lower
    if safe_upper is not None:
        properties["level-upper"] = safe_upper
    if safe_lower is not None and safe_upper is not None:
        properties["level-value"] = safe_json_number(0.5 * (safe_lower + safe_upper))
    elif safe_lower is not None:
        properties["level-value"] = safe_lower
    elif safe_upper is not None:
        properties["level-value"] = safe_upper
    return properties


def get_contourf_levels(levels, extend):
    """Return a list of (label, lower, upper) for each contourf band.

    `lower` / `upper` are floats or None for open-ended bands (extend='min'/'max'/'both').
    `label` is a human-readable string preserved for the GeoJSON `title` property.
    """
    finite_levels = [float(level) for level in levels]
    bands = []
    for i in range(len(finite_levels) - 1):
        lower = finite_levels[i]
        upper = finite_levels[i + 1]
        bands.append((f"{lower:.2f}-{upper:.2f}", lower, upper))
    if extend == 'both':
        return [
            (f"<{finite_levels[0]:.2f}", None, finite_levels[0]),
            *bands,
            (f">{finite_levels[-1]:.2f}", finite_levels[-1], None),
        ]
    if extend == 'max':
        return [*bands, (f">{finite_levels[-1]:.2f}", finite_levels[-1], None)]
    if extend == 'min':
        return [(f"<{finite_levels[0]:.2f}", None, finite_levels[0]), *bands]
    return bands
