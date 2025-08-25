#!/usr/bin/python3.4
# -*- encoding: utf-8 -*-
"""Helper module for transformation of matplotlib.contour(f) to GeoJSON."""
import enum

from geojson import MultiPolygon
import numpy as np

from .vertices import get_vertices_from_path


class Orientation(enum.IntEnum):
    CW = enum.auto()
    CCW = enum.auto()

def orientation(vertices) -> Orientation:
    """
    Determines the orientation of a closed polygon using the signed area (shoelace formula).
    Returns Orientation.CCW for counter-clockwise, Orientation.CW for clockwise.
    """
    # Remove duplicate closing vertex if present
    if np.all(vertices[0] == vertices[-1]):
        vertices = vertices[:-1]
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area > 0:
        return Orientation.CCW
    else:
        return Orientation.CW

def multi_polygon(path, min_angle_deg, ndigits):
    # It seems matplotlib emits polygons in either CW or CCW order.
    # We detect which order the first polygon has, and uses this
    # as the ring, with polygons of the other winding order as
    # holes. If order is reversed compared to the conventions,
    # we reverse the order of the polygon so rings have CCW order.
    orientation_for_keep = None
    polygons = []
    for linestring in path.to_polygons():
        if min_angle_deg:
            linestring = keep_high_angle(linestring, min_angle_deg)
        if ndigits:
            linestring = np.around(linestring, ndigits)

        handedness = orientation(linestring)
        if len(polygons) == 0:
            orientation_for_keep = handedness
        if orientation_for_keep != Orientation.CCW:
            linestring = linestring[::-1, :]

        if handedness == orientation_for_keep:
            polygons.append([linestring.tolist()])
        else:
            # This is a hole, which we assume belong
            # to the previous polygon
            polygons[-1].extend([linestring.tolist()])

    return MultiPolygon(coordinates=polygons)


def unit_vector(vector):
    """Return the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle(v1, v2):
    """Return the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def keep_high_angle(vertices, min_angle_deg):
    """Keep vertices with angles higher then given minimum."""
    accepted = []
    v = vertices
    v1 = v[1] - v[0]
    accepted.append((v[0][0], v[0][1]))
    for i in range(1, len(v) - 2):
        v2 = v[i + 1] - v[i - 1]
        diff_angle = np.fabs(angle(v1, v2) * 180.0 / np.pi)
        if diff_angle > min_angle_deg:
            accepted.append((v[i][0], v[i][1]))
            v1 = v[i] - v[i - 1]
    accepted.append((v[-1][0], v[-1][1]))
    return np.array(accepted, dtype=vertices.dtype)


def set_contourf_properties(stroke_width, fcolor, fill_opacity, level, unit):
    """Set property values for Polygon."""
    return {
        "stroke": fcolor,
        "stroke-width": stroke_width,
        "stroke-opacity": 1,
        "fill": fcolor,
        "fill-opacity": fill_opacity,
        "title": "{} {}".format(level, unit)
    }


def get_contourf_levels(levels, extend):
    mid_levels = ["%.2f" % levels[i] + '-' + "%.2f" % levels[i+1] for i in range(len(levels)-1)]
    if extend == 'both':
        return ["<%.2f" % levels[0], *mid_levels, ">%.2f" % levels[-1]]
    elif extend == 'max':
        return [*mid_levels, ">%.2f" % levels[-1]]
    elif extend == 'min':
        return ["<%.2f" % levels[0], *mid_levels]
    else:
        return mid_levels
