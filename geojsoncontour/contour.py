"""Transform matplotlib.contour(f) to GeoJSON."""

import warnings

import geojson
import numpy as np
from matplotlib.colors import rgb2hex
from geojson import Feature, LineString
from geojson import Polygon, FeatureCollection

from .utilities.multipoly import (
    remove_consecutive_duplicate_vertices,
    multi_polygon,
    keep_high_angle,
    safe_json_number,
    set_contourf_properties,
    get_contourf_levels
)
from .utilities.vertices import get_vertices_from_path


_WGS84_LON_RANGE = (-180.0, 180.0)
_WGS84_LAT_RANGE = (-90.0, 90.0)
_RFC7946_RECOMMENDED_PRECISION = 6


def _warn_precision(ndigits):
    if ndigits is None:
        warnings.warn(
            "ndigits=None preserves full float64 precision. "
            "RFC 7946 §11.2 recommends no more than 6 decimal places.",
            stacklevel=3,
        )
    elif ndigits > _RFC7946_RECOMMENDED_PRECISION:
        warnings.warn(
            f"ndigits={ndigits} exceeds the RFC 7946 §11.2 recommendation of "
            f"{_RFC7946_RECOMMENDED_PRECISION} decimal places.",
            stacklevel=3,
        )


def _warn_paths_outside_wgs84(paths):
    """Emit a single warning if any path vertex is outside the WGS 84 range.

    Performed once per top-level call by sampling each Path's bounding box,
    which is cheap relative to walking every emitted ring/line and avoids the
    nested ``warnings.catch_warnings`` contexts that previous versions used.
    """
    for path in paths:
        vertices = path.vertices
        if vertices.size == 0:
            continue
        lon = vertices[:, 0]
        lat = vertices[:, 1]
        if (
            lon.min() < _WGS84_LON_RANGE[0]
            or lon.max() > _WGS84_LON_RANGE[1]
            or lat.min() < _WGS84_LAT_RANGE[0]
            or lat.max() > _WGS84_LAT_RANGE[1]
        ):
            warnings.warn(
                "Coordinates outside the WGS 84 range "
                f"(lon {_WGS84_LON_RANGE}, lat {_WGS84_LAT_RANGE}); "
                "RFC 7946 §4 requires WGS 84 longitude/latitude.",
                stacklevel=3,
            )
            return


def contour_to_geojson(contour, geojson_filepath=None, min_angle_deg=None,
                       ndigits=5, unit='', stroke_width=1, geojson_properties=None, strdump=False,
                       serialize=True):
    """Transform matplotlib.contour to geojson."""
    _warn_precision(ndigits)
    paths = contour.get_paths()
    _warn_paths_outside_wgs84(paths)
    line_features = []
    colors = contour.get_edgecolors()
    levels = contour.levels
    for contour_index, (path, color, level) in enumerate(zip(paths, colors, levels)):
        for coordinates in get_vertices_from_path(path):
            if min_angle_deg:
                coordinates = keep_high_angle(coordinates, min_angle_deg)
            if ndigits is not None:
                coordinates = np.around(coordinates, ndigits)
            coordinates = remove_consecutive_duplicate_vertices(coordinates)
            if len(coordinates) < 2:
                continue
            if np.all(np.equal(coordinates, coordinates[0])):
                # Matplotlib sometimes emits empty paths which
                # can be ignored
                continue
            line = LineString(coordinates.tolist())
            level_value = safe_json_number(level)
            properties = {
                "stroke-width": stroke_width,
                "stroke": rgb2hex(color),
                "title": f"{level:.2f} {unit}" if level_value is not None else f"{unit}".strip(),
                "level-index": contour_index,
            }
            if level_value is not None:
                properties["level-value"] = level_value
            if geojson_properties:
                properties.update(geojson_properties)
            line_features.append(Feature(geometry=line, properties=properties))

    feature_collection = FeatureCollection(line_features)
    return _render_feature_collection(feature_collection, geojson_filepath, strdump, serialize)


def contourf_to_geojson_overlap(contourf, geojson_filepath=None, min_angle_deg=None,
                                ndigits=5, unit='', stroke_width=1, fill_opacity=.9,
                                geojson_properties=None, strdump=False, serialize=True):
    """Transform matplotlib.contourf to geojson with overlapping filled contours."""
    _warn_precision(ndigits)
    paths = contourf.get_paths()
    _warn_paths_outside_wgs84(paths)
    polygon_features = []
    contourf_levels = get_contourf_levels(contourf.levels, contourf.extend)
    contourf_colors = contourf.get_facecolor()
    for level_index, (path, level_info, color) in enumerate(
        zip(paths, contourf_levels, contourf_colors)
    ):
        title, lower, upper = level_info
        polygon = multi_polygon(path, min_angle_deg, ndigits)
        if not polygon.coordinates:
            continue
        fcolor = rgb2hex(color)
        properties = set_contourf_properties(
            stroke_width, fcolor, fill_opacity, title, unit,
            level_index=level_index, level_lower=lower, level_upper=upper,
        )
        if geojson_properties:
            properties.update(geojson_properties)

        # Split MultiPolygons into individual Polygon features for "overlap" style
        for poly_coords in polygon.coordinates:
            feature = Feature(geometry=Polygon(poly_coords), properties=properties)
            polygon_features.append(feature)
    feature_collection = FeatureCollection(polygon_features)
    return _render_feature_collection(feature_collection, geojson_filepath, strdump, serialize)


def contourf_to_geojson(contourf, geojson_filepath=None, min_angle_deg=None,
                        ndigits=5, unit='', stroke_width=1, fill_opacity=.9, fill_opacity_range=None,
                        geojson_properties=None, strdump=False, serialize=True):
    """Transform matplotlib.contourf to geojson with MultiPolygons."""
    _warn_precision(ndigits)
    paths = contourf.get_paths()
    _warn_paths_outside_wgs84(paths)
    polygon_features = []
    contourf_levels = get_contourf_levels(contourf.levels, contourf.extend)
    contourf_colors = contourf.get_facecolor()
    if fill_opacity_range:
        variable_opacity = True
        min_opacity, max_opacity = fill_opacity_range
        opacity_steps = max(len(contourf_levels) - 1, 1)
        opacity_increment = (max_opacity - min_opacity) / opacity_steps
    else:
        variable_opacity = False
    for contour_index, (path, level_info, color) in enumerate(
        zip(paths, contourf_levels, contourf_colors)
    ):
        title, lower, upper = level_info
        polygon = multi_polygon(path, min_angle_deg, ndigits)
        if not polygon.coordinates:
            continue
        fcolor = rgb2hex(color)
        current_fill_opacity = fill_opacity
        if variable_opacity:
            current_fill_opacity = min_opacity + contour_index * opacity_increment
        properties = set_contourf_properties(
            stroke_width, fcolor, current_fill_opacity, title, unit,
            level_index=contour_index, level_lower=lower, level_upper=upper,
        )
        if geojson_properties:
            properties.update(geojson_properties)
        feature = Feature(geometry=polygon, properties=properties)
        polygon_features.append(feature)
    feature_collection = FeatureCollection(polygon_features)
    return _render_feature_collection(feature_collection, geojson_filepath, strdump, serialize)


def _render_feature_collection(feature_collection, geojson_filepath, strdump, serialize):
    if not serialize:
        return feature_collection
    if strdump or not geojson_filepath:
        return geojson.dumps(feature_collection, sort_keys=True, separators=(',', ':'))
    with open(geojson_filepath, 'w') as fileout:
        geojson.dump(feature_collection, fileout, sort_keys=True, separators=(',', ':'))
