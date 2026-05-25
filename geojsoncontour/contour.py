"""Transform matplotlib.contour(f) to GeoJSON."""

import geojson
import numpy as np
from matplotlib.colors import rgb2hex
from geojson import Feature, LineString
from geojson import Polygon, FeatureCollection

from .utilities.multipoly import (
    remove_consecutive_duplicate_vertices,
    multi_polygon,
    keep_high_angle,
    set_contourf_properties,
    get_contourf_levels
)
from .utilities.vertices import get_vertices_from_path


def contour_to_geojson(contour, geojson_filepath=None, min_angle_deg=None,
                       ndigits=5, unit='', stroke_width=1, geojson_properties=None, strdump=False,
                       serialize=True):
    """Transform matplotlib.contour to geojson."""
    line_features = []
    paths = contour.get_paths()
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
            properties = {
                "stroke-width": stroke_width,
                "stroke": rgb2hex(color),
                "title": f"{level:.2f} {unit}",
                "level-value": float(f"{level:.6f}"),
                "level-index": contour_index
            }
            if geojson_properties:
                properties.update(geojson_properties)
            line_features.append(Feature(geometry=line, properties=properties))

    feature_collection = FeatureCollection(line_features)
    return _render_feature_collection(feature_collection, geojson_filepath, strdump, serialize)


def contourf_to_geojson_overlap(contourf, geojson_filepath=None, min_angle_deg=None,
                                ndigits=5, unit='', stroke_width=1, fill_opacity=.9,
                                geojson_properties=None, strdump=False, serialize=True):
    """Transform matplotlib.contourf to geojson with overlapping filled contours."""
    polygon_features = []
    contourf_levels = get_contourf_levels(contourf.levels, contourf.extend)
    contourf_colors = contourf.get_facecolor()
    for path, level, color in zip(contourf.get_paths(), contourf_levels, contourf_colors):
        polygon = multi_polygon(path, min_angle_deg, ndigits)
        if not polygon.coordinates:
            continue
        fcolor = rgb2hex(color)
        properties = set_contourf_properties(stroke_width, fcolor, fill_opacity, level, unit)
        if geojson_properties:
            properties.update(geojson_properties)

        # Split MultiPolygons into individual Polygon features for "overlap" style
        # multi_polygon returns a MultiPolygon geometry, coordinates are [ [ [x,y], [x,y] (hole) ], ... ]
        for poly_coords in polygon.coordinates:
            feature = Feature(geometry=Polygon(poly_coords), properties=properties)
            polygon_features.append(feature)
    feature_collection = FeatureCollection(polygon_features)
    return _render_feature_collection(feature_collection, geojson_filepath, strdump, serialize)


def contourf_to_geojson(contourf, geojson_filepath=None, min_angle_deg=None,
                        ndigits=5, unit='', stroke_width=1, fill_opacity=.9, fill_opacity_range=None,
                        geojson_properties=None, strdump=False, serialize=True):
    """Transform matplotlib.contourf to geojson with MultiPolygons."""
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
    for contour_index, (path, level, color) in enumerate(
        zip(contourf.get_paths(), contourf_levels, contourf_colors)
    ):
        polygon = multi_polygon(path, min_angle_deg, ndigits)
        if not polygon.coordinates:
            continue
        fcolor = rgb2hex(color)
        current_fill_opacity = fill_opacity
        if variable_opacity:
            current_fill_opacity = min_opacity + contour_index * opacity_increment
        properties = set_contourf_properties(
            stroke_width, fcolor, current_fill_opacity, level, unit
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
