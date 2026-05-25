import os
import unittest
import json

import numpy
import matplotlib as mpl
mpl.use('Agg')  # create plots without running X-server
import matplotlib.pyplot as plt
import geojsoncontour


class TestContourToGeoJson(unittest.TestCase):
    dirname = os.path.dirname(__file__)
    geojson_file = os.path.join(dirname, 'test1.geojson')
    geojson_properties_file = os.path.join(dirname, 'test_properties.geojson')
    geojson_file_contourf = os.path.join(dirname, 'contourf.geojson')
    geojson_file_multipoly = os.path.join(dirname, 'multipolycontourf.geojson')

    def tearDown(self):
        plt.close('all')


    @classmethod
    def setUpClass(cls):
        cls.config = ContourPlotConfig(level_lower=0.0, level_upper=202.0, unit='[unit]')
        if os.path.exists(cls.geojson_file):
            os.remove(cls.geojson_file)
        if os.path.exists(cls.geojson_properties_file):
            os.remove(cls.geojson_properties_file)
        if os.path.exists(cls.geojson_file_contourf):
            os.remove(cls.geojson_file_contourf)
        if os.path.exists(cls.geojson_file_multipoly):
            os.remove(cls.geojson_file_multipoly)

    def create_contour(self):
        latrange, lonrange, Z = TestContourToGeoJson.create_grid_data()
        figure = plt.figure()
        ax = figure.add_subplot(111)
        return ax.contour(
            lonrange, latrange, Z,
            levels=self.config.levels,
            cmap=self.config.colormap
        )

    def create_contourf(self):
        latrange, lonrange, Z = TestContourToGeoJson.create_grid_data()
        figure = plt.figure()
        ax = figure.add_subplot(111)
        return ax.contourf(
            lonrange, latrange, Z,
            levels=self.config.levels,
            cmap=self.config.colormap
        )

    def test_matplotlib_contour_to_geojson(self):
        contours = self.create_contour()
        ndigits = 3
        geojsoncontour.contour_to_geojson(
            contour=contours,
            geojson_filepath=self.geojson_file,
            min_angle_deg=self.config.min_angle_between_segments,
            ndigits=ndigits,
            unit=self.config.unit,
            stroke_width=5
        )
        self.assertTrue(os.path.exists(self.geojson_file))
        with open(self.geojson_file) as filein:
            result = json.load(filein)
        self.assertGreater(len(result['features']), 0)
        self.assertTrue(
            all(feature['geometry']['type'] == 'LineString' for feature in result['features'])
        )
        self.assertTrue(
            all(feature['properties']['stroke-width'] == 5 for feature in result['features'])
        )
        os.remove(self.geojson_file)

    def test_matplotlib_contour_to_geojson_none_min_angle(self):
        contours = self.create_contour()
        ndigits = 3
        geojsoncontour.contour_to_geojson(
            contour=contours,
            geojson_filepath=self.geojson_file,
            min_angle_deg=None,
            ndigits=ndigits,
            unit=self.config.unit,
            stroke_width=5
        )
        self.assertTrue(os.path.exists(self.geojson_file))
        os.remove(self.geojson_file)

    def test_return_string_if_destination_file_not_provided(self):
        contours = self.create_contour()
        ndigits = 3
        result = geojsoncontour.contour_to_geojson(
            contour=contours,
            min_angle_deg=self.config.min_angle_between_segments,
            ndigits=ndigits,
            unit=self.config.unit,
            stroke_width=5
        )
        self.assertTrue(isinstance(result, str))

    def test_return_string_if_strdump_argument_provided(self):
        contours = self.create_contour()
        ndigits = 3
        result = geojsoncontour.contour_to_geojson(
            geojson_filepath=self.geojson_file,
            strdump=True,
            contour=contours,
            min_angle_deg=self.config.min_angle_between_segments,
            ndigits=ndigits,
            unit=self.config.unit,
            stroke_width=5
        )
        self.assertTrue(isinstance(result, str))

    def test_return_python_object_if_serialize_argument_false(self):
        contours = self.create_contour()
        ndigits = 3
        result = geojsoncontour.contour_to_geojson(
            serialize=False,
            geojson_filepath=self.geojson_file,
            strdump=True,
            contour=contours,
            min_angle_deg=self.config.min_angle_between_segments,
            ndigits=ndigits,
            unit=self.config.unit,
            stroke_width=5
        )
        self.assertTrue(isinstance(result, dict))
        self.assertEqual(result["type"], "FeatureCollection")

    def test_contour_to_geojson_extra_properties(self):
        contour = self.create_contour()
        ndigits = 3
        geojson_properties = {
            'description': 'A description',
            'stroke-opacity': 1.0
        }
        geojsoncontour.contour_to_geojson(
            contour=contour,
            geojson_filepath=self.geojson_properties_file,
            min_angle_deg=self.config.min_angle_between_segments,
            ndigits=ndigits,
            unit=self.config.unit,
            stroke_width=5,
            geojson_properties=geojson_properties
        )
        self.assertTrue(os.path.exists(self.geojson_properties_file))
        with open(self.geojson_properties_file) as filein:
            result = json.load(filein)
        self.assertGreater(len(result['features']), 0)
        self.assertTrue(
            all(feature['properties']['description'] == 'A description'
                for feature in result['features'])
        )
        self.assertTrue(
            all(feature['properties']['stroke-opacity'] == 1.0
                for feature in result['features'])
        )
        os.remove(self.geojson_properties_file)

    def test_matplotlib_contourf_to_geojson(self):
        contourf = self.create_contourf()
        ndigits = 3
        geojsoncontour.contourf_to_geojson(
            contourf=contourf,
            geojson_filepath=self.geojson_file_multipoly,
            min_angle_deg=self.config.min_angle_between_segments,
            ndigits=ndigits,
            unit=self.config.unit
        )
        self.assertTrue(os.path.exists(self.geojson_file_multipoly))
        with open(self.geojson_file_multipoly) as filein:
            result = json.load(filein)
        self.assertGreater(len(result['features']), 0)
        self.assertTrue(
            all(feature['geometry']['type'] == 'MultiPolygon'
                for feature in result['features'])
        )
        os.remove(self.geojson_file_multipoly)

    def test_matplotlib_contourf_to_geojson_overlap(self):
        contourf = self.create_contourf()
        ndigits = 3
        geojsoncontour.contourf_to_geojson_overlap(
            contourf=contourf,
            geojson_filepath=self.geojson_file_contourf,
            min_angle_deg=self.config.min_angle_between_segments,
            ndigits=ndigits,
            unit=self.config.unit
        )
        self.assertTrue(os.path.exists(self.geojson_file_contourf))
        with open(self.geojson_file_contourf) as filein:
            result = json.load(filein)
        self.assertGreater(len(result['features']), 0)
        self.assertTrue(
            all(feature['geometry']['type'] == 'Polygon' for feature in result['features'])
        )
        os.remove(self.geojson_file_contourf)

    def test_matplotlib_contour_keeps_two_point_boundary_line(self):
        x = numpy.array([0.0, 1.0])
        y = numpy.array([0.0, 1.0])
        x, y = numpy.meshgrid(x, y)
        z = x
        contour = plt.contour(x, y, z, levels=[0.5])

        result = geojsoncontour.contour_to_geojson(
            contour=contour,
            ndigits=3,
            serialize=False
        )

        self.assertEqual(len(result['features']), 1)
        coordinates = result['features'][0]['geometry']['coordinates']
        self.assertEqual(len(coordinates), 2)
        self.assertEqual(coordinates, [[0.5, 1.0], [0.5, 0.0]])

    def test_matplotlib_contourf_min_angle_keeps_rectangle_corners(self):
        x = numpy.array([0.0, 1.0])
        y = numpy.array([0.0, 1.0])
        x, y = numpy.meshgrid(x, y)
        z = x
        contourf = plt.contourf(x, y, z, levels=[0.25, 0.75])

        result = geojsoncontour.contourf_to_geojson(
            contourf=contourf,
            min_angle_deg=15,
            ndigits=3,
            serialize=False
        )

        shell = result['features'][0]['geometry']['coordinates'][0][0]
        self.assertEqual(len(shell), 5)
        self.assertAlmostEqual(self.polygon_area(shell), 0.5)

    def test_matplotlib_contourf_assigns_holes_to_containing_shells(self):
        x = numpy.linspace(-4, 4, 161)
        y = numpy.linspace(-4, 4, 161)
        x, y = numpy.meshgrid(x, y)
        z = (
            numpy.exp(-((x + 1.7) ** 2 + y ** 2) / 0.7)
            + numpy.exp(-((x - 1.7) ** 2 + y ** 2) / 0.7)
            - 0.55 * numpy.exp(-(x ** 2 + y ** 2) / 0.18)
        )
        contourf = plt.contourf(x, y, z, levels=[0.05, 0.18, 0.45])
        self.assertTrue(any(len(path.to_polygons()) > 1 for path in contourf.get_paths()))

        result = geojsoncontour.contourf_to_geojson(
            contourf=contourf,
            ndigits=3,
            serialize=False
        )

        polygons_with_holes = [
            polygon
            for feature in result['features']
            for polygon in feature['geometry']['coordinates']
            if len(polygon) > 1
        ]
        self.assertGreater(len(polygons_with_holes), 0)
        for polygon in polygons_with_holes:
            shell = polygon[0]
            self.assertGreater(self.polygon_area(shell), 0)
            for hole in polygon[1:]:
                self.assertLess(self.polygon_area(hole), 0)
                self.assertTrue(self.point_in_ring(hole[0], shell))

    def test_matplotlib_contourf_removes_duplicate_gridline_vertices(self):
        x = numpy.linspace(0, 1, 5)
        y = numpy.linspace(0, 1, 5)
        x, y = numpy.meshgrid(x, y)
        z = x
        contourf = plt.contourf(x, y, z, levels=[0.25, 0.75])
        self.assertTrue(
            any(
                self.has_consecutive_duplicate_vertices(linestring)
                for path in contourf.get_paths()
                for linestring in path.to_polygons()
            )
        )

        result = geojsoncontour.contourf_to_geojson(
            contourf=contourf,
            ndigits=3,
            serialize=False
        )

        rings = [
            ring
            for feature in result['features']
            for polygon in feature['geometry']['coordinates']
            for ring in polygon
        ]
        self.assertTrue(rings)
        self.assertTrue(
            all(not self.has_consecutive_duplicate_vertices(ring) for ring in rings)
        )

    def test_matplotlib_contourf_opacity_range_reaches_maximum_band(self):
        x = numpy.linspace(0, 2, 3)
        y = numpy.array([0.0, 1.0])
        x, y = numpy.meshgrid(x, y)
        z = x
        contourf = plt.contourf(x, y, z, levels=[0.0, 1.0, 2.0])

        result = geojsoncontour.contourf_to_geojson(
            contourf=contourf,
            fill_opacity_range=(0.2, 0.8),
            serialize=False
        )

        opacities = [
            feature['properties']['fill-opacity']
            for feature in result['features']
        ]
        self.assertEqual(opacities, [0.2, 0.8])

    @staticmethod
    def create_grid_data():
        grid_size = 1.0
        lat_min = -90.0
        lat_max = 90.0
        lon_min = -180.0
        lon_max = 180.0
        latrange = numpy.arange(lat_min, lat_max, grid_size)
        lonrange = numpy.arange(lon_min, lon_max, grid_size)
        X, Y = numpy.meshgrid(lonrange, latrange)
        Z = numpy.sqrt(X*X + Y*Y)
        return latrange, lonrange, Z

    def test_orientation_order_gh31(self):
        # Flipping x should still result in CCW orientation
        # of final polygon
        x = numpy.linspace(0, 10, 14)[::-1]
        y = numpy.linspace(10, 20, 15)
        x, y = numpy.meshgrid(x, y)
        z = numpy.sin(x) * numpy.cos(y)
        contourf = plt.contourf(x, y, z)
        mp = geojsoncontour.contourf_to_geojson(contourf, ndigits=3)

    @staticmethod
    def polygon_area(coordinates):
        coordinates = numpy.asarray(coordinates)
        if numpy.allclose(coordinates[0], coordinates[-1]):
            coordinates = coordinates[:-1]
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        return 0.5 * numpy.sum(x * numpy.roll(y, -1) - numpy.roll(x, -1) * y)

    @staticmethod
    def point_in_ring(point, ring):
        x, y = point
        inside = False
        for start, end in zip(ring[:-1], ring[1:]):
            x0, y0 = start
            x1, y1 = end
            if (y0 > y) == (y1 > y):
                continue
            intersection_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < intersection_x:
                inside = not inside
        return inside

    @staticmethod
    def has_consecutive_duplicate_vertices(coordinates):
        coordinates = numpy.asarray(coordinates)
        return any(
            numpy.allclose(coordinates[index], coordinates[index - 1])
            for index in range(1, len(coordinates))
        )

class ContourPlotConfig(object):
    def __init__(self, level_lower=0.0, level_upper=100.0, colormap=plt.cm.jet, unit=''):  # jet, jet_r, YlOrRd, gist_rainbow
        self.n_contours = 10
        self.min_angle_between_segments = 15
        self.level_lower = level_lower
        self.level_upper = level_upper
        self.colormap = colormap
        self.unit = unit
        self.levels = numpy.linspace(
            start=self.level_lower,
            stop=self.level_upper,
            num=self.n_contours
        )


if __name__ == '__main__':
    unittest.main()
