import unittest

import lib.area
import lib.point

class TestAreaFunctions(unittest.TestCase):
    def test_square_covered_by(self):
        # make a big circle that covers a small square
        p_zero = lib.point.Point(0, 0, 0)
        c = lib.area.Circle(p_zero, 100)

        # covered case
        s = lib.area.Square(p_zero, 10)
        self.assertEqual(s.covered_by(c), lib.area.COVERED_ENUM.FULL, 'square fully covered by circle')

        # partial case
        s = lib.area.Square(p_zero, 100)
        self.assertEqual(s.covered_by(c), lib.area.COVERED_ENUM.PARTIAL, 'square partially covered by circle')

        # uncovered case
        p_far_away = lib.point.Point(1000, 1000, 0)
        s = lib.area.Square(p_far_away, 100)
        self.assertEqual(s.covered_by(c), lib.area.COVERED_ENUM.EMPTY, 'square partially covered by circle')
        pass

    def test_quadtree(self):
        # basic circle area estimation test
        p_zero = lib.point.Point(0, 0, 0)
        c = lib.area.Circle(p_zero, 1)

        p_square_llp = lib.point.Point(-1, -1, 0)
        s = lib.area.Square(p_square_llp, 2)

        actual_area = c.area()

        full_squares, partial_squares = lib.area.quadtree([c], s, limit=0.01)

        lower_limit = full_squares
        upper_limit = full_squares + partial_squares
        actual_area = c.area()

        self.assertLessEqual(lower_limit, actual_area, f"lower estimate equal or below actual area: {lower_limit} <= {actual_area}")
        self.assertGreaterEqual(upper_limit, actual_area, f"upper estimate equal or above actual area: {upper_limit} >= {actual_area}")

        # 2 disjoint circles of radius 1
        p_left = lib.point.Point(-3, 0, 0)
        p_right = lib.point.Point(3, 0, 0)

        c_left = lib.area.Circle(p_left, 1)
        c_right = lib.area.Circle(p_right, 1)

        p_square_llp = lib.point.Point(-4, -4, 0)

        s = lib.area.Square(p_square_llp, 8)

        full_squares, partial_squares = lib.area.quadtree([c_left, c_right], s, limit=0.01)

        actual_area = c_left.area() + c_right.area()

        lower_limit = full_squares
        upper_limit = full_squares + partial_squares

        self.assertLessEqual(lower_limit, actual_area, f"lower estimate equal or below actual area: {lower_limit} <= {actual_area}")
        self.assertGreaterEqual(upper_limit, actual_area, f"upper estimate equal or above actual area: {upper_limit} >= {actual_area}")

        # 2 circles, one contained within the other

        c_inner = lib.area.Circle(p_zero, 0.5)
        c_outer = lib.area.Circle(p_zero, 1)

        p_square_llp = lib.point.Point(-1, -1, 0)
        s = lib.area.Square(p_square_llp, 2)

        full_squares, partial_squares = lib.area.quadtree([c_inner, c_outer], s, limit=0.01)

        lower_limit = full_squares
        upper_limit = full_squares + partial_squares
        actual_area = c_outer.area()

        self.assertLessEqual(lower_limit, actual_area, f"lower estimate equal or below actual area: {lower_limit} <= {actual_area}")
        self.assertGreaterEqual(upper_limit, actual_area, f"upper estimate equal or above actual area: {upper_limit} >= {actual_area}")

        # 2 overlapping circles (single point, total area ~2pi)
        p_left = lib.point.Point(-1, 0, 0)
        p_right = lib.point.Point(1, 0, 0)

        c_left = lib.area.Circle(p_left, 1)
        c_right = lib.area.Circle(p_right, 1)

        # oversize but should work
        p_square_llp = lib.point.Point(-4, -4, 0)
        s = lib.area.Square(p_square_llp, 8)

        full_squares, partial_squares = lib.area.quadtree([c_left, c_right], s, limit=0.01)

        lower_limit = full_squares
        upper_limit = full_squares + partial_squares
        actual_area = c_left.area() + c_right.area()

        self.assertLessEqual(lower_limit, actual_area, f"lower estimate equal or below actual area: {lower_limit} <= {actual_area}")
        self.assertGreaterEqual(upper_limit, actual_area, f"upper estimate equal or above actual area: {upper_limit} >= {actual_area}")

        # 2 overlapping circles (need to calculate correct area)
        p_left = lib.point.Point(-0.5, 0, 0)
        p_right = lib.point.Point(0.5, 0, 0)

        c_left = lib.area.Circle(p_left, 1)
        c_right = lib.area.Circle(p_right, 1)

        # oversize but should work
        p_square_llp = lib.point.Point(-4, -4, 0)
        s = lib.area.Square(p_square_llp, 8)

        full_squares, partial_squares = lib.area.quadtree([c_left, c_right], s, limit=0.01)
        # TODO: compute the correct area for this and add an assert. Could probably
        # do it by hand dividing the overlapping circles into a polygon (square?)
        # and 2 circle arcs/segments.
        #print(f"{full_squares=}, {partial_squares=}. estimate: {full_squares + partial_squares/2} +/-{partial_squares/2}")

