'''functions for estimating the area coverage of a meshtastic network
'''
from enum import Enum
import logging
from math import pi

from lib.config import Config
from lib.node import NodeConfig
from lib.point import Point

logger = logging.getLogger(__name__)

LIMIT_CASE_AREA = 10 # meters squared

class COVERED_ENUM(Enum):
    FULL = 'FULL'
    EMPTY = 'EMPTY'
    PARTIAL = 'PARTIAL'

class Circle:
    def __init__(self, center: Point, radius: float):
        self.center = center
        self.radius = radius

    def area(self):
        return pi * (self.radius**2)

class Square:
    def __init__(self, lower_left_point, side_length):
        '''Points labeled clockwise starting from lower left (min x, y)
        B+--+C
         |  |
         |  |
        A+--+D
        '''
        llp = lower_left_point
        self.A = Point(llp.x, llp.y, 0)
        self.B = Point(llp.x, llp.y + side_length, 0)
        self.C = Point(llp.x + side_length, llp.y + side_length, 0)
        self.D = Point(llp.x + side_length, llp.y, 0)
        self.side_length = side_length

    def area(self):
        return self.side_length ** 2

    def contains(self, p: Point) -> bool:
        if self.A.x <= p.x <= self.D.x and \
            self.A.y <= p.y <= self.B.y:
            return True
        else:
            return False

    def covered_by(self, c: Circle) -> COVERED_ENUM:
        '''check 4 corners of square for containment within circle c.
        Return enum indicating fully covered, empty (not covered), or
        partially covered
        '''
        A_cov = self.A.euclidean_distance(c.center) <= c.radius
        B_cov = self.B.euclidean_distance(c.center) <= c.radius
        C_cov = self.C.euclidean_distance(c.center) <= c.radius
        D_cov = self.D.euclidean_distance(c.center) <= c.radius
        if A_cov and B_cov and C_cov and D_cov:
            # c completely covers this square
            return COVERED_ENUM.FULL
        elif not (A_cov or B_cov or C_cov or D_cov):
            # circle may be within square. Check if circle center is within
            # square and if so return partial. Otherwise we cannot be covered
            if self.contains(c.center):
                return COVERED_ENUM.PARTIAL
            else:
                return COVERED_ENUM.EMPTY
        else:
            # some corners covered by circle, others not
            return COVERED_ENUM.PARTIAL

def quadtree(circles: [Circle], begin_square: Square, limit=LIMIT_CASE_AREA) -> (float, float):
    '''estimate algorithm: subdivide an area into squares, and check if their
    corner points are within circles to determine for every square if it is:
    - contained within any circle
    - not contained within any circle
    - partial

    Lower bound estimate of network area is covered by full squares.
    Upper is full + partial, while partial is 'error'. Recurse on partial
    squares until desired granularity is reached/error below desired amount.

    Because of the recursion I suspect tracking total error will be tricky, so just
    define a smallest allowable area of a square.

    Important assumption: you begin with a square that fully contains all the
    circles you are measuring the area of. If not, you will get incorrect results.

    Arguments:
    circles: list of circles to estimate area of
    begin_square: square to begin estimation with (unknown if full/partial/empty)

    Returns:
    (full, partial) -- area of all computed full/partial sub-squares
    '''
    # base case: if we're less than limit area, check our coverage and return
    # check corner points against all circles.
    # - full: return full
    # - empty: return empty
    # partial: split up into sub-squares, recurse, add/return result
    logger.debug(f"quadtree called with {len(circles)} circles & square sidelength {begin_square.side_length}")
    full_area = 0
    partial_area = 0
    begin_square_area = begin_square.area()
    partially_covered = False
    for c in circles:
        cov = begin_square.covered_by(c)
        if cov == COVERED_ENUM.FULL:
            # we are fully covered by a circle. compute area & return.
            # no recursion is possible (done)
            logger.debug(f"\tfully covered")
            full_area = begin_square_area
            return (full_area, partial_area)
        elif cov == COVERED_ENUM.PARTIAL:
            # a circle partially covers us, so we can't be uncovered by
            # all circles. Will need to recurse.
            partially_covered = True
    if not partially_covered:
        # not fully or partially covered by any circle. Uncovered by all
        # circles, compute area & return. no recursion possible (done).
        logger.debug(f"\tnot covered")
        return (0, 0)

    # if we reach here, we must be partially covered. unless we've hit the
    # limit, subdivide & recurse. If we have hit the limit, return our
    # area as partial.
    logger.debug(f"\tnot covered")
    if begin_square.side_length <= limit:
        logger.debug(f"\tlimit reached")
        return (0, begin_square_area)
    else:
        logger.debug(f"\trecursing...")
        # split into 4 squares, recurse, sum & return totals
        new_length = begin_square.side_length / 2
        A = Square(begin_square.A, new_length)
        B_point = Point(begin_square.A.x, begin_square.A.y + new_length, 0)
        B = Square(B_point, new_length)
        C_point = Point(begin_square.A.x + new_length, begin_square.A.y + new_length, 0)
        C = Square(C_point, new_length)
        D_point = Point(begin_square.A.x + new_length, begin_square.A.y, 0)
        D = Square(D_point, new_length)

        res = [quadtree(circles, s, limit) for s in [A, B, C, D]]

        for r in res:
            full_area += r[0]
            partial_area += r[1]

        return (full_area, partial_area)
    pass

def estimate_coverage_area(conf: Config, node_confs: [NodeConfig]) -> (float, float):
    '''use an algorithm of our choosing (currently 'quadtree') to estimate
    the area covered by the given network. The sim Config and node configs are
    used to calculate estimated coverage circles for each node, which are
    then used to estimate the network's coverage area.

    Arguments:
    conf -- Config object describing the simulation
    node_confs -- list of NodeConfig objects describing the network

    Returns:
    (area, error) -- estimated area covered by the network in meters^2, and error
    amount. Upper/lower bounds are `error` amount in either direction from the
    estimate.
    '''
    # estimate coverage areas of nodes based on model
    node_coverage_circles = []
    for n in node_confs:
        r = n.estimate_max_range(conf)
        c = Circle(n.position, r)
        node_coverage_circles.append(c)

    # determine bounding square for network
    left_limits = [c.center.x - c.radius for c in node_coverage_circles]
    right_limits = [c.center.x + c.radius for c in node_coverage_circles]
    lower_limits = [c.center.y - c.radius for c in node_coverage_circles]
    upper_limits = [c.center.y + c.radius for c in node_coverage_circles]

    min_x = min(left_limits)
    max_x = max(right_limits)

    min_y = min(lower_limits)
    max_y = max(upper_limits)

    bounding_square_llp = Point(min_x, min_y, 0)

    side_length = max(max_x - min_x, max_y - min_y)
    bounding_square = Square(bounding_square_llp, side_length)

    # compute estimate
    lower_estimate, error = quadtree(node_coverage_circles, bounding_square)

    # choose midpoint of interval, provide error amount
    return (lower_estimate + error/2, error/2)
