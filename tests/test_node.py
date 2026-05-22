import unittest

import lib.node

class TestNodeConf(unittest.TestCase):
    def test_reject_rssi_and_pathloss_between_identical_nodes(self):

        from lib.config import CONFIG
        from lib.point import Point
        conf = CONFIG

        # reasonable values
        nodeconf = lib.node.NodeConfig(0, Point(0, 0, 1), 1, 30, 902e6)

        with self.assertRaises(ValueError, msg="cannot compute rssi/pathloss between the same nodes (by id)"):
            nodeconf.compute_rssi_and_pathloss_to(nodeconf, conf)

    def test_require_nodes_have_positive_height_agl(self):
        from lib.point import Point

        p_zero = Point(0, 0, 0)
        p_neg = Point(0, 0, -1)
        with self.assertRaises(ValueError, msg="NodeConf must be given position with positive height agl"):
            n = lib.node.NodeConfig(0, p_zero, 1, 30, 902e6)

        with self.assertRaises(ValueError, msg="NodeConf must be given position with positive height agl"):
            n = lib.node.NodeConfig(0, p_neg, 1, 30, 902e6)
