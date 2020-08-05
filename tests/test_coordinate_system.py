import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from bop.coordinate_system import CoordinateSystem, Position


class TestCoordinateSystem(unittest.TestCase):

    def test_equality_of_coordinate_systems(self):
        cs = CoordinateSystem(np.random.random((3, 3)))
        rot = R.from_matrix(np.eye(3))
        rot_cs = CoordinateSystem(rot.apply(cs.axes))
        np.testing.assert_almost_equal(rot_cs.axes, cs.axes)


class TestPosition(unittest.TestCase):

    def test_eq(self):
        self.assertTrue(Position((1, 0, 0)) == Position((1, 0, 0)))

    def test_ne(self):
        self.assertFalse(Position((1, 1, 0)) == Position((1, 0, 0)))

    def test_add_eq(self):
        pos1 = Position((1, 0, 0))
        pos2 = Position((1, 1, 0))
        pos3 = Position((2, 1, 0))
        self.assertTrue(pos1 + pos2 == pos3)

    def test_sub_eq(self):
        pos1 = Position((1, 0, 0))
        pos2 = Position((1, 1, 0))
        pos3 = Position((0, 1, 0))
        self.assertTrue(pos2 - pos1 == pos3)

    def test_get_distance(self):
        pos1 = Position((1, 0, 0))
        pos2 = Position((1, -1.5, 0))
        dist = pos1.get_distance(pos2)
        self.assertTrue(dist == 1.5)


if __name__ == '__main__':
    unittest.main()
