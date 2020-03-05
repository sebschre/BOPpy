import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from bop.coordinate_system import CoordinateSystem


class TestCoordinateSystem(unittest.TestCase):

    def test_eq(self):
        cs = CoordinateSystem()
        rot = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
        rot_cs = CoordinateSystem(rot.apply(cs.axes))
        self.assertTrue(rot_cs == cs)
