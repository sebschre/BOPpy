import numpy as np
from typing import Tuple


class CoordinateSystem:
    """
    The coordinate system
    """
    def __init__(self,
                 axes: Tuple[
                     Tuple[float, float, float],
                     Tuple[float, float, float],
                     Tuple[float, float, float]
                 ] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        """
        :param axes:
        """
        if np.linalg.det(axes) != 0:
            self.axes = axes
        else:
            raise ValueError("has to be invertible")

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        """
        Two coordinate systems are considered equal,
        if they can be transformed into each other by a rotation
        :param other:
        :return:
        """
        rot = np.dot(other.axes, self.axes)
        if np.all(np.dot(np.transpose(rot), rot) == np.eye(3)):
            if np.linalg.det(rot) == 1:
                return True
        return False

    def __repr__(self):
        return f"CoordSys: {self.axes}"
