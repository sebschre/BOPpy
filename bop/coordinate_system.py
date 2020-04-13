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
            self.axes = np.array(axes)
        else:
            raise ValueError("has to be invertible")

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other: 'CoordinateSystem'):
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


class Position:

    def __init__(self,
                 pos_frac: Tuple[float, float, float],
                 cs: CoordinateSystem = CoordinateSystem()):
        self.pos_frac = np.array(pos_frac)
        self.coordinate_system = cs

    @property
    def pos_global(self):
        return np.dot(self.coordinate_system.axes, self.pos_frac)

    @pos_global.setter
    def pos_global(self, value: Tuple[float, float, float]):
        self.pos_frac = np.dot(np.linalg.inv(self.coordinate_system.axes), value)

    def __repr__(self):
        return f"{self.pos_global}"

    def __eq__(self, other: 'Position'):
        return np.all(self.pos_global == other.pos_global)

    def __ne__(self, other: 'Position'):
        return not self == other

    def __add__(self, other: 'Position'):
        pos_new = self.pos_global + other.pos_global
        return Position(pos_new)

    def __sub__(self, other: 'Position'):
        pos_new = self.pos_global - other.pos_global
        return Position(pos_new)

    def get_distance(self, other: 'Position') -> float:
        return np.sqrt(np.sum((self.pos_global - other.pos_global)**2))


class PBCPosition(Position):
    """
    Position class that considers periodic boundary conditions
    """
    pass
