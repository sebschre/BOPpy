"""Analytic Bond-Order Potential Calculator"""

from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList


class BOPcalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.neighborlist = self.nl = NeighborList([0.5 * self.rc_list] * len(atoms), self_interaction=False)
