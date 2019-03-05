from ase.atoms import Atoms
from .bopatom import BOPAtom
import numbers


class BOPAtoms(Atoms):

    def __init__(self, onsite_levels, **kwargs):
        # only use named arguments to avoid confusion with order of parent class constructor arguments
        super().__init__(**kwargs)
        self.set_array('onsite_levels', onsite_levels, dtype='float')

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')
            return BOPAtom(bopatoms=self, index=i)
        else:
            return super().__getitem__(i)
