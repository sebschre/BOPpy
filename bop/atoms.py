from ase.atom import Atom, atomproperty, names, chemical_symbols
from ase.atoms import Atoms
import numbers
import networkx as nx


names['onsite_level']             = ('onsite_levels', 0.0)
names['number_valence_orbitals']  = ('numbers_valence_orbitals', 5)  # pure-d valence
names['number_valence_electrons'] = ('numbers_valence_electrons', 7.0)
names['stoner_integral']          = ('stoner_integrals', 0.76)


class BOPAtom(Atom):
    onsite_level             = atomproperty('onsite_level', 'Atomic onsite level')
    number_valence_orbitals  = atomproperty('number_valence_orbitals', 'Number of valence orbtials')
    number_valence_electrons = atomproperty('number_valence_electrons', 'Number of valence electrons')
    stoner_integral          = atomproperty('stoner_integral', 'Stoner integral')

    def __init__(self, *args,
                 onsite_level=None,
                 number_valence_orbitals=None,
                 number_valence_electrons=None,
                 stoner_integral=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if self.atoms is None:
            # This atom is not part of any Atoms object:
            self.data['onsite_level'] = onsite_level
            self.data['number_valence_orbitals'] = number_valence_orbitals
            self.data['number_valence_electrons'] = number_valence_electrons
            self.data['stoner_integral'] = stoner_integral


class BOPAtoms(Atoms):

    def __init__(self, *args, onsite_levels=None, **kwargs):
        # only use named arguments to avoid confusion with order of parent class constructor arguments
        super().__init__(*args, **kwargs)
        self.graph = nx.Graph()
        for atom in self:
            self.graph.add_node(atom)
        self.set_array('onsite_levels', onsite_levels, dtype='float')

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')
            return BOPAtom(atoms=self, index=i)
        else:
            return super().__getitem__(i)
