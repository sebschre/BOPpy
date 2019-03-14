from ase.atom import Atom, atomproperty, names, chemical_symbols
from ase.atoms import *
from ase.neighborlist import NeighborList
import numpy as np
import numbers
import networkx as nx


names['onsite_level']             = ('onsite_levels', 0.0)
names['number_valence_orbitals']  = ('numbers_valence_orbitals', 5)  # pure-d valence
names['number_valence_electrons'] = ('numbers_valence_electrons', 7.0)
names['stoner_integral']          = ('stoner_integrals', 0.76)


class BOPAtom(Atom):
    """ A BOPAtom class
    """
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
        Atom.__init__(self, *args, **kwargs)
        if self.atoms is None:
            # This atom is not part of any Atoms object:
            self.data['onsite_level'] = onsite_level
            self.data['number_valence_orbitals'] = number_valence_orbitals
            self.data['number_valence_electrons'] = number_valence_electrons
            self.data['stoner_integral'] = stoner_integral

    def __repr__(self):
        return 'BOP'+super().__repr__()

    def __hash__(self):
        """
        This should return a hash function that does not change during lifetime,
        i.e. hashing the position array is not an option
        TODO: avoid hash collision
        :return:
        """
        if isinstance(self.index, numbers.Integral):
            return self.index
        else:
            return super().__hash__()

    def __eq__(self, other):
        """Check for equality of two BOPAtom objects.
        """
        if not isinstance(other, Atom):
            return False
        # return self.data == other.data
        return self.symbol == other.symbol and \
               np.all(self.position == other.position)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


def get_bopatoms(atoms: Atoms):
    atoms.__class__ = BOPAtoms
    atoms._init_bopatoms()
    return atoms


class BOPAtoms(Atoms):
    """ A BOPAtoms class
    """

    def __init__(self, *args,
                 onsite_levels=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._init_bopatoms(onsite_levels)

    def _init_bopatoms(self, onsite_levels=None):
        self._update_nl()
        self.set_array('onsite_levels', onsite_levels, dtype='float')
        self.graph = nx.Graph()
        self._update_graph_nodes()
        self._update_graph_edges()

    def _update_nl(self):
        cutoff = 3
        self.nl = NeighborList([cutoff] * len(self), skin=0.3, self_interaction=True)
        self.nl.update(self)

    def _update_graph_nodes(self):
        for atom in self:
            self.graph.add_node(atom.index, atom=atom)

    def _update_graph_edges(self):
        all_distances = self.get_all_distances(mic=True)
        for node in self.graph.nodes:
            indices, offsets = self.nl.get_neighbors(node)
            distances = self.get_distances(node, indices)
            print([self[i].number_valence_orbitals for i in indices])
            bonds_ddsigma = np.exp(-distances)
            bonds_ddpi = np.exp(-distances)
            bonds_ddelta = np.exp(-distances)
            self.graph.add_edges_from(
                [(node, other, {'bond': bond}) for (other, bond) in zip(indices, bonds_ddsigma)]
            )

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')
            return BOPAtom(atoms=self, index=i)
        else:
            return super().__getitem__(i)
