import numpy as np
import networkx as nx
import itertools
from periodictable import elements
from bop.coordinate_system import Position
from typing import List, Iterator, TypeVar, Tuple


class BOPAtom:

    def __init__(self, position: Position, element_name: str):
        if element_name not in (el.symbol for el in elements):
            raise ValueError(f"Initialized BOPAtom with undefined element_name {element_name}")
        self.element_name = element_name
        if type(position) is not Position:
            position = Position(position)
        self.position = position
        self.onsite_level = None
        self.number_valence_orbitals = None
        self.number_valence_electrons = None
        self.stoner_integral = None
        self.charge_penalty = None

    def __repr__(self):
        return f"BOPAtom: {self.element_name} at {self.position}"


class BOPGraph(nx.DiGraph):

    def __init__(self, atom_list: List[BOPAtom]):
        super(BOPGraph, self).__init__()
        self.add_nodes_from(atom_list)
        # nx.adjacency_matrix(self)**L

    def update_edges(self, cutoff=3):
        for (pair, distance) in self._get_distances():
            if distance <= cutoff:
                self.add_edge(*pair)
            else:
                if self.has_edge(*pair):
                    self.remove_edge(*pair)

    def _get_distances(self) -> Iterator[Tuple[Tuple[BOPAtom, BOPAtom], float]]:
        """
        TODO: return pairs only once
        :return:
        """
        for pair in circular_pairwise(self.nodes):
            yield (pair, pair[0].position.get_distance(pair[1].position))

    def _init_bopatoms(self, onsite_levels=None):
        self._update_nl()
        self.set_array('onsite_levels', onsite_levels, dtype='float')
        self.graph = nx.Graph()
        self._update_graph_nodes()
        self._update_graph_edges()

    def _update_nl(self):
        cutoff = 3
        #self.nl = NeighborList([cutoff] * len(self), skin=0.3, self_interaction=True)
        #self.nl.update(self)

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

    #def __getitem__(self, i):
    #    if isinstance(i, numbers.Integral):
    #        natoms = len(self)
    #        if i < -natoms or i >= natoms:
    #            raise IndexError('Index out of range.')
    #        return BOPAtom(atoms=self, index=i)
    #    else:
    #        return super().__getitem__(i)


T = TypeVar('T')


def circular_pairwise(it: Iterator[T]) -> Iterator[Tuple[T, T]]:
    """
    https://stackoverflow.com/a/36918890/573256
    :return:
    """
    if hasattr(it, '__iter__'):
        first, snd = itertools.tee(it)
        second = itertools.cycle(snd)
        next(second)
        return zip(first, second)
    else:
        second = itertools.cycle(it)
        next(second)
        return zip(it, second)
