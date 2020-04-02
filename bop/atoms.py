import itertools
from enum import Enum
from typing import List, Iterator, TypeVar, Tuple, Union, Container, Iterable, Set, Dict, Mapping, Callable, FrozenSet, Hashable
import numpy as np
import networkx as nx
from periodictable import elements
from bop.coordinate_system import Position


class ValenceOrbitalType(Enum):
    S = 1
    P = 3
    D = 5


class ValenceOrbital:

    def __init__(self,
                 stoner_integral: float = None,
                 charge_penalty: float = None):
        self.stoner_integral = stoner_integral
        self.charge_penalty = charge_penalty


class AtomType:
    """
    The type of atom with element_name, number_valence_electrons, valence_orbital_dict and identifier (ident).
    """

    def __init__(self,
                 element_name: str,
                 number_valence_electrons: float = 10.0,
                 valence_orbital_dict: Dict[ValenceOrbitalType, ValenceOrbital] = {},
                 ident: int = 0
                 ):
        if element_name not in (el.symbol for el in elements):
            raise ValueError(f"Initialized AtomType with undefined element_name {element_name}")
        self.element_name = element_name
        self.number_valence_electrons = number_valence_electrons
        self.valence_orbital_dict = valence_orbital_dict
        self.ident = ident

    def __eq__(self, other: 'AtomType'):
        return self.element_name == other.element_name \
               and self.ident == other.ident \
               and self.valence_orbital_dict == other.valence_orbital_dict

    def __ne__(self, other: 'AtomType'):
        return not self == other

    def __repr__(self):
        return f"{self.element_name}({self.ident})"


class BondDefinitions:

    def __init__(self, atom_types: Iterable[AtomType]):
        self.atom_types = set(atom_types)

    def __contains__(self, atom_type: AtomType):
        return atom_type in self.atom_types

    def get_bond_func(self, atom_type1: AtomType, atom_type2: AtomType) -> Callable:
        """
        TODO: make this a property?
        :param atom_type1:
        :param atom_type2:
        :return:
        """
        if atom_type1 not in self.atom_types or atom_type2 not in self.atom_types:
            raise ValueError(f"atom type {atom_type1} or {atom_type2} not in BondDefinitions")
        return lambda x: np.exp(-x)


class BOPAtom:

    def __init__(self,
                 position: Union[Position, Tuple[float, float, float]],
                 atom_type: Union[AtomType, str],
                 onsite_levels: Dict[ValenceOrbitalType, float] = {}):
        if type(position) is not Position:
            position = Position(position)
        self.position = position
        if type(atom_type) is not AtomType:
            atom_type = AtomType(atom_type)
        self.atom_type = atom_type
        if atom_type.valence_orbital_dict.keys() != onsite_levels.keys():
            raise ValueError(f"Onsite levels {onsite_levels} do not match orbitals {atom_type.valence_orbital_dict}")

    def __repr__(self):
        return f"{self.atom_type} at {self.position}"


class BOPGraph(nx.Graph):

    def __init__(self,
                 atom_list: List[BOPAtom],
                 bond_definitions: BondDefinitions = None):
        super(BOPGraph, self).__init__()
        self.add_nodes_from(atom_list)
        self.bond_definitions = bond_definitions
        # nx.adjacency_matrix(self)**L

    def update_bond_definitions(self, bond_definitions: BondDefinitions) -> None:
        self.bond_definitions = bond_definitions

    def update_edges(self, cutoff=3) -> None:
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
