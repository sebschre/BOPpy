import itertools
from enum import Enum
from typing import List, Iterator, TypeVar, Tuple, Union, Container, Iterable, Set, Dict, Mapping, Callable, FrozenSet, \
    Hashable
import collections
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
import igraph as igr
from periodictable import elements
from bop.coordinate_system import Position


class ValenceOrbitalType(Enum):
    """

    """
    S = 1
    P = 3
    D = 5


class ValenceOrbitalParameter:
    """

    """

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
                 valence_orbital_dict: Dict[ValenceOrbitalType, ValenceOrbitalParameter] = {},
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


class GraphCalculator(ABC):

    @property
    @abstractmethod
    def nodes(self) -> List[BOPAtom]:
        pass

    @property
    @abstractmethod
    def edges(self):
        pass

    @abstractmethod
    def add_nodes_from(self, node_list: List[BOPAtom]) -> None:
        pass

    @abstractmethod
    def add_edge(self, node1: BOPAtom, node2: BOPAtom) -> None:
        pass

    @abstractmethod
    def has_edge(self, node1: BOPAtom, node2: BOPAtom) -> bool:
        pass

    @abstractmethod
    def remove_edge(self, node1: BOPAtom, node2: BOPAtom) -> None:
        pass

    @abstractmethod
    def _neighbors(self, node: BOPAtom) -> List[BOPAtom]:
        pass

    def depth_limited_search(self, initial_node: BOPAtom, depth: int, self_counting=True):
        max_depth = depth

        def __recursion(node, depth_remaining: int, sc=self_counting):
            nonlocal max_depth
            level = max_depth - depth_remaining + 1
            if depth_remaining > 0:
                if sc:
                    # TODO: is this necessary if (node, node) is an edge?
                    yield (level, (node, node))
                    yield from __recursion(node, depth_remaining - 1)
                for neighbor in self._neighbors(node):
                    yield (level, (node, neighbor))  # TODO: yield edge?
                    yield from __recursion(neighbor, depth_remaining - 1)

        return __recursion(initial_node, depth)


class NxGraphCalculator(GraphCalculator):

    def __init__(self):
        self.__graph = nx.Graph()

    @property
    def nodes(self) -> List[BOPAtom]:
        return list(self.__graph.nodes)

    @property
    def edges(self):
        return self.__graph.edges

    def add_nodes_from(self, node_list: List[BOPAtom]) -> None:
        return self.__graph.add_nodes_from(node_list)

    def add_edge(self, node1: BOPAtom, node2: BOPAtom) -> None:
        self.__graph.add_edge(node1, node2)

    def has_edge(self, node1: BOPAtom, node2: BOPAtom) -> bool:
        return self.__graph.has_edge(node1, node2)

    def remove_edge(self, node1: BOPAtom, node2: BOPAtom) -> None:
        return self.__graph.remove_edge(node1, node2)

    def _neighbors(self, node: BOPAtom) -> List[BOPAtom]:
        return self.__graph.neighbors(node)


class IGraphCalculator(GraphCalculator):

    def __init__(self):
        self.__graph = igr.Graph()

    @property
    def nodes(self) -> List[BOPAtom]:
        return [v['name'] for v in self.__graph.vs]

    @property
    def edges(self):
        return self.__graph.get_edgelist()

    def __bopatom_to_vertex(self, node: BOPAtom) -> igr.Vertex:
        vertices = [v for v in self.__graph.vs if v['name'] == node]
        if vertices:
            return vertices[0]
        else:
            return None

    def add_nodes_from(self, node_list: List[BOPAtom]) -> None:
        self.__graph.add_vertices(node_list)

    def add_edge(self, node1: BOPAtom, node2: BOPAtom) -> None:
        v = [self.__bopatom_to_vertex(x) for x in (node1, node2)]
        self.__graph.add_edge(*v)

    def has_edge(self, node1: BOPAtom, node2: BOPAtom) -> bool:
        v = [self.__bopatom_to_vertex(x) for x in (node1, node2)]
        return self.__graph.get_eid(*v, error=False) != -1

    def remove_edge(self, node1: BOPAtom, node2: BOPAtom) -> None:
        v = [self.__bopatom_to_vertex(x) for x in (node1, node2)]
        edge_id = self.__graph.get_eid(*v, error=False)
        self.__graph.delete_edges(edge_id)

    def _neighbors(self, node: BOPAtom):
        vertex = self.__bopatom_to_vertex(node)
        neighbor_indices = self.__graph.neighbors(vertex)
        neighbor_nodes = self.__graph.vs[neighbor_indices]["name"]
        return neighbor_nodes


class BOPGraph:

    def __init__(self, atom_list: List[BOPAtom], graph_calc: GraphCalculator, bond_definitions: BondDefinitions = None,
                 **attr):
        super().__init__(**attr)
        self._graph_calc = graph_calc  # TODO: unexpected behavior if graph_calc was initialized before
        self._graph_calc.add_nodes_from(atom_list)
        self.bond_definitions = bond_definitions
        # nx.adjacency_matrix(self)**L

    def update_edges(self, cutoff=3) -> None:
        for (pair, distance) in self._get_distances():
            if distance <= cutoff:
                if not self._graph_calc.has_edge(*pair):
                    self._graph_calc.add_edge(*pair)
            else:
                if self._graph_calc.has_edge(*pair):
                    self._graph_calc.remove_edge(*pair)

    def _get_distances(self) -> Iterator[Tuple[Tuple[BOPAtom, BOPAtom], float]]:
        """
        TODO: return pairs only once
        :return:
        """
        for pair in circular_pairwise(self._graph_calc.nodes):
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
