import functools
import itertools
from abc import ABC, abstractmethod
from typing import TypeVar, List, Iterable, Iterator, Tuple
from bop.atoms import BOPAtom
from bop.graph_calculator import GraphCalculator, Node
from numba import jit
import scipy
from scipy.spatial.transform import Rotation as R
import numpy as np
import networkx as nx


class NodeInteractionCalculator(ABC):

    @abstractmethod
    def get_interaction(self, node1: Node, node2: Node) -> scipy.sparse.spmatrix:
        pass
        # return scipy.sparse.dia_matrxi(np.zeros(1, 1))


class BOPAtomInteractionCalculator(NodeInteractionCalculator):

    def get_interaction(self, atom1: BOPAtom, atom2: BOPAtom) -> scipy.sparse.spmatrix:
        if atom1 != atom2:
            data = np.array([[1, 2, 3, 4]])
            offsets = np.array([0])
            sparse_matrix = scipy.sparse.dia_matrix(np.random.rand(4, 4), dtype=np.float)
        else:
            sparse_matrix = scipy.sparse.dia_matrix(np.random.rand(4, 4), dtype=np.float)
        return sparse_matrix


class BOPGraph:

    def __init__(self, atom_list: List[BOPAtom], graph_calc: GraphCalculator,
                 node_interaction_calc: NodeInteractionCalculator = None):
        # TODO: unexpected behavior if graph_calc was initialized before
        self._graph_calc = graph_calc
        self._graph_calc.add_nodes_from(atom_list)
        self.node_interaction_calc = node_interaction_calc
        self.saved_hop_paths = dict()
        # nx.adjacency_matrix(self)**L

    def update_edges(self, cutoff=3) -> None:
        def __node_interaction(n1, n2):
            return self.node_interaction_calc.get_interaction(n1, n2) if self.node_interaction_calc else None

        # first add the "onsite hops"
        for node in self._graph_calc.nodes:
            if not self._graph_calc.has_edge(node, node):
                onsite_hop = __node_interaction(node, node)
                self._graph_calc.add_edge(
                    node, node, distance=0, hop=onsite_hop, rot=R.from_quat([1, 0, 0, 0])
                )
        # then find all neighboring nodes and add edges if distance <= cutoff
        for (pair, distance) in self._get_distances():
            if distance <= cutoff:
                if not self._graph_calc.has_edge(*pair):
                    hop = __node_interaction(*pair)
                    self._graph_calc.add_edge(
                        *pair, distance=distance, hop=hop, rot=R.from_quat([1, 0, 0, 0])
                    )
            else:
                if self._graph_calc.has_edge(*pair):
                    self._graph_calc.remove_edge(*pair)

    def compute_interference_path(self, from_node: Node, to_node: Node, depth: int):
        return functools.reduce(
            lambda x, y: x + y,
            [_multiply_hops_in_path(x) for x in self._graph_calc.all_paths_from_to(from_node, to_node, depth)]
        )

    def compute_interference_path_2(self, from_atom: BOPAtom, to_atom: BOPAtom, depth: int):
        valence_dimension_in = sum([x.value for x in from_atom.atom_type.valence_orbital_dict.keys()])
        valence_dimension_out = sum([x.value for x in to_atom.atom_type.valence_orbital_dict.keys()])
        valence_dimension_in = 3
        valence_dimension_out = 3
        mult_paths = np.zeros((valence_dimension_in, valence_dimension_out))
        for path in self._graph_calc.all_paths_from_to(from_atom, to_atom, depth):
            atom_tuple = tuple([x[0] for i, x in enumerate(path.edges) if i == 0] + [x[1] for x in path.edges])
            mult_paths += self.__multiply_hops_in_path(atom_tuple)
        return mult_paths

        #     edges_in_path = path.edges(data=True)
        #     # check if this path has been computed before
        #     nodes_in_path = tuple(x[0] for x in edges_in_path) + (to_node, )
        #     # find longest previously computed and saved interference path
        #     nodes_key = nodes_in_path
        #     while len(nodes_key) > 2:
        #         if nodes_key not in self.saved_hop_paths.keys():
        #             nodes_key = nodes_key[:-1]
        #         else:
        #             break
        #     if len(nodes_key) > 2:
        #         saved_interf_path = self.saved_hop_paths[nodes_key]
        #         remaining_hops = list(edges_in_path)[len(nodes_key)-1:]
        #     for node1, node2, edge_data in path.edges(data=True):
        #         if (node1, node2) not in self.saved_hop_paths.keys():
        #             self.saved_hop_paths[(node1, node2)] = _multiply_arrays(edge_data['hop'].toarray())
        #             print(self.saved_hop_paths)
        # print('global')
        # print(self.saved_hop_paths)

    def __multiply_hops_in_path(self, atom_tuple: Tuple[BOPAtom]) -> np.array:
        # atom_tuple = [x[0] for i, x in enumerate(path.edges) if i == 0] + [x[1] for x in path.edges]
        atom_ids_in_path = tuple([x.atom_id for x in atom_tuple])
        if atom_ids_in_path in self.saved_hop_paths.keys():
            return self.saved_hop_paths[atom_ids_in_path]
        total_length = len(atom_tuple)
        if total_length == 2:
            atom_selection = tuple(atom_tuple)
            # hop = path[atom_selection[0]][atom_selection[1]][1]['hop']
            # hop = self._graph_calc.edges[atom_selection[0]][atom_selection[1]][1]['hop']
            # self.saved_hop_paths[atom_selection] = hop
            # return hop
            return np.eye(3)
        elif total_length == 3:
            return np.eye(3)
        length_to_check = total_length - 1
        while length_to_check > total_length / 2:
            for i in range(0, total_length - length_to_check):
                atom_selection = atom_tuple[i:i + length_to_check]
                atom_ids_in_selection = tuple([x.atom_id for x in atom_selection])
                if atom_ids_in_selection in self.saved_hop_paths.keys():
                    left_atoms = atom_tuple[0:i]
                    left_hops = self.__multiply_hops_in_path(left_atoms)
                    self.saved_hop_paths[left_atoms] = left_hops
                    right_atoms = atom_tuple[i + length_to_check:]
                    right_hops = self.__multiply_hops_in_path(right_atoms)
                    self.saved_hop_paths[right_atoms] = right_hops
                    # return np.dot(np.dot(left_hops, self.saved_hop_paths[atom_selection]), right_hops)
                    return np.eye(3)
            length_to_check -= 1
        if total_length % 2 == 0:
            left_atoms = atom_tuple[:int(total_length / 2)]
            right_atoms = atom_tuple[int(total_length / 2):]
        else:
            left_atoms = atom_tuple[:int((total_length + 1) / 2)]
            right_atoms = atom_tuple[int((total_length + 1) / 2):]
        left_hops = self.__multiply_hops_in_path(left_atoms)
        self.saved_hop_paths[left_atoms] = left_hops
        right_hops = self.__multiply_hops_in_path(right_atoms)
        self.saved_hop_paths[right_atoms] = right_hops
        return np.dot(left_hops, right_hops)

    def compute_all_interference_paths(self, depth: int):
        # TODO: improve performance... i < j
        for node1 in self._graph_calc.nodes:
            for node2 in self._graph_calc.nodes:
                self.compute_interference_path(from_node=node1, to_node=node2, depth=depth)

    def _get_distances(self) -> Iterator[Tuple[Tuple[Node, Node], float]]:
        """
        TODO: return pairs only once
        :return:
        """
        for pair in circular_pairwise(self._graph_calc.nodes):
            yield (pair, pair[0].get_distance(pair[1]))



@jit(nopython=True)
def numba_dot(m1, m2):
    return np.dot(m1, m2)


@jit(nopython=True)
def _multiply_arrays_njit(arrays: np.ndarray) -> np.ndarray:
    arr_result = arrays[0]
    for i in range(1, len(arrays)):
        arr_result = numba_dot(arr_result, arrays[i])
    return arr_result


def _multiply_arrays(arrays: np.ndarray) -> np.ndarray:
    return functools.reduce(np.dot, arrays)


def _multiply_hops_in_path(path: nx.MultiDiGraph, with_njit=False) -> np.ndarray:
    __hop_list = [x[2]['hop'].toarray() for x in path.edges(data=True) if x[2]['hop'] is not None]
    if with_njit:
        return _multiply_arrays_njit(np.array(__hop_list))
    else:
        return _multiply_arrays(np.array(__hop_list))


T = TypeVar('T')


def circular_pairwise(it: Iterable[T]) -> Iterator[Tuple[T, T]]:
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
