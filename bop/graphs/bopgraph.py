import functools
import itertools
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List, Tuple, TypeVar

import networkx as nx
import numpy as np
import scipy
from numba import jit
from scipy.spatial.transform import Rotation as R

from bop.graphs.calculator import GraphCalculator
from bop.nodes.node import Node


class NodeInteractionCalculator(ABC):
    @abstractmethod
    def get_interaction(self, node1: Node, node2: Node) -> scipy.sparse.spmatrix:
        pass
        # return scipy.sparse.dia_matrxi(np.zeros(1, 1))


class BOPAtomInteractionCalculator(NodeInteractionCalculator):
    def get_interaction(self, node1: Node, node2: Node) -> scipy.sparse.spmatrix:
        if node1 != node2:
            data = np.array([[1, 2, 3, 4]])
            offsets = np.array([0])
            sparse_matrix = scipy.sparse.dia_matrix(
                np.random.rand(4, 4), dtype=np.float64
            )
        else:
            sparse_matrix = scipy.sparse.dia_matrix(
                np.random.rand(4, 4), dtype=np.float64
            )
        return sparse_matrix


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
    __hop_list = [
        x[2]["hop"].toarray() for x in path.edges(data=True) if x[2]["hop"] is not None
    ]
    if with_njit:
        return _multiply_arrays_njit(np.array(__hop_list))
    else:
        return _multiply_arrays(np.array(__hop_list))


T = TypeVar("T")


def circular_pairwise(it: Iterable[T]) -> Iterator[Tuple[T, T]]:
    """
    https://stackoverflow.com/a/36918890/573256
    :return:
    """
    if hasattr(it, "__iter__"):
        first, snd = itertools.tee(it)
        second = itertools.cycle(snd)
        next(second)
        return zip(first, second)
    else:
        second = itertools.cycle(it)
        next(second)
        return zip(it, second)


class BOPGraph:
    def __init__(
        self,
        node_list: List[Node],
        graph_calc: GraphCalculator,
        node_interaction_calc: NodeInteractionCalculator = None,
    ):
        # TODO: unexpected behavior if graph_calc was initialized before
        self._graph_calc = graph_calc
        self._graph_calc.add_nodes_from(node_list)
        self.node_interaction_calc = node_interaction_calc
        # nx.adjacency_matrix(self)**L

    def update_edges(self, cutoff=3) -> None:
        def __node_interaction(n1, n2):
            return (
                self.node_interaction_calc.get_interaction(n1, n2)
                if self.node_interaction_calc
                else None
            )

        # first add the "onsite hops"
        for node in self._graph_calc.nodes:
            if not self._graph_calc.has_edge(node, node):
                onsite_hop = __node_interaction(node, node)
                self._graph_calc.add_edge(
                    node,
                    node,
                    distance=0,
                    hop=onsite_hop,
                    rot=R.from_quat([1, 0, 0, 0]),
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
            [
                _multiply_hops_in_path(x)
                for x in self._graph_calc.all_paths_from_to(from_node, to_node, depth)
            ],
        )

    def _get_distances(self) -> Iterator[Tuple[Tuple[Node, Node], float]]:
        """
        TODO: return pairs only once
        :return:
        """
        for pair in circular_pairwise(self._graph_calc.nodes):
            yield (pair, pair[0].get_distance(pair[1]))
