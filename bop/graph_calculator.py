import functools
import itertools
from abc import ABC, abstractmethod
from typing import List, Iterator, TypeVar, Tuple, Iterable

import igraph as igr
import networkx as nx
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R


class Node(ABC):

    @abstractmethod
    def get_distance(self, other: 'Node'):
        pass


class GraphCalculator(ABC):

    @property
    @abstractmethod
    def nodes(self) -> List[Node]:
        pass

    @property
    @abstractmethod
    def edges(self):
        pass

    @abstractmethod
    def add_nodes_from(self, node_list: List[Node]) -> None:
        pass

    @abstractmethod
    def add_edge(self, node1: Node, node2: Node, **attr) -> None:
        pass

    @abstractmethod
    def has_edge(self, node1: Node, node2: Node) -> bool:
        pass

    @abstractmethod
    def remove_edge(self, node1: Node, node2: Node) -> None:
        pass

    @abstractmethod
    def _neighbors(self, node: Node) -> List[Node]:
        pass

    def depth_limited_search(self, initial_node: Node, depth: int) -> Iterator[Tuple[Node, Node, int]]:
        max_depth = depth

        def __recursion(node, depth_remaining: int):
            nonlocal max_depth
            level = max_depth - depth_remaining + 1
            if depth_remaining > 0:
                for neighbor in self._neighbors(node):
                    yield (node, neighbor, level)
                    yield from __recursion(neighbor, depth_remaining - 1)

        yield from __recursion(initial_node, depth)

    def all_paths(self, initial_node: Node, depth: int):
        path = [None] * depth
        for level, edge in self.depth_limited_search(initial_node, depth=depth):
            path[level - 1] = edge
            if level == depth:
                yield path

    @abstractmethod
    def all_paths_from_to(self, from_node: Node, to_node: Node, depth_limit: int) -> Iterable[nx.MultiDiGraph]:
        pass
        # for path in self.all_paths(initial_node=initial_node, depth=depth):
        #     if path[-1][-1] == final_node:
        #         yield path


class NxGraphCalculator(GraphCalculator):

    def __init__(self):
        self.__graph = nx.Graph()

    @property
    def nodes(self) -> List[Node]:
        return list(self.__graph.nodes)

    @property
    def edges(self):
        return self.__graph.edges

    def add_nodes_from(self, node_list: List[Node]) -> None:
        return self.__graph.add_nodes_from(node_list)

    def add_edge(self, node1: Node, node2: Node, **attr) -> None:
        self.__graph.add_edge(node1, node2, **attr)

    def _dfs_multi_edge_tree(self, source=None, depth_limit=None, reverse_count_from: int = None) -> nx.MultiDiGraph:
        """
        TODO: find nicer solution than reverse_count_from
        :param source:
        :param depth_limit:
        :param reverse_count_from:
        :return:
        """
        tree = nx.MultiDiGraph()
        if source is None:
            tree.add_nodes_from(self.__graph)
        else:
            tree.add_node(source)
        if reverse_count_from:
            for node1, node2, depth in self.depth_limited_search(source, depth_limit):
                tree.add_edge(node2, node1, reverse_count_from -
                              depth + 1, **self.edges[node1, node2])
        else:
            for node1, node2, depth in self.depth_limited_search(source, depth_limit):
                tree.add_edge(node1, node2, depth, **self.edges[node1, node2])
        return tree

    def get_edge(self, node1: Node, node2: Node):
        return self.__graph.edges([node1, node2])

    def has_edge(self, node1: Node, node2: Node) -> bool:
        return self.__graph.has_edge(node1, node2)

    def remove_edge(self, node1: Node, node2: Node) -> None:
        return self.__graph.remove_edge(node1, node2)

    def _neighbors(self, node: Node) -> List[Node]:
        return self.__graph.neighbors(node)

    def _connection_trees_from_to(
            self, from_node: Node, to_node: Node, depth_limit: int) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
        if depth_limit % 2 == 0:
            depth1, depth2 = int(depth_limit / 2), int(depth_limit / 2)
        else:
            depth1, depth2 = int((depth_limit + 1) /
                                 2), int((depth_limit - 1) / 2)
        tree1 = self._dfs_multi_edge_tree(from_node, depth_limit=depth1)
        tree2 = self._dfs_multi_edge_tree(
            to_node, depth_limit=depth2, reverse_count_from=depth_limit)
        return tree1, tree2

    def _connection_graph_from_to(self, from_node: Node, to_node: Node, depth_limit: int) -> nx.MultiDiGraph:
        (tree1, tree2) = self._connection_trees_from_to(
            from_node, to_node, depth_limit)
        return nx.compose(tree1, tree2)

    def all_paths_from_to(self, from_node: Node, to_node: Node, depth_limit: int) -> Iterable[nx.MultiDiGraph]:
        """
        TODO: come up with more performant solution
        :param from_node:
        :param to_node:
        :param depth_limit:
        :return:
        """
        tree_full = self._connection_graph_from_to(
            from_node, to_node, depth_limit)

        def __recursion(node_from: Node, depth_level: int, path: nx.MultiDiGraph):
            for _, node_to, depth in tree_full.out_edges(node_from, keys=True):
                if depth == depth_level:
                    path.add_edge(node_from, node_to, depth_level,
                                  **self.edges[node_from, node_to])
                    if depth_level == depth_limit:
                        yield path
                    else:
                        yield from __recursion(node_to, depth_level + 1, path)
                    path.remove_edge(node_from, node_to, depth_level)

        yield from __recursion(from_node, 1, nx.MultiDiGraph())


class IGraphCalculator(GraphCalculator):

    def __init__(self):
        self.__graph = igr.Graph()

    @property
    def nodes(self) -> List[Node]:
        return [v['name'] for v in self.__graph.vs()]

    @property
    def edges(self):
        return self.__graph.get_edgelist()

    def get_edge(self, node1: Node, node2: Node):
        pass

    def __node_to_vertex(self, node: Node) -> igr.Vertex:
        vertices = [v for v in self.__graph.vs() if v['name'] == node]
        if vertices:
            return vertices[0]
        else:
            return None

    def add_nodes_from(self, node_list: List[Node]) -> None:
        self.__graph.add_vertices(node_list)

    def add_edge(self, node1: Node, node2: Node, **attr) -> None:
        v = [self.__node_to_vertex(x) for x in (node1, node2)]
        self.__graph.add_edge(*v)

    def has_edge(self, node1: Node, node2: Node) -> bool:
        v = [self.__node_to_vertex(x) for x in (node1, node2)]
        return self.__graph.get_eid(*v, error=False) != -1

    def remove_edge(self, node1: Node, node2: Node) -> None:
        v = [self.__node_to_vertex(x) for x in (node1, node2)]
        edge_id = self.__graph.get_eid(*v, error=False)
        self.__graph.delete_edges(edge_id)

    def _neighbors(self, node: Node):
        vertex = self.__node_to_vertex(node)
        neighbor_indices = self.__graph.neighbors(vertex)
        neighbor_nodes = self.__graph.vs()[neighbor_indices]["name"]
        return neighbor_nodes

    def all_paths_from_to(self, initial_node: Node, final_node: Node, depth: int):
        pass


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
            sparse_matrix = scipy.sparse.dia_matrix((data, offsets), shape=(4, 4))
        else:
            sparse_matrix = scipy.sparse.dia_matrix(np.eye(4))
        return sparse_matrix


class BOPGraph:

    def __init__(self, node_list: List[Node], graph_calc: GraphCalculator,
                 node_interaction_calc: NodeInteractionCalculator = None):
        # TODO: unexpected behavior if graph_calc was initialized before
        self._graph_calc = graph_calc
        self._graph_calc.add_nodes_from(node_list)
        self.node_interaction_calc = node_interaction_calc
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

    @staticmethod
    def _multiply_hops_in_path(path: nx.MultiDiGraph) -> np.ndarray:
        __hop_list = [x[2]['hop'].toarray() for x in path.edges(data=True) if x[2]['hop'] is not None]
        return functools.reduce(lambda hop1, hop2: np.dot(hop1, hop2), __hop_list)

    def compute_interference_path(self, from_node: Node, to_node: Node, depth: int):
        return functools.reduce(
            lambda x, y: x + y,
            [self._multiply_hops_in_path(x) for x in self._graph_calc.all_paths_from_to(from_node, to_node, depth)]
        )

    def _get_distances(self) -> Iterator[Tuple[Tuple[Node, Node], float]]:
        """
        TODO: return pairs only once
        :return:
        """
        for pair in circular_pairwise(self._graph_calc.nodes):
            yield (pair, pair[0].get_distance(pair[1]))


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
