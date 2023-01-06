from typing import Iterable, List, Tuple

import networkx as nx

from bop.graphs.calculator import GraphCalculator
from bop.nodes.node import Node


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

    def _dfs_multi_edge_tree(
        self, source=None, depth_limit=None, reverse_count_from: int = None
    ) -> nx.MultiDiGraph:
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
                tree.add_edge(
                    node2,
                    node1,
                    reverse_count_from - depth + 1,
                    **self.edges[node1, node2]
                )
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
        self, from_node: Node, to_node: Node, depth_limit: int
    ) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
        if depth_limit % 2 == 0:
            depth1, depth2 = int(depth_limit / 2), int(depth_limit / 2)
        else:
            depth1, depth2 = int((depth_limit + 1) / 2), int((depth_limit - 1) / 2)
        tree1 = self._dfs_multi_edge_tree(from_node, depth_limit=depth1)
        tree2 = self._dfs_multi_edge_tree(
            to_node, depth_limit=depth2, reverse_count_from=depth_limit
        )
        return tree1, tree2

    def _connection_graph_from_to(
        self, from_node: Node, to_node: Node, depth_limit: int
    ) -> nx.MultiDiGraph:
        (tree1, tree2) = self._connection_trees_from_to(from_node, to_node, depth_limit)
        return nx.compose(tree1, tree2)

    def all_paths_from_to(
        self, from_node: Node, to_node: Node, depth_limit: int
    ) -> Iterable[nx.MultiDiGraph]:
        """
        TODO: implement more performant solution
        :param from_node:
        :param to_node:
        :param depth_limit:
        :return:
        """
        tree_full = self._connection_graph_from_to(from_node, to_node, depth_limit)

        def __recursion(node_from: Node, depth_level: int, path: nx.MultiDiGraph):
            for _, node_to, depth in tree_full.out_edges(node_from, keys=True):
                if depth == depth_level:
                    path.add_edge(
                        node_from,
                        node_to,
                        depth_level,
                        **self.edges[node_from, node_to]
                    )
                    if depth_level == depth_limit:
                        yield path
                    else:
                        yield from __recursion(node_to, depth_level + 1, path)
                    path.remove_edge(node_from, node_to, depth_level)

        yield from __recursion(from_node, 1, nx.MultiDiGraph())
