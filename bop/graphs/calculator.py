from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List, Tuple

import networkx as nx

from bop.nodes.node import Node


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

    def depth_limited_search(
        self, initial_node: Node, depth: int
    ) -> Iterator[Tuple[Node, Node, int]]:
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
    def all_paths_from_to(
        self, from_node: Node, to_node: Node, depth_limit: int
    ) -> Iterable[nx.MultiDiGraph]:
        pass
        # for path in self.all_paths(initial_node=initial_node, depth=depth):
        #     if path[-1][-1] == final_node:
        #         yield path
