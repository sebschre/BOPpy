from typing import List

import igraph as igr

from bop.graphs.calculator import GraphCalculator
from bop.nodes.node import Node


class IGraphCalculator(GraphCalculator):
    def __init__(self):
        self.__graph = igr.Graph()

    @property
    def nodes(self) -> List[Node]:
        return [v["name"] for v in self.__graph.vs()]

    @property
    def edges(self):
        return self.__graph.get_edgelist()

    def get_edge(self, node1: Node, node2: Node):
        pass

    def __node_to_vertex(self, node: Node) -> igr.Vertex:
        vertices = [v for v in self.__graph.vs() if v["name"] == node]
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
