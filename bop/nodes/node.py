from abc import ABC, abstractmethod


class Node(ABC):
    @abstractmethod
    def get_distance(self, other: "Node"):
        pass
