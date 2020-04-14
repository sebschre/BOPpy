import itertools
import copy
from enum import Enum
from typing import List, Iterator, TypeVar, Tuple, Union, Container, Iterable, Set, Dict, Mapping, Callable, FrozenSet, \
    Hashable, Generator
import collections
from abc import ABC, abstractmethod
import numpy as np
from periodictable import elements
from bop.coordinate_system import Position
from bop.graph_calculator import Node, GraphCalculator, NxGraphCalculator, IGraphCalculator


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


class BOPAtom(Node):

    atom_number = 0

    def __init__(self,
                 position: Union[Position, Tuple[float, float, float]],
                 atom_type: Union[AtomType, str],
                 onsite_levels: Dict[ValenceOrbitalType, float] = {}):
        BOPAtom.atom_number += 1
        self.atom_id = BOPAtom.atom_number
        if type(position) is not Position:
            position = Position(position)
        self.position = position
        if type(atom_type) is not AtomType:
            atom_type = AtomType(atom_type)
        self.atom_type = atom_type
        if atom_type.valence_orbital_dict.keys() != onsite_levels.keys():
            raise ValueError(f"Onsite levels {onsite_levels} do not match orbitals {atom_type.valence_orbital_dict}")
        else:
            self.onsite_levels = onsite_levels

    def __del__(self):
        BOPAtom.atom_number -= 1

    def __repr__(self):
        # return f"{self.atom_type} at {self.position}"
        return f"Atom {self.atom_id}"
    
    def get_distance(self, other: 'BOPAtom') -> float:
        return self.position.get_distance(other.position)
