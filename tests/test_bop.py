import unittest
from bop.atoms import *


class TestBOPAtom(unittest.TestCase):

    def test_init(self):
        bopatom = BOPAtom((0, 0, 1), 'Fe')
        self.assertTrue(bopatom.__class__ == BOPAtom)


class TestBOPGraph(unittest.TestCase):

    def test_init(self):
        a1 = BOPAtom((0, 0, 1), 'Fe')
        a2 = BOPAtom((0, 1, 0), 'Fe')
        BOPGraph([a1, a2])

    def test_distances(self):
        a1 = BOPAtom((0, 0, 1), 'Fe')
        a2 = BOPAtom((0, 1, 0), 'Fe')
        G = BOPGraph([a1, a2])
        distances = list(G.get_distances())
        print(distances)
        #self.assertTrue(pos2 - pos1 == pos3)


if __name__ == '__main__':
    unittest.main()
