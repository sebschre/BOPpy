import unittest
from bop.atoms import *


class TestBOPAtom(unittest.TestCase):

    def test_init(self):
        bopatom = BOPAtom(Position((0, 0, 1)), 'Fe')
        self.assertTrue(bopatom.__class__ == BOPAtom)


class TestBOPGraph(unittest.TestCase):

    def setUp(self):
        self.a1 = BOPAtom(Position((0, 0, 1)), 'Fe')
        self.a2 = BOPAtom(Position((0, 1, 0)), 'Fe')
        self.G = BOPGraph([self.a1, self.a2])

    def test_distances(self):
        distances = list(self.G._get_distances())
        self.assertAlmostEqual(distances[0][1], np.sqrt(2))

    def test_init_edges(self):
        self.G.update_edges(cutoff=1)
        self.assertFalse(self.G.edges)


if __name__ == '__main__':
    unittest.main()
