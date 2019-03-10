import unittest
from atoms import *


class TestBOPAtoms(unittest.TestCase):

    def test_init(self):
        bopatoms = BOPAtoms('Fe')
        self.assertTrue(bopatoms.__class__ == BOPAtoms)

    def test_init_nodes(self):
        bopatoms = BOPAtoms('Fe')
        self.assertTrue(bopatoms.graph.nodes[0]['atom'].__class__ == BOPAtom)


if __name__ == '__main__':
    unittest.main()