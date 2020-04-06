import unittest
from bop.atoms import *


class TestBOPAtom(unittest.TestCase):

    def test_init_explicit(self):
        bopatom = BOPAtom(Position((0, 0, 1)), AtomType('Fe'))
        self.assertTrue(bopatom.__class__ == BOPAtom)

    def test_init_implicit(self):
        bopatom = BOPAtom((0, 0, 1), 'Fe')
        self.assertTrue(bopatom.__class__ == BOPAtom)

    def test_init_with_valence(self):
        orbitals = {ValenceOrbitalType.S: ValenceOrbitalParameter(1.0, 1.0)}
        onsite_levels = {ValenceOrbitalType.S: 0.0}
        bopatom = BOPAtom(
            Position((0, 0, 1)),
            AtomType('Fe', valence_orbital_dict=orbitals), onsite_levels=onsite_levels
        )
        self.assertTrue(bopatom.__class__ == BOPAtom)

    def test_init_with_wrong_onsite_levels(self):
        orbitals = {ValenceOrbitalType.S: ValenceOrbitalParameter(1.0, 1.0)}
        onsite_levels = {ValenceOrbitalType.D: 0.0}
        with self.assertRaises(ValueError):
            BOPAtom(Position((0, 0, 1)), AtomType('Fe', valence_orbital_dict=orbitals), onsite_levels=onsite_levels)


class TestBOPGraph(unittest.TestCase):

    def setUp(self):
        self.a1 = BOPAtom(Position((0, 0, 1)), 'Fe')
        self.a2 = BOPAtom(Position((0, 1, 0)), 'Fe')
        self.a3 = BOPAtom(Position((2, 1, 0)), 'Fe')
        self.a4 = BOPAtom(Position((2, 2, 0)), 'Fe')
        self.bg = BOPGraph([self.a1, self.a2, self.a3, self.a4])

    def test_distances(self):
        distances = list(self.bg._get_distances())
        self.assertAlmostEqual(distances[0][1], np.sqrt(2))

    def test_init_edges(self):
        self.bg.update_edges(cutoff=0.1)
        self.assertFalse(self.bg.graph_calc.edges)

    def test_dfs(self):
        self.bg.update_edges(cutoff=2)
        # nx.relabel.convert_node_labels_to_integers(self.G)
        # print(nx.dfs_tree(self.G, source=self.a1, depth_limit=3).edges())


if __name__ == '__main__':
    unittest.main()
