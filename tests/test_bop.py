import unittest
import time
import numpy as np
from bop.atoms import BOPAtom, ValenceOrbitalParameter, ValenceOrbitalType, AtomType
from bop.coordinate_system import Position
from bop.graph_calculator import GraphCalculator, NxGraphCalculator, IGraphCalculator, BOPGraph


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


class TestBOPGraphWithNxGraph(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        self.a1 = BOPAtom(Position((0, 0, 1)), 'Fe')
        self.a2 = BOPAtom(Position((0, 1, 0)), 'Fe')
        self.a3 = BOPAtom(Position((2, 1, 0)), 'Fe')
        self.a4 = BOPAtom(Position((2, 2, 0)), 'Fe')
        self.bg = BOPGraph([self.a1, self.a2, self.a3, self.a4], graph_calc=NxGraphCalculator())

    def tearDown(self):
        t = time.time() - self.startTime
        print(f"{self.id()}: \t{t:.4f}s")

    def test_distances(self):
        distances = list(self.bg._get_distances())
        self.assertAlmostEqual(distances[0][1], np.sqrt(2))

    def test_init_edges(self):
        self.bg.update_edges(cutoff=0.1)
        # self.assertFalse(self.bg._graph_calc.edges)

    def test_dfs(self):
        self.bg.update_edges(cutoff=2)
        # nx.relabel.convert_node_labels_to_integers(self.G)
        # print(nx.dfs_tree(self.G, source=self.a1, depth_limit=3).edges())

    def test_all_paths_from_to_1(self):
        self.bg.update_edges(cutoff=2)
        self.assertIs(len(list(self.bg._graph_calc.all_paths_from_to(self.a1, self.a2, depth_limit=2))), 2)

    def test_all_paths_from_to_2(self):
        self.bg.update_edges(cutoff=2)
        self.assertIs(len(list(self.bg._graph_calc.all_paths_from_to(self.a1, self.a2, depth_limit=3))), 5)

    def test_all_paths_from_to(self):
        self.bg.update_edges(cutoff=2)
        for path in self.bg._graph_calc.all_paths_from_to(self.a1, self.a2, depth_limit=4):
            print(path.edges(data=True))

    def test_connection_graph(self):
        self.bg.update_edges(cutoff=2)
        self.bg._graph_calc._connection_graph_from_to(self.a1, self.a2, depth_limit=6)


class TestBOPGraphWithIGraph(unittest.TestCase):
    """
    TODO: merge testing with different graph calculators
    """
    def setUp(self):
        self.a1 = BOPAtom(Position((0, 0, 1)), 'Fe')
        self.a2 = BOPAtom(Position((0, 1, 0)), 'Fe')
        self.a3 = BOPAtom(Position((2, 1, 0)), 'Fe')
        self.a4 = BOPAtom(Position((2, 2, 0)), 'Fe')
        self.bg = BOPGraph([self.a1, self.a2, self.a3, self.a4], graph_calc=IGraphCalculator())

    def test_distances(self):
        distances = list(self.bg._get_distances())
        self.assertAlmostEqual(distances[0][1], np.sqrt(2))

    def test_init_edges(self):
        self.bg.update_edges(cutoff=0.1)
        # self.assertFalse(self.bg._graph_calc.edges)

    def test_dfs(self):
        self.bg.update_edges(cutoff=2)
        # nx.relabel.convert_node_labels_to_integers(self.G)
        # print(nx.dfs_tree(self.G, source=self.a1, depth_limit=3).edges())

    #def test_dls(self):
    #    self.bg.update_edges(cutoff=2)
    #    for n in self.bg._graph_calc.depth_limited_search(self.a1, 3):
    #        pass

    #def test_all_paths(self):
    #    self.bg.update_edges(cutoff=2)
    #    for path in self.bg._graph_calc.all_paths(self.a1, 3):
    #        pass


if __name__ == '__main__':
    suite_nxgraph = unittest.TestLoader().loadTestsFromTestCase(TestBOPGraphWithNxGraph)
    suite_bopatoms = unittest.TestLoader().loadTestsFromTestCase(TestBOPAtom)
    unittest.TextTestRunner(verbosity=0).run(suite_nxgraph)
    unittest.TextTestRunner(verbosity=0).run(suite_bopatoms)
    unittest.main()
