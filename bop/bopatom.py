from ase.atom import Atom, atomproperty, names, chemical_symbols


names['onsite_level']             = ('onsite_levels', 0.0)
names['number_valence_orbitals']  = ('numbers_valence_orbitals', 5)  # pure-d valence
names['number_valence_electrons'] = ('numbers_valence_electrons', 7.0)
names['stoner_integral']          = ('stoner_integrals', 0.76)


class BOPAtom(Atom):
    onsite_level             = atomproperty('onsite_level', 'Atomic onsite level')
    number_valence_orbitals  = atomproperty('number_valence_orbitals', 'Number of valence orbtials')
    number_valence_electrons = atomproperty('number_valence_electrons', 'Number of valence electrons')
    stoner_integral          = atomproperty('stoner_integral', 'Stoner integral')

    def __init__(self, bopatoms=None, index=None):
        super().__init__(atoms=bopatoms, index=index)
