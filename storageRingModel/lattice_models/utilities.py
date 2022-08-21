from particle_tracer_lattice import ParticleTracerLattice



real_numer=(int,float)


class LockedDict(dict):

    def __init__(self, dictionary: dict):
        super().__init__(dictionary)
        self._isKeyUsed = {}
        self._reset_Use_Counter()

    def _reset_Use_Counter(self):
        """Reset dictionary that records if a parameter was used"""
        for key in super().keys():
            self._isKeyUsed[key] = False

    def __setitem__(self, key, item):

        raise Exception("this dictionary cannot have new items added")

    def pop(self, *args):
        raise Exception("entries cannot be removed from dictionary")

    def clear(self):
        raise Exception("dictionary cannot be cleared")

    def __delete__(self, instance):
        raise Exception("dictionary cannot be deleted except by garbage collector")

    def __getitem__(self, key):
        """Get key, and record that it was accesed to later it can be checked wether every value was accessed"""
        assert key in self._isKeyUsed.keys()
        self._isKeyUsed[key] = True
        return super().__getitem__(key)

    def super_Special_Change_Item(self, key, item):

        assert key in super().keys()
        assert type(item) in real_numer and item >= 0.0
        super().__setitem__(key, item)

    def assert_all_entries_accesed(self):
        for value in self._isKeyUsed.values():
            assert value  # value must have been used

    def assert_All_Entries_Accessed_And_Reset_Counter(self):
        """Check that every value in the dictionary was accesed, and reset counter"""

        for value in self._isKeyUsed.values():
            assert value  # value must have been used
        self._reset_Use_Counter()


class RingGeometryError(Exception):
    pass


class InjectorGeometryError(Exception):
    pass

def assert_combiners_are_same(lattice_injector: ParticleTracerLattice, lattice_ring: ParticleTracerLattice) -> None:
    """Combiner from injector and ring must have the same shared characteristics, as well as have the expected
    parameters"""

    assert lattice_injector.combiner.output_offset == lattice_ring.combiner.output_offset
    assert lattice_injector.combiner.ang < 0 < lattice_ring.combiner.ang