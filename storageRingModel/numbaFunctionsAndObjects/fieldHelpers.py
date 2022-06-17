from numbaFunctionsAndObjects.benderHalbachFieldHelper import SegmentedBenderSimFieldHelper_Numba, spec_Bender_Halbach
from numbaFunctionsAndObjects.benderIdealFieldHelper import BenderIdealFieldHelper_Numba, spec_Bender_Ideal
from numbaFunctionsAndObjects.combinerHalbachFieldHelper import CombinerHalbachLensSimFieldHelper_Numba, \
    spec_Combiner_Halbach
from numbaFunctionsAndObjects.combinerIdealFieldHelper import CombinerIdealFieldHelper_Numba, spec_Combiner_Ideal
from numbaFunctionsAndObjects.combinerSimFieldHelper import CombinerSimFieldHelper_Numba, spec_Combiner_Sim
from numbaFunctionsAndObjects.driftFieldHelper import DriftFieldHelper_Numba, spec_Drift
from numbaFunctionsAndObjects.halbachLensFieldHelper import LensHalbachFieldHelper_Numba, spec_Lens_Halbach
from numbaFunctionsAndObjects.idealLensFieldHelper import IdealLensFieldHelper, spec_Ideal_Lens
from numbaFunctionsAndObjects.utilities import jitclass_Wrapper


def get_Ideal_lens_Field_Helper(params):
    return jitclass_Wrapper(params, IdealLensFieldHelper, spec_Ideal_Lens)


def get_Drift_Field_Helper(params):
    return jitclass_Wrapper(params, DriftFieldHelper_Numba, spec_Drift)


def get_Combiner_Halbach_Field_Helper(params):
    return jitclass_Wrapper(params, CombinerHalbachLensSimFieldHelper_Numba, spec_Combiner_Halbach)


def get_Halbach_Lens_Helper(params):
    return jitclass_Wrapper(params, LensHalbachFieldHelper_Numba, spec_Lens_Halbach)


def get_Combiner_Ideal(params):
    return jitclass_Wrapper(params, CombinerIdealFieldHelper_Numba, spec_Combiner_Ideal)


def get_Combiner_Sim(params):
    return jitclass_Wrapper(params, CombinerSimFieldHelper_Numba, spec_Combiner_Sim)


def get_Halbach_Bender(params):
    return jitclass_Wrapper(params, SegmentedBenderSimFieldHelper_Numba, spec_Bender_Halbach)


def get_Bender_Ideal(params):
    return jitclass_Wrapper(params, BenderIdealFieldHelper_Numba, spec_Bender_Ideal)
