from typing import Union

import numpy as np

from helper_tools import is_close_all
from lattice_elements.elements import Drift, BenderSim, CombinerLensSim, HalbachLensSim, \
    LensIdeal, BenderIdeal, CombinerIdeal, CombinerSim
from storage_ring_placement.shapes import Line, Kink, CappedSlicedBend, Bend, LineWithAngledEnds
from storage_ring_placement.storage_ring_geometry import StorageRingGeometry
from storage_ring_placement.storage_ring_geometry_solver import StorageRingGeometryConstraintsSolver


# todo: The output offset stuff for bender is a 0th order approximation only. go to 1st at least


def _get_Target_Radii(PTL) -> float:
    """Find what radius is the target radius for each bending element. For now this insists that all are the same"""

    radii = []
    for element in PTL:
        if type(element) is BenderSim:
            radii.append(element.ro)
    for radius in radii[1:]:
        assert radius == radii[0]  # different target radii is not supported now
    return radii[0]


def _kink_From_Combiner(combiner: Union[CombinerLensSim, CombinerIdeal]) -> Kink:
    """From an element in the ParticleTraceLattice, build a geometric shape object"""

    L1 = combiner.Lb  # from kink to next bender
    L2 = combiner.La  # from previous bender to kin
    inputAng = combiner.ang
    input_offset = combiner.input_offset
    output_offset = combiner.output_offset
    L1 += -(input_offset + output_offset) / np.tan(inputAng)
    L2 += - input_offset * np.sin(inputAng) + (input_offset + output_offset) / np.sin(inputAng)
    return Kink(-combiner.ang, L2, L1)


def _cappedSlicedBend_From_HalbachBender(bender: BenderSim) -> CappedSlicedBend:
    """From an element in the ParticleTraceLattice, build a geometric shape object"""

    length_seg, L_cap, radius, num_lenses = bender.Lm, bender.L_cap, bender.ro, bender.num_lenses
    magnet_depth = bender.rp + bender.magnet_width + bender.output_offset  # todo: why does this have output_offset??
    return CappedSlicedBend(length_seg, num_lenses, magnet_depth, L_cap, radius)


def solve_Floor_Plan(PTL, constrain: bool) -> StorageRingGeometry:
    """Use a ParticleTracerLattice to construct a geometric representation of the storage ring. The geometric
    representation is of the ideal orbit of a particle loosely speaking, not the centerline of elements."""
    assert not (constrain and PTL.lattice_type == 'injector')
    elements = []
    first_el = None
    for i, el_PTL in enumerate(PTL):
        if type(el_PTL) in (HalbachLensSim, LensIdeal):
            constrained = True if el_PTL in PTL.linear_elements_to_constrain else False
            elements.append(Line(el_PTL.L, constrained=constrained))
        elif type(el_PTL) is Drift:
            elements.append(LineWithAngledEnds(el_PTL.L, el_PTL.input_tilt_angle, el_PTL.output_tilt_angle))
        elif type(el_PTL) in (CombinerLensSim, CombinerIdeal, CombinerSim):
            elements.append(_kink_From_Combiner(el_PTL))
        elif type(el_PTL) is BenderSim:
            elements.append(_cappedSlicedBend_From_HalbachBender(el_PTL))
        elif type(el_PTL) is BenderIdeal:
            elements.append(Bend(el_PTL.ro, el_PTL.ang))
        else:
            raise Exception
        if i == 0:
            first_el = elements[0]

    n_in_Initial = -np.array([np.cos(PTL.initial_ang), np.sin(PTL.initial_ang)]) if PTL.initial_ang != -np.pi \
        else np.array([1.0, 0.0])
    pos_in_Initial = np.array(PTL.initial_location)
    first_el.place(pos_in_Initial, n_in_Initial)

    storage_ring = StorageRingGeometry(elements)
    if constrain:
        targetRadii = _get_Target_Radii(PTL)
        solver = StorageRingGeometryConstraintsSolver(storage_ring, targetRadii)
        storage_ring = solver.make_valid_storage_ring()
    else:
        storage_ring.build()
    return storage_ring


def _build_Lattice_Bending_Element(bender: Union[BenderIdeal, BenderSim],
                                   shape: Union[Bend, CappedSlicedBend]):
    """Given a geometric shape object, fill the geometric attributes of an Element object. """

    assert type(bender) in (BenderIdeal, BenderSim) and type(shape) in (Bend, CappedSlicedBend)
    bender.rb = shape.radius - bender.output_offset  # get the bending radius back from orbit radius
    if type(bender) is BenderSim:
        bender.num_lenses = shape.num_lenses
    bender.r1 = np.array([*shape.pos_in, 0])
    bender.r2 = np.array([*shape.pos_out, 0])
    bender.nb = np.array([*shape.norm_in, 0])
    bender.ne = np.array([*shape.norm_out, 0])
    bender.r0 = np.array([*shape.benderCenter, 0])
    n = -shape.norm_in
    theta = np.arctan2(n[1], n[0])
    if theta < 0:
        theta += np.pi * 2
    bender.theta = theta


def _build_Lattice_Combiner_Element(combiner: Union[CombinerLensSim, CombinerIdeal, CombinerSim], shape: Kink):
    """Given a geometric shape object, fill the geometric attributes of an Element object. """

    assert type(combiner) in (CombinerLensSim, CombinerIdeal, CombinerSim)
    assert type(shape) is Kink

    n_out_perp = -np.flip(shape.norm_out) * np.array([-1, 1])
    r2 = (shape.pos_out + n_out_perp * combiner.output_offset)
    combiner.r2 = np.array([*r2, 0.0])
    r1 = r2 + -shape.norm_out * combiner.Lb + shape.norm_in * combiner.La
    combiner.r1 = np.array([*r1, 0])
    combiner.nb = np.array([*shape.norm_in, 0])
    combiner.ne = np.array([*shape.norm_out, 0])
    theta = np.arctan2(shape.norm_out[1], shape.norm_out[0]) - np.pi
    theta = theta + 2 * np.pi  # conventino
    combiner.theta = theta
    rot = combiner.theta
    combiner.R_Out = np.asarray([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])  # the rotation matrix for
    rot = -rot
    combiner.R_In = np.asarray(
        [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])  # np.linalg.inv(combiner.R_Out)


def _build_Lattice_Lens_Or_Drift(element: Union[Drift, HalbachLensSim, LensIdeal],
                                 shape: Union[Line, LineWithAngledEnds]):
    """Given a geometric shape object, fill the geometric attributes of an Element object. """

    assert type(element) in (Drift, HalbachLensSim, LensIdeal) and type(shape) in (Line, LineWithAngledEnds)
    if shape.constrained:
        element.set_length(shape.length)

    element.r1 = np.array([*shape.pos_in, 0])
    element.r2 = np.array([*shape.pos_out, 0])
    element.nb = np.array([*shape.norm_in, 0])
    element.ne = np.array([*shape.norm_out, 0])
    if type(shape) is Line:
        theta = np.arctan2(shape.norm_out[1], shape.norm_out[0])
    elif type(shape) is LineWithAngledEnds:
        n = shape.input_to_output_unit_vec()
        theta = np.arctan2(n[1], n[0])
    else:
        raise NotImplementedError
    if theta < 0:
        theta += np.pi * 2
    element.theta = theta
    element.R_Out = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    element.R_In = np.asarray([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])


def is_particle_tracer_lattice_closed(PTL) -> bool:
    """Check that the lattice is closed. """

    elPTL_First, elPTL_Last = PTL.el_list[0], PTL.el_list[-1]
    closedTolerance = 1e-11
    if not is_close_all(elPTL_First.nb, -1 * elPTL_Last.ne, closedTolerance):  # normal vector must be same
        return False
    if not is_close_all(elPTL_First.r1, elPTL_Last.r2, closedTolerance):
        return False
    return True


def update_and_place_elements_from_floor_plan(PTL, floor_plan):
    for i, (el_PTL, el_Geom) in enumerate(zip(PTL.el_list, floor_plan)):
        if type(el_PTL) in (LensIdeal, HalbachLensSim, Drift):
            _build_Lattice_Lens_Or_Drift(el_PTL, el_Geom)
        elif type(el_PTL) in (CombinerLensSim, CombinerIdeal, CombinerSim):
            _build_Lattice_Combiner_Element(el_PTL, el_Geom)
        elif type(el_PTL) in (BenderSim, BenderIdeal):
            _build_Lattice_Bending_Element(el_PTL, el_Geom)
        else:
            raise NotImplementedError
