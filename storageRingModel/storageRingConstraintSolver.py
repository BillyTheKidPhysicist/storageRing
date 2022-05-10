from typing import Union
from helperTools import *
from storageRingGeometryModules.shapes import Line,Kink,CappedSlicedBend,Bend
from storageRingGeometryModules.storageRingGeometry import StorageRingGeometry
from storageRingGeometryModules.storageRingGeometrySolver import StorageRingGeometryConstraintsSolver
import numpy as np
from elementPT import Drift,HalbachBenderSimSegmented,CombinerHalbachLensSim,HalbachLensSim,LensIdeal,BenderIdeal,\
    CombinerIdeal,CombinerSim



#todo: The output offset stuff for bender is a 0th order approximation only. go to 1st at least



def _get_Target_Radii(PTL)-> float:
    """Find what radius is the target radius for each bending element. For now this insists that all are the same"""

    radii=[]
    for element in PTL:
        if type(element) is HalbachBenderSimSegmented:
            radii.append(element.ro)
    for radius in radii[1:]:
        assert radius==radii[0] #different target radii is not supported now
    return radii[0]

def _kink_From_Combiner(combiner: Union[CombinerHalbachLensSim, CombinerIdeal])->Kink:
    """From an element in the ParticleTraceLattice, build a geometric shape object"""
    
    L1=combiner.Lb #from kink to next bender
    L2=combiner.La #from previous bender to kin
    inputAng = combiner.ang
    inputOffset = combiner.inputOffset
    outputOffset=combiner.outputOffset
    L1 += -(inputOffset+outputOffset) / np.tan(inputAng)
    L2 +=- inputOffset * np.sin(inputAng) + (inputOffset+outputOffset) / np.sin(inputAng)
    return Kink(-combiner.ang,L2,L1)

def _cappedSlicedBend_From_HalbachBender(bender: HalbachBenderSimSegmented)->CappedSlicedBend:
    """From an element in the ParticleTraceLattice, build a geometric shape object"""

    lengthSegment, Lcap, radius, numMagnets = bender.Lseg, bender.Lcap, bender.ro, bender.numMagnets
    magnetDepth = bender.rp + bender.yokeWidth + bender.outputOffset
    return CappedSlicedBend(lengthSegment,numMagnets,magnetDepth,Lcap,radius)

def _build_Storage_Ring_Geometry_From_PTL(PTL, constrain: bool)->StorageRingGeometry:
    """Use a ParticleTracerLattice to construct a geometric representation of the storage ring. The geometric
    representation is of the ideal orbit of a particle loosely speaking, not the centerline of elements."""

    if constrain is True: assert PTL.latticeType=='storageRing'
    elements=[]
    firstEl=None
    for i,el_PTL in enumerate(PTL):
        if type(el_PTL) in (Drift,HalbachLensSim,LensIdeal):
            constrained=True if el_PTL in PTL.linearElementsToConstraint else False
            elements.append(Line(el_PTL.L,constrained=constrained))
        elif type(el_PTL) in (CombinerHalbachLensSim,CombinerIdeal,CombinerSim):
            elements.append(_kink_From_Combiner(el_PTL))
        elif type(el_PTL) is HalbachBenderSimSegmented:
            elements.append(_cappedSlicedBend_From_HalbachBender(el_PTL))
        elif type(el_PTL) is BenderIdeal:
            elements.append(Bend(el_PTL.ro,el_PTL.ang))
        else: raise Exception
        if i==0:
            firstEl=elements[0]
    firstEl.place(np.array([0,0]),-np.pi)
    storageRing=StorageRingGeometry(elements)
    if constrain==True:
        targetRadii=_get_Target_Radii(PTL)
        solver=StorageRingGeometryConstraintsSolver(storageRing, targetRadii)
        storageRing=solver.make_Valid_Storage_Ring()
    else:
        storageRing.build()
    return storageRing

def _build_Lattice_Bending_Element(bender: Union[BenderIdeal, HalbachBenderSimSegmented], shape: Union[Bend, CappedSlicedBend]):
    """Given a geometric shape object, fill the geometric attributes of an Element object. """

    assert type(bender) in (BenderIdeal,HalbachBenderSimSegmented) and type(shape) in (Bend,CappedSlicedBend)
    bender.rb = shape.radius - bender.outputOffset  # get the bending radius back from orbit radius
    if type(bender) is HalbachBenderSimSegmented:
        bender.numMagnets = shape.numMagnets
        bender.build_Post_Constrained()
    bender.r1 = np.array([*shape.pos_in, 0])
    bender.r2 = np.array([*shape.pos_out, 0])
    bender.nb = np.array([*shape.n_in, 0])
    bender.ne = np.array([*shape.n_out, 0])
    bender.r0 = np.array([*shape.benderCenter, 0])
    n = -shape.n_in
    theta = np.arctan2(n[1], n[0])
    if theta < 0:
        theta += np.pi * 2
    bender.theta = theta
    rot = bender.theta - bender.ang + np.pi / 2
    bender.ROut = np.asarray(
        [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])  # the rotation matrix for
    rot = -rot
    bender.RIn = np.asarray(
        [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])  # np.linalg.inv(bender.ROut)
    
def _build_Lattice_Combiner_Element(combiner: Union[CombinerHalbachLensSim, CombinerIdeal, CombinerSim], shape: Kink):
    """Given a geometric shape object, fill the geometric attributes of an Element object. """

    assert type(combiner) in (CombinerHalbachLensSim,CombinerIdeal,CombinerSim)
    assert type(shape) is Kink
    
    n_out_perp = -np.flip(shape.n_out) * np.array([-1, 1])
    r2 = (shape.pos_out + n_out_perp * combiner.outputOffset)
    combiner.r2 = np.array([*r2, 0.0])
    r1 = r2 + -shape.n_out * combiner.Lb + shape.n_in * combiner.La
    combiner.r1 = np.array([*r1, 0])
    combiner.nb = np.array([*shape.n_in, 0])
    combiner.ne = np.array([*shape.n_out, 0])
    theta = np.arctan2(shape.n_out[1], shape.n_out[0]) - np.pi
    theta = theta + 2 * np.pi  # conventino
    combiner.theta = theta
    rot = combiner.theta
    combiner.ROut = np.asarray([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])  # the rotation matrix for
    rot = -rot
    combiner.RIn = np.asarray([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])  # np.linalg.inv(combiner.ROut)

def _build_Lattice_Lens_Or_Drift(element: Union[Drift, HalbachLensSim, LensIdeal], shape: Line):
    """Given a geometric shape object, fill the geometric attributes of an Element object. """

    assert type(element) in (Drift,HalbachLensSim,LensIdeal) and type(shape) is Line
    if shape.constrained == True: element.set_Length(shape.length)

    element.r1 = np.array([*shape.pos_in, 0])
    element.r2 = np.array([*shape.pos_out, 0])
    element.nb = np.array([*shape.n_in, 0])
    element.ne = np.array([*shape.n_out, 0])
    theta = np.arctan2(shape.n_out[1], shape.n_out[0])
    if theta < 0:
        theta += np.pi * 2
    element.theta = theta
    element.ROut = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    element.RIn = np.asarray([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])

def is_Particle_Tracer_Lattice_Closed(PTL)-> bool:
    """Check that the lattice is closed. """

    elPTL_First,elPTL_Last=PTL.elList[0],PTL.elList[-1]

    if not iscloseAll(elPTL_First.nb,-1*elPTL_Last.ne,1e-12): #normal vector must be same
        return False
    if not iscloseAll(elPTL_First.r1,elPTL_Last.r2,1e-12):
        return False
    return True

def build_Particle_Tracer_Lattice(PTL,constrain: bool)-> None:
    """
    Build up a ParticleTracerLattice (PTL) in place. Apply constraints if indicated. To build the lattice, elements are
    iterated through build one at a time based on a StorageRingGeometry object that is used to represent and solve the
    geometry.

    :param PTL: type of ParticleTracerLattice. A class that hold the elements used for particle tracing. Can be an
        injector type lattice, or a ring type. If constraints are applied, the ring type lattice must be built to be
        closed
    :param constrain: Wether to solve for constraints (ie, must be closed if a ring lattice)
    :return: None
    """

    #todo: There's an inconsistency in how r1/r2/ne/nb is handled with the storage ring
    assert not (constrain and PTL.latticeType=='injector')
    storageRingGeometry=_build_Storage_Ring_Geometry_From_PTL(PTL,constrain)
    for i,(el_PTL,el_Geom) in enumerate(zip(PTL.elList,storageRingGeometry)):
        if type(el_Geom) is Line:
            _build_Lattice_Lens_Or_Drift(el_PTL, el_Geom)
        elif type(el_Geom) is Kink:
            _build_Lattice_Combiner_Element(el_PTL, el_Geom)
        elif type(el_Geom) in (CappedSlicedBend,Bend):
            _build_Lattice_Bending_Element(el_PTL, el_Geom)
        else: raise ValueError

    if PTL.latticeType=='storageRing' and constrain:
        assert is_Particle_Tracer_Lattice_Closed(PTL)
