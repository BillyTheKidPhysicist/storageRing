import scipy.interpolate as spi
import time
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from typing import Optional,Union
import warnings
import fastNumbaMethodsAndClass
import pandas as pd
import numba
from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from HalbachLensClass import SegmentedBenderHalbach as _HalbachBenderFieldGenerator
from constants import MASS_LITHIUM_7,BOLTZMANN_CONSTANT,BHOR_MAGNETON,SIMULATION_MAGNETON
from numba.typed import List
from scipy.spatial.transform import Rotation as Rot



#todo: this is just awful. I need to clean this up more in line with what I know about clean code now

# from profilehooks import profile



TINY_STEP=1e-9
TINY_OFFSET = 1e-12  # tiny offset to avoid out of bounds right at edges of element

def full_Arctan(q):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    phi = np.arctan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi
def is_Even(x):
    assert type(x) is int
    return True if x%2==0 else False

class Element:
    """
    Base class for other elements. Contains universal attributes and methods.

    An element is the fundamental component of a neutral atom storage ring/injector. An arrangment of elements is called
    a lattice, as is done in accelerator physics. Elements are intended to be combined together such that particles can
    smoothly move from one to another, and many class variables serves this purpose. An element also contains methods
    for force vectors and magnetic potential at a point in space. It will also contain methods to generate fields values
    and construct itself, which is not always trivial.
    """
    def __init__(self, PTL,ang: float=0.0,build: bool=True,L: Optional[float]=None):
        # self.theta = None  # angle that describes an element's rotation in the xy plane.
        # SEE EACH ELEMENT FOR MORE DETAILS
        # -Straight elements like lenses and drifts: theta=0 is the element's input at the origin and the output pointing
        # east. for theta=90 the output is pointing up.
        # -Bending elements without caps: at theta=0 the outlet is at (bending radius,0) pointing south with the input
        # at some angle counterclockwise. a 180 degree bender would have the inlet at (-bending radius,0) pointing south.
        # force is a continuous function of r and theta, ie a revolved cross section of a hexapole
        # -Bending  elements with caps: same as without caps, but keep in mind that the cap on the output would be BELOW
        # y=0
        # combiner: theta=0 has the outlet at the origin and pointing to the west, with the inlet some distance to the right
        # and pointing in the NE direction
        self.PTL = PTL  # particle tracer lattice object. Used for various constants
        self.nb = None  # normal vector to beginning (clockwise sense) of element.
        self.ne = None  # normal vector to end (clockwise sense) of element
        self.r0 = None  # coordinates of center of bender, minus any caps
        self.ROut = None  # 2d matrix to rotate a vector out of the element's reference frame
        self.RIn = None  # 2d matrix to rotate a vector into the element's reference frame
        self.r1 = None  # 3D coordinates of beginning (clockwise sense) of element in lab frame
        self.r2 = None  # 3D coordinates of ending (clockwise sense) of element in lab frame
        self.SO = None  # the shapely object for the element. These are used for plotting, and for finding if the coordinates
        # # are inside an element that can't be found with simple geometry
        self.SO_Outer=None #shapely object that represents the outer edge of the element
        self.outerHalfWidth=None #outer diameter/width of the element, where applicable. For example, outer diam of lens is the
        # #bore radius plus magnets and mount material radial thickness
        self.ang = ang  # bending angle of the element. 0 for lenses and drifts
        self.L=L
        self.index=None
        self.Lo = None  # length of orbit for particle. For lenses and drifts this is the same as the length. This is a nominal
        # # value because for segmented benders the path length is not simple to compute
        self.bumpOffset=0.0 # the transverse translational. Only applicable to bump lenses now
        self.bumpVector=np.zeros(3) #positoin vector of the bump displacement. zero vector for no bump amount
        self.outputOffset = 0.0  # some elements have an output offset, like from bender's centrifugal force or
        # #lens combiner
        self.fieldFact = 1.0  # factor to modify field values everywhere in space by, including force
        self.fastFieldHelper=fastNumbaMethodsAndClass.BaseClassFieldHelper_Numba()
        self.maxCombinerAng=.2 #because the field value is a right rectangular prism, it must extend to past the
        # #end of the tilted combiner. This maximum value is used to set that extra extent, and can't be exceede by ang
        if build==True:
            self.build()
    def perturb_Element(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float):
        """
        perturb the alignment of the element relative to the vacuum tube. The vacuum tube will remain unchanged, but
        the element will be shifted, and therefore the force it applies will be as well. This is modeled as shifting
        and rotating the supplied coordinates to force and magnetic field function, then rotating the force

        :param shiftY: Shift in the y direction in element frame
        :param shiftZ: Shift in the z direction in the element frame
        :param rotY: Rotation about y axis of the element
        :param rotZ: Rotation about z axis of the element
        :return:
        """
        self.fastFieldHelper.update_Element_Perturb_Params(shiftY,shiftZ,rotY,rotZ)
    def magnetic_Potential(self, qEl:np.ndarray)->float:
        """
        Return magnetic potential energy at position qEl.

        Return magnetic potential energy of a lithium atom in simulation units, where the mass of a lithium-7 atom is 
        1kg, at cartesian 3D coordinate qEl in the local element frame. This is done by calling up fastFieldHelper, a
        jitclass, which does the actual math/interpolation.
         
        :param qEl: 3D cartesian position vector in local element frame, numpy.array([x,y,z])
        :return: magnetic potential energy of a lithium atom in simulation units, float
        """
        return self.fastFieldHelper.magnetic_Potential(*qEl) #will raise NotImplementedError if called
    def force(self, qEl:np.ndarray)->np.ndarray:
        """
        Return force at position qEl.

        Return 3D cartesian force of a lithium at cartesian 3D coordinate qEl in the local element frame. Force vector
        has simulation units where lithium-7 mass is 1kg. This is done by calling up fastFieldHelper, a
        jitclass, which does the actual math/interpolation.
        
         
        :param qEl: 3D cartesian position vector in local element frame,numpy.array([x,y,z])
        :return: New 3D cartesian force vector, numpy.array([Fx,Fy,Fz])
        """
        return np.asarray(self.fastFieldHelper.force(*qEl)) #will raise NotImplementedError if called

    def transform_Lab_Coords_Into_Global_Orbit_Frame(self, qLab:np.ndarray, cumulativeLength:float)->np.ndarray:
        """
        Generate coordinates in the non-cartesian global orbit frame that grows cumulatively with revolutions, from
        observer/lab cartesian coordinates. 
        
        :param qLab: 3D cartesian position vector in observer/lab frame,numpy.array([x,y,z])
        :param cumulativeLength: total length in orbit frame traveled so far. For a series of linear elements this
        would simply be the sum of their length, float
        :return: New 3D global orbit frame position, numpy.ndarray([x,y,z])
        """
        qEl = self.transform_Lab_Coords_Into_Element_Frame(qLab)
        qOrbit = self.transform_Element_Coords_Into_Local_Orbit_Frame(qEl)
        qOrbit[0] = qOrbit[0] + cumulativeLength  # longitudinal component grows
        return qOrbit

    def transform_Lab_Coords_Into_Element_Frame(self, qLab:np.ndarray)->np.ndarray:
        """
        Generate local cartesian element frame coordinates from cartesian observer/lab frame coordinates

        :param qLab: 3D cartesian position vector in observer/lab frame,numpy.array([x,y,z])
        :return: New 3D cartesian element frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError
    def transform_Element_Coords_Into_Lab_Frame(self,qEl:np.ndarray)->np.ndarray:
        """
        Generate cartesian observer/lab frame coordinates from local cartesian element frame coordinates

        :param qEl: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: New 3D cartesian observer/lab frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError
    def transform_Orbit_Frame_Into_Lab_Frame(self,qOrbit:np.ndarray)->np.ndarray:
        """
        Generate global cartesian observer/lab frame coords from non-cartesian local orbit frame coords. Orbit coords
        are similiar to the Frenet-Serret Frame.

        :param qOrbit: 3D non-cartesian orbit frame position, numpy.ndarray([so,xo,yo]). so is the distance along
            the orbit trajectory. xo is in the xy lab plane, yo is perpdindicular Not necessarily the same as the
            distance along the center of the element.
        :return: New 3D cartesian observer/lab frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError
    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl:np.ndarray)->np.ndarray:
        """
        Generate non-cartesian local orbit frame coords from local cartesian element frame coords. Orbit coords are
        similiar to the Frenet-Serret Frame.

        :param qEl: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: New 3D non-cartesian orbit frame position, numpy.ndarray([so,xo,yo]). so is the distance along
            the orbit trajectory. xo is in the xy lab plane, yo is perpdindicular Not necessarily the same as the
            distance along the center of the element.
        """
        raise NotImplementedError
    def transform_Lab_Frame_Vector_Into_Element_Frame(self, vecLab:np.ndarray)->np.ndarray:
        """
        Generate element frame vector from observer/lab frame vector. 
        
        :param vecLab: 3D cartesian vector in observer/lab frame,numpy.array([vx,vy,vz]) 
        :return: 3D cartesian vector in element frame,numpy.array([vx,vy,vz])
        """""
        vecNew = vecLab.copy()  # prevent editing
        vecNew[:2]=self.RIn@vecNew[:2]
        return vecNew
    def transform_Element_Frame_Vector_Into_Lab_Frame(self, vecEl:np.ndarray)->np.ndarray:
        """
        Generate observer/lab frame vector from element frame vector. 

        :param vecEl: 3D cartesian vector in element frame,numpy.array([vx,vy,vz]) 
        :return: 3D cartesian vector in observer/lab frame,numpy.array([vx,vy,vz])
        """""
        vecNew = vecEl.copy()  #prevent editing
        vecNew[:2]=self.ROut@vecNew[:2]
        return vecNew

    def is_Coord_Inside(self, qEl:np.ndarray)->bool:
        """
        Check if a 3D cartesian element frame coordinate is contained within an element's vacuum tube

        :param qEl: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: True if the coordinate is inside, False if outside
        """
        return self.fastFieldHelper.is_Coord_Inside_Vacuum(*qEl) #will raise NotImplementedError if called

    def build(self)->None:
        """build the element either by simulating fields, loading fields, or using analytic results. Called in __init__"""
        raise NotImplementedError
    def shape_Field_Data_3D(self, data:np.ndarray)->tuple:
        """
        Shape 3D field data for fast linear interpolation method

        Take an array with the shape (n,7) where n is the number of points in space. Each row
        must have the format [x,y,z,gradxB,gradyB,gradzB,B] where B is the magnetic field norm at x,y,z and grad is the
        partial derivative. The data must be from a 3D grid of points with no missing points or any other funny business
        and the order of points doesn't matter. Return arrays are raveled for use by fast interpolater

        :param data: (n,7) numpy array of points originating from a 3d grid
        :return: tuple of 7 arrays, first 3 are grid edge coords (x,y,z) and last 4 are flattened field values
        (Fx,Fy,Fz,V)
        """
        assert data.shape[1]==7 and len(data)>2**3
        xArr=np.unique(data[:,0])
        yArr=np.unique(data[:,1])
        zArr=np.unique(data[:,2])

        numx=xArr.shape[0]
        numy=yArr.shape[0]
        numz=zArr.shape[0]
        FxMatrix=np.empty((numx,numy,numz))
        FyMatrix=np.empty((numx,numy,numz))
        FzMatrix=np.empty((numx,numy,numz))
        VMatrix=np.zeros((numx,numy,numz))
        xIndices=np.argwhere(data[:,0][:,None]==xArr)[:,1]
        yIndices=np.argwhere(data[:,1][:,None]==yArr)[:,1]
        zIndices=np.argwhere(data[:,2][:,None]==zArr)[:,1]
        FxMatrix[xIndices,yIndices,zIndices]=-SIMULATION_MAGNETON*data[:,3]
        FyMatrix[xIndices,yIndices,zIndices]=-SIMULATION_MAGNETON*data[:,4]
        FzMatrix[xIndices,yIndices,zIndices]=-SIMULATION_MAGNETON*data[:,5]
        VMatrix[xIndices,yIndices,zIndices]=SIMULATION_MAGNETON*data[:,6]
        VFlat,FxFlat,FyFlat,FzFlat=np.ravel(VMatrix),np.ravel(FxMatrix),np.ravel(FyMatrix),np.ravel(FzMatrix)
        return xArr,yArr,zArr,FxFlat,FyFlat,FzFlat,VFlat

    def shape_Field_Data_2D(self, data:np.ndarray)->tuple:
        """2D version of shape_Field_Data_3D. Data must be shape (n,5), with each row [x,y,Fx,Fy,V]"""
        assert data.shape[1]==5 and len(data)>2**3
        xArr = np.unique(data[:, 0])
        yArr = np.unique(data[:, 1])
        numx = xArr.shape[0]
        numy = yArr.shape[0]
        BGradxMatrix = np.zeros((numx, numy))
        BGradyMatrix = np.zeros((numx, numy))
        B0Matrix = np.zeros((numx, numy))
        xIndices = np.argwhere(data[:, 0][:, None] == xArr)[:, 1]
        yIndices = np.argwhere(data[:, 1][:, None] == yArr)[:, 1]

        BGradxMatrix[xIndices, yIndices] = data[:, 2]
        BGradyMatrix[xIndices, yIndices] = data[:, 3]
        B0Matrix[xIndices, yIndices] = data[:, 4]
        FxMatrix = -SIMULATION_MAGNETON * BGradxMatrix
        FyMatrix = -SIMULATION_MAGNETON * BGradyMatrix
        VMatrix=SIMULATION_MAGNETON * B0Matrix
        VFlat,FxFlat,FyFlat=np.ravel(VMatrix),np.ravel(FxMatrix), np.ravel(FyMatrix)
        return xArr, yArr,FxFlat,FyFlat,VFlat 


class LensIdeal(Element):
    """
    Ideal model of lens with hard edge. Force inside is calculated from field at pole face and bore radius as
    F=2*ub*r/rp**2 where rp is bore radius, and ub the simulation bohr magneton where the mass of lithium7=1kg.
    This will prevent energy conservation because of the absence of fringe fields between elements to reduce
    forward velocity. Interior vacuum tube is a cylinder
    """
    def __init__(self, PTL, L:float, Bp:float, rp:float, ap:float,bumpOffset:float =0.0,build=True):
        """
        :param PTL: Instance of ParticleTracerLatticeClass
        :param L: Total length of element and lens, m. Not always the same because of need to contain fringe fields
        :param Bp: Magnetic field at the pole face, T.
        :param rp: Bore radius, m. Distance from center of magnet to the magnetic material
        :param ap: Aperture of bore, m. Typically is the radius of the vacuum tube
        :param bumpOffset: dubious
        :param fillParams: Wether to carry out filling the element parameters. dubious
        """
        # fillParams is used to avoid filling the parameters in inherited classes
        super().__init__(PTL,build=False,L=L)
        self.Bp = Bp
        self.rp = rp
        self.ap = ap  # size of apeture radially
        self.shape = 'STRAIGHT'  # The element's geometry
        self.bumpOffset=bumpOffset
        if build==True:
            self.build()
    def set_BpFact(self,BpFact:float)->None:
        """update the magnetic field multiplication factor. This is used to simulate changing field values"""
        self.fieldFact=BpFact
        self.K = self.fieldFact*(2 * self.Bp * SIMULATION_MAGNETON / self.rp ** 2)  # 'spring' constant
    def build(self)->float:
        """Update class parameters"""
        self.K = self.fieldFact*(2 * self.Bp * SIMULATION_MAGNETON / self.rp ** 2)  # 'spring' constant
        if self.L is not None:
            self.Lo = self.L
        self.fastFieldHelper=fastNumbaMethodsAndClass.IdealLensFieldHelper_Numba(self.L,self.K,self.ap)

    def transform_Lab_Coords_Into_Element_Frame(self, qLab:np.ndarray)->np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew=qLab.copy()
        qNew-=self.r1
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def transform_Element_Coords_Into_Lab_Frame(self,qEl:np.ndarray)->np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = qEl.copy()
        qNew=self.transform_Element_Frame_Vector_Into_Lab_Frame(qNew)
        qNew+=self.r1
        return qNew
    def transform_Orbit_Frame_Into_Lab_Frame(self,q:np.ndarray)->np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew=q.copy()
        qNew[:2]=self.ROut@qNew[:2]
        qNew+=self.r1
        return qNew
    def transform_Element_Coords_Into_Local_Orbit_Frame(self, q:np.ndarray)->np.ndarray:
        """Overrides abstract method from Element. Element and orbit frame is identical"""
        return q.copy()

    def set_Length(self, L:float)->None:
        """this is used typically for setting the length after satisfying constraints"""
        assert L>0.0
        self.L = L
        self.Lo = self.L

class Drift(LensIdeal):
    """
    Simple model of free space. Effectively a cylinderical vacuum tube
    """
    def __init__(self, PTL, L:float, ap:float,build=True):
        """
        :param PTL: Instance of ParticleTracerLatticeClass
        :param L: Total length of element and lens, m. Not always the same because of need to contain fringe fields
        :param ap: Aperture of bore, m. Typically is the radius of the vacuum tube
        """
        super().__init__(PTL, L, 0, np.inf, ap,build=False) #set Bp to zero and bore radius to infinite
        self.fastFieldHelper=fastNumbaMethodsAndClass.DriftFieldHelper_Numba(L,ap)
        if build==True:
            self.build()
    def build(self)->None:
        self.Lo = self.L

class BenderIdeal(Element):
    """
        Element representing a bender/waveguide. Base class for other benders

        Simple ideal model of bending/waveguide element modeled as a toroid. The force is linearly proportional to the
        particle's minor radius in the toroid. In this way, it works as a bent lens. Particles will travel through the
        bender displaced from the center of the  potential (minor radius=0.0) because of the centrifugal effect. Thus,
        to minimize oscillations, the nominal particle trajectory is not straight through the bender, but offset a small
        distance. For the ideal bender, this offset can be calculated analytically. Because there are no fringe fields,
        energy conservation is not expected

        Attributes
        ----------
        Bp: Magnetic field at poleface of bender bore, Teslas.

        rp: Radius (minor) to poleface of bender bore, meters.

        ap: Radius (minor) of aperture bender bore, meters. Effectively the vacuum tube inner radius

        rb: Nominal ending radius of bender/waveguide, meters. This is major radius of the toroid. Note that atoms will
            revolve at a slightly larger radius because of centrifugal effect

        shape: Gemeotric shape of element used for placement. ParticleTracerLatticeClass uses this to assemble lattice

        ro: Orbit bending radius, meter. Larger than self.rb because of centrifugal effect

        segmented: Wether the element is made up of discrete segments, or is continuous. Used in
            ParticleTracerLatticeClass
        """
    def __init__(self, PTL, ang:float, Bp:float, rp:float, rb:float, ap:float, build=True):
        if all(arg is not None for arg in [ap,rb,ang])==True:
            assert ap<rp<rb/2.0 and 0.0<ang<2*np.pi
        self.Bp:float  = Bp
        self.rp:float = rp
        self.ap:float = ap
        self.rb:float = rb
        self.K:Optional[float] = None
        self.shape:str = 'BEND'
        self.ro: Optional[float] =None  # bending radius of orbit, ie rb + rOffset.
        self.segmented:bool = False  # wether the element is made up of discrete segments, or is continuous
        super().__init__(PTL, ang=ang,build=False)
        if build==True:
            self.build()
    def build(self)->None:
        self.K = (2 * self.Bp * SIMULATION_MAGNETON / self.rp ** 2)  # 'spring' constant
        self.outputOffset = sqrt(self.rb ** 2 / 4 + self.PTL.v0Nominal ** 2 / self.K) - self.rb / 2#self.output_Offset(self.rb)
        self.ro = self.rb + self.outputOffset
        if self.ang is not None:  # calculation is being delayed until constraints are solved
            self.L = self.rb * self.ang
            self.Lo = self.ro * self.ang
        self.fastFieldHelper=fastNumbaMethodsAndClass.BenderIdealFieldHelper_Numba(self.ang,self.K,self.rp,self.rb,self.ap)
    def transform_Lab_Coords_Into_Element_Frame(self, qLab:np.ndarray)->np.ndarray:
        """Overrides abstract method from Element."""
        qNew = qLab - self.r0
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def transform_Element_Coords_Into_Lab_Frame(self,qEl:np.ndarray)->np.ndarray:
        """Overrides abstract method from Element."""
        qNew=qEl.copy()
        qNew=self.transform_Element_Frame_Vector_Into_Lab_Frame(qNew)
        qNew=qNew+self.r0
        return qNew

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl:np.ndarray)->np.ndarray:
        """Overrides abstract method from Element."""
        qo = qEl.copy()
        phi = self.ang - full_Arctan(qo)  # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = sqrt(qEl[0] ** 2 + qEl[1] ** 2) - self.ro
        qo[0] = qos
        qo[1] = qox
        return qo

    def transform_Orbit_Frame_Into_Lab_Frame(self,qOrbit:np.ndarray)->np.ndarray:
        """Overrides abstract method from Element."""
        xo,yo,zo=qOrbit
        phi=self.ang-xo/self.ro
        xLab=self.ro*np.cos(phi)
        yLab=self.ro*np.sin(phi)
        zLab=zo
        qLab=np.asarray([xLab,yLab,zLab])
        qLab[:2]=self.ROut@qLab[:2]
        qLab+=self.r0
        return qLab

class CombinerIdeal(Element):
    # combiner: This is is the element that bends the two beams together. The logic is a bit tricky. It's geometry is
    # modeled as a straight section, a simple square, with a segment coming of at the particle in put at an angle. The
    # angle is decided by tracing particles through the combiner and finding the bending angle.
    def __init__(self, PTL, Lm: float, c1: float, c2: float, apL: float,apR: float,apZ: float, mode: str,
                 sizeScale: float,build: bool=True):
        Element.__init__(self,PTL, build=False)
        assert mode in ('injector', 'storageRing')
        self.fieldFact=-1.0 if mode=='injector' else 1.0
        self.sizeScale = sizeScale  # the fraction that the combiner is scaled up or down to. A combiner twice the size would
        # use sizeScale=2.0
        self.apR = apR
        self.apL = apL
        self.apz=apZ
        self.ap=None
        self.Lm = Lm
        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet
        self.c1 = c1
        self.c2 = c2
        self.space = 0  # space at the end of the combiner to account for fringe fields

        self.shape = 'COMBINER_SQUARE'
        self.inputOffset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0

        if build==True:
            self.build()

    def build(self)->None:
        self.apR,self.apL,self.apz,self.Lm=[val*self.sizeScale for val in (self.apR,self.apL,self.apz,self.Lm)]
        self.c1,self.c2=self.c1/self.sizeScale,self.c2/self.sizeScale
        self.Lb = self.Lm  # length of segment after kink after the inlet
        self.fastFieldHelper = fastNumbaMethodsAndClass.CombinerIdealFieldHelper_Numba(self.c1, self.c2,np.nan, self.Lb,
                                                                            self.apL,self.apR, np.nan, np.nan)
        inputAngle, inputOffset, qTracedArr = self.compute_Input_Angle_And_Offset()
        self.Lo = np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.ang = inputAngle
        self.inputOffset = inputOffset
        self.La = .5*(self.apR+self.apL) * np.sin(self.ang)
        self.L = self.La * np.cos(self.ang) + self.Lb  # TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING
        self.fastFieldHelper = fastNumbaMethodsAndClass.CombinerIdealFieldHelper_Numba(self.c1, self.c2, self.La,self.Lb,
                                                                                       self.apL,self.apR, self.apz, self.ang)

    def compute_Input_Angle_And_Offset(self, inputOffset:float=0.0, h:float=1e-6)->tuple:
        # this computes the output angle and offset for a combiner magnet.
        # NOTE: for the ideal combiner this gives slightly inaccurate results because of lack of conservation of energy!
        # NOTE: for the simulated bender, this also give slightly unrealisitc results because the potential is not allowed
        # to go to zero (finite field space) so the the particle will violate conservation of energy
        # limit: how far to carry the calculation for along the x axis. For the hard edge magnet it's just the hard edge
        # length, but for the simulated magnets, it's that plus twice the length at the ends.
        # h: timestep
        # lowField: wether to model low or high field seekers
        if isinstance(self,CombinerHexapoleSim):
            assert 0.0 <= inputOffset < self.ap
        q = np.asarray([0.0, -inputOffset, 0.0])
        p = np.asarray([self.PTL.v0Nominal, 0.0, 0.0])
        coordList = []  # Array that holds particle coordinates traced through combiner. This is used to find lenght

        # #of orbit.
        def force(x):
            if isinstance(self,CombinerHexapoleSim):
                if x[0] < self.Lm + self.space and sqrt(x[1] ** 2 + x[2] ** 2) > self.ap:
                    return np.empty(3) * np.nan
            return np.asarray(self.fastFieldHelper.force_NoSearchInside(x[0], x[1], x[2]))

        limit = self.Lm + 2 * self.space
        forcePrev = force(q)  # recycling the previous force value cut simulation time in half
        while True:
            F = forcePrev
            F[2] = 0.0  # exclude z component, ideally zero
            a = F
            q_n = q + p * h + .5 * a * h ** 2
            if q_n[0] > limit:  # if overshot, go back and walk up to the edge assuming no force
                dr = limit - q[0]
                dt = dr / p[0]
                q = q + p * dt
                coordList.append(q)
                break
            F_n = force(q_n)
            assert np.all(np.isnan(F_n) == False)

            F_n[2] = 0.0
            a_n = F_n  # accselferation new or accselferation sub n+1
            p_n = p + .5 * (a + a_n) * h
            q = q_n
            p = p_n
            forcePrev = F_n
            coordList.append(q)
        qArr = np.asarray(coordList)
        outputAngle = np.arctan2(p[1], p[0])
        outputOffset = q[1]
        if isinstance(self,CombinerHexapoleSim):
            lensCorner = np.asarray([self.space + self.Lm, -self.ap, 0.0])
            minSep = np.min(np.linalg.norm(qArr - lensCorner, axis=1))
            return outputAngle, outputOffset, qArr, minSep
        else:
            return outputAngle, outputOffset, qArr
    def compute_Trajectory_Length(self, qTracedArr:np.ndarray)->float:
        # to find the trajectory length model the trajectory as a bunch of little deltas for each step and add up their
        # length
        x = qTracedArr[:, 0]
        y = qTracedArr[:, 1]
        xDelta = np.append(x[0],x[1:] - x[:-1])  # have to add the first value to the length of difference because
        # it starts at zero
        yDelta = np.append(y[0], y[1:] - y[:-1])
        dLArr = np.sqrt(xDelta ** 2 + yDelta ** 2)
        Lo = np.sum(dLArr)
        return Lo
    def transform_Lab_Coords_Into_Element_Frame(self, qLab:np.ndarray)->np.ndarray:
        qEl = self.transform_Lab_Frame_Vector_Into_Element_Frame(qLab-self.r2) #a simple vector trick
        return qEl

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl:np.ndarray)->np.ndarray:
        # NOTE: THIS NOT GOING TO BE CORRECT IN GENERALY BECAUSE THE TRAJECTORY IS NOT SMOOTH AND I HAVE NOT WORKED IT OUT
        # YET
        qo = qEl.copy()
        qo[0] = self.Lo - qo[0]
        qo[1] = 0  # qo[1]
        return qo
    def transform_Element_Coords_Into_Lab_Frame(self,q:np.ndarray)->np.ndarray:
        qNew=q.copy()
        qNew[:2]=self.ROut@qNew[:2]+self.r2[:2]
        return qNew
    def transform_Orbit_Frame_Into_Lab_Frame(self,qo:np.ndarray)->np.ndarray:
        qNew=qo.copy()
        qNew[0]=-qNew[0]
        qNew[:2]=self.ROut@qNew[:2]
        qNew+=self.r1
        return qNew


class CombinerSim(CombinerIdeal):
    def __init__(self, PTL, combinerFile, mode,sizeScale=1.0,build=True):
        # PTL: particle tracing lattice object
        # combinerFile: File with data with dimensions (n,6) where n is the number of points and each row is
        # (x,y,z,gradxB,gradyB,gradzB,B). Data must have come from a grid. Data must only be from the upper quarter
        # quadrant, ie the portion with z>0 and x< length/2
        #mode: wether the combiner is functioning as a loader, or a circulator.
        # sizescale: factor to scale up or down all dimensions. This modifies the field strength accordingly, ie
        # doubling dimensions halves the gradient
        assert mode in ('injector','storageRing')
        Lm = .187
        apL = .015
        apR = .025
        apZ = 6e-3
        super().__init__(PTL, Lm, np.nan, np.nan, apL, apR, apZ, mode, sizeScale, build=False)
        self.fringeSpace = 5 * 1.1e-2
        self.combinerFile = combinerFile
        if build==True:
            self.build()

    def build(self)->None:
        self.space = self.fringeSpace * self.sizeScale  # extra space past the hard edge on either end to account for fringe fields
        self.apL = self.apL * self.sizeScale
        self.apR = self.apR * self.sizeScale
        self.apz = self.apz * self.sizeScale
        data = np.asarray(pd.read_csv(self.combinerFile, delim_whitespace=True, header=None))

        # use the new size scaling to adjust the provided data
        data[:, :3] = data[:, :3] * self.sizeScale  # scale the dimensions
        data[:, 3:6] = data[:, 3:6] / self.sizeScale  # scale the field gradient
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input
        fieldData=List(self.shape_Field_Data_3D(data))
        self.fastFieldHelper=fastNumbaMethodsAndClass.CombinerSimFieldHelper_Numba(fieldData,np.nan,self.Lb,self.Lm,
                                                                        self.space,self.apL,self.apR,self.apz,np.nan,self.fieldFact)
        inputAngle, inputOffset, qTracedArr = self.compute_Input_Angle_And_Offset()
        # TODO: I'M PRETTY SURE i CAN CONDENSE THIS WITH THE COMBINER IDEAL
         # 0.07891892567413786
        # to find the length
        self.Lo = self.compute_Trajectory_Length(
            qTracedArr)  # np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.L = self.Lo  # TODO: WHAT IS THIS DOING?? is it used anywhere
        self.ang = inputAngle
        y0 = inputOffset
        x0 = self.space
        theta = inputAngle
        self.La = (y0 + x0 / np.tan(theta)) / (np.sin(theta) + np.cos(theta) ** 2 / np.sin(theta))

        self.inputOffset = inputOffset-np.tan(inputAngle) * self.space  # the input offset is measured at the end of the hard edge
        self.fastFieldHelper = fastNumbaMethodsAndClass.CombinerSimFieldHelper_Numba(fieldData, self.La, self.Lb,
                                                            self.Lm,self.space, self.apL, self.apR,self.apz, self.ang,self.fieldFact)
        self.update_Field_Fact(self.fieldFact)
    def update_Field_Fact(self,fieldStrengthFact)->None:
        self.fastFieldHelper.fieldFact=fieldStrengthFact
        self.fieldFact=fieldStrengthFact


class HalbachBenderSimSegmented(BenderIdeal):
    #todo: a feature to prevent the interpolation introducing ridiculous fields into the bore by ending inside the
    #magnet
    #this element is a model of a bending magnet constructed of segments. There are three models from which data is
    # extracted required to construct the element. All exported data must be in a grid, though it the spacing along
    #each dimension may be different.
    #1:  A model of the repeating segments of magnets that compose the bulk of the bender. A magnet, centered at the
    #bending radius, sandwiched by other magnets (at the appropriate angle) to generate the symmetry. The central magnet
    #is position with z=0, and field values are extracted from z=0-TINY_STEP to some value that extends slightly past
    #the tilted edge. See docs/images/HalbachBenderSimSegmentedImage1.png
    #2: A model of the magnet between the last magnet, and the inner repeating section. This is required becasuse I found
    #that the assumption that I could jump straight from the outwards magnet to the unit cell portion was incorrect,
    #the force was very discontinuous. To model this I the last few segments of a bender, then extrac the field from
    #z=0 up to a little past halfway the second magnet. Make sure to have the x bounds extend a bit to capture
    # #everything. See docs/images/HalbachBenderSimSegmentedImage2.png
    #3: a model of the input portion of the bender. This portions extends half a magnet length past z=0. Must include
    #enough extra space to account for fringe fields. See docs/images/HalbachBenderSimSegmentedImage3.png
    def __init__(self, PTL,Lm:float,rp:float,numMagnets:float,rb:float,extraSpace:float,rOffsetFact:float):
        super().__init__(PTL, None, None, rp, rb, None, build=False)
        self.PTL=PTL
        self.rb=rb
        self.space=extraSpace
        self.Lm=Lm
        self.rp=rp
        self.Lseg = self.Lm + self.space * 2
        self.magnetWidth = rp * np.tan(2 * np.pi / 24) * 2
        self.yokeWidth = self.magnetWidth
        self.ucAng = None
        self.rOffsetFact=rOffsetFact #factor to times the theoretic optimal bending radius by
        self.fringeFracOuter=1.5 #multiple of bore radius to accomodate fringe field
        self.Lcap=self.Lm/2+self.fringeFracOuter*self.rp
        self.numMagnets = numMagnets
        self.segmented=True
        self.RIn_Ang=None
        self.M_uc=None
        self.M_ang=None
        self.numPointsBoreAp=20 #This many points should span the bore ap for good field sampling
        self.longitudinalCoordSpacing = self.rp/self.numPointsBoreAp #Spacing through unit cell. invariant scale
        self.numModelLenses=5 #number of lenses in halbach model to represent repeating system. Testing has shown
        #this to be optimal
        self.cap = True
        self.K = None #spring constant of field strength to set the offset of the lattice
        self.K_Func=None #function that returns the spring constant as a function of bending radii. This is used in the
        #constraint solver
        self.build()

    def build(self)->None:
        """Overrides abstract method from Element"""
        if self.numMagnets is not None:
            self.build_Pre_Constraint()
            self.build_Post_Constrained()
        else:
            self.build_Pre_Constraint()

    def compute_Aperture(self)->None:
        #beacuse the bender is segmented, the maximum vacuum tube allowed is not the bore of a single magnet
        #use simple geoemtry of the bending radius that touches the top inside corner of a segment
        vacuumTubeThickness=1e-3
        radiusCorner=np.sqrt((self.rb-self.rp)**2+(self.Lseg/2)**2)
        apMaxGeom=self.rb-radiusCorner-vacuumTubeThickness #max aperture without clipping magnet
        safetyFactor=.95
        apMaxGoodField=safetyFactor*self.numPointsBoreAp*self.rp/(self.numPointsBoreAp+np.sqrt(2))  #max aperture
        # without particles seeing field interpolation reaching into magnetic materal. Will not be exactly true for
        # several reasons (using int, and non equal grid in xy), so I include a smallsafety factor
        apMax=min([apMaxGeom,apMaxGoodField])
        assert apMax/self.rp>.8 #Not confident results work outside ap~rp
        return apMax

    def set_BpFact(self,BpFact:float):
        assert 0.0<=BpFact
        self.fieldFact=BpFact

    def build_Pre_Constraint(self)->None:
        def find_K(rb:float):
            ucAngTemp=np.arctan(self.Lseg/(2*(rb-self.rp-self.yokeWidth))) #value very near final value, good
            #approximation
            lens = _HalbachBenderFieldGenerator(self.rp, rb, ucAngTemp,self.Lm, numLenses=3)
            xArr = np.linspace(-self.rp/3, self.rp/3) + rb
            coords = np.asarray(np.meshgrid(xArr, 0, 0)).T.reshape(-1, 3)
            FArr = SIMULATION_MAGNETON*lens.BNorm_Gradient(coords)[:, 0]
            xArr -= rb
            m, b = np.polyfit(xArr, FArr, 1)  # fit to a line y=m*x+b, and only use the m component
            return m
        rArr=np.linspace(self.rb*.95,self.rb*1.05,num=10)
        kList = []
        for rb in rArr:
            kList.append(find_K(rb))
        kArr = np.asarray(kList)
        a, b, c = np.polyfit(rArr, kArr, 2)
        self.K_Func=lambda r: a * r ** 2 + b * r + c
        self.outputOffsetFunc = lambda r:  self.rOffsetFact*(sqrt(
            r ** 2 / 16 + self.PTL.v0Nominal ** 2 / (2 * self.K_Func(r))) - r / 4)  # this accounts for energy loss

    def build_Post_Constrained(self)->None:
        self.ap=self.compute_Aperture()
        assert self.rb-self.rp-self.yokeWidth>0.0
        self.ucAng = np.arctan(self.Lseg / (2 * (self.rb - self.rp - self.yokeWidth)))
        #500um works very well, but 1mm may be acceptable
        numModelLenes=3 #3 turns out to be a good number
        assert numModelLenes%2==1
        self.ang = 2 * self.numMagnets * self.ucAng
        self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
        m = np.tan(self.ucAng)
        self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        m = np.tan(self.ang / 2)
        self.M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        self.K=self.K_Func(self.rb)
        self.build_Fast_Field_Helper()
        self.fill_rOffset_And_Dependent_Params(self.outputOffsetFunc(self.rb))

    def build_Fast_Field_Helper(self)->None:
        """compute field values and build fast numba helper"""
        fieldDataSeg = self.generate_Segment_Field_Data()
        fieldDataInternal = self.generate_Internal_Fringe_Field_Data()
        fieldDataCap = self.generate_Cap_Field_Data()
        assert np.all(fieldDataCap[0]==fieldDataInternal[0]) and np.all(fieldDataCap[1]==fieldDataInternal[1])
        self.fastFieldHelper=fastNumbaMethodsAndClass.SegmentedBenderSimFieldHelper_Numba(fieldDataSeg,
            fieldDataInternal,fieldDataCap,self.ap,self.ang,self.ucAng,self.rb,self.numMagnets,self.Lcap,self.M_uc,
                                                                                          self.M_ang,self.RIn_Ang)
        self.fastFieldHelper.force(self.rb+1e-3,1e-3,1e-3) #force numba to compile
        self.fastFieldHelper.magnetic_Potential(self.rb+1e-3,1e-3,1e-3) #force numba to compile

    def make_Grid_Coords(self,xMin:float,xMax:float,zMin:float,zMax:float)->np.ndarray:
        """Make Array of points that the field will be evaluted at for fast interpolation. only x and s values change.
        """
        numPointsX=int(self.numPointsBoreAp*(xMax-xMin)/self.ap)
        yMin,yMax = -(self.rp + TINY_STEP),TINY_STEP #same for every part of bender
        numPointsY = int(self.numPointsBoreAp * (yMax - yMin) / self.ap)
        numPointsZ=int((zMax-zMin)/self.longitudinalCoordSpacing)
        assert numPointsX/numPointsY>=(xMax-xMin)/(yMax-yMin) # should be at least this ratio
        xArrArgs,yArrArgs,zArrArgs=(xMin,xMax,numPointsX),(yMin,yMax,numPointsY),(zMin,zMax,numPointsZ)
        coordArrList=[np.linspace(arrArgs[0],arrArgs[1],arrArgs[2]) for arrArgs in (xArrArgs,yArrArgs,zArrArgs)]
        gridCoords=np.asarray(np.meshgrid(*coordArrList)).T.reshape(-1,3)
        return gridCoords

    def fill_rOffset_And_Dependent_Params(self,rOffset:float)->None:
        #this needs a seperate function because it is called when finding the optimal rOffset rather than rebuilding
        #the entire element
        self.outputOffset=rOffset
        self.ro=self.rb+self.outputOffset
        self.L=self.ang*self.rb
        self.Lo=self.ang*self.ro+2*self.Lcap
    def update_rOffset_Fact(self,rOffsetFact:float)->None:
        self.rOffsetFact=rOffsetFact
        self.fill_rOffset_And_Dependent_Params(self.outputOffsetFunc(self.rb))
    def generate_Cap_Field_Data(self)->tuple:
        #x and y bounds should match with internal fringe bounds
        xMin=(self.rb-self.rp)*np.cos(2*self.ucAng)-TINY_STEP
        xMax=self.rb+self.rp+TINY_STEP
        zMin=-self.Lcap-TINY_STEP
        zMax=TINY_STEP
        fieldCoords=self.make_Grid_Coords(xMin,xMax,zMin,zMax)
        validIndices=np.sqrt((fieldCoords[:,0]-self.rb)**2+fieldCoords[:,1]**2)<self.rp
        lens=_HalbachBenderFieldGenerator(self.rp,self.rb,self.ucAng,self.Lm,
                                                numLenses=self.numModelLenses,positiveAngleMagnetsOnly=True)
        return self.compute_Valid_Field_Data(lens,fieldCoords,validIndices)

    def is_Valid_Internal_Fringe(self,coord0:np.ndarray)->float:
        """Return True if coord does NOT enter magnetic material, else False"""
        xzAngle = np.arctan2(coord0[2], coord0[0])
        coord=coord0.copy()
        assert -2*TINY_STEP/self.rb<=xzAngle<3*self.ucAng
        if self.ucAng < xzAngle <= 3 * self.ucAng:
            rotAngle=2*self.ucAng if xzAngle<=2*self.ucAng else 3*self.ucAng
            coord = Rot.from_rotvec(np.asarray([0.0, rotAngle, 0.0])).as_matrix() @ coord
        return np.sqrt((coord[0] - self.rb) ** 2 + coord[1] ** 2) < self.rp
    def generate_Internal_Fringe_Field_Data(self)->tuple:
        """An magnet slices are required to model the region going from the cap to the repeating unit cell,otherwise
        there is too large of an energy discontinuity"""
        #x and y bounds should match with cap bounds
        xMin=(self.rb-self.rp)*np.cos(2*self.ucAng)-TINY_STEP  #inward enough to account for the tilt
        xMax=self.rb+self.rp+TINY_STEP
        zMin=-TINY_STEP
        zMax=np.tan(2*self.ucAng)*(self.rb+self.ap)+TINY_STEP
        fieldCoords=self.make_Grid_Coords(xMin,xMax,zMin,zMax)
        validIndices=[self.is_Valid_Internal_Fringe(coord) for coord in fieldCoords]
        lens = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                                         numLenses=self.numModelLenses, positiveAngleMagnetsOnly=True)
        return self.compute_Valid_Field_Data(lens,fieldCoords,validIndices)

    def generate_Segment_Field_Data(self)->tuple:
        """Internal repeating unit cell segment. This is modeled as a tilted portion with angle self.ucAng to the
        z axis, with its bottom face at z=0 alinged with the xy plane"""
        xMin=(self.rb-self.ap)*np.cos(self.ucAng)-TINY_STEP
        xMax=self.rb+self.ap+TINY_STEP
        zMin=-TINY_STEP
        zMax=np.tan(self.ucAng)*(self.rb+self.ap)+TINY_STEP
        fieldCoords=self.make_Grid_Coords(xMin,xMax,zMin,zMax)
        validIndices=np.sqrt((fieldCoords[:,0]-self.rb)**2+fieldCoords[:,1]**2)<self.rp
        lens = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                                                          numLenses=self.numModelLenses+2)
        return self.compute_Valid_Field_Data(lens,fieldCoords,validIndices)

    def compute_Valid_Field_Data(self,lens:_HalbachBenderFieldGenerator,fieldCoords:np.ndarray,
                                 validIndices:Union[list,tuple,np.ndarray])->tuple:
        BNormGradArr, BNormArr = np.zeros((len(fieldCoords), 3)) * np.nan, np.zeros(len(fieldCoords)) * np.nan
        BNormGradArr[validIndices], BNormArr[validIndices] = lens.BNorm_Gradient(fieldCoords[validIndices],
                                                                                 returnNorm=True)
        fieldData = np.column_stack((fieldCoords, BNormGradArr, BNormArr))
        return self.shape_Field_Data_3D(fieldData)

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl:np.ndarray)->np.ndarray:
        # q: element coordinates (x,y,z)
        # returns qo: the coordinates in the orbit frame (s,xo,yo)

        qo = qEl.copy()
        angle = full_Arctan(qo)#np.arctan2(qo[1], qo[0])
        if angle < 0:  # restrict range to between 0 and 2pi
            angle += 2 * np.pi

        if angle < self.ang:  # if particle is in the bending angle section. Could still be outside though
            phi = self.ang - angle  # angle swept out by particle in trajectory. This is zero
            # when the particle first enters
            qox = sqrt(qEl[0] ** 2 + qEl[1] ** 2) - self.ro
            qo[0] = self.ro * phi + self.Lcap  # include the distance traveled throught the end cap
            qo[1] = qox
        else:  # if particle is outside of the bending segment angle so it could be in the caps, or elsewhere
            if (self.rb - self.ap < qEl[0] < self.rb + self.ap) and (0 > qEl[1] > -self.Lcap):  # If inside the cap on
                # the eastward side
                qo[0] = self.Lcap + self.ang * self.ro + (-qEl[1])
                qo[1] = qEl[0] - self.ro
            else:
                qTest = qEl.copy()
                qTest[0] = self.RIn_Ang[0, 0] * qEl[0] + self.RIn_Ang[0, 1] * qEl[1]
                qTest[1] = self.RIn_Ang[1, 0] * qEl[0] + self.RIn_Ang[1, 1] * qEl[1]
                # if in the westard side cap
                if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0):
                    qo[0] = self.Lcap - qTest[1]
                    qo[1] = qTest[0] - self.ro
                else:  # if in neither then it must be outside
                    qo[:] = np.nan
        return qo
    def _get_Shapely_Object_Of_Bore(self):
        """Shapely object of bore in x,z plane with y=0. Not of vacuum tube, but of largest possible bore. For two
        unit cells."""
        from shapely.geometry import Polygon
        bore=Polygon([(self.rb+self.rp,0.0),(self.rb+self.rp,(self.rb+self.rp)*np.tan(self.ucAng)),
        ((self.rb+self.rp)*np.cos(self.ucAng*2),
         (self.rb+self.rp)*np.sin(self.ucAng*2)),
        ((self.rb-self.rp)*np.cos(self.ucAng*2),(self.rb-self.rp)*np.sin(self.ucAng*2))
                      ,(self.rb-self.rp,(self.rb-self.rp)*np.tan(self.ucAng)),(self.rb-self.rp,0.0)])
        return bore
class HalbachLensSim(LensIdeal):
    def __init__(self,PTL, rpLayers:Union[float,tuple],L: float,apFrac: float,bumpOffset: float,
                 magnetWidth: Union[float,tuple],build: bool=True):
        #if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
        #to accomdate the new rp such as force values and positions
        if isinstance(rpLayers,(float,int)):
            rpLayers=(rpLayers,)
            if magnetWidth is not None: #todo: when would it be none?
                assert isinstance(magnetWidth,(int,float))
                magnetWidth=(magnetWidth,)
        elif isinstance(rpLayers,tuple):
            if magnetWidth is not None:
                assert isinstance(magnetWidth,tuple)
        else: raise TypeError
        self.numGridPointsZ = 30
        self.numGridPointsXY = 20
        rp=min(rpLayers)
        self.apMax=(rp-TINY_STEP)*(1-np.sqrt(2)/self.numGridPointsXY) #from geometric arguments of grid inside circle.
        #imagine two concentric rings on a grid, such that no grid box which has a portion outside the outer ring
        #has any portion inside the inner ring. This is to prevent interpolation reaching into magnetic material
        ap=self.apMax if apFrac is None else apFrac*rp
        assert ap<=self.apMax
        super().__init__(PTL, L, None, rp, ap, bumpOffset,build=False)
        self.fringeFracOuter=1.5
        self.L=L
        self.bumpOffset=bumpOffset
        self.Lo=None
        self.magnetWidth=magnetWidth
        self.rpLayers=rpLayers #can be multiple bore radius for different layers
        self.fringeFracInnerMin=4.0 #if the total hard edge magnet length is longer than this value * rp, then it can
        #can safely be modeled as a magnet "cap" with a 2D model of the interior
        self.lengthEffective=None #if the magnet is very long, to save simulation
        #time use a smaller length that still captures the physics, and then model the inner portion as 2D
        self.Lcap=None
        self.extraFieldLength=None #extra field added to end of lens to account misalignment
        self.force_Func_Outer=None #function that returns 3D vector of force values towards the end of the magnet, if
        #the magent is short, then it returns the values for one half, otherwise symmetry is used  to model interior as
        #2D
        self.magnetic_Potential_Func_Fringe = None
        self.magnetic_Potential_Func_Inner = None
        self.fieldFact = 1.0 #factor to multiply field values by for tunability
        if build==True:
            self.build()

    def set_Length(self,L:float)->None:
        assert L>0.0
        self.L=L
        self.build()
    def set_extraFieldLength(self)->None:
        """Set factor that extends field interpolation along length of lens to allow for misalignment. If misalignment
        is too large for good field region, extra length is clipped"""
        jitterAmp=self.PTL.jitterAmp*np.sqrt(2) #consider circular aperture
        assert jitterAmp<self.L/10.0 #small angles
        maxJitterAmp=self.apMax-self.ap
        tiltMax=jitterAmp/self.L if jitterAmp<maxJitterAmp else maxJitterAmp/self.L
        self.extraFieldLength=self.rp*tiltMax*1.1 #safety factor for approximations
    def build(self)->None:
        """Overrides abstract method from Element"""
        if self.L is None: #defer building until constraints are satisified
            return
        self.set_extraFieldLength()
        self.fill_Geometric_Params()
        self.build_Field_Helper()
        F_edge=np.linalg.norm(self.force(np.asarray([0.0,self.ap/2,.0])))
        F_center=np.linalg.norm(self.force(np.asarray([self.Lcap,self.ap/2,.0])))
        assert F_edge/F_center<.01
    def fill_Geometric_Params(self)->None:
        """Compute dependent geometric values"""
        self.Lm=self.L-2*self.fringeFracOuter*max(self.rpLayers)  #hard edge length of magnet
        assert self.Lm>0.0
        self.Lo=self.L
        self.lengthEffective=min(self.fringeFracInnerMin*max(self.rpLayers),
                                 self.Lm)  #if the magnet is very long, to save simulation
        #time use a smaller length that still captures the physics, and then model the inner portion as 2D
        self.Lcap=self.lengthEffective/2+self.fringeFracOuter*max(self.rpLayers)
        maximumMagnetWidth=tuple(rp*np.tan(2*np.pi/24)*2 for rp in self.rpLayers)
        if self.magnetWidth is None:
            self.magnetWidth=maximumMagnetWidth
        else:
            assert np.all(np.array(self.magnetWidth)<maximumMagnetWidth)
        mountThickness=1e-3 #outer thickness of mount, likely from space required by epoxy and maybe clamp
        self.outerHalfWidth=max(self.rpLayers)+self.magnetWidth[np.argmax(self.rpLayers)] +mountThickness
    def make_Grid_Coord_Arrays(self)->tuple:
        """
        because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
        """
        yArr_Quadrant = np.linspace(-TINY_OFFSET, self.ap-TINY_OFFSET, self.numGridPointsXY)
        xArr_Quadrant = -yArr_Quadrant.copy()
        zMin = -TINY_OFFSET # inside the lens
        zMax = self.Lcap+TINY_OFFSET + self.extraFieldLength  # outside the lens
        zArr = np.linspace(zMin, zMax, self.numGridPointsZ)
        return xArr_Quadrant,yArr_Quadrant,zArr
    def make_2D_Field_Data(self,lens:_HalbachLensFieldGenerator,xArr:np.ndarray,
                           yArr:np.ndarray)->Union[np.ndarray,None]:
        """
        Make 2d field data for interpolation.

        This comes from the center of the lens, and models a continuous segment homogenous in x (element frame)
        that allows for constructing very long lenses. If lens is too short, return None

        :param lens: lens object to compute fields values
        :param xArr: Grid edge x values of quarter of plane
        :param yArr: Grid edge y values of quarter of plane
        :return: Either 2d array of field data, or None
        """
        if self.lengthEffective<self.Lm: #if total magnet length is large enough to ignore fringe fields for interior
            # portion inside then use a 2D plane to represent the inner portion to save resources
            planeCoords=np.asarray(np.meshgrid(xArr,yArr,0)).T.reshape(-1,3)
            validIndices=np.linalg.norm(planeCoords,axis=1)<=self.rp
            BNormGrad,BNorm=np.zeros((len(validIndices),3))*np.nan,np.ones(len(validIndices))*np.nan
            BNormGrad[validIndices],BNorm[validIndices] = lens.BNorm_Gradient(planeCoords[validIndices],returnNorm=True)
            # BNormGrad,BNorm=lens.BNorm_Gradient(planeCoords,returnNorm=True)
            data2D=np.column_stack((planeCoords[:,:2],BNormGrad[:,:2],BNorm)) #2D is formated as
            # [[x,y,z,B0Gx,B0Gy,B0],..]
        else:
            data2D=None
        return data2D
    def make_3D_Field_Data(self,lens:_HalbachLensFieldGenerator,xArr:np.ndarray,yArr:np.ndarray,
                           zArr:np.ndarray)->np.ndarray:
        """
        Make 3d field data for interpolation from end of lens region

        If the lens is sufficiently long compared to bore radius then this is only field data from the end region
        (fringe frields and interior near end) because the interior region is modeled as a single plane to exploit
        longitudinal symmetry. Otherwise, it is exactly half of the lens and fringe fields

        :param lens: lens object to compute fields values
        :param xArr: Grid edge x values of quarter of plane
        :param yArr: Grid edge y values of quarter of plane
        :param zArr: Grid edge z values of half of lens, or region near end if long enough
        :return: 2D array of field data
        """
        volumeCoords=np.asarray(np.meshgrid(xArr,yArr,zArr)).T.reshape(-1,3) #note that these coordinates can have
        #the wrong value for z if the magnet length is longer than the fringe field effects. This is intentional and
        #input coordinates will be shifted in a wrapper function
        validXY=np.linalg.norm(volumeCoords[:,:2],axis=1)<=self.rp
        validZ=volumeCoords[:,2]>=self.Lm/2
        validIndices = np.logical_or(validXY,validZ)
        BNormGrad,BNorm=np.zeros((len(validIndices),3))*np.nan,np.ones(len(validIndices))*np.nan
        BNormGrad[validIndices],BNorm[validIndices] = lens.BNorm_Gradient(volumeCoords[validIndices],returnNorm=True)
        data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        return data3D
    def make_Field_Data(self)->tuple:
        lens = _HalbachLensFieldGenerator(self.lengthEffective, self.magnetWidth, self.rpLayers)
        xArr_Quadrant, yArr_Quadrant, zArr=self.make_Grid_Coord_Arrays()
        data2D=self.make_2D_Field_Data(lens,xArr_Quadrant,yArr_Quadrant)
        data3D=self.make_3D_Field_Data(lens,xArr_Quadrant,yArr_Quadrant,zArr)
        return data2D,data3D
    def build_Field_Helper(self)->None:
        """Generate magnetic field gradients and norms for """
        data2D, data3D = self.make_Field_Data()
        xArrEnd,yArrEnd,zArrEnd,FxArrEnd,FyArrEnd,FzArrEnd,VArrEnd=self.shape_Field_Data_3D(data3D)
        if data2D is not None: #if no inner plane being used
            xArrIn,yArrIn,FxArrIn,FyArrIn,VArrIn=self.shape_Field_Data_2D(data2D)
        else:
            xArrIn,yArrIn,FxArrIn, FyArrIn,VArrIn=[np.ones(1)*np.nan]*5
        fieldData=List([xArrEnd,yArrEnd,zArrEnd,FxArrEnd,FyArrEnd,FzArrEnd,VArrEnd,xArrIn,yArrIn,FxArrIn,FyArrIn,VArrIn])
        self.fastFieldHelper=fastNumbaMethodsAndClass.LensHalbachFieldHelper_Numba(fieldData,self.L,self.Lcap,self.ap,
                                                                                   self.extraFieldLength)
        self.fastFieldHelper.force(1e-3,1e-3,1e-3) #force compile
        self.fastFieldHelper.magnetic_Potential(1e-3,1e-3,1e-3) #force compile
    def update_Field_Fact(self,fieldStrengthFact:float)->None:
        """Update value used to model magnet strength tunability. fieldFact multiplies force and magnetic potential to
        model increasing or reducing magnet strength """
        self.fastFieldHelper.fieldFact=fieldStrengthFact
        self.fieldFact=fieldStrengthFact
    def perturb_Element(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float)->None:
        """Overrides abstract method from Element. Add catches for ensuring particle stays in good field region of
        interpolation"""
        print(self)
        assert abs(rotZ)<.05 and abs(rotZ)<.05 #small angle
        totalShiftY=shiftY+rotZ*self.L
        totalShiftZ=shiftZ+rotY*self.L
        totalShift=np.sqrt(totalShiftY**2+totalShiftZ**2)
        maxShift=self.apMax-self.ap #max shift is moving current aperture to maximum good field aperture
        if totalShift>maxShift:
            print('Misalignment is moving particles to bad field region, misalingment will be clipped')
            reductionFact=.9*maxShift/totalShift #safety factor
            print('proposed', totalShift, 'new', .9*maxShift)
            shiftY,shiftZ,rotY,rotZ=[val*reductionFact for val in [shiftY,shiftZ,rotY,rotZ]]
        self.fastFieldHelper.update_Element_Perturb_Params(shiftY,shiftZ,rotY,rotZ)

class CombinerHexapoleSim(CombinerIdeal):#,LensIdeal): #use inheritance here
    def __init__(self, PTL, Lm:float, rp:float, loadBeamDiam:float,layers:int, mode:str,build:bool=True):
        #PTL: object of ParticleTracerLatticeClass
        #Lm: hardedge length of magnet.
        #loadBeamDiam: Expected diameter of loading beam. Used to set the maximum combiner bending
        #layers: Number of concentric layers
        #mode: wether storage ring or injector. Injector uses high field seeking, storage ring used low field seeking
        assert  mode in ('storageRing','injector')
        CombinerIdeal.__init__(self,PTL,Lm,None,None,None,None,None,mode,1.0,build=False)
        # LensIdeal.__init__(self,PTL,None,None,None,None,build=False)
        # HalbachLensSim
        self.numGridPointsZ = 30
        self.numGridPointsXY = 30
        self.outerFringeFrac = 1.5

        self.Lm = Lm
        self.rp = rp
        self.layers=layers
        self.ap=.9*self.rp
        self.loadBeamDiam=loadBeamDiam
        self.PTL = PTL
        self.fieldFact=-1.0 if mode=='injector' else 1.0
        self.space=None
        self.extraFieldLength=0.0
        self.apMax=None

        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet

        self.shape = 'COMBINER_CIRCULAR'
        self.inputOffset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0
        self.lens=None
        if build==True:
            self.build()

    def build(self)->None:
        rpList = []
        magnetWidthList = []
        for _ in range(self.layers):
            rpList.append(self.rp + sum(magnetWidthList))
            nextMagnetWidth = (self.rp + sum(magnetWidthList)) * np.tan(2 * np.pi / 24) * 2
            magnetWidthList.append(nextMagnetWidth)
        self.space = max(rpList) * self.outerFringeFrac
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input in a straight line. This is that section
        self.lens = _HalbachLensFieldGenerator(self.Lm, tuple(magnetWidthList), tuple(rpList))
        inputAngle, inputOffset, trajectoryLength = self.compute_Orbit_Characteristics()

        self.Lo = trajectoryLength  # np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.L = self.Lo  # TODO: WHAT IS THIS DOING?? is it used anywhere
        self.ang = inputAngle
        y0 = inputOffset
        x0 = self.space
        theta = inputAngle
        self.La = (y0 + x0 / np.tan(theta)) / (np.sin(theta) + np.cos(theta) ** 2 / np.sin(theta))
        self.inputOffset = inputOffset - np.tan(
            inputAngle) * self.space  # the input offset is measured at the end of the hard edge
        fieldData = self.make_Field_Data(self.La,self.ang)
        self.fastFieldHelper = fastNumbaMethodsAndClass.CombinerHexapoleSimFieldHelper_Numba(fieldData, self.La,
                                self.Lb,self.Lm, self.space,self.ap, self.ang,self.fieldFact,self.extraFieldLength)

        self.fastFieldHelper.force(1e-3, 1e-3, 1e-3)  # force compile
        self.fastFieldHelper.magnetic_Potential(1e-3, 1e-3, 1e-3)  # force compile



        # F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        # F_center = np.linalg.norm(self.force(np.asarray([self.Lm/2 + self.space, self.ap / 2, .0])))
        # assert F_edge / F_center < .01

    def make_Grid_Coords_Arrays(self,La:float,ang:float)->tuple:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant


        yMax = self.ap + (La + self.ap * np.sin(abs(ang))) * np.sin(abs(ang))
        yMax=np.clip(yMax,self.rp,np.inf)
        numY=self.numGridPointsXY
        numX=int(self.numGridPointsXY*self.ap/yMax)
        zMax=self.compute_Valid_zMax(La,ang)

        yArr_Quadrant = np.linspace(-TINY_OFFSET, yMax, numY)  # this remains y in element frame
        xArr_Quadrant = np.linspace(-(self.rp - TINY_OFFSET), TINY_OFFSET, numX)  # this becomes x in element frame
        zArr = np.linspace(-TINY_OFFSET, zMax + self.extraFieldLength, num=self.numGridPointsZ)
        return xArr_Quadrant,yArr_Quadrant,zArr
    def compute_Valid_zMax(self,La:float,ang:float)->float:
        """Interpolation points inside magnetic material are set to nan. This can cause a problem near externel face of
        combiner because particles may see np.nan when they are actually in a valid region. To circumvent, zMax is
        chosen such that the first z point above the lens is just barely above it, and vacuum tube is configured to
        respect that. See fastNumbaMethodsAndClasses.CombinerHexapoleSimFieldHelper_Numba.is_Coord_Inside_Vacuum"""
        firstValidPointSpacing=1e-6
        maxLength = (self.Lb + (La + self.ap * np.sin(abs(ang))) * np.cos(abs(ang)))
        symmetryPlaneX = self.Lm / 2 + self.space  # field symmetry plane location. See how force is computed
        zMax=maxLength-symmetryPlaneX #subtle. The interpolation must extend to long enough to account for the
        #combiner not being symmetric, but the interpolation field being symmetric. See how force symmetry is handled
        zMax+=self.extraFieldLength
        pointSpacing=zMax/(self.numGridPointsZ-1)
        lastPointInLensIndex=int((self.Lm/2)/pointSpacing) #last point in magnetic material
        distToJustOutsideLens=firstValidPointSpacing+self.Lm/2-lastPointInLensIndex*pointSpacing #just outside material
        extraSpacePerPoint=distToJustOutsideLens/lastPointInLensIndex
        zMax+=extraSpacePerPoint*(self.numGridPointsZ-1)
        assert abs((lastPointInLensIndex*zMax/(self.numGridPointsZ-1)-self.Lm/2)-1e-6),1e-12
        return zMax

    def make_Field_Data(self,La:float,ang:float)->tuple:
        xArr,yArr,zArr = self.make_Grid_Coords_Arrays(La,ang)
        self.apMax = self.rp - np.sqrt((xArr[1] - xArr[0]) ** 2 + (yArr[1] - yArr[0]) ** 2)
        assert self.ap < self.apMax
        volumeCoords = np.asarray(np.meshgrid(xArr, yArr, zArr)).T.reshape(-1, 3)
        BNormGrad, BNorm = np.zeros((len(volumeCoords), 3)) * np.nan, np.zeros(len(volumeCoords)) * np.nan
        validIndices = np.logical_or(np.linalg.norm(volumeCoords[:, :2], axis=1) <= self.rp,
                                     volumeCoords[:, 2] >= self.Lm / 2)  # tricky
        BNormGrad[validIndices], BNorm[validIndices] = self.lens.BNorm_Gradient(volumeCoords[validIndices],
                                                                                returnNorm=True)
        data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        fieldData = self.shape_Field_Data_3D(data3D)
        return fieldData

    def compute_Orbit_Characteristics(self)->tuple:
        LaMax = (self.ap + self.space / np.tan(self.maxCombinerAng)) / (np.sin(self.maxCombinerAng) +
                                                np.cos(self.maxCombinerAng) ** 2 / np.sin(self.maxCombinerAng))
        fieldData=self.make_Field_Data(LaMax,self.maxCombinerAng)
        self.fastFieldHelper = fastNumbaMethodsAndClass.CombinerHexapoleSimFieldHelper_Numba(fieldData, np.nan, self.Lb,
                                                self.Lm,self.space, self.ap,np.nan,self.fieldFact,self.extraFieldLength)
        self.outputOffset = self.find_Ideal_Offset()
        inputAngle, inputOffset, qTracedArr, minSep = self.compute_Input_Angle_And_Offset(self.outputOffset)
        trajectoryLength = self.compute_Trajectory_Length(qTracedArr)
        assert np.abs(inputAngle) < self.maxCombinerAng  # tilt can't be too large or it exceeds field region.
        assert inputAngle * self.fieldFact > 0  # satisfied if low field is positive angle and high is negative.
        # Sometimes this can happen because the lens is to long so an oscilattory behaviour is required by injector
        return inputAngle, inputOffset, trajectoryLength

    def update_Field_Fact(self,fieldStrengthFact)->None:
        self.fastFieldHelper.fieldFact=fieldStrengthFact
        self.fieldFact=fieldStrengthFact
    def set_extraFieldLength(self)->None:
        """Set factor that extends field interpolation along length of lens to allow for misalignment. If misalignment
        is too large for good field region, extra length is clipped"""
        jitterAmp=self.PTL.jitterAmp*np.sqrt(2) #consider circular aperture
        assert jitterAmp<self.L/10.0 #small angles
        maxJitterAmp=self.apMax-self.ap
        tiltMax=jitterAmp/self.L if jitterAmp<maxJitterAmp else maxJitterAmp/self.L
        self.extraFieldLength=self.rp*tiltMax*1.1 #safety factor for approximations
    def perturb_Element(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float)->None:
        """Overrides abstract method from Element. Add catches for ensuring particle stays in good field region of
        interpolation"""
        assert abs(rotZ)<.05 and abs(rotZ)<.05 #small angle
        totalShiftY=shiftY+rotZ*self.L
        totalShiftZ=shiftZ+rotY*self.L
        totalShift=np.sqrt(totalShiftY**2+totalShiftZ**2)
        maxShift=self.apMax-self.ap #max shift is moving current aperture to maximum good field aperture
        if totalShift>maxShift:
            print('Misalignment is moving particles to bad field region, misalingment will be clipped')
            reductionFact=.9*maxShift/totalShift #safety factor
            print('proposed', totalShift, 'new', .9*maxShift)
            shiftY,shiftZ,rotY,rotZ=[val*reductionFact for val in [shiftY,shiftZ,rotY,rotZ]]
        self.fastFieldHelper.update_Element_Perturb_Params(shiftY,shiftZ,rotY,rotZ)
    def find_Ideal_Offset(self)->float:
        #use newton's method to find where the minimum seperation between atomic beam PATH and lens is equal to the
        #beam diameter for INJECTED beam. This requires modeling high field seekers. A larger output offset produces
        # a smaller input seperation, and a larger loading/circulating beam angular speration. Particle is traced
        # backwards from the end of the combiner to the input. Uses forward difference.
        fieldFact0=self.fieldFact
        self.update_Field_Fact(-1.0)

        #try and find an initial gradient that works
        keepTrying=True
        dxInitial=self.ap/2.0
        sepWithNoOffset=self.ap #maximum seperation occurs with no added offset
        maxTries=30
        numTries=0
        while keepTrying:
            try:
                grad=(self.compute_Input_Angle_And_Offset(dxInitial)[-1]-sepWithNoOffset)/dxInitial
                keepTrying=False
            except:
                dxInitial*=.5
            numTries+=1
            assert numTries<maxTries

        x=0.0 #initial value of output offset
        f=self.ap #initial value of lens/atom seperation. This should be equal to input deam diamter/2 eventuall
        i,iterMax=0,10 #to prevent possibility of ifnitne loop
        tolAbsolute=1e-4 #m
        targetSep=self.loadBeamDiam/2
        while(abs(f-targetSep)>tolAbsolute):
            deltaX=-.8*(f-targetSep)/grad # I like to use a little damping
            x=x+deltaX
            angle,offset,qArr,fNew = self.compute_Input_Angle_And_Offset(x)
            assert angle<0 #This doesn't work if the beam is exiting upwards. This can happpen physical of course,
            #but shouldn't happen
            grad=(fNew-f)/deltaX
            f=fNew
            i+=1
            assert i<iterMax
        assert x>0
        self.update_Field_Fact(fieldFact0)
        return x


class geneticLens(LensIdeal):
    def __init__(self, PTL, geneticLens, ap):
        # if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
        # to accomdate the new rp such as force values and positions
        # super().__init__(PTL, geneticLens.length, geneticLens.maximum_Radius(), np.nan,np.nan,'injector',fillParams=False)
        raise NotImplementedError #under construction still
        super().__init__(PTL, geneticLens.length, None, geneticLens.maximum_Radius(), ap, 0.0, fillParams=False)
        self.fringeFracOuter = 4.0
        self.L = geneticLens.length + 2 * self.fringeFracOuter * self.rp
        self.Lo = None
        self.shape = 'STRAIGHT'
        self.lens = geneticLens
        assert self.lens.minimum_Radius() >= ap
        self.fringeFracInnerMin = np.inf  # if the total hard edge magnet length is longer than this value * rp, then it can
        # can safely be modeled as a magnet "cap" with a 2D model of the interior
        self.lengthEffective = None  # if the magnet is very long, to save simulation
        # time use a smaller length that still captures the physics, and then model the inner portion as 2D

        self.magnetic_Potential_Func_Fringe = None
        self.magnetic_Potential_Func_Inner = None
        self.fieldFact = 1.0  # factor to multiply field values by for tunability
        if self.L is not None:
            self.build()

    def set_Length(self, L):
        assert L > 0.0
        self.L = L
        self.build()

    def build(self):
        # todo: more robust way to pick number of points in element. It should be done by using the typical lengthscale
        # of the bore radius

        numPointsLongitudinal = 31
        numPointsTransverse = 31

        self.Lm = self.L - 2 * self.fringeFracOuter * self.rp  # hard edge length of magnet
        assert np.abs(self.Lm - self.lens.length) < 1e-6
        assert self.Lm > 0.0
        self.Lo = self.L

        numXY = numPointsTransverse
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
        yArr_Quadrant = np.linspace(-TINY_STEP, self.ap + TINY_STEP, numXY)
        xArr_Quadrant = np.linspace(-(self.ap + TINY_STEP), TINY_STEP, numXY)

        zMin = -TINY_STEP
        zMax = self.L / 2 + TINY_STEP
        zArr = np.linspace(zMin, zMax, num=numPointsLongitudinal)  # add a little extra so interp works as expected

        # assert (zArr[-1]-zArr[-2])/self.rp<.2, "spatial step size must be small compared to radius"
        assert len(xArr_Quadrant) % 2 == 1 and len(yArr_Quadrant) % 2 == 1
        assert all((arr[-1] - arr[-2]) / self.rp < .1 for arr in [xArr_Quadrant, yArr_Quadrant]), "" \
                                                                                                  "spatial step size must be small compared to radius"

        volumeCoords = np.asarray(np.meshgrid(xArr_Quadrant, yArr_Quadrant, zArr)).T.reshape(-1,
                                                                                             3)  # note that these coordinates can have
        # the wrong value for z if the magnet length is longer than the fringe field effects. This is intentional and
        # input coordinates will be shifted in a wrapper function
        BNormGrad, BNorm = self.lens.BNorm_Gradient(volumeCoords, returnNorm=True)
        data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        self.fill_Field_Func(data3D)
        # self.compile_fast_Force_Function()

        # F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        # F_center = np.linalg.norm(self.force(np.asarray([self.L / 2, self.ap / 2, .0])))
        # assert F_edge / F_center < .01

    def force(self, q, searchIsCoordInside=True):
        raise Exception("under construction")

        # if np.isnan(F[0])==False:
        #     if q[0]<2*self.rp*self.fringeFracOuter or q[0]>self.L-2*self.rp*self.fringeFracOuter:
        #         return np.zeros(3)
        # F = self.fieldFact * np.asarray(F)
        # return F

    def fill_Field_Func(self, data):
        interpF, interpV = self.make_Interp_Functions(data)

        # wrap the function in a more convenietly accesed function
        @numba.njit(numba.types.UniTuple(numba.float64, 3)(numba.float64, numba.float64, numba.float64))
        def force_Func(x, y, z):
            Fx0, Fy0, Fz0 = interpF(-z, y, x)
            Fx = Fz0
            Fy = Fy0
            Fz = -Fx0
            return Fx, Fy, Fz

        self.force_Func = force_Func
        self.magnetic_Potential_Func = lambda x, y, z: interpV(-z, y, x)

    def magnetic_Potential(self, q):
        # this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
        x, y, z = q
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if self.is_Coord_Inside(q) == False:
            raise Exception(ValueError)

        if 0 <= x <= self.L / 2:
            x = self.L / 2 - x
            V = self.magnetic_Potential_Func(x, y, z)
        elif self.L / 2 < x:
            x = x - self.L / 2
            V = self.magnetic_Potential_Func(x, y, z)
        else:
            raise Exception(ValueError)
        return V
