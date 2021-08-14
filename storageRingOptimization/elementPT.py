import numpy as np
from math import sqrt
import warnings
import fastElementNUMBAFunctions
from InterpFunction import generate_3DInterp_Function_NUMBA, generate_2DInterp_Function_NUMBA
import pandas as pd
import numba
from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from HalbachLensClass import SegmentedBenderHalbach\
    as _SegmentedBenderHalbachLensFieldGenerator


# from profilehooks import profile


# Notes:
# --begining and ending of elements refer to the geometric sense in the lattice. For example, the beginning of the lens
# is the part encountered going clockwise, and the end is the part exited going clockwise for a particle being traced
# --Simulated fields come from COMSOL, and the data must be exported exactly correclty or else the reshaping will not work
# and it can be a huge pain in the but. I could have simply exported the data and used unstructured interpolation, but that
# is very very slow to inialize and query. So data must be in grid format. I also do not use the scipy linearNDInterpolater
# because it is not fast enough, and instead use a function I found in a stack exchange question. Someone made a github
# repository for it. This method gives the same results (except at the edges, where the logic fails, but scipy
# seems to still give reasonable ansers) and take about 5us per evaluatoin instead of 200us.
# --There are at least 3 frames of reference. The first is the lab frame, the second is the element frame, and the third
# which is only found in segmented bender, is the unit cell frame.

@numba.njit(numba.float64(numba.float64[:]))
def fast_Arctan2(q):
    phi = np.arctan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


class Element:
    def __init__(self, PTL):
        self.theta = None  # angle that describes an element's rotation in the xy plane.
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
        # are inside an element that can't be found with simple geometry
        self.ang = 0  # bending angle of the element. 0 for lenses and drifts
        self.Lm = None  # hard edge length of magnet along line through the bore
        self.L = None  # length of magnet along line through the bore
        self.K = None  # 'spring constant' of element. For some this comes from comsol fields.
        self.Lo = None  # length of orbit for particle. For lenses and drifts this is the same as the length. This is a nominal
        # value because for segmented benders the path length is not simple to compute
        self.index = None  # elements position in lattice
        self.cap = False  # wether there is a cap or not present on the element. Cap simulates fringe fields
        self.comsolExtraSpace = None  # extra space in comsol files to exported grids. this can be used to find dimensions
        self.ap = None  # apeture of element. most elements apetures are assumed to be square
        self.apz = None  # apeture in the z direction. all but the combiner is symmetric, so there apz is the same as ap
        self.apL = None  # apeture on the 'left' as seen going clockwise. This is for the combiner
        self.apR = None  # apeture on the'right'.
        self.type = None  # gemetric tupe of magnet, STRAIGHT,BEND or COMBINER. This is used to generalize how the geometry
        # constructed in particleTracerLattice
        self.sigma=None # the transverse translational. Only applicable to bump lenses now
        self.sim = None  # wether the field values are from simulations
        self.fieldFact = 1.0  # factor to modify field values everywhere in space by, including force
        self.fast_Numba_Force_Function=None #function that takes in only position and returns force. This is based on
        #arguments at the time of calling the function compile_Fast_Numba_Force_Function
    def compile_Fast_Numba_Force_Function(self):
        #function that generates the fast_Numba_Force_Function based on existing parameters. It compiles it as well
        #by passing dummy arguments
        pass
    def magnetic_Potential(self, q):
        # Function that returns the magnetic potential in the element's frame
        return 0.0

    def transform_Lab_Coords_Into_Global_Orbit_Frame(self, q, cumulativeLength):
        # Change the lab coordinates into the particle's orbit frame in the lab. This frame cumulatively grows with
        #revolutions.
        q = self.transform_Lab_Coords_Into_Element_Frame(q)  # first change lab coords into element frame
        qo = self.transform_Element_Coords_Into_Local_Orbit_Frame(q)  # then element frame into orbit frame
        qo[0] = qo[0] + cumulativeLength  # orbit frame is in the element's frame, so the preceding length needs to be
        # accounted for
        return qo

    def transform_Lab_Coords_Into_Element_Frame(self, q):
        # this is overwridden by all other elements
        pass

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, q):
        # for straight elements (lens and drift), element and orbit frame are identical. This is the local orbit frame.
        #does not grow with growing revolution
        # q: 3D coordinates in element frame
        return q.copy()
    def transform_Lab_Frame_Vector_Into_Element_Frame(self, vec):
        # vec: 3D vector in lab frame to rotate into element frame
        vecNew = vec.copy()  # copying prevents modifying the original value
        vecx = vecNew[0]
        vecy = vecNew[1]
        vecNew[0] = vecx * self.RIn[0, 0] + vecy * self.RIn[0, 1]
        vecNew[1] = vecx * self.RIn[1, 0] + vecy * self.RIn[1, 1]
        return vecNew

    def transform_Element_Frame_Vector_Into_Lab_Frame(self, vec):
        # rotate vector out of element frame into lab frame
        # vec: vector in
        vecNew = vec.copy()  # copy input vector to not modify the original
        vecx = vecNew[0];
        vecy = vecNew[1]
        vecNew[0] = vecx * self.ROut[0, 0] + vecy * self.ROut[0, 1]
        vecNew[1] = vecx * self.ROut[1, 0] + vecy * self.ROut[1, 1]
        return vecNew

    def set_Length(self, L):
        # this is used typically for setting the length after satisfying constraints
        self.L = L
        self.Lo = L

    def is_Coord_Inside(self, q):
        return None

    def make_Interp_Functions(self, data):
        # This method takes an array data with the shape (n,6) where n is the number of points in space. Each row
        # must have the format [x,y,z,gradxB,gradyB,gradzB,B] where B is the magnetic field norm at x,y,z and grad is the
        # partial derivative. The data must be from a 3D grid of points with no missing points or any other funny business
        # and the order of points doesn't matter
        xArr = np.unique(data[:, 0])
        yArr = np.unique(data[:, 1])
        zArr = np.unique(data[:, 2])

        numx = xArr.shape[0]
        numy = yArr.shape[0]
        numz = zArr.shape[0]
        BGradxMatrix = np.empty((numx, numy, numz))
        BGradyMatrix = np.empty((numx, numy, numz))
        BGradzMatrix = np.empty((numx, numy, numz))
        B0Matrix = np.zeros((numx, numy, numz))
        xIndices = np.argwhere(data[:, 0][:, None] == xArr)[:, 1]
        yIndices = np.argwhere(data[:, 1][:, None] == yArr)[:, 1]
        zIndices = np.argwhere(data[:, 2][:, None] == zArr)[:, 1]
        BGradxMatrix[xIndices, yIndices, zIndices] = data[:, 3]
        BGradyMatrix[xIndices, yIndices, zIndices] = data[:, 4]
        BGradzMatrix[xIndices, yIndices, zIndices] = data[:, 5]
        B0Matrix[xIndices, yIndices, zIndices] = data[:, 6]
        interpFx = generate_3DInterp_Function_NUMBA(-self.PTL.u0 * BGradxMatrix, xArr, yArr, zArr)
        interpFy = generate_3DInterp_Function_NUMBA(-self.PTL.u0 * BGradyMatrix, xArr, yArr, zArr)
        interpFz = generate_3DInterp_Function_NUMBA(-self.PTL.u0 * BGradzMatrix, xArr, yArr, zArr)
        interpV = generate_3DInterp_Function_NUMBA(self.PTL.u0 * B0Matrix, xArr, yArr, zArr)
        return interpFx, interpFy, interpFz, interpV


class LensIdeal(Element):
    # ideal model of lens with hard edge. Force inside is calculated from field at pole face and bore radius as
    # F=2*ub*r/rp**2 where rp is bore radius, and ub is bore magneton.
    def __init__(self, PTL, L, Bp, rp, ap, fillParams=True):
        # fillParams is used to avoid filling the parameters in inherited classes
        super().__init__(PTL)
        self.Bp = Bp  # field strength at pole face
        self.rp = rp  # bore radius
        self.L = L  # lenght of magnet
        self.ap = ap  # size of apeture radially
        self.type = 'STRAIGHT'  # The element's geometry

        if fillParams == True:
            self.fill_Params()
    def set_BpFact(self,BpFact):
        #update the magnetic field multiplication factor. This is used to simulate changing field values, or making
        #the bore larger
        self.fieldFact=BpFact
        self.K = self.fieldFact*(2 * self.Bp * self.PTL.u0 / self.rp ** 2)  # 'spring' constant
    def fill_Params(self):
        self.K = self.fieldFact*(2 * self.Bp * self.PTL.u0 / self.rp ** 2)  # 'spring' constant
        if self.L is not None:
            self.Lo = self.L
        self.compile_Fast_Numba_Force_Function()

    def magnetic_Potential(self, q):
        # potential energy at provided coordinates
        # q coords in element frame
        r = sqrt(q[1] ** 2 + +q[2] ** 2)
        if r < self.ap:
            return .5*self.K * r ** 2
        else:
            return 0.0
    def calculate_Steps_To_Collision(self,q,p,h):
        #based on the current position and trajectory, in the element frame, how many timesteps would be required
        # to have a collision
        r0=self.ap
        y,z=q[1:]
        vy,vz=p[1:]
        #time to collision with wall transversally
        dtT=(-vy*y - vz*z + sqrt(r0**2*vy**2 + r0**2*vz**2 - vy**2*z**2 + 2*vy*vz*y*z - vz**2*y**2))/(vy**2 + vz**2)
        dtL=(-self.r2[0]-q[0])/p[0] #longitudinal time to collision
        dt=min(dtT,dtL)
        return dt/h

    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew = q.copy()  # CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        qNew[0] = qNew[0] - self.r1[0]
        qNew[1] = qNew[1] - self.r1[1]
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def transform_Element_Coords_Into_Lab_Frame(self,qEl):
        qNew = qEl.copy()  # CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        qNew=self.transform_Element_Frame_Vector_Into_Lab_Frame(qNew)
        qNew[0] = qNew[0] +self.r1[0]
        qNew[1] = qNew[1] +self.r1[1]
        return qNew

    def force(self, q):
        # note: for the perfect lens, in it's frame, there is never force in the x direction. Force in x is always zero
        F=np.zeros(3)
        if 0 <= q[0] <= self.L and q[1] ** 2 + q[2] ** 2 < self.ap**2:
            F[1] = -self.K * q[1]
            F[2] = -self.K * q[2]
            return F
        else:
            return np.asarray([np.nan,np.nan,np.nan])
    def compile_Fast_Numba_Force_Function(self):
        L = self.L
        ap = self.ap
        K = self.K
        forceNumba = fastElementNUMBAFunctions.lens_Ideal_Force_NUMBA
        @numba.njit()
        def force_NUMBA_Wrapper(q):
            return forceNumba(q, L, ap, K)
        self.fast_Numba_Force_Function=force_NUMBA_Wrapper
        self.fast_Numba_Force_Function(np.zeros(3))


    def set_Length(self, L):
        # this is used typically for setting the length after satisfying constraints
        self.L = L
        self.Lo = self.L

    def is_Coord_Inside(self, q):
        # check with fast geometric arguments if the particle is inside the element. This won't necesarily work for all
        # elements. If True is retured, the particle is inside. If False is returned, it is defintely outside. If none is
        # returned, it is unknown
        # q: coordinates to test
        if not 0 <= q[0] <= self.L:
            return False
        else:
            if q[1] ** 2 + q[2] ** 2 < self.ap**2:
                return True
            else:
                return False

class BumpsLensIdeal(LensIdeal):
    def __init__(self, PTL, L, Bp, rp,sigma,ap, fillParams=True):
        super().__init__(PTL, L, Bp, rp, ap, fillParams=True)
        self.sigma=sigma #the amount of vertical shift for bumping the beam over

class Drift(LensIdeal):
    def __init__(self, PTL, L, ap):
        super().__init__(PTL, L, 0, np.inf, ap)

    def force(self, q):
        # note: for the perfect lens, in it's frame, there is never force in the x direction. Force in x is always zero
        if 0 <= q[0] <= self.L and q[1] ** 2 + q[2] ** 2 < self.ap**2:
            return np.zeros(3)
        else:
            return np.asarray([np.nan,np.nan,np.nan])


class BenderIdeal(Element):
    def __init__(self, PTL, ang, Bp, rp, rb, ap, fillParams=True):
        # this is the base model for the bending elements, of which there are several kinds. To maintain generality, there
        # are numerous methods and variables that are not necesary here, but are used in other child classes. For example,
        # this element has no caps, but a cap length of zero is still used.
        super().__init__(PTL)
        self.sim = False
        self.ang = ang  # total bending angle of bender
        self.Bp = Bp  # field strength at pole
        self.rp = rp  # bore radius of magnet
        self.ap = ap  # radial apeture size
        self.rb = rb  # bending radius of magnet. This is tricky because this is the bending radius down the center, but the
        # actual trajectory of the particles is offset a little out from this
        self.type = 'BEND'
        self.rOffset = None  # the offset to the bending radius because of centrifugal force. There are two versions, one that
        # accounts for reduced speed from energy conservation for actual fields, and one that doesn't
        self.ro = None  # bending radius of orbit, ie rb + rOffset.
        self.rOffsetFunc = None  # a function that returns the value of the offset of the trajectory for a given bending radius
        self.segmented = False  # wether the element is made up of discrete segments, or is continuous
        self.capped = False  # wether the element has 'caps' on the inlet and outlet. This is used to model fringe fields
        self.Lcap = 0  # length of caps

        if fillParams == True:
            self.fill_Params()

    def fill_Params(self):
        self.K = (2 * self.Bp * self.PTL.u0 / self.rp ** 2)  # 'spring' constant
        self.rOffsetFunc = lambda rb: sqrt(rb ** 2 / 4 + self.PTL.v0Nominal ** 2 / self.K) - rb / 2
        self.rOffset = self.rOffsetFunc(self.rb)
        self.ro = self.rb + self.rOffset
        if self.ang is not None:  # calculation is being delayed until constraints are solved

            self.L = self.rb * self.ang
            self.Lo = self.ro * self.ang
    def magnetic_Potential(self, q):
        # potential energy at provided coordinates
        # q coords in element frame
        r = sqrt(q[0] ** 2 + +q[1] ** 2)
        if self.rb + self.rp > r > self.rb - self.rp and np.abs(q[2]) < self.ap:
            rNew = sqrt((r - self.rb) ** 2 + q[2] ** 2)
            return self.Bp * self.PTL.u0 * rNew ** 2 / self.rp ** 2
        else:
            return 0.0

    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew = q - self.r0
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def transform_Element_Coords_Into_Lab_Frame(self,qEl):
        qNew=qEl.copy()
        qNew=self.transform_Element_Frame_Vector_Into_Lab_Frame(qNew)
        qNew=qNew+self.r0
        return qNew

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, q):
        # q: element coords
        # returns a 3d vector in the orbit frame. First component is distance along trajectory, second is radial displacemnt
        # from the nominal orbit computed with centrifugal force, and the third is the z axis displacemnt.
        qo = q.copy()
        phi = self.ang - fast_Arctan2(qo)  # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
        qo[0] = qos
        qo[1] = qox
        return qo

    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame
        F = np.zeros(3)
        phi = fast_Arctan2(q)
        if phi < self.ang:
            r = sqrt(q[0] ** 2 + q[1] ** 2)  # radius in x y frame
            F0 = -self.K * (r - self.rb)  # force in x y plane
            F[0] = np.cos(phi) * F0
            F[1] = np.sin(phi) * F0
            F[2] = -self.K * q[2]
        else:
            F = np.asarray([np.nan,np.nan,np.nan])
        return F
    def is_Coord_Inside(self, q):
        phi = fast_Arctan2(q)
        if phi < 0:  # constraint to between zero and 2pi
            phi += 2 * np.pi
        if phi <=self.ang:  # if particle is in bending segment
            rh = sqrt(q[0] ** 2 + q[1] ** 2) - self.rb  # horizontal radius
            r = sqrt(rh ** 2 + q[2] ** 2)  # particle displacement from center of apeture
            if r > self.ap:
                return False
            else:
                return True
        else:
            return False


class CombinerIdeal(Element):
    # combiner: This is is the element that bends the two beams together. The logic is a bit tricky. It's geometry is
    # modeled as a straight section, a simple square, with a segment coming of at the particle in put at an angle. The
    # angle is decided by tracing particles through the combiner and finding the bending angle.
    def __init__(self, PTL, Lm, c1, c2, ap, mode,sizeScale, fillsParams=True):
        super().__init__(PTL)
        self.sim = False
        self.mode = mode
        self.sizeScale = sizeScale  # the fraction that the combiner is scaled up or down to. A combiner twice the size would
        # use sizeScale=2.0
        self.ap = ap * self.sizeScale
        self.apR = self.ap * self.sizeScale
        self.apL = self.ap * self.sizeScale
        self.Lm = Lm * self.sizeScale
        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet
        self.c1 = c1 / self.sizeScale
        self.c2 = c2 / self.sizeScale
        self.space = 0  # space at the end of the combiner to account for fringe fields

        self.type = 'COMBINER'
        self.inputOffset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0
        if fillsParams == True:
            self.fill_Params()

    def fill_Params(self):

        self.Lb = self.Lm  # length of segment after kink after the inlet
        if self.mode=='injector': #if part of the injection system, atoms will be in high field seeking state
            lowField=False
            self.fieldFact=-1.0 #reverse field to model high field seeker
        else:
            lowField=True

        inputAngle, inputOffset, qTracedArr = self.compute_Input_Angle_And_Offset()
        self.Lo = np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.ang = inputAngle
        self.inputOffset = inputOffset

        self.apz = self.ap / 2
        self.La = self.ap * np.sin(self.ang)
        self.L = self.La * np.cos(self.ang) + self.Lb  # TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING

    def compute_Input_Angle_And_Offset(self, h=1e-7):
        # this computes the output angle and offset for a combiner magnet.
        # NOTE: for the ideal combiner this gives slightly inaccurate results because of lack of conservation of energy!
        # NOTE: for the simulated bender, this also give slightly unrealisitc results because the potential is not allowed
        # to go to zero (finite field space) so the the particle will violate conservation of energy
        # limit: how far to carry the calculation for along the x axis. For the hard edge magnet it's just the hard edge
        # length, but for the simulated magnets, it's that plus twice the length at the ends.
        # h: timestep
        # lowField: wether to model low or high field seekers
        q = np.asarray([0.0, 0.0, 0.0])
        p = np.asarray([self.PTL.v0Nominal, 0.0, 0.0])
        coordList = []  # Array that holds particle coordinates traced through combiner. This is used to find lenght
        # #of orbit.


        force = lambda x: self.force(x,searchIsCoordInside=False)
        limit = self.Lm + 2 * self.space

        forcePrev = force(q)  # recycling the previous force value cut simulation time in half
        while True:
            F = forcePrev
            F[2] = 0.0  # exclude z component, ideally zero
            a = F
            q_n = q + p * h + .5 * a * h ** 2
            F_n = force(q_n)
            F_n[2] = 0.0
            a_n = F_n  # accselferation new or accselferation sub n+1
            p_n = p + .5 * (a + a_n) * h
            if q_n[0] > limit:  # if overshot, go back and walk up to the edge assuming no force
                dr = limit - q[0]
                dt = dr / p[0]
                q = q + p * dt
                coordList.append(q)
                break
            q = q_n
            p = p_n
            forcePrev = F_n
            coordList.append(q)
        outputAngle = np.arctan2(p[1], p[0])
        outputOffset = q[1]
        return outputAngle, outputOffset, np.asarray(coordList)

    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew = q.copy()
        qNew = qNew - self.r2
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, q):
        # NOTE: THIS NOT GOING TO BE CORRECT IN GENERALY BECAUSE THE TRAJECTORY IS NOT SMOOTH AND I HAVE NOT WORKED IT OUT
        # YET
        qo = q.copy()
        qo[0] = self.Lo - qo[0]
        qo[1] = 0  # qo[1]
        return qo

    def force(self, q,searchIsCoordInside=True):
        # force at point q in element frame
        # q: particle's position in element frame
        if searchIsCoordInside==True:
            if self.is_Coord_Inside(q) is False:
                return np.asarray([np.nan])
        F = np.zeros(3)  # force vector starts out as zero
        if 0<q[0] < self.Lb:
            B0 = sqrt((self.c2 * q[2]) ** 2 + (self.c1 + self.c2 * q[1]) ** 2)
            F[1] = self.PTL.u0 * self.c2 * (self.c1 + self.c2 * q[1]) / B0
            F[2] = self.PTL.u0 * self.c2 ** 2 * q[2] / B0
        return F*self.fieldFact
    def magnetic_Potential(self, q):
        V0=0
        if 0<q[0] < self.Lb:
            V0 = self.PTL.u0*sqrt((self.c2 * q[2]) ** 2 + (self.c1 + self.c2 * q[1]) ** 2)
        return V0


    def is_Coord_Inside(self, q):
        # q: coordinate to test in element's frame
        if not -self.apz <= q[2] <= self.apz:  # if outside the z apeture (vertical)
            return False
        elif 0 <= q[0] <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner. Simple square apeture
            if -self.apL < q[1] < self.apR:  # if inside the y (width) apeture
                return True
        elif q[0] < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            m = np.tan(self.ang)
            Y1 = m * q[0] + (self.apR - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * q[0] + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * q[0] + (-self.apL - m * self.Lb)
            if np.sign(m)<0.0 and (q[1] < Y1 and q[1] > Y2 and q[1] > Y3): #if the inlet is tilted 'down'
                return True
            elif np.sign(m)>0.0 and (q[1] < Y1 and q[1] < Y2 and q[1] > Y3): #if the inlet is tilted 'up'
                return True
            else:
                return False

    def transform_Element_Coords_Into_Lab_Frame(self,q):
        qNew=q.copy()
        qNew[:2]=self.ROut@qNew[:2]+self.r2[:2]
        return qNew


class CombinerSim(CombinerIdeal):
    def __init__(self, PTL, combinerFile, mode,sizeScale=1.0):
        # PTL: particle tracing lattice object
        # combinerFile: File with data with dimensions (n,6) where n is the number of points and each row is
        # (x,y,z,gradxB,gradyB,gradzB,B). Data must have come from a grid. Data must only be from the upper quarter
        # quadrant, ie the portion with z>0 and x< length/2
        #mode: wether the combiner is functioning as a loader, or a circulator.
        # sizescale: factor to scale up or down all dimensions. This modifies the field strength accordingly, ie
        # doubling dimensions halves the gradient
        assert mode == 'injector' or mode == 'storageRing'
        Lm = .187
        apL = .015
        apR = .025
        fringeSpace = 5 * 1.1e-2
        apz = 6e-3
        print('fix the issue that the bounds extend to left and right pass the magnet')
        super().__init__(PTL, Lm, np.nan, np.nan, np.nan,mode, sizeScale,
                         fillsParams=False)  # TODO: replace all the Nones with np.nan
        self.sim = True

        self.space = fringeSpace * self.sizeScale  # extra space past the hard edge on either end to account for fringe fields
        self.apL = apL * self.sizeScale
        self.apR = apR * self.sizeScale
        self.apz = apz * self.sizeScale


        self.data = None
        self.combinerFile = combinerFile
        self.force_Func=None
        self.magnetic_Potential_Func = None
        self.fill_Params()

    def fill_Params(self):
        if self.mode=='injector': #if part of the injection system, atoms will be in high field seeking state
            self.fieldFact=-1.0
        self.data = np.asarray(pd.read_csv(self.combinerFile, delim_whitespace=True, header=None))

        # use the new size scaling to adjust the provided data
        self.data[:, :3] = self.data[:, :3] * self.sizeScale  # scale the dimensions
        self.data[:, 3:6] = self.data[:, 3:6] / self.sizeScale  # scale the field gradient
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input
        interpFx,interpFy, interpFz, funcV = self.make_Interp_Functions(self.data)
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def force_Func(x,y,z):
            Fx=interpFx(x,y,z)
            Fy=interpFy(x,y,z)
            Fz=interpFz(x,y,z)
            return Fx,Fy,Fz
        self.force_Func=force_Func
        self.magnetic_Potential_Func = lambda x, y, z: self.fieldFact*funcV(x, y, z)
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
        self.data = None  # to save memory and pickling time
        self.compile_Fast_Numba_Force_Function()
        # print(self.inputOffsetLoad,self.inputOffset)
        # print(self.ang,self.angLoad)

    def compute_Trajectory_Length(self, qTracedArr):
        # TODO: CHANGE THAT X DOESN'T START AT ZERO
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


    def force(self, q,searchIsCoordInside=True):
        # this function uses the symmetry of the combiner to extract the force everywhere.
        #I believe there are some redundancies here that could be trimmed to save time.
        if searchIsCoordInside==False: # the inlet length of the combiner, La, can only be computed after tracing
            #through the combiner. Thus, it should be set to zero to use the force function for tracing purposes
            La=0.0
        else:
            La=self.La
        F=fastElementNUMBAFunctions.combiner_Sim_Force_NUMBA(q, La,self.Lb,self.Lm,self.space,self.ang,
                                            self.apz,self.apL,self.apR,searchIsCoordInside,self.force_Func)
        F=self.fieldFact*np.asarray(F)
        return F
    def compile_Fast_Numba_Force_Function(self):
        forceNumba = fastElementNUMBAFunctions.combiner_Sim_Force_NUMBA
        La=self.La
        Lb=self.Lb
        Lm=self.Lm
        space=self.space
        ang=self.ang
        apz=self.apz
        apL=self.apL
        apR=self.apR
        force_Func=self.force_Func
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64[:]))
        def force_NUMBA_Wrapper(q):
            return forceNumba(q,La,Lb,Lm,space,ang,apz,apL,apR,True,force_Func)
        self.fast_Numba_Force_Function=force_NUMBA_Wrapper
        self.fast_Numba_Force_Function(np.zeros(3)) #force compile by passing a dummy argument

    def magnetic_Potential(self, q):
        # this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
        qNew = q.copy()
        if 0 <= qNew[0] <= (self.Lm / 2 + self.space):  # if the particle is in the first half of the magnet
            if qNew[2] < 0:  # if particle is in the lower plane
                qNew[2] = -qNew[2]  # flip position to upper plane
        if (self.Lm / 2 + self.space) < qNew[0]:  # if the particle is in the last half of the magnet
            qNew[0] = (self.Lm / 2 + self.space) - (
                        qNew[0] - (self.Lm / 2 + self.space))  # use the reflection of the particle
            if qNew[2] < 0:  # if in the lower plane, need to use symmetry
                qNew[2] = -qNew[2]
        return self.magnetic_Potential_Func(*qNew)


class BenderIdealSegmented(BenderIdeal):
    # -very similiar to ideal bender, but force is not a continuous
    # function of theta and r. It is instead a series of discrete magnets which are represented as a unit cell. A
    # full magnet would be modeled as two unit cells, instead of a single unit cell, to exploit symmetry and thus
    # save memory. Half the time the symetry is exploited by using a simple rotation, the other half of the time the
    # symmetry requires a reflection, then rotation.
    def __init__(self, PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space, rOffsetFact, ap, fillParams=True):
        super().__init__(PTL, None, Bp, rp, rb, ap, fillParams=False)
        self.numMagnets = numMagnets
        self.Lm = Lm
        self.space = space
        self.yokeWidth = yokeWidth
        self.ucAng = None
        self.segmented = True
        self.cap = False
        self.ap = ap
        self.rOffsetFact = rOffsetFact
        self.RIn_Ang = None
        self.Lseg = None
        self.M_uc = None  # matrix for reflection used in exploting segmented symmetry. This is 'inside' a single magnet element
        self.M_ang = None  # matrix for reflection used in exploting segmented symmetry. This is reflecting out from the unit cell
        if fillParams == True:
            self.fill_Params()

    def fill_Params(self):
        super().fill_Params()
        self.rOffsetFunc = lambda rb: self.rOffsetFact * (sqrt(
            rb ** 2 / 4 + self.PTL.v0Nominal ** 2 / self.K) - rb / 2)
        self.rOffset = self.rOffsetFunc(self.rb)
        self.Lseg = self.Lm + 2 * self.space
        if self.numMagnets is not None:
            self.ucAng = np.arctan((self.Lm / 2 + self.space) / (self.rb - self.yokeWidth - self.rp))
            self.ang = 2 * self.ucAng * self.numMagnets
            self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
            self.Lo = self.ro * self.ang
            m = np.tan(self.ucAng)
            self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)

    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew = q - self.r0
        qx = qNew[0]
        qy = qNew[1]
        qNew[0] = qx * self.RIn[0, 0] + qy * self.RIn[0, 1]
        qNew[1] = qx * self.RIn[1, 0] + qy * self.RIn[1, 1]
        return qNew

    def transform_Lab_Frame_Vector_Into_Element_Frame(self, vec):
        # vec: 3D vector in lab frame to rotate into element frame
        vecx = vec[0]
        vecy = vec[1]
        vec[0] = vecx * self.RIn[0, 0] + vecy * self.RIn[0, 1]
        vec[1] = vecx * self.RIn[1, 0] + vecy * self.RIn[1, 1]
        return vec

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, q):
        qo = q.copy()
        phi = self.ang - fast_Arctan2(q) # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
        qo[0] = qos
        qo[1] = qox
        return qo

    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame

        quc = self.transform_Element_Coords_Into_Unit_Cell_Frame(q)  # get unit cell coords
        if quc[1] < self.Lm / 2:  # if particle is inside the magnet region
            self.F[0] = -self.K * (quc[0] - self.rb)
            self.F[2] = -self.K * quc[2]
            self.F = self.transform_Unit_Cell_Force_Into_Element_Frame(self.F,
                                                                       q)  # transform unit cell coordinates into
            # element frame
        else:
            self.F = np.zeros(3)
        return self.F.copy()

    # @profile() #.497
    def transform_Unit_Cell_Force_Into_Element_Frame(self, F, q):
        # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
        # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
        # or leaving the element interface as mirror images of each other.
        # F: Force to be rotated out of unit cell frame
        # q: particle's position in the element frame where the force is acting
        FNew = F.copy()  # copy input vector to not modify the original
        return fastElementNUMBAFunctions.transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(FNew, q, self.M_uc, self.ucAng)



    # @profile()
    def transform_Element_Coords_Into_Unit_Cell_Frame(self, q):
        # As particle leaves unit cell, it does not start back over at the beginning, instead is turns around so to speak
        # and goes the other, then turns around again and so on. This is how the symmetry of the unit cell is exploited.
        # q: particle coords in element frame
        # returnUCFirstOrLast: return 'FIRST' or 'LAST' if the coords are in the first or last unit cell. This is typically
        # used for including unit cell fringe fields
        # return cythonFunc.transform_Element_Coords_Into_Unit_Cell_Frame_CYTH(qNew, self.ang, self.ucAng)
        return fastElementNUMBAFunctions.transform_Element_Coords_Into_Unit_Cell_Frame_NUMBA(q,self.ang,self.ucAng)


    def transform_Element_Coords_Into_Unit_Cell_Frame_NUMBA(self,q):
        return fastElementNUMBAFunctions.transform_Element_Coords_Into_Unit_Cell_Frame_NUMBA(q,self.ang,self.ucAng)


class BenderIdealSegmentedWithCap(BenderIdealSegmented):
    def __init__(self, PTL, numMagnets, Lm, Lcap, Bp, rp, rb, yokeWidth, space, rOffsetfact, ap, fillParams=True):
        super().__init__(PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space, rOffsetfact, ap, fillParams=False)
        self.Lcap = Lcap
        self.cap = True
        if fillParams == True:
            self.fill_Params()

    def fill_Params(self):
        super().fill_Params()
        if self.numMagnets is not None:
            if self.ang > 3 * np.pi / 2 or self.ang < np.pi / 2 or self.Lcap > self.rb - self.rp * 2:  # this is done so that finding where the particle is inside the
                # bender is not a huge chore. There is almost no chance it would have this shape anyways. Changing this
                # would affect force, orbit coordinates and isinside at least
                raise Exception('DIMENSIONS OF BENDER ARE OUTSIDE OF ACCEPTABLE BOUNDS')
            self.Lo = self.ro * self.ang + 2 * self.Lcap

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, q):
        # q: element coordinates (x,y,z)
        # returns qo: the coordinates in the orbit frame (s,xo,yo)

        qo = q.copy()
        angle = fast_Arctan2(qo)#np.arctan2(qo[1], qo[0])
        if angle < 0:  # restrict range to between 0 and 2pi
            angle += 2 * np.pi

        if angle < self.ang:  # if particle is in the bending angle section. Could still be outside though
            phi = self.ang - angle  # angle swept out by particle in trajectory. This is zero
            # when the particle first enters
            qox = sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
            qo[0] = self.ro * phi + self.Lcap  # include the distance traveled throught the end cap
            qo[1] = qox
        else:  # if particle is outside of the bending segment angle so it could be in the caps, or elsewhere
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap):  # If inside the cap on
                # the eastward side
                qo[0] = self.Lcap + self.ang * self.ro + (-q[1])
                qo[1] = q[0] - self.ro
            else:
                qTest = q.copy()
                qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
                qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
                # if in the westard side cap
                if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0):
                    qo[0] = self.Lcap - qTest[1]
                    qo[1] = qTest[0] - self.ro
                else:  # if in neither then it must be outside
                    qo[:] = np.nan
        return qo

    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame (x,y,z)
        if self.is_Coord_Inside(q)==False:
            return np.asarray([np.nan,np.nan,np.nan])
        F=np.zeros(3)
        phi = fast_Arctan2(q)  # numba version
        if phi < self.ang:  # if inside the bbending segment
            return super().force(q)
        elif phi > self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap):  # If inside the cap on
                # the eastward side
                F[0] = -self.K * (q[0] - self.rb)
            else:  # if in the westward segment maybe
                qTest = q.copy()
                qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
                qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
                if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (
                        self.Lcap > qTest[1] > 0):  # definitely in the
                    # westard segment
                    forcex = -self.K * (qTest[0] - self.rb)
                    F[0] = self.RIn_Ang[0, 0] * forcex
                    F[1] = -self.RIn_Ang[1, 0] * forcex
                else:
                    F = np.zeros(3)
                    warnings.warn('PARTICLE IS OUTSIDE ELEMENT')
        return F

    # @profile()
    def is_Coord_Inside(self, q):
        # q: particle's position in element frame
        phi = fast_Arctan2(q)  # calling a fast numba version that is global
        if phi < self.ang:  # if particle is inside bending angle region
            if (sqrt(q[0]**2+q[1] ** 2)-self.rb)**2 + q[2] ** 2 < self.ap**2:
                return True
            else:
                return False
        else:  # if outside bender's angle range
            if (q[0]-self.rb)**2+q[2]**2 <= self.ap**2 and (0 >= q[1] >= -self.Lcap):  # If inside the cap on
                # eastward side
                return True
            else:
                qTestx = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
                qTesty = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
                if (qTestx-self.rb)**2+q[2]**2 <= self.ap**2 and (self.Lcap >= qTesty >= 0):  # if on the westwards side
                    return True
                else:  # if not in either cap, then outside the bender
                    return False


class HalbachBenderSimSegmentedWithCap(BenderIdealSegmentedWithCap):
    #this element is a model of a bending magnet constructed of segments. There are three models from which data is
    # extracted required to construct the element. All exported data must be in a grid, though it the spacing along
    #each dimension may be different.
    #1:  A model of the repeating segments of magnets that compose the bulk of the bender. A magnet, centered at the
    #bending radius, sandwiched by other magnets (at the appropriate angle) to generate the symmetry. The central magnet
    #is position with z=0, and field values are extracted from z=0-1e-6 to some value that extends slightly past
    #the tilted edge. See docs/images/HalbachBenderSimSegmentedWithCapImage1.png
    #2: A model of the magnet between the last magnet, and the inner repeating section. This is required becasuse I found
    #that the assumption that I could jump straight from the outwards magnet to the unit cell portion was incorrect,
    #the force was very discontinuous. To model this I the last few segments of a bender, then extrac the field from
    #z=0 up to a little past halfway the second magnet. Make sure to have the x bounds extend a bit to capture
    # #everything. See docs/images/HalbachBenderSimSegmentedWithCapImage2.png
    #3: a model of the input portion of the bender. This portions extends half a magnet length past z=0. Must include
    #enough extra space to account for fringe fields. See docs/images/HalbachBenderSimSegmentedWithCapImage3.png
    def __init__(self, PTL,Lm,rp,numMagnets,rb,extraSpace,rOffsetFact,apFrac):
        # super().__init__(PTL, numMagnets, Lm, Lcap, None, rp, rb, yokeWidth, extraSpace, rOffsetFact, ap,
        #                  fillParams=False)
        super().__init__(None,None,None,None,None,None,None,None,None,None,None,fillParams=False)
        self.sim = True
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
        self.ap = rp*apFrac
        self.K = None #spring constant of field strength to set the offset of the lattice
        self.Fx_Func_Seg = None
        self.Fy_Func_Seg = None
        self.Fz_Func_Seg = None
        self.Force_Func_Seg=None
        self.magnetic_Potential_Func_Seg = None
        self.Fx_Func_Cap = None
        self.Fy_Func_Cap = None
        self.Fz_Func_Cap = None
        self.Force_Func_Cap=None
        self.magnetic_Potential_Func_Cap = None
        self.Fx_Func_Internal_Fringe = None
        self.Fy_Func_Internal_Fringe = None
        self.Fz_Func_Internal_Fringe = None
        self.Force_Func_Internal_Fringe=None
        self.magnetic_Potential_Func_Fringe = None
        self.K_Func=None #function that returns the spring constant as a function of bending radii. This is used in the
        #constraint solver
        if numMagnets is not None:
            self.fill_Params_Pre_Constraint()
            self.fill_Params_Post_Constrained()
        else:
            self.fill_Params_Pre_Constraint()

    def set_BpFact(self,BpFact):
        self.fieldFact=BpFact

    def fill_Params_Pre_Constraint(self):
        def find_K(rb):
            ucAngTemp=np.arctan(self.Lseg/(2*(rb-self.rp-self.yokeWidth))) #value very near final value, good
            #approximation
            lens = _SegmentedBenderHalbachLensFieldGenerator(self.rp, rb, ucAngTemp,self.Lm, numLenses=3)
            xArr = np.linspace(-self.rp/3, self.rp/3) + rb
            coords = np.asarray(np.meshgrid(xArr, 0, 0)).T.reshape(-1, 3)
            FArr = self.PTL.u0*lens.BNorm_Gradient(coords)[:, 0]
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
        self.rOffsetFunc = lambda r:  self.rOffsetFact*(sqrt(
            r ** 2 / 16 + self.PTL.v0Nominal ** 2 / (2 * self.K_Func(r))) - r / 4)  # this accounts for energy loss

    def fill_Params_Post_Constrained(self):
        self.ucAng = np.arctan(self.Lseg / (2 * (self.rb - self.rp - self.yokeWidth)))
        spatialStepSize=1e-3 #target step size for spatial field interpolate

        #fill periodic segment data
        numXY=2*(int(2*self.ap/spatialStepSize)//2)+1 #to ensure it is odd
        numZ=2*(int(np.tan(self.ucAng)*(self.rb+self.rp)/spatialStepSize)//2)+1
        xyArr=np.linspace(-self.ap-1e-6,self.ap+1e-6,num=numXY)
        zArr=np.linspace(-1e-6,np.tan(self.ucAng)*(self.rb+self.rp)+1e-6,num=numZ)
        coords=np.asarray(np.meshgrid(xyArr,xyArr,zArr)).T.reshape(-1,3)
        coords[:,0]+=self.rb #add the bending radius to the coords
        lensSegmentedSymmetry = _SegmentedBenderHalbachLensFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm, numLenses=3)
        #using just 3 magnets makes a big difference
        BNormGradArr,BNormArr=lensSegmentedSymmetry.BNorm_Gradient(coords,returnNorm=True)
        dataSeg=np.column_stack((coords,BNormGradArr,BNormArr))
        self.fill_Force_Func_Seg(dataSeg)
        self.K=self.K_Func(self.rb)


        # #fill first segment magnet that comes before the repeating segments that can be modeled the same
        lensFringe = _SegmentedBenderHalbachLensFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                                                          numLenses=3,inputOnly=True)

        x1=-(self.ap+1.5*(self.rb-self.ap)*(1-np.cos(2*self.ucAng))) #Inwards enough to account for tilted magnet
        x2=self.ap+1e-6
        numX=2*(int((x2-x1)/spatialStepSize)//2)+1
        numY=2*(int(2*self.ap//spatialStepSize)//2)+1
        numZ=2*(int(np.tan(2*self.ucAng)*(self.rb+self.ap)/spatialStepSize)//2)+1
        xArr=np.linspace(x1,x2,num=numX)*1.2+self.rb
        yArr=np.linspace(-self.ap-1e-6,self.ap+1e-6,num=numY)
        zArr=np.linspace(-1e-6,np.tan(2*self.ucAng)*(self.rb+self.ap)+1e-6,num=numZ)

        coords = np.asarray(np.meshgrid(xArr, yArr, zArr)).T.reshape(-1, 3)
        BNormGradArr,BNormArr=lensFringe.BNorm_Gradient(coords,returnNorm=True)
        dataCap=np.column_stack((coords,BNormGradArr,BNormArr))
        self.fill_Force_Func_Internal_Fringe(dataCap)


        #fill the first magnet and its fringe field
        lensFringe = _SegmentedBenderHalbachLensFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                                                          numLenses=3,inputOnly=True)
        numXY=2*(int(2*self.ap/spatialStepSize)//2)+1
        numZ=2*(int(self.Lcap/spatialStepSize)//2)+1
        xyArr=np.linspace(-self.ap-1e-6,self.ap+1e-6,num=numXY)
        zArr=np.linspace(1e-6,-self.Lcap-1e-6,num=numZ)
        coords = np.asarray(np.meshgrid(xyArr,xyArr, zArr)).T.reshape(-1, 3)
        coords[:,0]+=self.rb
        BNormGradArr,BNormArr=lensFringe.BNorm_Gradient(coords,returnNorm=True)
        dataFringe=np.column_stack((coords,BNormGradArr,BNormArr))
        self.fill_Field_Func_Cap(dataFringe)

        self.rOffset = self.rOffsetFunc(self.rb)

        self.ro = self.rb + self.rOffset
        self.ang = 2 * self.numMagnets * self.ucAng
        self.L = self.ang * self.rb
        self.Lo = self.ang * self.ro + 2 * self.Lcap
        self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
        m = np.tan(self.ucAng)
        self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        m = np.tan(self.ang / 2)
        self.M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        self.compile_Fast_Numba_Force_Function()

    def fill_Field_Func_Cap(self,dataCap):
        interpFx, interpFy, interpFz, interpV = self.make_Interp_Functions(dataCap)
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def Force_Func_Cap_Wrap(x,y,z):
            Fx=interpFx(x, -z, y)
            Fy=interpFz(x, -z, y)
            Fz=-interpFy(x, -z, y)
            return Fx,Fy,Fz
        self.Force_Func_Cap=Force_Func_Cap_Wrap
        self.magnetic_Potential_Func_Cap = lambda x, y, z: interpV(x, -z, y)

    def fill_Force_Func_Internal_Fringe(self,dataFringe):
        interpFx, interpFy, interpFz, interpV = self.make_Interp_Functions(dataFringe)
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def force_Internal_Wrap(x,y,z):
            Fx=interpFx(x, -z, y)
            Fy=interpFz(x, -z, y)
            Fz=-interpFy(x, -z, y)
            return Fx,Fy,Fz
        self.Force_Func_Internal_Fringe=force_Internal_Wrap
        self.magnetic_Potential_Func_Fringe = lambda x, y, z: interpV(x, -z, y)

    def fill_Force_Func_Seg(self,dataSeg):
        interpFx, interpFy, interpFz, interpV = self.make_Interp_Functions(dataSeg)
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def force_Seg_Wrap(x,y,z):
            Fx=interpFx(x, -z, y)
            Fy=interpFz(x, -z, y)
            Fz=-interpFy(x, -z, y)
            return Fx,Fy,Fz
        self.Force_Func_Seg=force_Seg_Wrap
        self.magnetic_Potential_Func_Seg = lambda x, y, z: interpV(x, -z, y)


    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame
        F= fastElementNUMBAFunctions.segmented_Bender_Sim_Force_NUMBA(q,self.ang,self.ucAng,self.numMagnets,self.rb,
                self.ap,self.M_ang,self.M_uc,self.RIn_Ang,self.Lcap,self.Force_Func_Seg,self.Force_Func_Internal_Fringe
                                                                          ,self.Force_Func_Cap)
        return self.fieldFact*np.asarray(F)
    def compile_Fast_Numba_Force_Function(self):
        ang=self.ang
        ucAng=self.ucAng
        M_uc=self.M_uc
        numMagnets=self.numMagnets
        rb=self.rb
        ap=self.ap
        M_ang=self.M_ang
        RIn_Ang=self.RIn_Ang
        Lcap=self.Lcap
        Force_Func_Seg=self.Force_Func_Seg
        Force_Func_Internal_Fringe=self.Force_Func_Internal_Fringe
        Force_Func_Cap=self.Force_Func_Cap
        forceNumba = fastElementNUMBAFunctions.segmented_Bender_Sim_Force_NUMBA
        @numba.njit()#numba.float64[:](numba.float64[:]))
        def force_NUMBA_Wrapper(q):
            return forceNumba(q, ang, ucAng, numMagnets, rb, ap, M_ang,M_uc, RIn_Ang, Lcap,Force_Func_Seg,
                                     Force_Func_Internal_Fringe, Force_Func_Cap)
        self.fast_Numba_Force_Function=force_NUMBA_Wrapper
        self.fast_Numba_Force_Function(np.zeros(3))


    def magnetic_Potential(self, q):
        # magnetic potential at point q in element frame
        # q: particle's position in element frame
        phi = fast_Arctan2(q)  # calling a fast numba version that is global
        V0 = 0.0
        if phi < self.ang:  # if particle is inside bending angle region
            revs = int((self.ang - phi) // self.ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position = 'FIRST'
            elif revs == self.numMagnets * 2 - 1 or revs == self.numMagnets * 2 - 2:
                position = 'LAST'
            else:
                position = 'INNER'
            if position == 'INNER':
                quc = self.transform_Element_Coords_Into_Unit_Cell_Frame(q)  # get unit cell coords
                V0 = self.magnetic_Potential_Func_Seg(*quc)
            elif position == 'FIRST' or position == 'LAST':
                V0 = self.magnetic_Potential_First_And_Last(q, position)
            else:
                warnings.warn('PARTICLE IS OUTSIDE LATTICE')
                V0=np.nan
        elif phi > self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap):  # If inside the cap on
                # eastward side
                x, y, z = q
                V0 = self.magnetic_Potential_Func_Cap(x, y, z)
            else:
                qTest = q.copy()
                qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
                qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
                if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (
                        self.Lcap > qTest[1] > 0):  # if on the westwards side
                    x, y, z = qTest
                    y = -y
                    V0 = self.magnetic_Potential_Func_Cap(x, y, z)
                else:  # if not in either cap
                    V0=np.nan
        return V0*self.fieldFact

    def magnetic_Potential_First_And_Last(self, q, position):
        qNew = q.copy()
        if position == 'FIRST':
            qx = qNew[0]
            qy = qNew[1]
            qNew[0] = self.M_ang[0, 0] * qx + self.M_ang[0, 1] * qy
            qNew[1] = self.M_ang[1, 0] * qx + self.M_ang[1, 1] * qy
            V0 = self.magnetic_Potential_Func_Fringe(*qNew)
        elif position == 'LAST':
            V0 = self.magnetic_Potential_Func_Fringe(*qNew)
        else:
            raise Exception('INVALID POSITION SUPPLIED')
        return V0*self.fieldFact

class HalbachLensSim(LensIdeal):
    def __init__(self,PTL, rp,L,apFrac):
        #if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
        #to accomdate the new rp such as force values and positions
        super().__init__(PTL, None, None, rp, None, fillParams=False)
        self.fringeFracOuter=1.5
        self.L=L
        self.Lm = None#L-2*self.fringeFracOuter*rp #hard edge length of magnet
        self.Lo=None
        self.rp=rp
        self.ap=rp*apFrac
        self.fringeFracInnerMin=4.0 #if the total hard edge magnet length is longer than this value * rp, then it can
        #can safely be modeled as a magnet "cap" with a 2D model of the interior
        self.lengthEffective=None #if the magnet is very long, to save simulation
        #time use a smaller length that still captures the physics, and then model the inner portion as 2D
        self.Lcap=None

        self.data2D = None
        self.data3D = None
        self.force_Func_Outer=None #function that returns 3D vector of force values towards the end of the magnet, if
        #the magent is short, then it returns the values for one half, otherwise symmetry is used  to model interior as
        #2D
        self.magnetic_Potential_Func_Fringe = None
        self.force_Func_Inner=None
        self.magnetic_Potential_Func_Inner = None
        self.fieldFact = 1.0 #factor to multiply field values by for tunability
        if self.L is not None:
            self.fill_Params()

    def set_Length(self,L):
        self.L=L
        self.fill_Params()
    def fill_Params(self,externalDataProvided=False):
        spatialStepSize=1e-3 #target step size in space for spatial interpolating grid


        self.Lm=self.L-2*self.fringeFracOuter*self.rp  #hard edge length of magnet
        self.Lo=self.L
        self.lengthEffective=min(self.fringeFracInnerMin*self.rp,
                                 self.Lm)  #if the magnet is very long, to save simulation
        #time use a smaller length that still captures the physics, and then model the inner portion as 2D
        self.Lcap=self.lengthEffective/2+self.fringeFracOuter*self.rp

        magnetWidth=self.rp*np.tan(2*np.pi/24)*2
        lens=_HalbachLensFieldGenerator(1,magnetWidth,self.rp,length=self.lengthEffective)


        numXY=2*(int(2*self.ap/spatialStepSize)//2)+1 #to ensure it is odd
        xyArr=np.linspace(-self.ap-1e-6,self.ap+1e-6,num=numXY) #add a little extra so the interp works correctly
        if self.lengthEffective<self.Lm: #if total magnet length is large enough to ignore fringe fields for interior
            # portion inside then use a 2D plane to represent the inner portion to save resources
            planeCoords=np.asarray(np.meshgrid(xyArr,xyArr,0)).T.reshape(-1,3)
            BNormGrad,BNorm=lens.BNorm_Gradient(planeCoords,returnNorm=True)
            self.data2D=np.column_stack((planeCoords[:,:2],BNormGrad[:,:2],BNorm)) #2D is formated as
            # [[x,y,z,B0Gx,B0Gy,B0],..]
            self.fill_Field_Func_2D()
        else:
            #still need to make the force function
            @numba.njit()
            def force_Func_Inner(x, y, z):
                return 0.0,0.0,0.0
            self.force_Func_Inner = force_Func_Inner
            self.magnetic_Potential_Func_Inner = lambda x, y, z: 0.0


        zMin=0
        zMax=self.Lcap
        numZ=2*(int(2*(zMax-zMin)/spatialStepSize)//2)+1 #to ensure it is odd
        zArr=np.linspace(zMin-1e-6,zMax+1e-6,num=numZ) #add a little extra so interp works as expected

        volumeCoords=np.asarray(np.meshgrid(xyArr,xyArr,zArr)).T.reshape(-1,3) #note that these coordinates can have
        #the wrong value for z if the magnet length is longer than the fringe field effects. This is intentional and
        #input coordinates will be shifted in a wrapper function
        BNormGrad = lens.BNorm_Gradient(volumeCoords)
        BNorm = lens.BNorm(volumeCoords)
        self.data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        self.fill_Field_Func_Cap()
        self.compile_Fast_Numba_Force_Function()


    def fill_Field_Func_Cap(self):
        interpFx, interpFy, interpFz, interpV = self.make_Interp_Functions(self.data3D)
        # wrap the function in a more convenietly accesed function
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def force_Func_Outer(x,y,z):
            Fx=interpFz(-z, y,x)
            Fy=interpFy(-z, y,x)
            Fz=-interpFx(-z, y,x)
            return Fx,Fy,Fz
        self.force_Func_Outer=force_Func_Outer
        self.magnetic_Potential_Func_Fringe = lambda x, y, z: interpV(-z, y, x)

    def fill_Field_Func_2D(self):
        #Data is provided for lens that points in the positive z, so the force functions need to be rotated
        xArr = np.unique(self.data2D[:, 0])
        yArr = np.unique(self.data2D[:, 1])
        numx = xArr.shape[0]
        numy = yArr.shape[0]
        BGradxMatrix = np.empty((numx, numy))
        BGradyMatrix = np.empty((numx, numy))
        B0Matrix = np.zeros((numx, numy))
        xIndices = np.argwhere(self.data2D[:, 0][:, None] == xArr)[:, 1]
        yIndices = np.argwhere(self.data2D[:, 1][:, None] == yArr)[:, 1]

        BGradxMatrix[xIndices, yIndices] = self.data2D[:, 2]
        BGradyMatrix[xIndices, yIndices] = self.data2D[:, 3]
        B0Matrix[xIndices, yIndices] = self.data2D[:, 4]

        interpFx=generate_2DInterp_Function_NUMBA(-self.PTL.u0*BGradxMatrix,xArr,yArr)
        interpFy=generate_2DInterp_Function_NUMBA(-self.PTL.u0*BGradyMatrix,xArr,yArr)
        interpV=generate_2DInterp_Function_NUMBA(self.PTL.u0*B0Matrix,xArr,yArr)
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def force_Func_Inner(x,y,z):
            Fx=0.0
            Fy=interpFy(-z, y) #model is rotated in particle tracing frame
            Fz=-interpFx(-z, y)
            return Fx,Fy,Fz
        self.force_Func_Inner=force_Func_Inner
        self.magnetic_Potential_Func_Inner = lambda x, y, z: interpV(-z, y)#[0][0]
    def set_BpFact(self,BpFact):
        self.fieldFact=BpFact
    def magnetic_Potential(self, q):
        x,y,z=q
        if q[0] <= self.Lcap:
            x = self.Lcap - x
            V0 = self.magnetic_Potential_Func_Fringe(x, y, z)
        elif self.Lcap < q[0] <= self.L - self.Lcap:
            V0 = self.magnetic_Potential_Func_Inner(x,y,z)
        elif q[0] <= self.L: #this one is tricky with the scaling
            x=self.Lcap-(self.L-x)
            V0 = self.magnetic_Potential_Func_Fringe(x, y, z)
        else:
            V0=0
        return V0*self.fieldFact



    def force(self, q):
        F= fastElementNUMBAFunctions.lens_Halbach_Force_NUMBA(q,self.Lcap,self.L,self.ap,self.force_Func_Inner
                                                                  ,self.force_Func_Outer)
        return self.fieldFact*np.asarray(F)
    def compile_Fast_Numba_Force_Function(self):
        forceNumba = fastElementNUMBAFunctions.lens_Halbach_Force_NUMBA
        Lcap=self.Lcap
        L=self.L
        ap=self.ap
        force_Func_Inner=self.force_Func_Inner
        force_Func_Outer=self.force_Func_Outer
        @numba.njit()
        def force_NUMBA_Wrapper(q):
            return forceNumba(q,Lcap,L,ap,force_Func_Inner,force_Func_Outer)
        self.fast_Numba_Force_Function=force_NUMBA_Wrapper
        self.fast_Numba_Force_Function(np.zeros(3)) #force compile by passing a dummy argument

class BumpsLensSimWithCaps(HalbachLensSim):
    def __init__(self, PTL, file2D, file3D,fringeFrac, L,rp, ap,sigma):
        super().__init__(PTL, file2D, file3D,fringeFrac, L,rp, ap)
        self.sigma=sigma #the amount of vertical shift for bumping the beam over