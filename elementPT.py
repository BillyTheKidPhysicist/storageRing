import numpy as np
import pyximport; pyximport.install(language_level=3,setup_args={'include_dirs':np.get_include()})
import cythonFunc
import numpy.linalg as npl
import warnings
import scipy.ndimage as spimg
from interp3d import interp_3d
import scipy.interpolate as spi
import pandas as pd
import time
import numpy.linalg as npl
import sys
import matplotlib.pyplot as plt
import numba
#from profilehooks import profile




#Notes:
#--begining and ending of elements refer to the geometric sense in the lattice. For example, the beginning of the lens
#is the part encountered going clockwise, and the end is the part exited going clockwise for a particle being traced
#--Simulated fields come from COMSOL, and the data must be exported exactly correclty or else the reshaping will not work
#and it can be a huge pain in the but. I could have simply exported the data and used unstructured interpolation, but that
# is very very slow to inialize and query. So data must be in grid format. I also do not use the scipy linearNDInterpolater
#because it is not fast enough, and instead use a function I found in a stack exchange question. Someone made a github
#repository for it. This method gives the same results (except at the edges, where the logic fails, but scipy
#seems to still give reasonable ansers) and take about 5us per evaluatoin instead of 200us.
#--There are at least 3 frames of reference. The first is the lab frame, the second is the element frame, and the third
# which is only found in segmented bender, is the unit cell frame.

@numba.njit(numba.float64(numba.float64[:]))
def fast_Arctan2(q):
    phi=np.arctan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi




class Element:
    def __init__(self,PTL):
        self.theta=None #angle that describes an element's rotation in the xy plane.
        #SEE EACH ELEMENT FOR MORE DETAILS
        #-Straight elements like lenses and drifts: theta=0 is the element's input at the origin and the output pointing
        #east. for theta=90 the output is pointing up.
        #-Bending elements without caps: at theta=0 the outlet is at (bending radius,0) pointing south with the input
        # at some angle counterclockwise. a 180 degree bender would have the inlet at (-bending radius,0) pointing south.
        # force is a continuous function of r and theta, ie a revolved cross section of a hexapole
        #-Bending  elements with caps: same as without caps, but keep in mind that the cap on the output would be BELOW
        #y=0
        #combiner: theta=0 has the outlet at the origin and pointing to the west, with the inlet some distance to the right
        #and pointing in the NE direction
        self.PTL=PTL #particle tracer lattice object. Used for various constants
        self.nb=None #normal vector to beginning (clockwise sense) of element.
        self.ne=None #normal vector to end (clockwise sense) of element
        self.r0=None #coordinates of center of bender, minus any caps
        self.ROut=None #2d matrix to rotate a vector out of the element's reference frame
        self.RIn = None #2d matrix to rotate a vector into the element's reference frame
        self.r1=None #3D coordinates of beginning (clockwise sense) of element
        self.r2=None #3D coordinates of ending (clockwise sense) of element
        self.SO = None #the shapely object for the element. These are used for plotting, and for finding if the coordinates
        #are inside an element that can't be found with simple geometry
        self.ang=0 #bending angle of the element. 0 for lenses and drifts
        self.Lm=None #hard edge length of magnet along line through the bore
        self.L=None #length of magnet along line through the bore
        self.K=None #'spring constant' of element. For some this comes from comsol fields.
        self.Lo=None #length of orbit for particle. For lenses and drifts this is the same as the length. This is a nominal
        #value because for segmented benders the path length is not simple to compute
        self.index=None #elements position in lattice
        self.cap=False #wether there is a cap or not present on the element. Cap simulates fringe fields
        self.comsolExtraSpace=.1e-3 #extra space in comsol files to exported grids. this can be used to find dimensions
        self.ap=None #apeture of element. most elements apetures are assumed to be square
        self.apz=None #apeture in the z direction. all but the combiner is symmetric, so there apz is the same as ap
        self.apL=None #apeture on the 'left' as seen going clockwise. This is for the combiner
        self.apR=None #apeture on the'right'.
        self.type=None #gemetric tupe of magnet, STRAIGHT,BEND or COMBINER. This is used to generalize how the geometry
        #constructed in particleTracerLattice
        self.sim=None #wether the field values are from simulations
        self.F=np.zeros(3) #object to hold the force to prevent constantly making new force vectors
    def magnetic_Potential(self,q):
        # Function that returns the magnetic potential in the element's frame
        return 0.0
    def transform_Lab_Coords_Into_Orbit_Frame(self, q, cumulativeLength):
        #Change the lab coordinates into the particle's orbit frame.
        q = self.transform_Lab_Coords_Into_Element_Frame(q) #first change lab coords into element frame
        qo = self.transform_Element_Coords_Into_Orbit_Frame(q) #then element frame into orbit frame
        qo[0] = qo[0] +cumulativeLength #orbit frame is in the element's frame, so the preceding length needs to be
        #accounted for
        return qo
    def transform_Lab_Coords_Into_Element_Frame(self,q):
        #this is overwridden by all other elements
        pass
    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        #for straight elements (lens and drift), element and orbit frame are identical
        #q: 3D coordinates in element frame
        return q.copy()

    def transform_Lab_Frame_Vector_Into_Element_Frame(self,vec):
        #vec: 3D vector in lab frame to rotate into element frame
        vecNew=vec.copy() #copying prevents modifying the original value
        vecx=vecNew[0];vecy=vecNew[1]
        vecNew[0] = vecx * self.RIn[0, 0] + vecy * self.RIn[0, 1]
        vecNew[1] = vecx * self.RIn[1, 0] + vecy * self.RIn[1, 1]
        return vecNew

    # @staticmethod
    # @numba.njit(numba.float64[:](numba.float64[:] ,numba.float64[: ,:]))
    # def transform_Lab_Frame_Vector_Into_Element_Frame_NUMBA(vecNew ,RIn):
    #    vec x =vecNew[0]
    #    vec y =vecNew[1]
    #    vecNew[0] = vecx * RIn[0, 0] + vecy * RIn[0, 1]
    #    vecNew[1] = vecx * RIn[1, 0] + vecy * RIn[1, 1]
    #    return vecNew
    def transform_Element_Frame_Vector_To_Lab_Frame(self, vec):
        # rotate vector out of element frame into lab frame
        #vec: vector in
        vecNew=vec.copy()#copy input vector to not modify the original
        vecx = vecNew[0];vecy = vecNew[1]
        vecNew[0] = vecx * self.ROut[0, 0] + vecy * self.ROut[0, 1]
        vecNew[1] = vecx * self.ROut[1, 0] + vecy * self.ROut[1, 1]
        return vecNew
    def set_Length(self,L):
        #this is used typically for setting the length after satisfying constraints
        self.L=L
        self.Lo=L

    def is_Coord_Inside(self, q):
        return None
    def make_Interp_Functions(self,data):
        #This method takes an array data with the shape (n,6) where n is the number of points in space. Each row
        #must have the format [x,y,z,gradxB,gradyB,gradzB,B] where B is the magnetic field norm at x,y,z and grad is the
        #partial derivative. The data must be from a 3D grid of points with no missing points or any other funny business
        #and the order of points doesn't matter
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
        interpFx = interp_3d.Interp3D(-self.PTL.u0 * BGradxMatrix, xArr, yArr, zArr)
        interpFy = interp_3d.Interp3D(-self.PTL.u0 * BGradyMatrix, xArr, yArr, zArr)
        interpFz = interp_3d.Interp3D(-self.PTL.u0 * BGradzMatrix, xArr, yArr, zArr)
        interpV=interp_3d.Interp3D(self.PTL.u0 * B0Matrix, xArr, yArr, zArr)
        return interpFx,interpFy,interpFz,interpV
class LensIdeal(Element):
    #ideal model of lens with hard edge. Force inside is calculated from field at pole face and bore radius as
    #F=2*ub*r/rp**2 where rp is bore radius, and ub is bore magneton.
    def __init__(self,PTL,L,Bp,rp,ap,fillParams=True):
        #fillParams is used to avoid filling the parameters in inherited classes
        super().__init__(PTL)
        self.Bp = Bp #field strength at pole face
        self.rp = rp #bore radius
        self.L = L #lenght of magnet
        self.ap = ap #size of apeture radially
        self.type='STRAIGHT' #The element's geometry
        if fillParams==True:
            self.fill_Params()
    def fill_Params(self):
        self.K = (2 * self.Bp * self.PTL.u0 / self.rp ** 2) #'spring' constant
        if self.L is not None:
            self.Lo=self.L
    def magnetic_Potential(self,q):
        #potential energy at provided coordinates
        #q coords in element frame
        r=np.sqrt(q[1]**2++q[2]**2)
        if r<self.ap:
            return self.Bp*self.PTL.u0*r**2/self.rp**2
        else:
            return 0.0
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew = q.copy()  # CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        qNew[0] = qNew[0] - self.r1[0]
        qNew[1] = qNew[1] - self.r1[1]
        qNew=self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def force(self,q):
        # note: for the perfect lens, in it's frame, there is never force in the x direction. Force in x is always zero
        self.F[1] = -self.K * q[1]
        self.F[2] = -self.K * q[2]
        return self.F.copy()
    def set_Length(self,L):
        # this is used typically for setting the length after satisfying constraints
        self.L=L
        self.Lo = self.L
    def is_Coord_Inside(self, q):
        # check with fast geometric arguments if the particle is inside the element. This won't necesarily work for all
        # elements. If True is retured, the particle is inside. If False is returned, it is defintely outside. If none is
        # returned, it is unknown
        # q: coordinates to test
        if not -self.ap<q[1]<self.ap or not -self.ap<q[2]<self.ap:
            return False
        elif q[0] < 0 or q[0] > self.L:
            return False
        else:
            return True

class Drift(LensIdeal):
    def __init__(self,PTL,L,ap):
        super().__init__(PTL,L,0,np.inf,ap)
    def force(self,q):
        return np.zeros(3)

class BenderIdeal(Element):
    def __init__(self,PTL,ang,Bp,rp,rb,ap,fillParams=True):
        #this is the base model for the bending elements, of which there are several kinds. To maintain generality, there
        #are numerous methods and variables that are not necesary here, but are used in other child classes. For example,
        #this element has no caps, but a cap length of zero is still used.
        super().__init__(PTL)
        self.sim=False
        self.ang=ang #total bending angle of bender
        self.Bp = Bp #field strength at pole
        self.rp = rp #bore radius of magnet
        self.ap = ap #radial apeture size
        self.rb=rb #bending radius of magnet. This is tricky because this is the bending radius down the center, but the
        #actual trajectory of the particles is offset a little out from this
        self.type='BEND'
        self.rOffset=None #the offset to the bending radius because of centrifugal force. There are two versions, one that
        #accounts for reduced speed from energy conservation for actual fields, and one that doesn't
        self.ro=None #bending radius of orbit, ie rb + rOffset.
        self.rOffsetFunc=None #a function that returns the value of the offset of the trajectory for a given bending radius
        self.segmented=False #wether the element is made up of discrete segments, or is continuous
        self.capped=False #wether the element has 'caps' on the inlet and outlet. This is used to model fringe fields
        self.Lcap=0 #length of caps

        if fillParams==True:
            self.fill_Params()
    def fill_Params(self):
        self.K = (2 * self.Bp * self.PTL.u0 / self.rp ** 2) #'spring' constant
        self.rOffsetFunc=lambda rb: np.sqrt(rb ** 2 / 4 + self.PTL.v0Nominal ** 2 / self.K) -rb / 2
        self.rOffset = self.rOffsetFunc(self.rb)
        self.ro=self.rb+self.rOffset
        if self.ang is not None: #calculation is being delayed until constraints are solved
            self.L  = self.rb * self.ang
            self.Lo = self.ro * self.ang

    def magnetic_Potential(self,q):
        #potential energy at provided coordinates
        #q coords in element frame
        r=np.sqrt(q[0]**2++q[1]**2)
        if self.rb+self.rp>r>self.rb-self.rp and np.abs(q[2])<self.ap:
            rNew=np.sqrt((r-self.rb)**2+q[2]**2)
            return self.Bp*self.PTL.u0*rNew**2/self.rp**2
        else:
            return 0.0

    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew=q-self.r0
        qNew=self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew

    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        #q: element coords
        #returns a 3d vector in the orbit frame. First component is distance along trajectory, second is radial displacemnt
        #from the nominal orbit computed with centrifugal force, and the third is the z axis displacemnt.
        qo = q.copy()
        phi = self.ang - np.arctan2(q[1], q[0])  # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = np.sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
        qo[0] = qos
        qo[1] = qox
        return qo
    def force(self, q):
        # force at point q in element frame
        #q: particle's position in element frame
        
        phi = fast_Arctan2(q)
        if phi<self.ang:
            r = np.sqrt(q[0] ** 2 + q[1] ** 2)  # radius in x y frame
            F0 = -self.K * (r - self.rb)  # force in x y plane
            self.F[0] = np.cos(phi) * F0
            self.F[1] = np.sin(phi) * F0
            self.F[2] = -self.K * q[2]
        else:
            warnings.warn("PARTICLE IS OUTSIDE ELEMENT")
            self.F=np.zeros(3)
        return self.F.copy()
    def is_Coord_Inside(self,q):
        if np.abs(q[2]) > self.ap:  # if clipping in z direction
            return False
        phi = np.arctan2(q[1], q[0])
        if (phi > self.ang and phi < 2 * np.pi) or phi < 0:  # if outside bender's angle range
            return False
        r = np.sqrt(q[0] ** 2 + q[1] ** 2)
        if r < self.rb - self.ap or r > self.rb + self.ap:
            return False
        return True

class CombinerIdeal(Element):
    # combiner: This is is the element that bends the two beams together. The logic is a bit tricky. It's geometry is
    # modeled as a straight section, a simple square, with a segment coming of at the particle in put at an angle. The
    # angle is decided by tracing particles through the combiner and finding the bending angle.
    def __init__(self,PTL,Lm,c1,c2,ap,sizeScale,fillsParams=True):
        super().__init__(PTL)
        self.sim=False
        self.sizeScale=sizeScale #the fraction that the combiner is scaled up or down to. A combiner twice the size would
        #use sizeScale=2.0
        self.ap = ap*self.sizeScale
        self.apR=self.ap*self.sizeScale
        self.apL=self.ap*self.sizeScale
        self.Lm=Lm*self.sizeScale
        self.La=None #length of segment between inlet and straight section inside the combiner. This length goes from
        #the center of the inlet to the center of the kink
        self.Lb=None #length of straight section after the kink after the inlet
        self.c1=c1/self.sizeScale
        self.c2=c2/self.sizeScale
        self.space=0 #space at the end of the combiner to account for fringe fields

        self.type='COMBINER'
        self.inputOffset=None #offset along y axis of incoming circulating atoms. a particle entering at this offset in
        #the y, with angle self.ang, will exit at x,y=0,0
        self.inputOffsetLoad=None #same as inputOffset, but for high field seekers being loaded into the ring
        self.angLoad=None #bending angle of combiner for particles being loaded into the ring
        self.LoLoad=None #trajectory length for particles being loaded into the ring in the combiner
        if fillsParams==True:
            self.fill_Params()
    def fill_Params(self):
        self.Lb=self.Lm #length of segment after kink after the inlet

        inputAngle, inputOffset,qTracedArr = self.compute_Input_Angle_And_Offset()
        self.Lo = np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.ang = inputAngle
        self.inputOffset = inputOffset

        inputAngleLoad, inputOffsetLoad, qTracedArr = self.compute_Input_Angle_And_Offset(lowField=False)
        self.LoLoad = np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.angLoad = inputAngleLoad
        self.inputOffsetLoad = inputOffsetLoad

        self.apz = self.ap / 2
        self.La = self.ap * np.sin(self.ang)
        self.L = self.La * np.cos(self.ang) + self.Lb #TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING

    def compute_Input_Angle_And_Offset(self,h=1e-6,lowField=True):
        #TODO: CAN i GET RID OF THIS LIMIT STUFF CLEANLY?

        # this computes the output angle and offset for a combiner magnet.
        # NOTE: for the ideal combiner this gives slightly inaccurate results because of lack of conservation of energy!
        #limit: how far to carry the calculation for along the x axis. For the hard edge magnet it's just the hard edge
        #length, but for the simulated magnets, it's that plus twice the length at the ends.
        #h: timestep
        #lowField: wether to model low or high field seekers
        # todo: make proper edge handling
        q = np.asarray([0.0, 0.0, 1e-3])
        p = np.asarray([self.PTL.v0Nominal, 0.0, 0.0])
        tempList=[] #Array that holds particle coordinates traced through combiner. This is used to find lenght
        # #of orbit.
        xList=[]
        yList=[]
        test=[]
        if lowField==True:
            force=self.force
        else:
            force = lambda x: -self.force(x)
        limit=self.Lm+2*self.space
        while True:
            F = force(q)
            a = F
            q_n = q + p * h + .5 * a * h ** 2
            F_n = force(q_n)
            a_n = F_n  # accselferation new or accselferation sub n+1
            p_n = p + .5 * (a + a_n) * h
            if q_n[0] > limit:  # if overshot, go back and walk up to the edge assuming no force
                dr = limit - q[0]
                dt = dr / p[0]
                q = q + p * dt
                tempList.append(q)
                break
            #test.append(self.magnetic_Potential(q))
            #xList.append(q[0])
            #yList.append(q[1])
            q = q_n
            p = p_n
            tempList.append(q)
        #plt.plot(xList,test)
        #plt.grid()
        #plt.show()
        outputAngle = np.arctan2(p[1], p[0])
        outputOffset = q[1]
        return outputAngle, outputOffset,np.asarray(tempList)
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew=q.copy()
        qNew = qNew - self.r2
        qNew=self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        # NOTE: THIS NOT GOING TO BE CORRECT IN GENERALY BECAUSE THE TRAJECTORY IS NOT SMOOTH AND I HAVE NOT WORKED IT OUT
        #YET
        qo=q.copy()
        qo[0] = self.Lo - qo[0]
        qo[1] = 0#qo[1]
        return qo
    def force(self, q):
        # force at point q in element frame
        #q: particle's position in element frame
        F = np.zeros(3)  # force vector starts out as zero
        if q[0] < self.Lb:
            B0 = np.sqrt((self.c2 * q[2]) ** 2 + (self.c1 + self.c2 * q[1]) ** 2)
            F[1] = self.PTL.u0 * self.c2 * (self.c1 + self.c2 * q[1]) / B0
            F[2] = self.PTL.u0 * self.c2 ** 2 * q[2] / B0
        return F

    def is_Coord_Inside(self,q):
        #q: coordinate to test in element's frame
        if not -self.apz<q[2]<self.apz: #if outside the z apeture
            return False
        elif 0<=q[0]<=self.Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner
            if -self.apL<q[1]<self.apR: #if inside the y (width) apeture
                return True
        elif q[0]<0:
            return False
        else:  # particle is in the bent section leading into combiner
            m=np.tan(self.ang)
            Y1=m*q[0]+(self.apR-m*self.Lb) #upper limit
            Y2 =(-1/m)*q[0]+self.La*np.sin(self.ang)+(self.Lb+self.La*np.cos(self.ang))/m
            Y3=m * q[0] + (-self.apL - m * self.Lb)
            if q[1]<Y1 and q[1]<Y2 and q[1]>Y3:
                return True
            else:
                return False

class CombinerSim(CombinerIdeal):
    def __init__(self,PTL,combinerFile,sizeScale=1.0):
        #PTL: particle tracing lattice object
        #combinerFile: File with data with dimensions (n,6) where n is the number of points and each row is
        # (x,y,z,gradxB,gradyB,gradzB,B). Data must have come from a grid. Data must only be from the upper quarter
        # quadrant, ie the portion with z>0 and x< length/2
        #sizescale: factor to scale up or down all dimensions. This modifies the field strength accordingly, ie
        #doubling dimensions halves the gradient
        Lm = .187
        apL=.015
        apR=.025
        fringeSpace=5*1.1e-2
        apz=6e-3
        super().__init__(PTL,Lm,np.nan,np.nan,np.nan,sizeScale,fillsParams=False) #TODO: replace all the Nones with np.nan
        self.sim=True
        self.space = fringeSpace*self.sizeScale  # extra space past the hard edge on either end to account for fringe fields
        self.apL=apL*self.sizeScale
        self.apR=apR*self.sizeScale
        self.apz=apz*self.sizeScale


        self.data=None
        self.combinerFile=combinerFile
        self.FxFunc=None
        self.FyFunc=None
        self.FzFunc=None
        self.magnetic_Potential_Func=None
        self.fill_Params()
    def fill_Params(self):
        self.data = np.asarray(pd.read_csv(self.combinerFile, delim_whitespace=True, header=None))
        #use the new size scaling to adjust the provided data
        self.data[:,:3]=self.data[:,:3]*self.sizeScale #scale the dimensions
        self.data[:,3:6]=self.data[:,3:6]/self.sizeScale #scale the field gradient
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input
        funcGradx,funcGrady,funcGradz,funcV=self.make_Interp_Functions(self.data)

        #self.magnetic_Potential=lambda x, y, z: funcGradx((x, y, z))
        self.FxFunc = lambda x, y, z: funcGradx((x, y, z))
        self.FyFunc = lambda x, y, z: funcGrady((x, y, z))
        self.FzFunc = lambda x, y, z: funcGradz((x, y, z))
        self.magnetic_Potential_Func = lambda x, y, z: funcV((x, y, z))


        #TODO: I'M PRETTY SURE i CAN CONDENSE THIS WITH THE COMBINER IDEAL
        inputAngle, inputOffset, qTracedArr = self.compute_Input_Angle_And_Offset() #0.07891892567413786

        #to find the length
        self.Lo = self.compute_Trajectory_Length(qTracedArr)#np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.L = self.Lo #TODO: WHAT IS THIS DOING?? is it used anywhere
        self.ang = inputAngle
        self.inputOffset=inputOffset-np.tan(inputAngle) * self.space  # the input offset is measured at the end of the hard
        # edge

        inputAngleLoad, inputOffsetLoad, qTracedArrLoad = self.compute_Input_Angle_And_Offset(lowField=False)
        self.LoLoad = self.compute_Trajectory_Length(qTracedArrLoad)
        self.angLoad = inputAngleLoad
        self.inputOffsetLoad = inputOffsetLoad

        # the inlet length needs to be long enough to extend past the fringe fields
        # TODO: MAKE EXACT, now it overshoots I think.
        self.La = self.space + np.tan(self.ang) * self.apR
        #self.Lo = self.La + self.Lb
        self.data=None #to save memory and pickling time

    def compute_Trajectory_Length(self, qTracedArr):
        #TODO: CHANGE THAT X DOESN'T START AT ZERO
        #to find the trajectory length model the trajectory as a bunch of little deltas for each step and add up their
        #length
        x = qTracedArr[:, 0]
        y = qTracedArr[:, 1]
        xDelta=np.append(x[0], x[1:] - x[:-1]) #have to add the first value to the length of difference because it starts
        #at zero
        yDelta = np.append(y[0], y[1:] - y[:-1])
        dLArr=np.sqrt(xDelta**2+yDelta**2)
        Lo=np.sum(dLArr)
        return Lo
    def force(self,q):
        #this function uses the symmetry of the combiner to extract the force everywhere.
        qNew=q.copy()
        xFact=1 #value to modify the force based on symmetry
        zFact=1
        if 0<=qNew[0]<=(self.Lm/2+self.space): #if the particle is in the first half of the magnet
            if qNew[2]<0: #if particle is in the lower plane
                qNew[2] = -qNew[2] #flip position to upper plane
                zFact=-1 #z force is opposite in lower half
        elif (self.Lm/2+self.space)<qNew[0]: #if the particle is in the last half of the magnet
            qNew[0]=(self.Lm/2+self.space)-(qNew[0]-(self.Lm/2+self.space)) #use the reflection of the particle
            xFact=-1 #x force is opposite in back plane
            if qNew[2]<0:  # if in the lower plane, need to use symmetry
                qNew[2] = -qNew[2]
                zFact=-1#z force is opposite in lower half
        self.F[0] = xFact*self.FxFunc(*qNew)
        self.F[1] = self.FyFunc(*qNew)
        self.F[2] = zFact*self.FzFunc(*qNew)
        return self.F
    def magnetic_Potential(self,q):
        #this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
        qNew=q.copy()
        if 0<=qNew[0]<=(self.Lm/2+self.space): #if the particle is in the first half of the magnet
            if qNew[2]<0: #if particle is in the lower plane
                qNew[2] = -qNew[2] #flip position to upper plane
        if (self.Lm/2+self.space)<qNew[0]: #if the particle is in the last half of the magnet
            qNew[0]=(self.Lm/2+self.space)-(qNew[0]-(self.Lm/2+self.space)) #use the reflection of the particle
            if qNew[2]<0:  # if in the lower plane, need to use symmetry
                qNew[2] = -qNew[2]
        return self.magnetic_Potential_Func(*qNew)

class BenderIdealSegmented(BenderIdeal):
    #-very similiar to ideal bender, but force is not a continuous
    #function of theta and r. It is instead a series of discrete magnets which are represented as a unit cell. A
    #full magnet would be modeled as two unit cells, instead of a single unit cell, to exploit symmetry and thus
    #save memory. Half the time the symetry is exploited by using a simple rotation, the other half of the time the
    #symmetry requires a reflection, then rotation.
    def __init__(self, PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space, rOffsetFact,ap,fillParams=True):
        super().__init__(PTL,None,Bp,rp,rb,ap,fillParams=False)
        self.numMagnets = numMagnets
        self.Lm = Lm
        self.space = space
        self.yokeWidth = yokeWidth
        self.ucAng = None
        self.segmented = True
        self.cap = False
        self.ap = ap
        self.rOffsetFact=rOffsetFact
        self.RIn_Ang = None
        self.Lseg=None
        self.M_uc=None #matrix for reflection used in exploting segmented symmetry. This is 'inside' a single magnet element
        self.M_ang=None #matrix for reflection used in exploting segmented symmetry. This is reflecting out from the unit cell
        if fillParams==True:
            self.fill_Params()

    def fill_Params(self):
        super().fill_Params()
        self.rOffsetFunc=lambda rb:self.rOffsetFact* np.sqrt(rb ** 2 / 4 + self.PTL.v0Nominal ** 2 / self.K) -rb / 2
        self.rOffset = self.rOffsetFunc(self.rb)
        self.Lseg=self.Lm+2*self.space
        if self.numMagnets is not None:
            self.ucAng = np.arctan((self.Lm / 2 + self.space) / (self.rb - self.yokeWidth - self.rp))
            self.ang = 2 * self.ucAng * self.numMagnets
            self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
            self.Lo = self.ro * self.ang
            m = np.tan(self.ucAng)
            self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew = q - self.r0
        qx=qNew[0]
        qy=qNew[1]
        qNew[0] = qx * self.RIn[0, 0] + qy * self.RIn[0, 1]
        qNew[1] = qx * self.RIn[1, 0] + qy * self.RIn[1, 1]
        return qNew

    def transform_Lab_Frame_Vector_Into_Element_Frame(self,vec):
        #vec: 3D vector in lab frame to rotate into element frame
        vecx=vec[0];vecy=vec[1]
        vec[0] = vecx * self.RIn[0, 0] + vecy * self.RIn[0, 1]
        vec[1] = vecx * self.RIn[1, 0] + vecy * self.RIn[1, 1]
        return vec


    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        qo = q.copy()
        phi = self.ang - np.arctan2(q[1], q[0])  # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = np.sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
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
            self.F = self.transform_Unit_Cell_Force_Into_Element_Frame(self.F, q)  # transform unit cell coordinates into
            # element frame
        else:
            self.F=np.zeros(3)
        return self.F.copy()
    #@profile() #.497
    def transform_Unit_Cell_Force_Into_Element_Frame(self, F, q):
        # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
        # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
        # or leaving the element interface as mirror images of each other.
        # F: Force to be rotated out of unit cell frame
        # q: particle's position in the element frame where the force is acting
        FNew = F.copy()  # copy input vector to not modify the original
        return cythonFunc.transform_Unit_Cell_Force_Into_Element_Frame(FNew,q,self.M_uc,self.ucAng)

    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:,:],numba.float64))
    def transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(FNew, q,M_uc,ucAng):
        # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
        # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
        # or leaving the element interface as mirror images of each other.
        # FNew: Force to be rotated out of unit cell frame
        # q: particle's position in the element frame where the force is acting

        phi = np.arctan2(q[1], q[0])  # the anglular displacement from output of bender to the particle. I use
        # output instead of input because the unit cell is conceptually located at the output so it's easier to visualize
        cellNum = int(phi // ucAng) + 1  # cell number that particle is in, starts at one
        if cellNum % 2 == 1:  # if odd number cell. Then the unit cell only needs to be rotated into that position
            rotAngle = 2 * (cellNum // 2) * ucAng
        else:  # otherwise it needs to be reflected. This is the algorithm for reflections
            Fx = FNew[0]
            Fy = FNew[1]
            FNew[0] = M_uc[0, 0] * Fx + M_uc[0, 1] * Fy
            FNew[1] = M_uc[1, 0] * Fx + M_uc[1, 1] * Fy
            rotAngle = 2 * ((cellNum - 1) // 2) * ucAng
        Fx = FNew[0]
        Fy = FNew[1]
        FNew[0] = Fx * np.cos(rotAngle) - Fy * np.sin(rotAngle)
        FNew[1] = Fx * np.sin(rotAngle) + Fy * np.cos(rotAngle)
        return FNew
    #@profile()
    def transform_Element_Coords_Into_Unit_Cell_Frame(self,q):
        #As particle leaves unit cell, it does not start back over at the beginning, instead is turns around so to speak
        #and goes the other, then turns around again and so on. This is how the symmetry of the unit cell is exploited.
        #q: particle coords in element frame
        # returnUCFirstOrLast: return 'FIRST' or 'LAST' if the coords are in the first or last unit cell. This is typically
        # used for including unit cell fringe fields
        qNew=q.copy()
        return cythonFunc.transform_Element_Coords_Into_Unit_Cell_Frame_CYTHON(qNew,self.ang,self.ucAng)
        #return self.transform_Element_Coords_Into_Unit_Cell_Frame_NUMBA(qNew,self.ang,self.ucAng)


    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:],numba.float64,numba.float64))
    def transform_Element_Coords_Into_Unit_Cell_Frame_NUMBA(qNew,ang,ucAng):
        phi=ang-np.arctan2(qNew[1],qNew[0])
        revs=int(phi//ucAng) #number of revolutions through unit cell
        if revs%2==0: #if even
            theta = phi - ucAng * revs
        else: #if odd
            theta = ucAng-(phi - ucAng * revs)
        r=np.sqrt(qNew[0]**2+qNew[1]**2)
        qNew[0]=r*np.cos(theta) #cartesian coords in unit cell frame
        qNew[1]=r*np.sin(theta) #cartesian coords in unit cell frame
        return qNew

    def is_Coord_Inside(self, q):
        if np.abs(q[2]) > self.ap:  # if clipping in z direction
            return False
        phi = np.arctan2(q[1], q[0])
        if phi < 0:  # constraint to between zero and 2pi
            phi += 2 * np.pi
        if phi < self.ang:
            r = np.sqrt(q[0] ** 2 + q[1] ** 2)
            if r < self.rb - self.ap or r > self.rb + self.ap:
                return False
        else:
            return False

        return True

class BenderIdealSegmentedWithCap(BenderIdealSegmented):
    def __init__(self,PTL,numMagnets,Lm,Lcap,Bp,rp,rb,yokeWidth,space,rOffsetfact,ap,fillParams=True):
        super().__init__(PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space,rOffsetfact, ap,fillParams=False)
        self.Lcap = Lcap
        self.cap=True
        if fillParams==True:
            self.fill_Params()
    def fill_Params(self):
        super().fill_Params()
        if self.numMagnets is not None:
            if self.ang>3*np.pi/2 or self.ang<np.pi/2 or self.Lcap>self.rb-self.rp*2: #this is done so that finding where the particle is inside the
                #bender is not a huge chore. There is almost no chance it would have this shape anyways. Changing this
                #would affect force, orbit coordinates and isinside at least
                raise Exception('DIMENSIONS OF BENDER ARE OUTSIDE OF ACCEPTABLE BOUNDS')
            self.Lo = self.ro * self.ang+2*self.Lcap


    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        qo = q.copy()
        angle = np.arctan2(qo[1], qo[0])
        if angle<0: #to use full 2pi with arctan2
            angle+=2*np.pi
        if angle>self.ang: #if particle is outside of the bending segment so it could be in the caps, or elsewhere
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap): #If inside the cap on
                #the eastward side
                qo[0] = self.Lcap + self.ang * self.ro + (-q[1])
                qo[1] = q[0] - self.ro
            qTest=q.copy()
            qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
            qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
            if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0):
                qo[0]=self.Lcap-qTest[1]
                qo[1]=qTest[0]-self.ro
        else:
            phi = self.ang - np.arctan2(q[1], q[0])  # angle swept out by particle in trajectory. This is zero
            # when the particle first enters
            ds = self.ro * phi+self.Lcap
            qos = ds
            qox = np.sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
            qo[0] = qos
            qo[1] = qox
        return qo

    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame
        phi = fast_Arctan2(q)
        if phi<self.ang: #if inside the bbending segment
            return super().force(q)
        elif phi>self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap): #If inside the cap on
                #the eastward side
                self.F[0]=-self.K*(q[0]-self.rb)
            else: #if in the westward segment maybe
                qTest=q.copy()
                qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
                qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
                if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0): #definitely in the 
                    #westard segment
                    forcex = -self.K * (qTest[0] - self.rb)
                    self.F[0]=self.RIn_Ang[0,0]*forcex
                    self.F[1]=-self.RIn_Ang[1,0]*forcex
                else:
                    self.F=np.zeros(3)
                    warnings.warn('PARTICLE IS OUTSIDE ELEMENT')
        return self.F.copy()
    #@profile()
    def is_Coord_Inside(self, q):
        #return self.is_Coord_Inside_NUMBA(q,self.RIn_Ang,self.ap,self.ang,self.rb,self.Lcap)
        return cythonFunc.is_Coord_Inside_CYTHON_BenderIdealSegmentedWithCap(q,self.RIn_Ang,self.ap,self.ang,self.rb,self.Lcap)

    @staticmethod
    @numba.njit(numba.boolean(numba.float64[:],numba.float64[:,:],numba.float64,numba.float64,numba.float64,numba.float64))
    def is_Coord_Inside_NUMBA(qNew,RIn_Ang,ap,ang,rb,Lcap):
        qx,qy,qz=qNew
        if not -ap<qz<ap:  # if clipping in z direction
            return False
        phi = fast_Arctan2(qNew)
        if phi<ang: #if inside the bending region
            r = np.sqrt(qx ** 2 + qy ** 2)
            if rb-ap<r<rb+ap:
                return True
        if phi>ang:  # if outside bender's angle range
            if (rb - ap < qx < rb + ap) and (0 > qy > -Lcap): #If inside the cap on
                #the eastward side
                return True
            qxTest = RIn_Ang[0, 0] * qx + RIn_Ang[0, 1] * qy
            qyTest = RIn_Ang[1, 0] * qx + RIn_Ang[1, 1] * qy
            if (rb - ap < qxTest < rb + ap) and (Lcap > qyTest > 0): #if inside on the westward side
                return True
        return False

class BenderSimSegmented(BenderIdealSegmented):
    def __init__(self, PTL, fileName,numMagnets, Lm, Bp, rp, rb, yokeWidth, space, ap,fillParams=True):
        super().__init__( PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space, ap,fillParams=False)
        self.fileName=fileName
        raise Exception('NOT IMPLEMENTED')

class BenderSimSegmentedWithCap(BenderIdealSegmentedWithCap):
    def __init__(self,PTL, fileSeg,fileCap,fileInternalFringe,Lm,Lcap,rp,K0,numMagnets,rb,extraSpace,yokeWidth,rOffsetFact,ap):
        super().__init__(PTL,numMagnets,Lm,Lcap,None,rp,rb,yokeWidth,extraSpace,rOffsetFact,ap,fillParams=False)
        self.sim=True

        self.fileSeg=fileSeg
        self.fileCap=fileCap
        self.fileInternalFringe=fileInternalFringe
        self.numMagnets=numMagnets
        self.extraSpace=extraSpace
        self.yokeWidth=yokeWidth
        self.ap=ap
        self.K0=K0
        self.ucAng=None
        self.dataSeg=None
        self.dataCap=None
        self.dataInternalFringe=None
        self.Fx_Func_Seg=None
        self.Fy_Func_Seg=None
        self.Fz_Func_Seg=None
        self.magnetic_Potential_Func_Seg =None
        self.Fx_Func_Cap=None
        self.Fy_Func_Cap=None
        self.Fz_Func_Cap=None
        self.magnetic_Potential_Func_Cap = None
        self.Fx_Func_Internal_Fringe = None
        self.Fy_Func_Internal_Fringe = None
        self.Fz_Func_Internal_Fringe = None
        self.magnetic_Potential_Func_Fringe = None
        self.fill_Params()
    def fill_Params(self):
        if self.ap is None:
            self.ap = self.rp * .9
        elif self.ap > self.rp:
            raise Exception('APETURE IS LARGER THAN BORE')
        self.Lseg=self.Lm+self.space*2
        self.rOffsetFunc = lambda rb: self.rOffsetFact * np.sqrt(rb ** 2 / 16 + self.PTL.v0Nominal ** 2 / (2 * self.K0)) - rb / 4 # this accounts for energy loss
        if self.dataSeg is None and self.fileSeg is not None:
            self.fill_Force_Func_Seg()
            rp=(self.dataSeg[:,0].max()-self.dataSeg[:,0].min()-2*self.comsolExtraSpace)/2
            if np.round(rp,6)!=np.round(self.rp,6):
                raise Exception('BORE RADIUS FROM FIELD FILE DOES NOT MATCH')
            self.check_K0()
            self.dataSeg=None #to save memory and pickling time
        if self.dataCap is None and self.fileCap is not None:
            self.fill_Force_Func_Cap()
            Lcap=self.dataCap[:,2].max()-self.dataCap[:,2].min()-self.comsolExtraSpace*2
            if np.round(self.Lcap,6)!=np.round(Lcap,6):
                raise Exception('CAP LENGTH FROM FIELD FILE DOES NOT MATCH INPUT CAP LENGTH')
            self.dataCap=None #to save memory and pickling time
        if self.dataInternalFringe is None and self.fileInternalFringe is not None:
            self.fill_Force_Func_Internal_Fringe()
            self.dataInternalFringe=None #to save memory and pickling time
        if self.numMagnets is not None:
            D = self.rb - self.rp - self.yokeWidth
            self.ucAng = np.arctan(self.Lseg / (2 * D))
            self.ang = 2 * self.numMagnets * self.ucAng
            self.rOffset=self.rOffsetFunc(self.rb)
            self.ro=self.rb+self.rOffset
            self.L=self.ang*self.rb
            self.Lo=self.ang*self.ro+2*self.Lcap
            self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
            m = np.tan(self.ucAng)
            self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2) #reflection matrix
            m = np.tan(self.ang / 2)
            self.M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2) #reflection matrix
    def fill_Force_Func_Cap(self):
        self.dataCap = np.asarray(pd.read_csv(self.fileCap, delim_whitespace=True, header=None))
        interpFx, interpFy, interpFz, interpV = self.make_Interp_Functions(self.dataCap)
        self.Fx_Func_Cap = lambda x, y, z:  interpFx((x, -z, y))
        self.Fy_Func_Cap = lambda x, y, z: interpFz((x, -z, y))
        self.Fz_Func_Cap = lambda x, y, z: -interpFy((x, -z, y))
        self.magnetic_Potential_Func_Cap=lambda x, y, z: interpV((x, -z, y))
    def fill_Force_Func_Internal_Fringe(self):
        self.dataInternalFringe = np.asarray(pd.read_csv(self.fileInternalFringe, delim_whitespace=True, header=None))
        interpFx, interpFy, interpFz, interpV = self.make_Interp_Functions(self.dataInternalFringe)
        self.Fx_Func_Internal_Fringe = lambda x, y, z: interpFx((x, -z, y))
        self.Fy_Func_Internal_Fringe = lambda x, y, z: interpFz((x, -z, y))
        self.Fz_Func_Internal_Fringe = lambda x, y, z: -interpFy((x, -z, y))
        self.magnetic_Potential_Func_Fringe=lambda x,y,z:interpV((x,-z,y))
    def fill_Force_Func_Seg(self):
        self.dataSeg = np.asarray(pd.read_csv(self.fileSeg, delim_whitespace=True, header=None))
        interpFx, interpFy, interpFz, interpV = self.make_Interp_Functions(self.dataSeg)
        self.Fx_Func_Seg = lambda x, y, z: interpFx((x, -z, y))
        self.Fy_Func_Seg = lambda x, y, z: interpFz((x, -z, y))
        self.Fz_Func_Seg = lambda x, y, z: -interpFy((x, -z, y))
        self.magnetic_Potential_Func_Seg= lambda x, y, z: interpV((x, -z, y))
    def check_K0(self):
        #use the fit to the gradient of the magnetic field to find the k value in F=-k*x
        xFit=np.linspace(-self.rp/2,self.rp/2,num=10000)+self.dataSeg[:,0].mean()
        yFit=[]
        for x in xFit:
            yFit.append(self.Fx_Func_Seg(x, 0, 0))
        xFit=xFit-self.dataSeg[:,0].mean()
        K = -np.polyfit(xFit, yFit, 1)[0] #fit to a line y=m*x+b, and only use the m component
        percDif=100.0*(K-self.K0)/self.K0
        if np.abs(percDif)<1.0: #must be within roughly 1%
            pass #k0 is sufficiently close
        else:
            print('K current',np.round(K),' K target', self.K0)
            raise Exception('K VALUE FALLS OUTSIDE ACCEPTABLE RANGE')

    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame
        phi = fast_Arctan2(q)#calling a fast numba version that is global
        if phi<self.ang: #if particle is inside bending angle region
            revs = int((self.ang-phi) // self.ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position='FIRST'
            elif revs == self.numMagnets * 2 - 1 or revs == self.numMagnets * 2 - 2:
                position='LAST'
            else:
                position='INNER'
            if position == 'INNER':
                quc = self.transform_Element_Coords_Into_Unit_Cell_Frame(q)  # get unit cell coords
                self.F[0] = self.Fx_Func_Seg(*quc)
                self.F[1] = self.Fy_Func_Seg(*quc)
                self.F[2] = self.Fz_Func_Seg(*quc)
                self.F = self.transform_Unit_Cell_Force_Into_Element_Frame(self.F, q)  # transform unit cell coordinates into
                    # element frame
            elif position =='FIRST' or position == 'LAST':
                self.F=self.force_First_And_Last(q,position)
            else:
                warnings.warn('PARTICLE IS OUTSIDE LATTICE')
                self.F=np.zeros(3)
        elif phi>self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap): #If inside the cap on
                #eastward side
                x,y,z=q
                x=x-self.rb
                self.F[0] = self.Fx_Func_Cap(x, y, z)
                self.F[1] = self.Fy_Func_Cap(x, y, z)
                self.F[2] = self.Fz_Func_Cap(x, y, z)
            else:
                qTestx = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
                qTesty = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
                if (self.rb - self.ap < qTestx < self.rb + self.ap) and (self.Lcap > qTesty > 0):#if on the westwards side
                    x,y,z=qTestx,qTesty,q[2]
                    x=x-self.rb
                    y=-y
                    self.F[0]=self.Fx_Func_Cap(x, y, z)
                    self.F[1]=self.Fy_Func_Cap(x, y, z)
                    self.F[2]=self.Fz_Func_Cap(x, y, z)
                    Fx = self.F[0]
                    Fy = self.F[1]
                    self.F[0] = self.M_ang[0, 0] * Fx + self.M_ang[0, 1] * Fy
                    self.F[1] = self.M_ang[1, 0] * Fx + self.M_ang[1, 1] * Fy
                else: #if not in either cap
                    warnings.warn('PARTICLE IS OUTSIDE LATTICE')
                    self.F = np.zeros(3)
        return self.F.copy()
    def force_First_And_Last(self,q,position):
        qNew=q.copy()
        if position=='FIRST':
            qx = qNew[0]
            qy = qNew[1]
            qNew[0] = self.M_ang[0, 0] * qx + self.M_ang[0, 1] * qy
            qNew[1] = self.M_ang[1, 0] * qx + self.M_ang[1, 1] * qy

            qNew[0]=qNew[0]-self.rb

            self.F[0] = self.Fx_Func_Internal_Fringe(*qNew)
            self.F[1] = self.Fy_Func_Internal_Fringe(*qNew)
            self.F[2] = self.Fz_Func_Internal_Fringe(*qNew)

            Fx = self.F[0]
            Fy = self.F[1]
            self.F[0] = self.M_ang[0, 0] * Fx + self.M_ang[0, 1] * Fy
            self.F[1] = self.M_ang[1, 0] * Fx + self.M_ang[1, 1] * Fy
        elif position=='LAST':
            qNew[0]=qNew[0]-self.rb
            self.F[0] = self.Fx_Func_Internal_Fringe(*qNew)
            self.F[1] = self.Fy_Func_Internal_Fringe(*qNew)
            self.F[2] = self.Fz_Func_Internal_Fringe(*qNew)
        else:
            raise Exception('INVALID POSITION SUPPLIED')
        return self.F.copy()
    def magnetic_Potential(self,q):
        # magnetic potential at point q in element frame
        # q: particle's position in element frame
        phi = fast_Arctan2(q)#calling a fast numba version that is global
        V0=0.0
        if phi<self.ang: #if particle is inside bending angle region
            revs = int((self.ang-phi) // self.ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position='FIRST'
            elif revs == self.numMagnets * 2 - 1 or revs == self.numMagnets * 2 - 2:
                position='LAST'
            else:
                position='INNER'
            if position == 'INNER':
                quc = self.transform_Element_Coords_Into_Unit_Cell_Frame(q)  # get unit cell coords
                V0=self.magnetic_Potential_Func_Seg(*quc)
            elif position =='FIRST' or position == 'LAST':
                V0=self.magnetic_Potential_First_And_Last(q,position)
            else:
                warnings.warn('PARTICLE IS OUTSIDE LATTICE')
                self.F=np.zeros(3)
        elif phi>self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap): #If inside the cap on
                #eastward side
                x,y,z=q
                x=x-self.rb
                V0=self.magnetic_Potential_Func_Cap(x,y,z)
            else:
                qTest=q.copy()
                qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
                qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
                if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0):#if on the westwards side
                    x,y,z=qTest
                    x=x-self.rb
                    y=-y
                    V0=self.magnetic_Potential_Func_Cap(x,y,z)
                else: #if not in either cap
                    warnings.warn('PARTICLE IS OUTSIDE LATTICE')
                    self.F = np.zeros(3)
        return V0
    def magnetic_Potential_First_And_Last(self,q,position):
        qNew=q.copy()
        if position=='FIRST':
            qx = qNew[0]
            qy = qNew[1]
            qNew[0] = self.M_ang[0, 0] * qx + self.M_ang[0, 1] * qy
            qNew[1] = self.M_ang[1, 0] * qx + self.M_ang[1, 1] * qy
            qNew[0]=qNew[0]-self.rb
            V0=self.magnetic_Potential_Func_Fringe(*qNew)
        elif position=='LAST':
            qNew[0]=qNew[0]-self.rb
            V0=self.magnetic_Potential_Func_Fringe(*qNew)
        else:
            raise Exception('INVALID POSITION SUPPLIED')
        return V0

class LensSimWithCaps(LensIdeal):
    def __init__(self, PTL, file2D, file3D, L, ap):
        super().__init__(PTL, None, None, None, None,fillParams=False)
        self.file2D=file2D
        self.file3D=file3D
        self.L=L
        self.ap=ap
        self.Lcap=None
        self.Linner=None
        self.data2D=None
        self.data3D=None
        self.Fx_Func_Fringe=None
        self.Fy_Func_Fringe = None
        self.Fz_Func_Fringe = None
        self.magnetic_Potential_Func_Fringe= None
        self.Fx_Func_Inner=None
        self.Fy_Func_Inner = None
        self.Fz_Func_Inner = None
        self.magnetic_Potential_Func_Inner = None
        self.fieldFact=1.0
        self.fill_Params()
    def fill_Params(self):
        if self.data3D is None and self.file3D is not None: #if data has not been loaded yet
            self.data3D = np.asarray(pd.read_csv(self.file3D, delim_whitespace=True, header=None))
            self.fill_Force_Func_Cap()
            self.Lcap = self.data3D[:,2].max() - self.data3D[:,2].min() - 2 * self.comsolExtraSpace
            self.data3D=False
        if self.data2D is None and self.file2D is not None: #if data has not been loaded yet
            self.data2D = np.asarray(pd.read_csv(self.file2D, delim_whitespace=True, header=None))
            self.fill_Force_Func_2D()
            self.rp=(self.data2D[:,0].max()-self.data2D[:,0].min()-2*self.comsolExtraSpace)/2
            if self.ap is None:
                self.ap=.9*self.rp
            self.data2D=False
        if self.L is not None and self.Lcap is not None:
            self.set_Length(self.L)

    def set_Length(self,L):
        self.L=L
        self.Linner=L-2*self.Lcap
        if self.Linner < 0:
            raise Exception('LENSES IS TOO SHORT TO ACCOMODATE FRINGE FIELDS')
        self.Lo = self.L
    def fill_Force_Func_Cap(self):
        interpFx,interpFy,interpFz,interpV=self.make_Interp_Functions(self.data3D)
        #wrap the function in a more convenietly accesed function
        self.Fx_Func_Fringe = lambda x, y, z: interpFz((-z, y , x))
        self.Fy_Func_Fringe = lambda x, y, z: interpFy((-z, y , x))
        self.Fz_Func_Fringe = lambda x, y, z: -interpFx((-z, y , x))
        self.magnetic_Potential_Func_Fringe= lambda x,y,z: interpV((-z, y , x))

    #
    def fill_Force_Func_2D(self):
        xArr=np.unique(self.data2D[:,0])
        yArr = np.unique(self.data2D[:, 1])
        numx=xArr.shape[0]
        numy=yArr.shape[0]
        interpX=spi.RectBivariateSpline(xArr,yArr,(-self.data2D[:, 2]* self.PTL.u0).reshape(numy, numx,order='F'),kx=1,ky=1)
        interpY = spi.RectBivariateSpline(xArr, yArr,(-self.data2D[:, 3]* self.PTL.u0).reshape(numy, numx,order='F'),kx=1,ky=1)
        interpV= spi.RectBivariateSpline(xArr, yArr,(self.data2D[:, 4]* self.PTL.u0).reshape(numy, numx,order='F'),kx=1,ky=1)
        self.Fx_Func_Inner = lambda x, y, z: 0.0
        self.Fy_Func_Inner = lambda x, y, z: interpY(-z, y)[0][0]
        self.Fz_Func_Inner = lambda x, y, z: -interpX(-z, y)[0][0]
        self.magnetic_Potential_Func_Inner=lambda x,y,z: interpV(-z,y)[0][0]

    def magnetic_Potential(self,q):
        if q[0]<=self.Lcap:
            x,y,z=q
            x=self.Lcap-x
            V0=self.magnetic_Potential_Func_Fringe(x,y,z)
        elif self.Lcap<q[0]<self.L-self.Lcap:
            V0=self.magnetic_Potential_Func_Inner(*q)
        elif q[0]<self.L:
            x,y,z=q
            x=x-(self.L-self.Lcap)
            V0=self.magnetic_Potential_Func_Fringe(x,y,z)
        else:
            warnings.warn('PARTICLE IS OUTSIDE ELEMENT')
            print('position is',q)
            V0=0
        V0=self.fieldFact*V0 #modify the magnetic field depending on how the magnet is tuend
        return V0
    def force(self,q):
        if q[0]<=self.Lcap:
            x,y,z=q
            x=self.Lcap-x
            self.F[0]= -self.Fx_Func_Fringe(x, y, z)
            self.F[1]= self.Fy_Func_Fringe(x, y, z)
            self.F[2]= self.Fz_Func_Fringe(x, y, z)
        elif self.Lcap<q[0]<self.L-self.Lcap:
            self.F[0]= 0.0
            self.F[1]= self.Fy_Func_Inner(*q)
            self.F[2]= self.Fz_Func_Inner(*q)
        elif self.L-self.Lcap<=q[0]<self.L:
            x,y,z=q
            x=x-(self.Linner+self.Lcap)
            self.F[0]= self.Fx_Func_Fringe(x, y, z)
            self.F[1]= self.Fy_Func_Fringe(x, y, z)
            self.F[2]= self.Fz_Func_Fringe(x, y, z)
        else:
            warnings.warn('PARTICLE IS OUTSIDE ELEMENT')
            self.F=np.zeros(3)
        self.F=self.fieldFact*self.F #modify the forces depending on how much the magnet is tuned
        return self.F.copy()