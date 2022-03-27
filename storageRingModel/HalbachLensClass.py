from numbers import Number
import time
from constants import MAGNETIC_PERMEABILITY
import numpy as np
from numpy.linalg import norm
import numba #type ignore
from scipy.spatial.transform import Rotation
from magpylib.magnet import Box as _Box
from magpylib import Collection
from typing import Union,Optional
from demag_functions import apply_demag

M_Default=1.018E6 #default magnetization value, SI. Magnetization for N48 grade

list_tuple_arr=Union[list,tuple,np.ndarray]

@numba.njit(numba.float64[:,:](numba.float64[:,:],numba.float64[:],numba.float64[:]))
def B_NUMBA(r,r0,m):
    r=r-r0  # convert to difference vector
    rNormTemp=np.sqrt(np.sum(r**2,axis=1))
    rNorm=np.empty((rNormTemp.shape[0],1))
    rNorm[:,0]=rNormTemp
    mrDotTemp=np.sum(m*r,axis=1)
    mrDot=np.empty((rNormTemp.shape[0],1))
    mrDot[:,0]=mrDotTemp
    Bvec=(MAGNETIC_PERMEABILITY/(4*np.pi))*(3*r*mrDot/rNorm**5-m/rNorm**3)
    return Bvec


class Sphere:

    def __init__(self, radius: float,M: float=M_Default):
        # angle: symmetry plane angle. There is a negative and positive one
        # radius: radius in inches
        #M: magnetization
        assert radius>0 and M>0
        self.angle: Optional[float] = None  # angular location of the magnet
        self.radius: float =radius
        self.volume: float=(4*np.pi/3)*self.radius**3 #m^3
        self.m0: float = M * self.volume  # dipole moment
        self.r0: Optional[np.ndarray] = None  # location of sphere
        self.n: Optional[np.ndarray] = None  # orientation
        self.m: Optional[np.ndarray] = None  # vector sphere moment
        self.theta: Optional[float] = None  # phi position
        self.phi: Optional[float] = None  # orientation of dipole. From lab z axis
        self.psi: Optional[float] = None  # orientation of dipole. in lab xy plane
        self.z: Optional[float] = None
        self.r: Optional[float] = None

    def position_Sphere(self, r: float, theta: float, z: float)->None:
        self.r,self.theta,self.z = r, theta, z
        assert not None in (theta,z,r)
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)
        self.r0 = np.asarray([x, y, self.z])

    def update_Size(self, radius: float)-> None:
        self.radius = radius
        self.volume = (4 * np.pi / 3) * self.radius ** 3
        M = 1.15e6  # magnetization density
        self.m0 = M * (4 / 3) * np.pi * self.radius ** 3  # dipole moment
        self.m = self.m0 * self.n  # vector sphere moment

    def orient(self, phi: float, psi: float)->None:
        # tilt the sphere in spherical coordinates. These are in the lab frame, there is no sphere frame
        #phi,psi =(pi/2,0) is along +x
        #phi,psi=(pi/2,pi/2) is along +y
        #phi,psi=(0.0,anything) is along +z
        self.phi = phi
        self.psi = psi
        self.n = np.asarray([np.sin(phi) * np.cos(psi), np.sin(phi) * np.sin(psi), np.cos(phi)])#x,y,z
        self.m = self.m0 * self.n

    def B(self, r:np.ndarray)-> np.ndarray:
        assert len(r.shape)==2 and r.shape[1]==3
        return B_NUMBA(r, self.r0, self.m)

    def B_Shim(self, r:np.ndarray, planeSymmetry: bool=True,negativeSymmetry: bool=True,rotationAngle: float=np.pi/3)->np.ndarray:
        # a single magnet actually represents 12 magnet
        # r: array of N position vectors to get field at. Shape (N,3)
        # planeSymmetry: Wether to exploit z symmetry or not
        # plt.quiver(self.r0[0],self.r0[1],self.m[0],self.m[1],color='r')
        arr = np.zeros(r.shape)
        arr += self.B(r)
        arr+=self.B_Symmetry(r,1,negativeSymmetry,rotationAngle,not planeSymmetry)
        arr+=self.B_Symmetry(r,2,negativeSymmetry,rotationAngle,not planeSymmetry)
        arr+=self.B_Symmetry(r,3,negativeSymmetry,rotationAngle,not planeSymmetry)
        arr+=self.B_Symmetry(r,4,negativeSymmetry,rotationAngle,not planeSymmetry)
        arr+=self.B_Symmetry(r,5,negativeSymmetry,rotationAngle,not planeSymmetry)

        if planeSymmetry == True:
            arr += self.B_Symmetry(r, 0, negativeSymmetry, rotationAngle,planeSymmetry)
            arr += self.B_Symmetry(r, 1, negativeSymmetry, rotationAngle,planeSymmetry)
            arr += self.B_Symmetry(r, 2, negativeSymmetry, rotationAngle,planeSymmetry)
            arr += self.B_Symmetry(r, 3, negativeSymmetry, rotationAngle,planeSymmetry)
            arr += self.B_Symmetry(r, 4, negativeSymmetry, rotationAngle,planeSymmetry)
            arr += self.B_Symmetry(r, 5, negativeSymmetry, rotationAngle,planeSymmetry)

        # plt.gca().set_aspect('equal')
        # plt.grid()
        # plt.show()
        return arr

    def B_Symmetry(self,r:np.ndarray, rotations: float,negativeSymmetry: float, rotationAngle: float,
                   planeReflection: float)->np.ndarray:
        rotAngle = rotationAngle * rotations
        M_Rot = np.array([[np.cos(rotAngle), -np.sin(rotAngle)], [np.sin(rotAngle), np.cos(rotAngle)]])
        r0Sym=self.r0.copy()
        r0Sym[:2]=M_Rot@r0Sym[:2]
        mSym =self.m.copy()
        mSym[:2] = M_Rot @ mSym[:2]
        if negativeSymmetry==True:
            mSym[:2] *= (-1) ** rotations
        if planeReflection == True:  # another dipole on the other side of the z=0 line
            r0Sym[2] = -r0Sym[2]
            mSym[-1] *= -1
        # plt.quiver(r0Sym[0], r0Sym[1], mSym[0], mSym[1])
        BVecArr = B_NUMBA(r, r0Sym, mSym)
        return BVecArr


class billyHalbachCollectionWrapper(Collection):
    magpyMagnetization_ToSI: float = 1 / (1e3 * MAGNETIC_PERMEABILITY)
    SI_MagnetizationToMagpy: float = 1/magpyMagnetization_ToSI
    meterTo_mm=1e3 #magpy takes distance in mm
    def __init__(self,*sources):
        super().__init__(sources)

    def rotate(self, rot, anchor=None, start=-1, increment=False):
        if anchor is None:
            raise NotImplementedError #not sure how to best deal with rotating collection about itself
        super().rotate(rot, anchor=0.0)

    def move(self,displacement: list_tuple_arr, start=-1, increment=False):
        displacement=[entry*self.meterTo_mm for entry in displacement]
        super().move(displacement)

    def B_Vec(self, evalCoords: np.ndarray)->np.ndarray:
        # r: Coordinates to evaluate at with dimension (N,3) where N is the number of evaluate points
        assert len(self)>0
        mTesla_To_Tesla=1e-3
        meterTo_mm=1e3
        evalCoords_mm = meterTo_mm * evalCoords
        BVec=mTesla_To_Tesla*self.getB(evalCoords_mm)
        return BVec
    def BNorm(self,evalCoords:np.ndarray)->np.ndarray:
        #r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        #Returns a either a (N,3) or (3) array, whichever matches the shape of the r array

        BVec=self.B_Vec(evalCoords)
        if len(evalCoords.shape)==1:
            return norm(BVec)
        else:
            return norm(BVec,axis=1)

    def BNorm_Gradient(self,evalCoords: np.ndarray,returnNorm: bool=False,dr: float=1e-7)->Union[np.ndarray,tuple]:
        #Return the gradient of the norm of the B field. use forward difference theorom
        #r: (N,3) vector of coordinates or (3) vector of coordinates.
        #returnNorm: Wether to return the norm as well as the gradient.
        #dr: step size
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(evalCoords.shape)==1:
            rEval=np.asarray([evalCoords])
        else:
            rEval=evalCoords.copy()
        def grad(index):
            coordb = rEval.copy()  # upper step
            coordb[:, index] += dr
            BNormB=self.BNorm(coordb)
            coorda = rEval.copy()  # upper step
            coorda[:, index] += -dr
            BNormA=self.BNorm(coorda)
            return (BNormB-BNormA)/(2*dr)
        BNormGradx=grad(0)
        BNormGrady=grad(1)
        BNormGradz=grad(2)
        if len(evalCoords.shape)==1:
            if returnNorm == True:
                BNormCenter=self.BNorm(rEval)
                return np.asarray([BNormGradx[0], BNormGrady[0], BNormGradz[0]]),BNormCenter[0]
            else:
                return np.asarray([BNormGradx[0],BNormGrady[0],BNormGradz[0]])
        else:
            if returnNorm==True:
                BNormCenter=self.BNorm(rEval)
                return np.column_stack((BNormGradx, BNormGrady, BNormGradz)),BNormCenter
            else:
                return np.column_stack((BNormGradx,BNormGrady,BNormGradz))
    def method_Of_Moments(self):
        apply_demag(self)


class Box(_Box):
    def __init__(self,mur: float=1.0,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mur=mur


class Layer(billyHalbachCollectionWrapper):
    # class object for a layer of the magnet. Uses the RectangularPrism object

    magpyMagnetization_ToSI: float = 1 / (1e3 * MAGNETIC_PERMEABILITY)
    SI_MagnetizationToMagpy: float = 1/magpyMagnetization_ToSI
    meterTo_mm=1e3 #magpy takes distance in mm

    numMagnetsInLayer = 12
    def __init__(self, rp: float,length: float,magnetWidth: float,position: list_tuple_arr=(0.0,0.0,0.0),
                 orientation: Optional[Rotation]=None, M: float=M_Default,mur: float=1.05,
                 rMagnetShift=None, thetaShift=None,phiShift=None, M_ShiftRelative=None,applyMethodOfMoments=False):
        #all SI units
        super().__init__()
        assert magnetWidth > 0.0 and length > 0.0  and M > 0.0
        assert isinstance(orientation,(type(None),Rotation))==True
        self.rMagnetShift: tuple=self.make_Tuple_If_None(rMagnetShift)
        self.thetaShift: tuple=self.make_Tuple_If_None(thetaShift)
        self.phiShift: tuple=self.make_Tuple_If_None(phiShift)
        self.M_ShiftRelative: tuple=self.make_Tuple_If_None(M_ShiftRelative)
        self.rp: tuple = (rp,)*self.numMagnetsInLayer
        self.mur=mur #relative permeability
        self.position=position
        self.orientation=orientation #orientation about body frame
        self.magnetWidth: float = magnetWidth
        self.length: float = length
        self.applyMethodOfMoments=applyMethodOfMoments
        self.M: float = M
        self.build()

    def make_Tuple_If_None(self,variable: Optional[tuple])-> tuple:
        variableTuple= (0.0,)*self.numMagnetsInLayer if variable is None else variable
        assert len(variableTuple)==self.numMagnetsInLayer
        return variableTuple

    def build(self)->None:
        # build the elements that form the layer. The 'home' magnet's center is located at x=r0+width/2,y=0, and its
        #magnetization points along positive x
        #how I do this is confusing
        thetaArr = np.linspace(0, 2 * np.pi, 12, endpoint=False) # location of 12 magnets.
        thetaArr+=np.asarray(self.thetaShift)
        phiArr =np.pi + np.arange(0, 12) * 2 * np.pi / 3 #direction of magnetization
        phiArr+=np.asarray(self.phiShift)
        rArr=self.rp+np.asarray(self.rMagnetShift)
        M_Arr=self.M*np.ones(self.numMagnetsInLayer)*(1.0+np.asarray(self.M_ShiftRelative))

        for r, phi,theta,M in zip(rArr,phiArr,thetaArr,M_Arr):
            rMagnetCenter = r + self.magnetWidth / 2
            xCenter, yCenter = rMagnetCenter * np.cos(theta), rMagnetCenter * np.sin(theta)
            R = Rotation.from_rotvec([0.0, 0.0, phi])
            M_MagpyUnit = self.M * self.SI_MagnetizationToMagpy  # uses units of mT for magnetization.
            dimension = np.asarray((self.magnetWidth, self.magnetWidth, self.length))*self.meterTo_mm
            position=(xCenter*self.meterTo_mm, yCenter*self.meterTo_mm, 0.0)
            box =Box(magnetization=(M_MagpyUnit, 0.0, 0.0), dimension=dimension,
                     position=position, orientation=R,mur=self.mur)
            self.add(box)
        if self.orientation is not None:
            self.rotate(self.orientation,anchor=0.0)
        self.move(self.position)
        if self.applyMethodOfMoments==True:
            self.method_Of_Moments()


class HalbachLens(billyHalbachCollectionWrapper):

    # magpyMagnetization_ToSI: float = 1 / (1e3 * MAGNETIC_PERMEABILITY)
    # SI_MagnetizationToMagpy: float = 1/magpyMagnetization_ToSI
    # meterTo_mm=1e3 #magpy takes distance in mm

    def __init__(self,rp: Union[float,tuple],magnetWidth: Union[float,tuple],length: float,
                 position: list_tuple_arr=(0.0,0.0,0.0), orientation: Optional[Rotation]=None,
                 M: float=M_Default,numSlices: int =1, applyMethodOfMoments=False):
        super().__init__()
        assert length > 0.0 and M > 0.0
        assert isinstance(orientation, (type(None), Rotation)) == True
        assert isinstance(rp,(float,tuple)) and isinstance(magnetWidth,(float,tuple))
        self.position = position
        self.orientation = orientation  # orientation about body frame
        self.rp: tuple=rp if isinstance(rp,tuple) else (rp,)
        self.magnetWidth: tuple = magnetWidth if isinstance(magnetWidth,tuple) else (magnetWidth,)
        self.length: float = length
        self.applyMethodOfMoments: bool = applyMethodOfMoments
        self.M: float = M
        self.numSlices=numSlices
        self.mur=1.05

        self.layerList: list[Layer]=[]
        self.build()

    def build(self):
        zArr, lengthArr = self.subdivide_Lens()
        for zLayer,length in zip(zArr,lengthArr):
            for radiusLayer,widthLayer in zip(self.rp,self.magnetWidth):
                layer=Layer(radiusLayer,length,widthLayer,M=self.M,position=(0,0,zLayer))
                self.add(layer)
                self.layerList.append(layer)

        if self.orientation is not None:
            self.rotate(self.orientation, anchor=0.0)
        self.move(self.position)

        if self.applyMethodOfMoments == True:
            self.method_Of_Moments()

    def subdivide_Lens(self):
        """To improve accuracu of magnetostatic method of moments, divide the layers into smaller layers """
        LArr=np.ones(self.numSlices)*self.length/self.numSlices
        zArr=np.cumsum(LArr)-self.length/2-.5*self.length/self.numSlices
        assert abs(np.sum(LArr)-self.length)<1e-12 and np.abs(np.mean(zArr))<1e-12 #length adds up and centered on 0
        return zArr,LArr


class SegmentedBenderHalbach(billyHalbachCollectionWrapper):
    #a model of odd number lenses to represent the symmetry of the segmented bender. The inner lens represents the fully
    #symmetric field
    def __init__(self,rp:float,rb: float,UCAngle: float,Lm: float,numLenses: int=3,
                 M: float=M_Default,positiveAngleMagnetsOnly: bool=False,applyMethodOfMoments=False,lensSlices: int=1):
        super().__init__()
        assert all(isinstance(value, Number) for value in (rp,rb,UCAngle,Lm))
        self.rp: float=rp #radius of bore of magnet, ie to the pole
        self.rb: float=rb #bending radius
        self.UCAngle: float=UCAngle #unit cell angle of a HALF single magnet, ie HALF the bending angle of a single magnet. It
        #is called the unit cell because obviously one only needs to use half the magnet and can use symmetry to
        #solve the rest
        self.Lm: float=Lm #length of single magnet
        self.M: float=M #magnetization, SI
        self.positiveAngleMagnetsOnly: bool=positiveAngleMagnetsOnly #This is used to model the cap amgnet, and the first full
        #segment. No magnets can be below z=0, but a magnet can be right at z=0. Very different behavious wether negative
        #or positive
        self.magnetWidth: float=rp * np.tan(2 * np.pi / 24) * 2 #set to size that exactly fits
        self.numLenses: int=numLenses #number of lenses in the model
        self.lensList: list=[] #list to hold lenses
        self.applyMethodsOfMoments=applyMethodOfMoments
        self.slices=lensSlices
        self._build()

    def _build(self)->None:
        self.lensList=[]
        if self.numLenses==1:
            if self.positiveAngleMagnetsOnly==True:
                raise Exception('Not applicable with only 1 magnet')
            angleArr=np.asarray([0.0])
        else:
            angleArr=np.linspace(-2*self.UCAngle*(self.numLenses-1)/2,2*self.UCAngle*(self.numLenses-1)/2,num=self.numLenses)
        if self.positiveAngleMagnetsOnly==True:
            angleArr=angleArr-angleArr.min()
        for i in range(angleArr.shape[0]):
            lens=HalbachLens(self.rp,self.magnetWidth,self.Lm,M=self.M,position=(self.rb,0.0,0.0),
                             numSlices=self.slices)

            R=Rotation.from_rotvec([0.0,-angleArr[i],0.0])
            lens.rotate(R,anchor=0)
            #my angle convention is unfortunately opposite what it should be here. positive theta
            # is clockwise about y axis in the xz plane looking from the negative side of y
            # lens.position(r0)
            self.lensList.append(lens)
            self.add(lens)
        if self.applyMethodsOfMoments==True:
            self.method_Of_Moments()