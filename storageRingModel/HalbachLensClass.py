from numbers import Number
import time
from constants import MAGNETIC_PERMEABILITY
import numpy as np
from numpy.linalg import norm
import numba #type ignore
from scipy.spatial.transform import Rotation
from magpylib.magnet import Box
from typing import Union,Optional

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


class RectangularPrism:
    #A right rectangular prism. Without any rotation the prism is oriented such that the 2 dimensions in the x,y plane
    #are equal, but the length, in the z plane, can be anything.

    #magpy uses non SI units, so I need to do some conversion
    magpyMagnetization_ToSI: float = 1 / (1e3 * MAGNETIC_PERMEABILITY)
    SI_MagnetizationToMagpy: float = 1/magpyMagnetization_ToSI
    meterTo_mm=1e3 #magpy takes distance in mm

    def __init__(self, rCenter: float ,theta: float,z: float,phi: float,width: float,length: float,M: float=M_Default,
                 ):
        #width: The width in the x,y plane without rotation, meters
        #lengthI: The length in the z plane without rotation, meters
        #M: magnetization, SI
        #MVec: direction of the magnetization vector
        # theta rotation is clockwise about y in my notation, originating at positive z
        # psi is counter clockwise around z
        assert width>0.0 and length >0.0  and M>0
        self.width= width
        self.length=length
        self.mur: float = 1.05  # recoil permeability
        self.M=M
        self.rCenter= rCenter #r in cylinderical coordinates, equals sqrt(x**2+y**2)
        self.x0=rCenter * np.cos(theta)
        self.y0=rCenter * np.sin(theta)
        self.z0 = z
        self.theta =theta #position angle in polar coordinates
        self.phi = phi # rotation about body z axis
        self.r0 = np.asarray([self.x0,self.y0,self.z0])
        self._magnet: Box=self._build_magpy_Prism()

    def _build_magpy_Prism(self)-> Box:
        """Build magpy prism object. This requires so units conversions, and forming vectors"""
        dimensions_mm =np.asarray([self.width, self.width, self.length]) * self.meterTo_mm
        position_mm =  np.asarray([self.x0, self.y0, self.z0]) * self.meterTo_mm
        R = Rotation.from_rotvec([0, 0, self.phi])
        M_MagpyUnit = self.M * self.SI_MagnetizationToMagpy  # uses units of mT for magnetization.
        magnet = Box(magnetization=(M_MagpyUnit, 0, 0.0), dimension=dimensions_mm, position=position_mm,
                     orientation=R)
        return magnet

    def B(self,evalCoords: np.ndarray)->np.ndarray:
        """B field vector at evalCoords. Can be shape (3) or (n,3). Return will have same shape except (1,3)->(3)"""
        assert len(evalCoords.shape) == 2 and evalCoords.shape[1] == 3
        evalCoords_mm = 1e3 * evalCoords
        BVec_mT = self._magnet.getB(evalCoords_mm)  # need to convert to mm
        BVec_T = 1e-3 * BVec_mT  # convert to tesla from milliTesla
        if len(BVec_T.shape) == 1:
            BVec_T = np.asarray([BVec_T])
        return BVec_T

    def B_Shim(self, r:np.ndarray, planeSymmetry: bool=True,negativeSymmetry: bool=True,
               rotationAngle: float=np.pi/3)->np.ndarray:
        # a single magnet actually represents 12 magnet
        # r: array of N position vectors to get field at. Shape (N,3)
        # planeSymmetry: Wether to exploit z symmetry or not
        # plt.quiver(self.r0[0],self.r0[1],np.cos(self.phi),np.sin(self.phi),color='r')
        print("I'm not sure this is working correctly with magnetic method of moments!")
        arr = np.zeros(r.shape)
        arr += self.B(r)
        arr+=self.B_Symmetry(r,1,negativeSymmetry,rotationAngle,False)
        arr+=self.B_Symmetry(r,2,negativeSymmetry,rotationAngle,False)
        arr+=self.B_Symmetry(r,3,negativeSymmetry,rotationAngle,False)
        arr+=self.B_Symmetry(r,4,negativeSymmetry,rotationAngle,False)
        arr+=self.B_Symmetry(r,5,negativeSymmetry,rotationAngle,False)

        if planeSymmetry == True:
            arr += self.B_Symmetry(r, 0, negativeSymmetry, rotationAngle,True)
            arr += self.B_Symmetry(r, 1, negativeSymmetry, rotationAngle,True)
            arr += self.B_Symmetry(r, 2, negativeSymmetry, rotationAngle,True)
            arr += self.B_Symmetry(r, 3, negativeSymmetry, rotationAngle,True)
            arr += self.B_Symmetry(r, 4, negativeSymmetry, rotationAngle,True)
            arr += self.B_Symmetry(r, 5, negativeSymmetry, rotationAngle,True)
        return arr

    def B_Symmetry(self,r:np.ndarray, rotations: float,negativeSymmetry: bool, rotationAngle: float,
                   planeSymmetry: bool):
        rotAngle = rotationAngle * rotations
        thetaSymm=rotAngle
        M_Rot = np.array([[np.cos(rotAngle), -np.sin(rotAngle)], [np.sin(rotAngle), np.cos(rotAngle)]])
        r0Sym=self.r0.copy()
        r0Sym[:2]=M_Rot@r0Sym[:2]
        if negativeSymmetry==True:
            thetaSymm+=rotations*np.pi
        if planeSymmetry==True:
            r0Sym[2] = -r0Sym[2]
        # plt.quiver(r0Sym[0], r0Sym[1], np.cos(self.phi+phiSymm),np.sin(self.phi+phiSymm))
        x,y,z=r0Sym
        rCenter=np.sqrt(x**2+y**2)
        theta=np.arctan2(y,x)
        magnet=RectangularPrism(rCenter,theta,z,self.phi+thetaSymm,self.width,self.length,self.M)
        B=magnet.B(r)
        return B

class Layer:
    # class object for a layer of the magnet. Uses the RectangularPrism object

    numMagnetsInLayer = 12
    def __init__(self, z, width, length,rp, M=M_Default, phase=0.0,rMagnetShift=None, thetaShift=None,phiShift=None,
                 M_ShiftRelative=None,applyMethodOfMoments=False):
        # z: z coordinate of the layer, meter. The layer is in the x,y plane. This is the location of the center
        # width: width of the rectangular prism in the xy plane
        # length: length of the rectangular prism in the z axis
        # M: magnetization, SI
        # spherePerDim: number of spheres per transvers dimension in each cube. Longitudinal number will be this times
        # a factor
        assert width > 0.0 and length > 0.0  and M > 0.0
        self.rMagnetShift: tuple=self.make_Tuple_If_None(rMagnetShift)
        self.thetaShift: tuple=self.make_Tuple_If_None(thetaShift)
        self.phiShift: tuple=self.make_Tuple_If_None(phiShift)
        self.M_ShiftRelative: tuple=self.make_Tuple_If_None(M_ShiftRelative)
        self.rp: tuple = (rp,)*self.numMagnetsInLayer
        self.z: float = z
        self.width: float = width
        self.length: float = length
        self.applyMethodOfMoments=applyMethodOfMoments

        self.M: float = M
        self.phase: float=phase #this will rotate the layer about z this amount
        self.RectangularPrismsList: list[RectangularPrism] = []  # list of RectangularPrisms
        self.build()
    def make_Tuple_If_None(self,variable: Optional[tuple])-> tuple:
        variableTuple= (0.0,)*self.numMagnetsInLayer if variable is None else variable
        assert len(variableTuple)==self.numMagnetsInLayer
        return variableTuple
    def build(self)->None:
        # build the elements that form the layer. The 'home' magnet's center is located at x=r0+width/2,y=0, and its
        #magnetization points along positive x
        #how I do this is confusing
        thetaArr = np.linspace(0, 2 * np.pi, 12, endpoint=False)+self.phase  # location of 12 magnets.
        thetaArr+=np.asarray(self.thetaShift)
        phiArr =self.phase+ np.pi + np.arange(0, 12) * 2 * np.pi / 3 #direction of magnetization
        phiArr+=np.asarray(self.phiShift)
        rArr=self.rp+np.asarray(self.rMagnetShift)
        M_Arr=self.M*np.ones(self.numMagnetsInLayer)*(1.0+np.asarray(self.M_ShiftRelative))
        for r, phi,theta,M in zip(rArr,phiArr,thetaArr,M_Arr):
            # x,y=r*np.cos(theta),r*np.sin(theta)
            # plt.quiver(x,y,np.cos(phi),np.sin(phi))
            rMagnetCenter = r + self.width / 2
            magnet = RectangularPrism(rMagnetCenter,theta,self.z,phi,self.width, self.length, M)
            self.RectangularPrismsList.append(magnet)
        if self.applyMethodOfMoments==True:
            self.solve_Method_Of_Moments()
        # plt.gca().set_aspect('equal')
        # plt.show()

    def B(self, r: np.ndarray)->np.ndarray:
        # r: Coordinates to evaluate at with dimension (N,3) where N is the number of evaluate points
        assert len(self.RectangularPrismsList)>0
        BArr=np.zeros(r.shape)
        for prism in self.RectangularPrismsList:
            BArr += prism.B(r)
        return BArr

    def solve_Method_Of_Moments(self):
        """Tweak magnetization of magnets based on magnetic fields of neighboring magnets, and self fields"""
        from demag_functions import apply_demag
        from magpylib import Collection
        prismList,chiList=[],[]
        for prism in self.RectangularPrismsList:
            prismList.append(prism._magnet)#extend([p._magnet._magnet for p in layer.RectangularPrismsList])
            chiList.append(prism.mur-1.0)
        col=Collection(prismList)
        apply_demag(col,chiList)

class HalbachLens:
    # class for a lens object. This is uses the layer object.
    # The lens will be positioned such that the center layer is at z=0. Can be tilted though

    def __init__(self,length:float,width:Union[float,tuple],rp:Union[float,tuple],M=M_Default,
                 applyMethodOfMoments=False,subdivide=False):
        assert type(width)==type(rp)
        if isinstance(width,float):
            width,rp=(width,),(rp,)
        assert all(isinstance(variable,tuple) for variable in (rp,width))
        assert all(val>0.0 for val in [*rp,*width])
        assert len(width)==len(rp)
        assert length>0.0 and M>0.0
        self.width: tuple=width
        self.rp: tuple=rp
        self.length: Optional[float]
        if length is None:
            self.length=width
        else:
            self.length=length
        self.layerList: list[Layer]=[] #object to hold layers
        self.applyMethodOfMoments: bool=applyMethodOfMoments
        self.subdivide: bool=subdivide
        self.M: float=M
        #euler angles. Order of operation is theta and then psi, done in the reference frame of the lens. In reality
        #the rotations are done to the evaluation point, then the resulting vector is rotated back
        self.phi: Optional[float]=None #rotation, through the y axis
        self.r0: np.ndarray=np.zeros(3) #location of center of magnet
        self.RInTheta: Optional[np.ndarray]=None #rotation matrix for the evaluation points to generate the final value
        self.ROutTheta: Optional[np.ndarray]=None #rotation matrix to rotate the vector out of the tilted frame to the original frame
        self._build()

    def position(self,r0: np.ndarray)->None:
        #position the magnet in space
        #r0: 3d position vector of the center of the magnet
        self.r0=r0

    def rotate(self,theta: float)->None:
        #rotate the magnet about its center
        self.phi=theta
        self.RInTheta=np.asarray([[np.cos(-theta),-np.sin(-theta)],[np.sin(-theta),np.cos(-theta)]]) #to rotate about the
        #y axis. Since the magnet is rotated by theta, the evaluation points need to be rotated by -theta
        self.ROutTheta = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #to rotate the
        #vector out after evaluating with the rotated coordinates

    def _build(self)->None:
        """Build layers forming the lens. Layers are concentric, or longitudinal"""
        self.layerList=[]
        if self.subdivide==True:
            zArr,lengthArr=self.subdivide_Lens()
        else:
            zArr,lengthArr=np.zeros(1),np.ones(1)*self.length
        for z,length in zip(zArr,lengthArr):
            for radiusLayer,widthLayer in zip(self.rp,self.width):
                layer=Layer(z,widthLayer,length,radiusLayer,M=self.M)
                self.layerList.append(layer)
        if self.applyMethodOfMoments==True:
            self.solve_Method_Of_Moments()
    def subdivide_Lens(self):
        """To improve accuracu of magnetostatic method of moments, divide the layers into smaller layers """
        numSlices=5
        LArr=np.ones(numSlices)*self.length/numSlices
        zArr=np.cumsum(LArr)-self.length/2-.5*self.length/numSlices
        assert abs(np.sum(LArr)-self.length)<1e-12 and np.abs(np.mean(zArr))<1e-12 #length adds up and centered on 0
        return zArr,LArr

    def solve_Method_Of_Moments(self):
        """Tweak magnetization of magnets based on magnetic fields of neighboring magnets, and self fields"""
        from demag_functions import apply_demag
        from magpylib import Collection
        prismList,chiList=[],[]
        for layer in self.layerList:
            prismList.extend([p._magnet for p in layer.RectangularPrismsList])
            chiList.extend([p.mur-1.0 for p in layer.RectangularPrismsList])
        col=Collection(prismList)
        apply_demag(col,chiList)


    def _transform_r(self,r:np.ndarray)->np.ndarray:
        #to evaluate the field from tilted or translated magnets, the evaluation point is instead tilted or translated,
        #then the vector is rotated back. This function handle the rotation and translation of the evaluation points
        #r: rows of coordinates, shape (N,3). Where N is the number of evaluation points
        if self.phi is not None:
            return self._transform_r_NUMBA(r,self.r0,self.ROutTheta)
        else:
            return r

    @staticmethod
    @numba.njit(numba.float64[:,:](numba.float64[:,:],numba.float64[:],numba.float64[:,:]))
    def _transform_r_NUMBA(r,r0,ROutTheta):
        rNew=r.copy()
        rNew=rNew-np.ones(rNew.shape)*r0 #need to move the coordinates towards where the evaluation will take
        #place
        for i in range(rNew.shape[0]):
            rx=rNew[i][0]
            rz=rNew[i][2]
            rNew[i][0]=ROutTheta[0,0]*rx+ROutTheta[0,1]*rz
            rNew[i][2]=ROutTheta[1,0]*rx+ROutTheta[1,1]*rz
        return rNew

    def _transform_Vector(self,v: np.ndarray)->np.ndarray:
        #to evaluate the field from tilted or translated magnets, the evaluation point is instead tilted or translated,
        #then the vector is rotated back. This function handles the rotation of the evaluated vector
        #v: rows of vectors, shape (N,3) where N is the number of vectors
        if self.phi is not None:
            return self._transform_Vector_NUMBA(v,self.RInTheta)
        else:
            return v

    @staticmethod
    @numba.njit(numba.float64[:,:](numba.float64[:,:],numba.float64[:,:]))
    def _transform_Vector_NUMBA(v: np.ndarray,RInTheta: np.ndarray)-> np.ndarray:
        vNew=v.copy()
        for i in range(vNew.shape[0]):
            vx=vNew[i][0]
            vz=vNew[i][2]
            vNew[i][0]=RInTheta[0,0]*vx+RInTheta[0,1]*vz
            vNew[i][2]=RInTheta[1,0]*vx+RInTheta[1,1]*vz
        return vNew

    def B_Vec(self,r: np.ndarray)->np.ndarray:
        #r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        #Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
        rEval=self._transform_r(rEval)
        BArr=np.zeros(rEval.shape)
        for layer in self.layerList:
            BArr+=layer.B(rEval)
        BArr=self._transform_Vector(BArr)
        if len(r.shape)==1:
            return BArr[0]
        else:
            return BArr

    def BNorm(self,r:np.ndarray)->np.ndarray:
        #r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        #Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
        BVec=self.B_Vec(rEval)
        if len(r.shape)==1:
            return norm(BVec)
        else:
            return norm(BVec,axis=1)

    def BNorm_Gradient(self,r: np.ndarray,returnNorm: bool=False,dr: float=1e-7)->Union[np.ndarray,tuple]:
        #Return the gradient of the norm of the B field. use forward difference theorom
        #r: (N,3) vector of coordinates or (3) vector of coordinates.
        #returnNorm: Wether to return the norm as well as the gradient.
        #dr: step size
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
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
        if len(r.shape)==1:
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


class SegmentedBenderHalbach(HalbachLens):
    #a model of odd number lenses to represent the symmetry of the segmented bender. The inner lens represents the fully
    #symmetric field
    def __init__(self,rp:float,rb: float,UCAngle: float,Lm: float,numLenses: int=3,
                 M: float=M_Default,positiveAngleMagnetsOnly: bool=False):
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
            lens=HalbachLens(self.Lm,(self.magnetWidth,),(self.rp,),M=self.M)
            x=self.rb*np.cos(angleArr[i]) #x coordinate of center of lens
            z=self.rb*np.sin(angleArr[i]) #z coordinate of center of lense
            r0=np.asarray([x,0,z])
            theta=angleArr[i]
            lens.rotate(-theta) #my angle convention is unfortunately opposite what it should be here. positive theta
            #is clockwise about y axis in the xz plane looking from the negative side of y
            lens.position(r0)
            self.lensList.append(lens)

    def B_Vec(self,r: np.ndarray)->np.ndarray:
        #r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        #Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval0=np.asarray([r])
        else:
            rEval0=r.copy()
        BArr=np.zeros(rEval0.shape)
        for lens in self.lensList:
            rEval=lens._transform_r(rEval0)
            for layer in lens.layerList:
                BArr += lens._transform_Vector(layer.B(rEval))
        if len(r.shape)==1:
            return BArr[0]
        else:
            return BArr