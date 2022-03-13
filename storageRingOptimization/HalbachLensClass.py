from numbers import Number
import time
from constants import MAGNETIC_PERMEABILITY
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import numba
from scipy.spatial.transform import Rotation
from magpylib.magnet import Box
from typing import Union

M_Default=1.018E6



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

class magpy_Prism:

    def __init__(self,x0,y0,z0,phi,width,length,M):
        self.magnet=self._build_magpy_Prism(x0,y0,z0,phi,width,length,M)

    def _build_magpy_Prism(self, x0: float, y0: float, z0: float, phi: float, width: float, length: float,M)->Box:
        dimensions_mm = 1e3 * np.asarray([width, width, length])
        position_mm = 1e3 * np.asarray([x0, y0, z0])
        R = Rotation.from_rotvec([0, 0, phi])
        M_MagpyUnit = 1e3 * M * MAGNETIC_PERMEABILITY  # uses units of mT for magnetization.
        magnet = Box(magnetization=(M_MagpyUnit, 0, 0.0), dimension=dimensions_mm, position=position_mm,
                     orientation=R)
        return magnet

    def B(self,evalCoords: np.ndarray)->np.ndarray:
        evalCoords_mm = 1e3 * evalCoords
        BVec_mT = self.magnet.getB(evalCoords_mm)  # need to convert to mm
        BVec_T = 1e-3 * BVec_mT  # convert to tesla from milliTesla
        return BVec_T


class Sphere:

    def __init__(self, radius: float,M: float=M_Default):
        # angle: symmetry plane angle. There is a negative and positive one
        # radius: radius in inches
        #M: magnetization
        assert radius>0 and M>0
        self.angle = None  # angular location of the magnet
        self.radius =radius
        self.volume=(4*np.pi/3)*self.radius**3 #m^3
        self.m0 = M * self.volume  # dipole moment
        self.r0 = None  # location of sphere
        self.n = None  # orientation
        self.m = None  # vector sphere moment
        self.phi = None  # phi position
        self.theta = None  # orientation of dipole. From local z axis
        self.psi = None  # orientation of dipole. in local xy plane
        self.z = None
        self.r = None

    def position_Sphere(self, r: float=None, phi: float=None, z: float=None)->None:
        if phi is not None:
            self.phi = phi
        if z is not None:
            self.z = z
        if r is not None:
            self.r = r
        assert not None in (phi,z,r)
        x = self.r * np.cos(self.phi)
        y = self.r * np.sin(self.phi)
        self.r0 = np.asarray([x, y, self.z])

    def update_Size(self, radius: float)-> None:
        self.radius = radius
        self.volume = (4 * np.pi / 3) * self.radius ** 3
        M = 1.15e6  # magnetization density
        self.m0 = M * (4 / 3) * np.pi * self.radius ** 3  # dipole moment
        self.m = self.m0 * self.n  # vector sphere moment

    def orient(self, theta: float, psi: float)->None:
        # tilt the sphere in spherical coordinates. These are in the lab frame, there is no sphere frame
        #theta,psi =(pi/2,0) is along +x
        #theta,psi=(pi/2,pi/2) is along +y
        #theta,psi=(0.0,anything) is along +z
        self.theta = theta
        self.psi = psi
        self.n = np.asarray([np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta)])#x,y,z
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

    def __init__(self, width: float,length: float,M: float=M_Default):
        #width: The width in the x,y plane without rotation, meters
        #lengthI: The length in the z plane without rotation, meters
        #M: magnetization, SI
        #MVec: direction of the magnetization vector
        # theta rotation is clockwise about y in my notation, originating at positive z
        # psi is counter clockwise around z
        assert width>0.0 and length >0.0  and M>0
        self.width = width
        self.length=length
        self.M=M
        self.r = None #r in cylinderical coordinates, equals sqrt(x**2+y**2)
        self.x=None #cartesian
        self.y=None #cartesian
        self.z = None #cartesian
        self.phi = None #position angle in polar coordinates
        self.r0 = None
        self.theta = None # rotation about body z axis
        self.magnet=None #holds the magpylib element

    def place(self, r:np.ndarray, theta: float, z: float,phi: float)->None:
        #currentyl no feature for tilting body, only rotating about z axis
        # r: the distance from x,y=0 from the center of the RectangularPrism
        #phi: position in polar/cylinderical coordinates
        #z: vertical distance in cylinderical coordinates
        #theta: rotation rectuangular prism about body axis
        self.r,self.theta,self.z,self.phi=r,theta,z,phi
        self.x = self.r * np.cos(self.theta)
        self.y = self.r * np.sin(self.theta)
        self.r0 = np.asarray([self.x, self.y, self.z])
        self.magnet=magpy_Prism(self.x,self.y,self.z,self.phi,self.width,self.length,self.M)

    def _build_RectangularPrism(self)->None:
        # build a rectangular prism made of multiple spheres
        # make rotation matrices
        # theta rotation is clockwise about y in my notation. psi is coutner clokwise around z
        #rotation matrices
        dimensions_mm=1e3*np.asarray([self.width,self.width,self.length])
        position_mm=1e3*np.asarray([self.x,self.y,self.z])
        R = Rotation.from_rotvec([0, 0, self.phi])
        M_MagpyUnit=1e3*self.M*MAGNETIC_PERMEABILITY #uses units of mT for magnetization.
        self.magnet=Box(magnetization=(M_MagpyUnit,0,0.0),dimension=dimensions_mm,position=position_mm,orientation=R)

    def B(self, evalCoords: np.ndarray)->None:
        # (n,3) array to evaluate coords at. SI
        assert len(evalCoords.shape)==2 and evalCoords.shape[1]==3
        BVec=self.magnet.B(evalCoords)
        if len(BVec.shape)==1:
            BVec=np.asarray([BVec])
        return BVec

    def B_Shim(self, r:np.ndarray, planeSymmetry: bool=True,negativeSymmetry: bool=True,
               rotationAngle: float=np.pi/3)->np.ndarray:
        # a single magnet actually represents 12 magnet
        # r: array of N position vectors to get field at. Shape (N,3)
        # planeSymmetry: Wether to exploit z symmetry or not
        # plt.quiver(self.r0[0],self.r0[1],np.cos(self.phi),np.sin(self.phi),color='r')
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
        phiSymm=rotAngle
        M_Rot = np.array([[np.cos(rotAngle), -np.sin(rotAngle)], [np.sin(rotAngle), np.cos(rotAngle)]])
        r0Sym=self.r0.copy()
        r0Sym[:2]=M_Rot@r0Sym[:2]
        if negativeSymmetry==True:
            phiSymm+=rotations*np.pi
        if planeSymmetry==True:
            r0Sym[2] = -r0Sym[2]
        # plt.quiver(r0Sym[0], r0Sym[1], np.cos(self.phi+phiSymm),np.sin(self.phi+phiSymm))
        magnet=magpy_Prism(*r0Sym,self.phi+phiSymm,self.width,self.length,self.M)
        B=magnet.B(r)
        return B

class Layer:
    # class object for a layer of the magnet. Uses the RectangularPrism object

    def __init__(self, z: float, width: float, length: float,rp: Union[float,tuple], M: float=M_Default,phase:float=0.0):
        # z: z coordinate of the layer, meter. The layer is in the x,y plane. This is the location of the center
        # width: width of the rectangular prism in the xy plane
        # length: length of the rectangular prism in the z axis
        # M: magnetization, SI
        # spherePerDim: number of spheres per transvers dimension in each cube. Longitudinal number will be this times
        # a factor
        maxLayerMagnets=12
        if isinstance(rp,tuple):
            assert len(rp)<=maxLayerMagnets
        else:
            assert rp>0.0
            rp=(rp,)
        assert width > 0.0 and length > 0.0  and M > 0.0
        self.z = z
        self.width = width
        self.length = length
        self.rp=rp
        self.M = M
        self.phase=phase #this will rotate the layer about z this amount
        self.RectangularPrismsList = []  # list of RectangularPrisms
        self.build()

    def build(self)->None:
        # build the elements that form the layer. The 'home' magnet's center is located at x=r0+width/2,y=0, and its
        #magnetization points along positive x
        thetaArr = np.linspace(0, 2 * np.pi, 12, endpoint=False)+self.phase  # location of 12 magnets.
        thetaArr=thetaArr.reshape(-1,len(self.rp)).T
        phiArr =self.phase+ np.pi + np.arange(0, 12) * 2 * np.pi / 3 #direction of magnetization
        phiArr=phiArr.reshape(-1,len(self.rp)).T
        for r,r_phi,r_theta in zip(self.rp,phiArr,thetaArr):
            for phi,theta in zip(r_phi,r_theta):
                # x,y=r*np.cos(theta),r*np.sin(theta)
                # plt.quiver(x,y,np.cos(phi),np.sin(phi))
                magnet = RectangularPrism(self.width, self.length, self.M)
                rCenter=r+self.width/2
                magnet.place(rCenter, theta, self.z, phi)
                self.RectangularPrismsList.append(magnet)
        # plt.gca().set_aspect('equal')
        # plt.show()

    def B(self, r: np.ndarray)->np.ndarray:
        # r: Coordinates to evaluate at with dimension (N,3) where N is the number of evaluate points
        assert len(self.RectangularPrismsList)>0
        BArr=np.zeros(r.shape)
        for prism in self.RectangularPrismsList:
            BArr += prism.B(r)
        return BArr


class HalbachLens:
    # class for a lens object. This is uses the layer object.
    # The lens will be positioned such that the center layer is at z=0. Can be tilted though

    def __init__(self,length:float,width:tuple,rp:tuple,M:float=M_Default):
        #note that M is tuned to  for spherePerDim=4
        # numLayers: Number of layers
        # width: Width of each Rectangular Prism in the layer, meter
        #length: the length of each layer, meter. If None, each layer is built of cubes
        # rp: bore radius of concentric layers
        #Br: remnant flux density
        #M: magnetization.
        assert all(isinstance(variable,tuple) for variable in (rp,width))
        assert all(val>0.0 for val in [*rp,*width])
        assert len(width)==len(rp)
        assert length>0.0 and M>0.0
        self.width=width
        self.rp=rp
        if length is None:
            self.length=width
        else:
            self.length=length
        self.layerList=[] #object to hold layers
        self.M=M
        #euler angles. Order of operation is theta and then psi, done in the reference frame of the lens. In reality
        #the rotations are done to the evaluation point, then the resulting vector is rotated back
        self.theta=None #rotation, through the y axis
        self.r0=np.zeros(3) #location of center of magnet
        self.RInTheta=None #rotation matrix for the evaluation points to generate the final value
        self.ROutTheta=None #rotation matrix to rotate the vector out of the tilted frame to the original frame

        self._build()

    def position(self,r0: np.ndarray)->None:
        #position the magnet in space
        #r0: 3d position vector of the center of the magnet
        self.r0=r0

    def rotate(self,theta: float)->None:
        #rotate the magnet about its center
        self.theta=theta
        self.RInTheta=np.asarray([[np.cos(-theta),-np.sin(-theta)],[np.sin(-theta),np.cos(-theta)]]) #to rotate about the
        #y axis. Since the magnet is rotated by theta, the evaluation points need to be rotated by -theta
        self.ROutTheta = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #to rotate the
        #vector out after evaluating with the rotated coordinates

    def _build(self)->None:
        #build the lens.
        self.layerList=[]
        z=0.0
        for radiusLayer,widthLayer in zip(self.rp,self.width):
            layer=Layer(z,widthLayer,self.length,radiusLayer,M=self.M)
            self.layerList.append(layer)

    def _transform_r(self,r:np.ndarray)->np.ndarray:
        #to evaluate the field from tilted or translated magnets, the evaluation point is instead tilted or translated,
        #then the vector is rotated back. This function handle the rotation and translation of the evaluation points
        #r: rows of coordinates, shape (N,3). Where N is the number of evaluation points
        if self.theta is not None:
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
        if self.theta is not None:
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
            return npl.norm(BVec)
        else:
            return npl.norm(BVec,axis=1)

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
    def __init__(self,rp:float,rb: float,UCAngle: np.ndarray,Lm: float,numLenses: int=3,
                 M: np.ndarray=1.03e6,positiveAngleMagnetsOnly: bool=False):
        assert all(isinstance(value, Number) for value in (rp,rb,UCAngle,Lm))
        self.rp=rp #radius of bore of magnet, ie to the pole
        self.rb=rb #bending radius
        self.UCAngle=UCAngle #unit cell angle of a HALF single magnet, ie HALF the bending angle of a single magnet. It
        #is called the unit cell because obviously one only needs to use half the magnet and can use symmetry to
        #solve the rest
        self.Lm=Lm #length of single magnet
        self.M=M #magnetization, SI
        self.positiveAngleMagnetsOnly=positiveAngleMagnetsOnly #This is used to model the cap amgnet, and the first full
        #segment. No magnets can be below z=0, but a magnet can be right at z=0. Very different behavious wether negative
        #or positive
        self.magnetWidth=rp * np.tan(2 * np.pi / 24) * 2 #set to size that exactly fits
        self.numLenses=numLenses #number of lenses in the model
        self.lensList=None #list to hold lenses
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
