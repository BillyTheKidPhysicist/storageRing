import time
import numpy as np
import numpy.linalg as npl
from interp3d import interp_3d
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numba
from profilehooks import profile

u0 = 4 * np.pi * 1e-7


def make_Interp_Functions(data):
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
    xIndices = np.argwhere(data[:, 0][:, None] == xArr)[:, 1]
    yIndices = np.argwhere(data[:, 1][:, None] == yArr)[:, 1]
    zIndices = np.argwhere(data[:, 2][:, None] == zArr)[:, 1]
    BGradxMatrix[xIndices, yIndices, zIndices] = data[:, 3]
    BGradyMatrix[xIndices, yIndices, zIndices] = data[:, 4]
    BGradzMatrix[xIndices, yIndices, zIndices] = data[:, 5]
    interpFx = interp_3d.Interp3D(BGradxMatrix, xArr, yArr, zArr)
    interpFy = interp_3d.Interp3D(BGradyMatrix, xArr, yArr, zArr)
    interpFz = interp_3d.Interp3D(BGradzMatrix, xArr, yArr, zArr)
    return interpFx, interpFy, interpFz


class RectangularPrism:
    #A right rectangular prism. Without any rotation the prism is oriented such that the 2 dimensions in the x,y plane
    #are equal, but the length, in the z plane, can be anything. not specified a cube is assumed.
    def __init__(self, width,length,M=1.15E6,MVec=np.asarray([1,0,0]),spherePerDim=6):
        #width: The width in the x,y plane without rotation, meters
        #lengthI: The length in the z plane without rotation, meters
        #M: magnetization, SI
        #MVec: direction of the magnetization vector
        #spherePerDim: number of spheres per transvers dimension in each cube. Longitudinal number will be this times
        #a factor
        # theta rotation is clockwise about y in my notation, originating at positive z
        # psi is counter clockwise around z
        self.width = width
        self.length=length
        self.M=M
        self.numSpherePerDim=spherePerDim
        if MVec[0]==0 and MVec[1]==0:
            self.Mpsi=0
        else:
            self.Mpsi=np.arctan2(MVec[1],MVec[0]) #magnetization psi direction.
        self.Mtheta=np.pi/2-np.arctan(MVec[2]/npl.norm(MVec[:2])) #magnetization theta direction


        self.n = None
        self.r = 0.0
        self.z = 0.0
        self.phi = 0.0
        self.r0 = np.asarray([self.r * np.cos(self.phi), self.r * np.sin(self.phi), self.z])
        self.theta = 0
        self.psi = 0
        self.sphereList = None

    def _build_RectangularPrism(self):
        # build a rectangular prism made of multiple spheres
        # make rotation matrices
        # theta rotation is clockwise about y in my notation. psi is coutner clokwise around z
        #rotation matrices
        Rtheta = np.asarray([[np.cos(self.theta), np.sin(self.theta)], [-np.sin(self.theta), np.cos(self.theta)]]) 
        Rpsi = np.asarray([[np.cos(self.psi), -np.sin(self.psi)], [np.sin(self.psi), np.cos(self.psi)]])
        rArr,radiusArr=self.generate_Positions_And_Radii_Ver2()
        self.sphereList = []
        i=0
        for r in rArr:
            sphere = Sphere(radiusInInches=radiusArr[i]/.0254,M=self.M)
            # rotate in theta
            r[[0, 2]] = Rtheta @ r[[0, 2]]
            r[:2] = Rpsi @ r[:2]
            sphere.r0 = r + self.r0
            sphere.orient(self.theta+self.Mtheta, self.psi+self.Mpsi)
            self.sphereList.append(sphere)
            i+=1

    def generate_Positions_And_Radii_Ver1(self):
        #create a model of a rectangular prism with a large sphere in the middle and spheres at each of the 8 corners
        #Returns an array of position vectors of the spheres, and an array of the radius of each sphere
        #only useful for a cube
        radiussphereCenter=self.width/2 #central sphere fits just inside the RectangularPrism
        volumeRemaining=self.width**3-(4/3)*np.pi*radiussphereCenter**3 #remaining volume of RectangularPrism after subtracting
        #center sphere
        radiusSphereCorners=((volumeRemaining/8)/(4*np.pi/3))**(1/3)

        # now do the spheres at the faces.
        # make list of starting positions
        r1 = np.asarray([0.0, 0.0, 0.0])
        r2 = np.asarray([self.width / 2, self.width / 2, self.length / 2])
        r3 = np.asarray([-self.width / 2, self.width / 2, self.length / 2])
        r4 = np.asarray([-self.width / 2, -self.width / 2, self.length / 2])
        r5 = np.asarray([self.width / 2, -self.width / 2, self.length / 2])
        r6=-r2
        r7=-r3
        r8=-r4
        r9=-r5
        radiusArr=np.asarray([radiussphereCenter])
        radiusArr=np.append(radiusArr,np.ones(8)*radiusSphereCorners)
        rList = [r1, r2, r3, r4, r5, r6, r7,r8,r9]
        return np.asarray(rList),radiusArr
    def generate_Positions_And_Radii_Ver2(self):
        #create a model of a rectangular prism with an array of spheres
        #Returns an array of position vectors of the spheres, and an array of the radius of each sphere
        numSpheresXYDim=self.numSpherePerDim #number of spheres in the XY dimension without rotation.
        numSpheresZDim=max(1,int(self.length*numSpheresXYDim/self.width)) #no less than 1
        numSpheres=numSpheresXYDim**2*numSpheresZDim
        radius=((self.width**2*self.length/numSpheres)/((4*np.pi/3)))**(1/3) #this is not the actual radius, a virtual
        # print(numSpheres)
        #radius to set the total magnetization of the sphere
        xySpacing=self.width/numSpheresXYDim  #the spacing between the spheres, including that I want a half gap at each
        #edge
        zSpacing=self.length/numSpheresZDim
        xyPosArr=np.linspace(-self.width/2+xySpacing/2,self.width/2-xySpacing/2,num=numSpheresXYDim)
        zPosArr=np.linspace(-self.length/2+zSpacing/2,self.length/2-zSpacing/2,num=numSpheresZDim)
        rArr=np.asarray(np.meshgrid(xyPosArr,xyPosArr,zPosArr)).T.reshape(-1,3)
        radiusArr=np.ones(rArr.shape[0])*radius

        return rArr,radiusArr

    def position(self, r=None, phi=None, z=None):
        # r: the distance from x,y=0 from the inner edge of the RectangularPrism


        if phi is not None:
            self.phi = phi
        if z is not None:
            self.z = z
        if r is not None:
            self.r = r + self.width / 2  # need to add the width of the RectangularPrism
        x = self.r * np.cos(self.phi)
        y = self.r * np.sin(self.phi)
        self.r0 = np.asarray([x, y, self.z])
        self._build_RectangularPrism()

    def orient(self, theta=0.0, psi=0.0):
        # tilt the RectangularPrism in spherical coordinates
        self.theta = theta
        self.psi = psi
        self._build_RectangularPrism()


    def B(self, r):
        assert len(r.shape)==2 and r.shape[1]==3
        BVec = np.zeros(r.shape)
        for sphere in self.sphereList:
            BVec += sphere.B(r)
        return BVec

    def B_Symmetric(self, r):
        # exploit the four fold symmetry of a 12 magnet hexapole
        arr = self.B(r)
        phi0 = self.phi
        self.position(phi=phi0 + np.pi / 2)
        arr += self.B(r)
        self.position(phi=phi0 + np.pi)
        arr += self.B(r)
        self.position(phi=phi0 + 3 * np.pi / 2)
        arr += self.B(r)
        self.position(phi=phi0)
        return arr


class Sphere:
    def __init__(self, radiusInInches=1.0 / 2,M=1.15e6):
        # angle: symmetry plane angle. There is a negative and positive one
        # radius: radius in inches
        #M: magnetization
        self.angle = None  # angular location of the magnet
        self.radius = radiusInInches * .0254  # meters. RADIUS!!!
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

    def position_Sphere(self, r=None, phi=None, z=None):
        if phi is not None:
            self.phi = phi
        if z is not None:
            self.z = z
        if r is not None:
            self.r = r
        x = self.r * np.cos(self.phi)
        y = self.r * np.sin(self.phi)
        self.r0 = np.asarray([x, y, self.z])

    def update_Size(self, radius):
        self.radius = radius
        self.volume = (4 * np.pi / 3) * self.radius ** 3
        M = 1.15e6  # magnetization density
        self.m0 = M * (4 / 3) * np.pi * self.radius ** 3  # dipole moment
        self.m = self.m0 * self.n  # vector sphere moment

    def orient(self, theta, psi):
        # tilt the sphere in spherical coordinates
        self.theta = theta
        self.psi = psi
        self.n = np.asarray([np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta)])
        self.m = self.m0 * self.n

    def vary_Amplitude(self, fact):
        self.m = fact * self.m0 * self.n  # vector sphere moment

    def BSlow(self, r):
        # magnetic field vector at a point in space
        # r: Coordinates of evaluation
        r = r - self.r0  # convert to difference vector
        if npl.norm(r) < self.radius:
            return np.nan
        rNorm = npl.norm(r)
        mrDot = np.sum(self.m * r)
        Bvec = 1e-7 * (3 * r * mrDot / rNorm ** 5 - self.m / rNorm ** 3)
        return Bvec

    def B(self, r):
        return self.B_NUMBA(r, self.r0, self.m)

    def B_Symmetric(self, r):
        arr = np.zeros(r.shape)
        arr += self.B(r)
        arr += self.B_Symetry(r, "counterclockwise", factors=0, fixedDipoleDirection=True, planeReflection=True)
        arr += self.B_Symetry(r, "counterclockwise", factors=1, fixedDipoleDirection=True)
        arr += self.B_Symetry(r, "counterclockwise", factors=1, fixedDipoleDirection=True, planeReflection=True)
        arr += self.B_Symetry(r, "counterclockwise", factors=2, fixedDipoleDirection=True)
        arr += self.B_Symetry(r, "counterclockwise", factors=2, fixedDipoleDirection=True, planeReflection=True)
        arr += self.B_Symetry(r, "counterclockwise", factors=3, fixedDipoleDirection=True)
        arr += self.B_Symetry(r, "counterclockwise", factors=3, fixedDipoleDirection=True, planeReflection=True)
        return arr
    #3.71
    @staticmethod
    @numba.njit(numba.float64[:,:](numba.float64[:,:],numba.float64[:],numba.float64[:]))
    def B_NUMBA(r, r0, m):
        r = r - r0  # convert to difference vector
        rNormTemp = np.sqrt(np.sum(r ** 2, axis=1))
        rNorm = np.empty((rNormTemp.shape[0], 1))
        rNorm[:, 0] = rNormTemp
        mrDotTemp = np.sum(m * r, axis=1)
        mrDot = np.empty((rNormTemp.shape[0], 1))
        mrDot[:, 0] = mrDotTemp
        Bvec = 1e-7 * (3 * r * mrDot / rNorm ** 5 - m / rNorm ** 3)
        return Bvec

    def B_Symetry(self, r, orientation, factors=1, flipDipole=False, angle=np.pi / 2, fixedDipoleDirection=False,
                  planeReflection=False):
        # orientation: String of "clockwise" or "counterclockwise" for orientation
        # factors: how many planes of symmetry to to reflect by. there are 6 total
        # fliSphere: wether to model the sphere as having the opposite orientation
        phi0 = np.arctan2(self.r0[1], self.r0[0])
        # choose the correct reflection angle.
        if orientation == 'clockwise':  # mirror across the clockwise plane
            phiSym = phi0 + (-angle) * factors  # angle to rotate the dipole position by
            deltaTheta = -angle * factors  # angle to rotate the dipole direction vector by
        elif orientation == 'counterclockwise':  # mirror across the counterclockwise plane
            phiSym = phi0 + angle * factors
            deltaTheta = angle * factors
        else:
            raise Exception('Improper orientation')
        xSym = npl.norm(self.r0[:2]) * np.cos(phiSym)
        ySym = npl.norm(self.r0[:2]) * np.sin(phiSym)
        rSym = np.asarray([xSym, ySym, self.r0[2]])
        mSym = self.m.copy()
        if fixedDipoleDirection == False:
            # rotate the dipole moment.
            MRot = np.array([[np.cos(deltaTheta), -np.sin(deltaTheta)], [np.sin(deltaTheta), np.cos(deltaTheta)]])
            mSym[:2] = MRot @ mSym[:2]
        if flipDipole == True:
            mSym = -mSym
        if planeReflection == True:  # another dipole on the other side of the z=0 line
            rSym[2] = -rSym[2]
        BVecArr = self.B_NUMBA(r, rSym, mSym)
        return BVecArr



class Layer:
    # class object for a layer of the magnet. Uses the RectangularPrism object
    def __init__(self, z,width,length,spherePerDim,M):
        # z: z coordinate of the layer, meter. The layer is in the x,y plane. This is the location of the center of the
        #width: width of the rectangular prism in the xy plane
        #length: length of the rectangular prism in the z axis
        #M: magnetization, SI
        #spherePerDim: number of spheres per transvers dimension in each cube. Longitudinal number will be this times
        #a factor
        self.z = z
        self.width=width
        self.length=length
        self.M=M
        self.numSpherePerDim=spherePerDim
        self.r1 = None  # radius values for the 3 kinds of magnets in each layer
        self.r2 = None  # radius values for the 3 kinds of magnets in each layer
        self.r3 = None  # radius values for the 3 kinds of magnets in each layer
        self.RectangularPrismsList = []  # list of RectangularPrisms

    def build(self, r1, r2, r3):
        # shape the layer. Create new RectangularPrisms for each reshpaing adds negligeable performance hit
        RectangularPrism1 = RectangularPrism(self.width,self.length,M=self.M,MVec=np.asarray([-1.0,0.0,0.0])
                                             ,spherePerDim=self.numSpherePerDim)
        RectangularPrism1.position(r=r1, phi=0, z=self.z)

        RectangularPrism2 = RectangularPrism(self.width,self.length,M=self.M,MVec=np.asarray([-1.0,0.0,0.0])
                                             ,spherePerDim=self.numSpherePerDim)
        RectangularPrism2.position(r=r2, phi=np.pi / 6, z=self.z)
        RectangularPrism2.orient(psi=(2*np.pi/3))

        RectangularPrism3 = RectangularPrism(self.width,self.length,M=self.M,MVec=np.asarray([-1.0,0.0,0.0])
                                             ,spherePerDim=self.numSpherePerDim)
        RectangularPrism3.position(r=r3, phi=-np.pi / 6, z=self.z)
        RectangularPrism3.orient(psi=-2*np.pi/3)

        self.RectangularPrismsList = [RectangularPrism1, RectangularPrism2, RectangularPrism3]
    def B(self, r):
        # r: Coordinates to evaluate at with dimension (N,3) where N is the number of evaluate points
        BArr=0
        for prism in self.RectangularPrismsList:
            BArr+= prism.B_Symmetric(r)
        return BArr

class HalbachLens:
    # class for a lens object. This is uses the layer object.
    # The lens will be positioned such that the center layer is at z=0. Can be tilted though

    def __init__(self, numLayers, width,rp,length=None,M=1.018e6,numSpherePerDim=2):
        #note that M is tuned to  for spherePerDim=4
        # numLayers: Number of layers
        # width: Width of each Rectangular Prism in the layer, meter
        #length: the length of each layer, meter. If None, each layer is built of cubes
        # rp: bore radius of every layer. If none, don't build.
        #Br: remnant flux density
        #M: magnetization.
        #spherePerDim: number of spheres per transvers dimension in each cube. Longitudinal number will be this times
        #a factor
        self.numLayers=numLayers
        self.width=width
        self.numSpherePerDim=numSpherePerDim
        if length is None:
            self.length=width
        else:
            self.length=length
        self.layerList=[] #object to hold layers
        self.layerArgs=[] #list of tuples of arguments for each layer
        self.M=M
        #euler angles. Order of operation is theta and then psi, done in the reference frame of the lens. In reality
        #the rotations are done to the evaluation point, then the resulting vector is rotated back
        self.theta=None #rotation, through the y axis
        self.r0=np.zeros(3) #location of center of magnet
        self.RInTheta=None #rotation matrix for the evaluation points to generate the final value
        self.ROutTheta=None #rotation matrix to rotate the vector out of the tilted frame to the original frame

        if numLayers==1:
            self.zArr=np.asarray([0])
        else:
            self.zArr=np.linspace(-(self.length/2+(self.numLayers-2)*self.length/2),
                                  (self.length/2+(self.numLayers-2)*self.length/2),num=self.numLayers)
        if rp is not None:
            self.set_Radius(rp)
    def position(self,r0):
        #position the magnet in space
        #r0: 3d position vector of the center of the magnet
        self.r0=r0
    def rotate(self,theta):
        #rotate the magnet about its center
        self.theta=theta
        self.RInTheta=np.asarray([[np.cos(-theta),-np.sin(-theta)],[np.sin(-theta),np.cos(-theta)]]) #to rotate about the
        #y axis. Since the magnet is rotated by theta, the evaluation points need to be rotated by -theta
        self.ROutTheta = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #to rotate the
        #vector out after evaluating with the rotated coordinates
    def _build(self):
        #build the lens.
        self.layerList=[]
        for i in range(self.numLayers):
            layer=Layer(self.zArr[i],self.width,self.length,M=self.M,spherePerDim=self.numSpherePerDim)
            layer.build(*self.layerArgs[i])
            self.layerList.append(layer)
    def update(self,layerArgs):
        #update the lens with new arguments. Requires all the arguments that the lens requires
        #layerArgs: List of tuple of arguments, one tuple per layer.
        self.layerArgs=layerArgs
        self._build()
    def set_Radius(self,r):
        #set the radius of the entire lens. There are 3 fundamental magnets in the halbach hexapole, and each can have a
        #unique radius, so they are set independently here. Allows for tunability in other applications
        #r: radius, meter
        self.layerArgs=[]
        for i in range(self.numLayers):
            self.layerArgs.append((r,r,r))
        self._build()
    def _transform_r(self,r):
        #to evaluate the field from tilted or translated magnets, the evaluation point is instead tilted or translated,
        #then the vector is rotated back. This function handle the rotation and translation of the evaluation points
        #r: rows of coordinates, shape (N,3). Where N is the number of evaluation points
        rNew=r.copy()
        rNew=rNew-np.ones(rNew.shape)*self.r0 #need to move the coordinates towards where the evaluation will take
        #place
        if self.theta is not None:
            for i in range(rNew.shape[0]):
                rNew[i][[0,2]]=rNew[i][[0,2]]@self.RInTheta

        return rNew
    def _transform_Vector(self,v):
        #to evaluate the field from tilted or translated magnets, the evaluation point is instead tilted or translated,
        #then the vector is rotated back. This function handles the rotation of the evaluated vector
        #v: rows of vectors, shape (N,3) where N is the number of vectors
        vNew=v.copy()
        if self.theta is not None:
            for i in range(vNew.shape[0]):
                vNew[i][[0,2]]=vNew[i][[0,2]]@self.ROutTheta
        return vNew

    def B_Vec(self,r):
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

    def BNorm(self,r):
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
    def BNorm_Gradient(self,r,returnNorm=False,dr=1e-7):
        #Return the gradient of the norm of the B field. use forward difference theorom
        #r: (N,3) vector of coordinates or (3) vector of coordinates.
        #returnNorm: Wether to return the norm as well as the gradient.
        #dr: step size
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
        BNormCenter=self.BNorm(rEval)
        def grad(index):
            coordb = rEval.copy()  # upper step
            coordb[:, index] += dr
            BNormB=self.BNorm(coordb)
            return (BNormB-BNormCenter)/dr
        BNormGradx=grad(0)
        BNormGrady=grad(1)
        BNormGradz=grad(2)
        if len(r.shape)==1:
            if returnNorm == True:
                return np.asarray([BNormGradx[0], BNormGrady[0], BNormGradz[0]]),BNormCenter[0]
            else:
                return np.asarray([BNormGradx[0],BNormGrady[0],BNormGradz[0]])
        else:
            if returnNorm==True:
                return np.column_stack((BNormGradx, BNormGrady, BNormGradz)),BNormCenter
            else:
                return np.column_stack((BNormGradx,BNormGrady,BNormGradz))
class DoubeLayerHalbachLens:
    #model of a halbach lens that is composed of two layers. Here they are taken to have the same magnets for the
    #inner and outer layer. This is simply modeled as two concentric halbach lenses
    def __init__(self, numLayers, width, rp, length=None, M=1.03e6, spherePerDim=4):
        lensInner=HalbachLens(numLayers,width,rp,length=length,M=M,spherePerDim=spherePerDim)
        rpOuterLayer=rp+width
        lensOuter=HalbachLens(numLayers,1.5*width,rpOuterLayer,length=length,M=M,spherePerDim=spherePerDim)
        self.lensList=[lensInner,lensOuter]
    def B_Vec(self,r):
        #r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        #Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
        BArr=np.zeros(rEval.shape)
        for lens in self.lensList:
            for layer in lens.layerList:
                BArr+=layer.B(rEval)
        if len(r.shape)==1:
            return BArr[0]
        else:
            return BArr
    def BNorm(self,r):
       return HalbachLens.BNorm(self,r)
    def BNorm_Gradient(self,r):
        return HalbachLens.BNorm_Gradient(self,r)


class SegmentedBenderHalbach(HalbachLens):
    #a model of odd number lenses to represent the symmetry of the segmented bender. The inner lens represents the fully
    #symmetric field
    def __init__(self,rp,rb,UCAngle,Lm,numLenses=3,magnetWidth=None,M=1.03e6,inputOnly=False):
        self.rp=rp #radius of bore of magnet, ie to the pole
        self.rb=rb #bending radius
        self.UCAngle=UCAngle #unit cell angle of a HALF single magnet, ie HALF the bending angle of a single magnet. It
        #is called the unit cell because obviously one only needs to use half the magnet and can use symmetry to
        #solve the rest
        self.Lm=Lm #length of single magnet
        self.M=M #magnetization, SI
        self.inputOnly=inputOnly #wether to model only the input of the bender, ie no magnets being added below the z=0
        #line, except for the magnet right at z=0
        if magnetWidth==None:
            self.magnetWidth=rp * np.tan(2 * np.pi / 24) * 2 #set to size that exactly fits
        else:
            self.magnetWidth=magnetWidth
        self.numLenses=numLenses #number of lenses in the model
        self.lensList=None #list to hold lenses
        self._build()
    def _build(self):
        self.lensList=[]
        if self.numLenses==1:
            angleArr=np.asarray([0.0])
        else:
            angleArr=np.linspace(-2*self.UCAngle*(self.numLenses-1)/2,2*self.UCAngle*(self.numLenses-1)/2,num=self.numLenses)
        if self.inputOnly==True:
            angleArr=angleArr[(self.numLenses-1)//2:]
        for i in range(angleArr.shape[0]):
            lens=HalbachLens(1,self.magnetWidth,self.rp,length=self.Lm,M=self.M)
            x=self.rb*np.cos(angleArr[i]) #x coordinate of center of lens
            z=self.rb*np.sin(angleArr[i]) #z coordinate of center of lense
            r0=np.asarray([x,0,z])
            theta=angleArr[i]
            lens.rotate(-theta) #my angle convention is unfortunately opposite what it should be here. positive theta
            #is clockwise about y axis in the xz plane looking from the negative side of y
            lens.position(r0)
            self.lensList.append(lens)

    def B_Vec(self,r):
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

# lensFringe = SegmentedBenderHalbachLensFieldGenerator(.01, 1.0, self.ucAng, self.Lm,
#                                                                           numLenses=3,inputOnly=True)


# dxArr=np.logspace(-5,-15,num=30)
# errorList=[]
# for dx in dxArr:
#     lens=HalbachLens(1,.0254,.05,length=.1)
#     testArr=npl.norm(lens.BNorm_Gradient(coords,dr=dx,method=1),axis=1)
#     error=1e2*np.sum(np.abs(testArr-sampleArr))/np.sum(sampleArr)
#     errorList.append(error)
#     print(dx,error)
#
# plt.title('Forward difference accuracy compared to central difference \n Central difference stepsize is 1e-6')
# plt.xlabel('Step size ,m')
# plt.ylabel("percent difference")
# plt.loglog(dxArr,errorList,marker='o')
# plt.grid()
# plt.show()




















# import scipy.optimize as spo
# numSpheresArr=np.asarray([1,2,3,4,5])
# costList=[]
# for numSpheres in numSpheresArr:
#     print('---------',numSpheres)
#     def cost(M):
#         lens=HalbachLens(1,.0254,.05,4*.05,M=M[0],spherePerDim=numSpheres)
#         BArr=lens.BNorm(particleCoords)
#         error=1e2*np.sum(np.abs(BArr-vals))/np.sum(vals)
#         # print(M[0],error)
#         return error
#     sol=spo.minimize(cost,np.asarray([1e6]))
#     print(sol)
#     costList.append(sol.fun)
#
# plt.title("Model accuracy compared to COMSOL versus sphere number")
# plt.xlabel('Number of spheres along xy dimensions')
# plt.ylabel('Percent error')
# plt.semilogy(numSpheresArr,costList)
# plt.grid()
# plt.show()

# lens=HalbachLens(1,.0254,.05,4*.05)
# MArr=np.linspace(.95,1.05,num=30)*1e6
# errorList=[]
# for M in MArr:
#     lens=HalbachLens(1,.0254,.05,4*.05,M=M)
#     BArr=lens.BNorm(particleCoords)
#     error=1e2*np.sum(np.abs(BArr-vals))/np.sum(vals)
#     errorList.append(error)
#     print(BArr.sum(),vals.sum())
# plt.plot(MArr,errorList)
# plt.show()


#
# L0=.5
# lens=HalbachLens(1,.0254,.05,length=L0)
# num=40
# rMax=.04
# posArr=np.linspace(-rMax,rMax,num=num)
# coordsList=[]
# for x in posArr:
#     for y in posArr:
#         if np.sqrt(x**2+y**2)<rMax:
#             coordsList.append([x,y])
# coords=np.asarray(coordsList)
# planeCoords=np.column_stack((coords,np.zeros(coords.shape[0])))
# zArr=np.linspace(3*.05,L0/2+5*.05,num=30)
# resList=[]
# sumList=[]
# for z in zArr:
#     planeCoords[:,2]=z
#     BSum = npl.norm(lens.BNorm_Gradient(planeCoords), axis=1).sum()
#     resList.append(BSum)
#     print(z,sum(resList))
#     sumList.append(sum(resList))
# resArr=np.asarray(sumList)
# resArr=np.abs(resArr-resArr[-1])
# resArr=100*resArr/resArr[0]
# plt.title('PercentField remainig of cumulative sum as a \n function of distance along magnet')
# plt.semilogy(((zArr-L0/2)/.05)[:-1],resArr[:-1],marker='x')
# plt.grid()
# plt.xlabel('Distance from magnet edge, multiple of bore radius')
# plt.ylabel('Percent of final value')
# plt.show()




# num=40
# rMax=.045
# posArr=np.linspace(-rMax,rMax,num=num)
# coordsList=[]
# posArrz=np.linspace(-.1,.1,num=num)
# for x in posArr:
#     for y in posArr:
#         for z in posArrz:
#             if np.sqrt(x**2+y**2)<rMax:
#                 coordsList.append([x,y,z])
# coords=np.asarray(coordsList)
#
# sphereNumArr=np.arange(2,12)
# resList=[]
# tList=[]
# for sphereNum in sphereNumArr:
#     print(sphereNum)
#     t=time.time()
#     lens=HalbachLens(1,.0254,.05,length=5*.0254,spherePerDim=sphereNum)
#     BArr=lens.BNorm(coords)
#     tList.append(time.time()-t)
#     resList.append(np.sum(BArr))
#
# plt.title('Time to solve')
# plt.xlabel('Number of spheres along xy dimensions')
# plt.ylabel('Time,seconds')
# plt.plot(sphereNumArr,tList,marker='x')
# plt.show()
#
# resArr=np.asarray(resList)
# resArr=100*np.abs(resArr-resArr[-1])/resArr[-1]
# plt.title('Error from \'actual\' model. \n percent error from last value over sum of values from test points')
# plt.grid()
# plt.xlabel('Number of spheres along xy dimensions')
# plt.ylabel('Percent error')
# plt.semilogy(sphereNumArr[:-1],resArr[:-1],marker='x')
# plt.show()
#
#
# num=40
# rMax=.04
# posArr=np.linspace(-rMax,rMax,num=num)
# coordsList=[]
# for x in posArr:
#     for y in posArr:
#         if np.sqrt(x**2+y**2)<rMax:
#             coordsList.append([x,y])
# planeCoords=np.asarray(coordsList)
# planeCoords=np.column_stack((planeCoords,np.zeros(planeCoords.shape[0])))
#
#
# # FracArr = np.linspace(1, 10, num=25)
# FracArr=np.arange(1,10.5,.5)
# FracArr = np.append(FracArr, 30)
# resList = []
# rp=.05
# for Frac in FracArr:
#     print(Frac)
#     lens = HalbachLens(1, .0254, rp, length=rp * Frac)
#     BNorm = npl.norm(lens.BNorm_Gradient(planeCoords),axis=1)
#     resList.append(np.sum(BNorm))
# resArr = np.asarray(resList)
# resArr = 100 * np.abs(resArr[-1] - resArr) / resArr[-1]
# plt.title('Percent difference of total field values from \'actual\' \n value at z=0')
# plt.xlabel('Magnet length as multiple of bore radius')
# plt.ylabel('Percent difference')
# plt.semilogy(FracArr[:-1], resArr[:-1],marker='x')
# plt.grid()
# plt.show()