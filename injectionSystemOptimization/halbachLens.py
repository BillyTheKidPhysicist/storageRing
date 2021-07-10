import time
import numpy as np
import numpy.linalg as npl
from interp3d import interp_3d
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


class Cube:
    def __init__(self, widthInInches=1.0,M=1.15E6):
        #M: magnetization
        self.width = widthInInches * .0254  # convert to meter
        self.n = None
        self.r = 0.0
        self.z = 0.0
        self.phi = 0.0
        self.r0 = np.asarray([self.r * np.cos(self.phi), self.r * np.sin(self.phi), self.z])
        self.theta = 0
        self.psi = 0
        self.M=M
        self.sphereList = None

    def _build_Cube(self):
        # build a cube made of 9 spheres, one at the center, and one at each corner.
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
            sphere.orient(self.theta, self.psi)
            self.sphereList.append(sphere)
            i+=1
    def generate_Positions_And_Radii_Ver1(self):
        #create a model of a cube with a large sphere in the middle and spheres at each of the 8 corners
        #Returns an array of position vectors of the spheres, and an array of the radius of each sphere
        radiussphereCenter=self.width/2 #central sphere fits just inside the cube
        volumeRemaining=self.width**3-(4/3)*np.pi*radiussphereCenter**3 #remaining volume of cube after subtracting
        #center sphere
        radiusSphereCorners=((volumeRemaining/8)/(4*np.pi/3))**(1/3)

        # now do the spheres at the faces.
        # make list of starting positions
        r1 = np.asarray([0.0, 0.0, 0.0])
        r2 = np.asarray([self.width / 2, self.width / 2, self.width / 2])
        r3 = np.asarray([-self.width / 2, self.width / 2, self.width / 2])
        r4 = np.asarray([-self.width / 2, -self.width / 2, self.width / 2])
        r5 = np.asarray([self.width / 2, -self.width / 2, self.width / 2])
        r6=-r2
        r7=-r3
        r8=-r4
        r9=-r5
        radiusArr=np.asarray([radiussphereCenter])
        radiusArr=np.append(radiusArr,np.ones(8)*radiusSphereCorners)
        rList = [r1, r2, r3, r4, r5, r6, r7,r8,r9]
        return np.asarray(rList),radiusArr
    def generate_Positions_And_Radii_Ver2(self):
        #create a model of a cube with an array of spheres
        #Returns an array of position vectors of the spheres, and an array of the radius of each sphere
        numSphersDim=6 #number of spheres in each dimension. total spheres is this cubed
        radius=((self.width**3/numSphersDim**3)/((4*np.pi/3)))**(1/3)
        posArr=np.linspace(-self.width/2,self.width/2,num=numSphersDim)
        rArr=np.asarray(np.meshgrid(posArr,posArr,posArr)).T.reshape(-1,3)
        radiusArr=np.ones(rArr.shape[0])*radius
        return rArr,radiusArr

    def position(self, r=None, phi=None, z=None):
        # r: the distance from x,y=0 from the inner edge of the cube
        if phi is not None:
            self.phi = phi
        if z is not None:
            self.z = z
        if r is not None:
            self.r = r + self.width / 2  # need to add the width of the cube
        x = self.r * np.cos(self.phi)
        y = self.r * np.sin(self.phi)
        self.r0 = np.asarray([x, y, self.z])
        self._build_Cube()

    def orient(self, theta, psi):
        # tilt the cube in spherical coordinates
        self.theta = theta
        self.psi = psi
        self._build_Cube()

    def B(self, r):
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

    def update_Size(self, radiusNewInches):
        self.radius = radiusNewInches * .0254
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
    # class object for a layer of the magnet. Uses the cube object
    def __init__(self, z,M=1.15e6):
        # z: z coordinate of the layer, meter. The layer is in the x,y plane. This is the location of the center of the
        #M: magnetization
        self.z = z
        self.M=M
        self.r1 = None  # radius values for the 3 kinds of magnets in each layer
        self.r2 = None  # radius values for the 3 kinds of magnets in each layer
        self.r3 = None  # radius values for the 3 kinds of magnets in each layer
        self.cubesList = None  # list of cubes

    def build(self, r1, r2, r3):
        # shape the layer. Create new cubes for each reshpaing adds negligeable performance hit
        cube1 = Cube(M=self.M)
        cube1.position(r=r1, phi=0, z=self.z)
        cube1.orient(-np.pi / 2, 0)

        cube2 = Cube(M=self.M)
        cube2.position(r=r2, phi=np.pi / 6, z=self.z)
        cube2.orient(np.pi / 2, -2 * np.pi / 6)
        #
        cube3 = Cube(M=self.M)
        cube3.position(r=r3, phi=-np.pi / 6, z=self.z)
        cube3.orient(np.pi / 2, 2 * np.pi / 6)
        self.cubesList = [cube1, cube2, cube3]

    def B(self, r):
        # r: Coordinates to evaluate at with dimension (N,3) where N is the number of evaluate points
        BArr = self.cubesList[0].B_Symmetric(r)
        BArr += self.cubesList[1].B_Symmetric(r)
        BArr += self.cubesList[2].B_Symmetric(r)
        return BArr

class Lens:
    # class for a lens object. This is uses the layer object.
    # The lens will be positioned such that the center layer is at z=0
    def __init__(self, numLayers, width,r=None,M=1.03e6):
        # numLayers: Number of layers
        # width: Width of each cube in the layer, inches
        # r: radius of each layer. If none, don't build.
        #Br: remnant flux density
        #M: magnetization.
        self.numLayers=numLayers
        self.width=width
        self.layerList=[] #object to hold layers
        self.layerArgs=[] #list of tuples of arguments for each layer
        self.M=M

        if numLayers==1:
            self.zArr=np.asarray([0])
        else:
            self.zArr=np.linspace(-(self.width/2+(self.numLayers-2)*self.width/2),
                                  (self.width/2+(self.numLayers-2)*self.width/2),num=self.numLayers)
            self.zArr=self.zArr*.0254 #convert to meters
        if r is not None:
            self.set_Radius(r)
            self._build()
    def _build(self):
        self.layerList=[]
        for i in range(self.numLayers):
            layer=Layer(self.zArr[i],M=self.M)
            layer.build(*self.layerArgs[i])
            self.layerList.append(layer)
    def update(self,layerArgs):
        #update the lens with new arguments. Requires all the arguments that the lens requires
        #layerArgs: List of tuple of arguments, one tuple per layer.
        self.layerArgs=layerArgs
        self._build()
    def set_Radius(self,r):
        #set the radius of the entire lens
        #r: radius, meter
        self.layerArgs=[]
        for i in range(self.numLayers):
            self.layerArgs.append((r,r,r))
        self._build()
    def B_Vec(self,r):
        #r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        #Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
        BArr=np.zeros(rEval.shape)
        for layer in self.layerList:
            BArr+=layer.B(rEval)
        if len(r.shape)==1:
            return BArr[0]
        else:
            return BArr
    def B_Norm(self,r):
        #r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        #Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
        BVec=self.B_Vec(rEval)
        if len(BVec.shape)==1:
            return npl.norm(BVec)
        else:
            return npl.norm(BVec,axis=1)
    def B_Grad(self,r):
        #Return the gradient of the norm of the B field. use central difference theorom
        #r: (N,3) vector of coordinates or (3) vector of coordinates.
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
        def grad(index):
            dr=1e-6 #step size to find the derivative
            coordb = rEval.copy()  # upper step
            coordb[:, index] += dr
            coorda = rEval.copy()  # lower step
            coorda[:, index] += -dr
            return (self.B_Norm(coordb)-self.B_Norm(coorda))/dr
        BGradx=grad(0)
        BGrady=grad(1)
        BGradz=grad(2)
        if len(r.shape)==1:
            return np.asarray([BGradx[0],BGrady[0],BGradz[0]])
        else:
            return np.column_stack((BGradx,BGrady,BGradz))