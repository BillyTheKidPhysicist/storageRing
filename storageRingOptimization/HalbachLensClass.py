import time
import numpy as np
import numpy.linalg as npl
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numba
# from profilehooks import profile

u0 = 4 * np.pi * 1e-7


@numba.njit(numba.float64[:,:](numba.float64[:,:],numba.float64[:],numba.float64[:]))
def B_NUMBA(r,r0,m):
    r=r-r0  # convert to difference vector
    rNormTemp=np.sqrt(np.sum(r**2,axis=1))
    rNorm=np.empty((rNormTemp.shape[0],1))
    rNorm[:,0]=rNormTemp
    mrDotTemp=np.sum(m*r,axis=1)
    mrDot=np.empty((rNormTemp.shape[0],1))
    mrDot[:,0]=mrDotTemp
    Bvec=1e-7*(3*r*mrDot/rNorm**5-m/rNorm**3)
    return Bvec


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
        return B_NUMBA(r, self.r0, self.m)

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
        BVecArr = B_NUMBA(r, rSym, mSym)
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
    # @profile
    def _transform_r(self,r):
        #to evaluate the field from tilted or translated magnets, the evaluation point is instead tilted or translated,
        #then the vector is rotated back. This function handle the rotation and translation of the evaluation points
        #r: rows of coordinates, shape (N,3). Where N is the number of evaluation points
        return self._transform_r_NUMBA(r,self.r0,self.theta,self.ROutTheta)
    @staticmethod
    @numba.njit()
    def _transform_r_NUMBA(r,r0,theta,ROutTheta):
        #todo: something is wrong here with the matrix multiplication
        rNew=r.copy()
        rNew=rNew-np.ones(rNew.shape)*r0 #need to move the coordinates towards where the evaluation will take
        #place
        if theta is not None:
            for i in range(rNew.shape[0]):
                rx=rNew[i][0]
                rz=rNew[i][2]
                rNew[i][0]=ROutTheta[0,0]*rx+ROutTheta[0,1]*rz
                rNew[i][2]=ROutTheta[1,0]*rx+ROutTheta[1,1]*rz
        return rNew
    # @profile
    def _transform_Vector(self,v):
        #todo: something seems wrong here with the matrix multiplixation
        #to evaluate the field from tilted or translated magnets, the evaluation point is instead tilted or translated,
        #then the vector is rotated back. This function handles the rotation of the evaluated vector
        #v: rows of vectors, shape (N,3) where N is the number of vectors
        return self._transform_Vector_NUMBA(v,self.theta,self.RInTheta)
        vNew=v.copy()
        if self.theta is not None:
            for i in range(vNew.shape[0]):
                vNew[i][[0,2]]=vNew[i][[0,2]]@self.ROutTheta
        return vNew
    @staticmethod
    @numba.njit()
    def _transform_Vector_NUMBA(v,theta,RInTheta):
        #todo: remove the none catches
        vNew=v.copy()
        if theta is not None:
            for i in range(vNew.shape[0]):
                vx=vNew[i][0]
                vz=vNew[i][2]
                vNew[i][0]=RInTheta[0,0]*vx+RInTheta[0,1]*vz
                vNew[i][2]=RInTheta[1,0]*vx+RInTheta[1,1]*vz
                # vNew[i][[0,2]]=RInTheta@vNew[i][[0,2]]
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
    def __init__(self,rp,rb,UCAngle,Lm,numLenses=3,magnetWidth=None,M=1.03e6,positiveAngleMagnetsOnly=False):
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
            if self.positiveAngleMagnetsOnly==True:
                raise Exception('Not applicable with only 1 magnet')
            angleArr=np.asarray([0.0])
        else:
            angleArr=np.linspace(-2*self.UCAngle*(self.numLenses-1)/2,2*self.UCAngle*(self.numLenses-1)/2,num=self.numLenses)
        if self.positiveAngleMagnetsOnly==True:
            angleArr=angleArr-angleArr.min()
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
# import time
# lens=SegmentedBenderHalbach(.01,1.0,.03,.01)
# xArr=np.linspace(-.1,.1,num=40)
# coords=np.asarray(np.meshgrid(xArr,xArr,xArr)).T.reshape(-1,3)
# # lens.BNorm_Gradient(coords,returnNorm=True)
# from profilehooks import profile
# @profile()
# def func():
#     print(np.sum(np.abs(lens.BNorm_Gradient(coords,returnNorm=True)[0]))) #2.596956744685721e-08
# func()
'''

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    4.265    4.265 HalbachLensClass.py:574(func)
        1    0.000    0.000    4.264    4.264 HalbachLensClass.py:453(BNorm_Gradient)
        4    0.001    0.000    4.262    1.066 HalbachLensClass.py:441(BNorm)
        4    0.003    0.001    4.255    1.064 HalbachLensClass.py:552(B_Vec)
       12    0.006    0.000    3.560    0.297 HalbachLensClass.py:304(B)
       36    0.017    0.000    3.555    0.099 HalbachLensClass.py:150(B_Symmetric)
      144    0.317    0.002    3.441    0.024 HalbachLensClass.py:143(B)
     1728    0.017    0.000    3.117    0.002 HalbachLensClass.py:221(B)
     1728    3.095    0.002    3.099    0.002 HalbachLensClass.py:13(B_NUMBA)
        3    0.001    0.000    2.531    0.844 HalbachLensClass.py:464(grad)
        2    0.000    0.000    0.674    0.337 dispatcher.py:402(_compile_for_args)
      8/2    0.000    0.000    0.674    0.337 dispatcher.py:929(compile)
      4/2    0.000    0.000    0.673    0.337 dispatcher.py:140(compile)
      4/2    0.000    0.000    0.673    0.337 dispatcher.py:147(_compile_cached)
      4/2    0.000    0.000    0.673    0.337 dispatcher.py:162(_compile_core)
      4/2    0.000    0.000    0.673    0.336 compiler.py:660(compile_extra)
'''