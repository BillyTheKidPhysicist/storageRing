from profilehooks import profile
import pandas as pd
from injectorAnalysis import Compactor
import time
import sys
from ParaWell import ParaWell
import numpy as np
import numpy.linalg as npl
from interp3d import interp_3d
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import numba
# from ParaWell import ParaWell
import scipy.optimize as spo
import numpy.polynomial.polynomial as npp
u0=4*np.pi*1e-7


def make_Interp_Functions(data):
    # This method takes an array data with the shape (n,6) where n is the number of points in space. Each row
    # must have the format [x,y,z,gradxB,gradyB,gradzB,B] where B is the magnetic field norm at x,y,z and grad is the
    # partial derivative. The data must be from a 3D grid of points with no missing points or any other funny business
    # and the order of points doesn't matter
    xArr=np.unique(data[:,0])
    yArr=np.unique(data[:,1])
    zArr=np.unique(data[:,2])
    numx=xArr.shape[0]
    numy=yArr.shape[0]
    numz=zArr.shape[0]
    BGradxMatrix=np.empty((numx,numy,numz))
    BGradyMatrix=np.empty((numx,numy,numz))
    BGradzMatrix=np.empty((numx,numy,numz))
    xIndices=np.argwhere(data[:,0][:,None]==xArr)[:,1]
    yIndices=np.argwhere(data[:,1][:,None]==yArr)[:,1]
    zIndices=np.argwhere(data[:,2][:,None]==zArr)[:,1]
    BGradxMatrix[xIndices,yIndices,zIndices]=data[:,3]
    BGradyMatrix[xIndices,yIndices,zIndices]=data[:,4]
    BGradzMatrix[xIndices,yIndices,zIndices]=data[:,5]
    interpFx=interp_3d.Interp3D(BGradxMatrix,xArr,yArr,zArr)
    interpFy=interp_3d.Interp3D(BGradyMatrix,xArr,yArr,zArr)
    interpFz=interp_3d.Interp3D(BGradzMatrix,xArr,yArr,zArr)
    return interpFx,interpFy,interpFz

class Sphere:
    def __init__(self,radiusInInches=1.0/2):
        #angle: symmetry plane angle. There is a negative and positive one
        #radius: radius in inches
        self.angle=None #angular location of the magnet
        self.radius=radiusInInches*.0254 #meters. RADIUS!!!
        M=3.5*1.15e6 #magnetization density
        self.m0=M*(4/3)*np.pi*self.radius**3 #dipole moment
        self.r0=None #location of sphere
        self.n=None #orientation
        self.m=None #vector sphere moment
        self.phi=None #phi position
        self.theta=None #orientation of dipole. From local z axis
        self.psi=None #orientation of dipole. in local xy plane
        self.z=None
        self.r=None
    def position_Sphere(self,r=None,phi=None,z=None):
        if phi is not None:
            self.phi=phi
        if z is not None:
            self.z=z
        if r is not None:
            self.r=r
        x = self.r*np.cos(self.phi)
        y = self.r*np.sin(self.phi)
        self.r0 = np.asarray([x, y, self.z])

    def update_Size(self,radiusNewInches):
        self.radius=radiusNewInches*.0254
        M=1.15e6 #magnetization density
        self.m0=M*(4/3)*np.pi*self.radius**3 #dipole moment
        self.m = self.m0 * self.n  # vector sphere moment

    def orient(self,theta,psi):
        #tilt the sphere in spherical coordinates
        self.theta=theta
        self.psi=psi
        self.n=np.asarray([np.sin(theta)*np.cos(psi),np.sin(theta)*np.sin(psi),np.cos(theta)])
        self.m = self.m0 * self.n
    def vary_Amplitude(self,fact):
        self.m = fact*self.m0 * self.n  # vector sphere moment
    def BSlow(self,r):
        #magnetic field vector at a point in space
        #r: Coordinates of evaluation
        r=r-self.r0 # convert to difference vector
        if npl.norm(r)<self.radius:
            return np.nan
        rNorm=npl.norm(r)
        mrDot=np.sum(self.m*r)
        Bvec=1e-7*(3*r*mrDot/rNorm**5 -self.m/rNorm**3)
        return Bvec
    def B(self,r):
        return self.B_NUMBA(r,self.r0,self.m)
    def B_Symmetric(self,r):
        arr=np.zeros(r.shape)
        arr+=self.B(r)
        arr+=self.B_Symetry(r,"counterclockwise",factors=0,fixedDipoleDirection=True,planeReflection=True)
        arr+=self.B_Symetry(r,"counterclockwise",factors=1,fixedDipoleDirection=True)
        arr+=self.B_Symetry(r,"counterclockwise",factors=1,fixedDipoleDirection=True,planeReflection=True)
        arr+=self.B_Symetry(r,"counterclockwise",factors=2,fixedDipoleDirection=True)
        arr+=self.B_Symetry(r,"counterclockwise",factors=2,fixedDipoleDirection=True,planeReflection=True)
        arr+=self.B_Symetry(r,"counterclockwise",factors=3,fixedDipoleDirection=True)
        arr+=self.B_Symetry(r,"counterclockwise",factors=3,fixedDipoleDirection=True,planeReflection=True)
        return arr
    @staticmethod
    @numba.njit(numba.float64[:,:](numba.float64[:,:],numba.float64[:],numba.float64[:]))
    def B_NUMBA(r,r0,m):
        r=r-r0 # convert to difference vector
        rNormTemp=np.sqrt(np.sum(r**2,axis=1))
        rNorm=np.empty((rNormTemp.shape[0],1))
        rNorm[:,0]=rNormTemp
        mrDotTemp=np.sum(m*r,axis=1)
        mrDot=np.empty((rNormTemp.shape[0],1))
        mrDot[:,0]=mrDotTemp
        Bvec=1e-7*(3*r*mrDot/rNorm**5 -m/rNorm**3)
        return Bvec
    def B_Symetry(self, r, orientation, factors=1,flipDipole=False,angle=np.pi/2,fixedDipoleDirection=False,
                  planeReflection=False):
        #return the magnetic field of a mirrored dipole reflected across the symmetry boundaries. There are two
        #orientation: String of "clockwise" or "counterclockwise" for orientation
        #factors: how many planes of symmetry to to reflect by. there are 6 total
        #fliSphere: wether to model the sphere as having the opposite orientation
        phi0=np.arctan2(self.r0[1],self.r0[0])
        #choose the correct reflection angle.
        if orientation=='clockwise': #mirror across the clockwise plane
            phiSym=phi0+(-angle)*factors #angle to rotate the dipole position by
            deltaTheta=-angle*factors #angle to rotate the dipole direction vector by
        elif orientation=='counterclockwise':#mirror across the counterclockwise plane
            phiSym=phi0+angle*factors
            deltaTheta=angle*factors
        else:
            raise Exception('Improper orientation')
        xSym = npl.norm(self.r0[:2]) * np.cos(phiSym)
        ySym = npl.norm(self.r0[:2]) * np.sin(phiSym)
        rSym=np.asarray([xSym,ySym,self.r0[2]])

        mSym = self.m.copy()
        if fixedDipoleDirection==False:
            #rotate the dipole moment.
            MRot=np.array([[np.cos(deltaTheta),-np.sin(deltaTheta)],[np.sin(deltaTheta),np.cos(deltaTheta)]])
            mSym[:2]=MRot@mSym[:2]
        if flipDipole==True:
            mSym=-mSym
        if planeReflection==True: #another dipole on the other side of the z=0 line
            rSym[2]=-rSym[2]
        BVecArr=self.B_NUMBA(r,rSym,mSym)
        return BVecArr
class ShimOptimizer:
    def __init__(self):
        self.magnetSize=1.0*.0254/4 #width of magnets, weather sphere or cube
        self.numLayers=28
        self.helper=ParaWell()
        
        self.lengthMagnet=self.numLayers*self.magnetSize
        self.boreRadius=.05
        self.vacuumTubeFrac=.8 #fraction of the bore that the vacuum tube takes up and thus the magnet cannot get closer
        #than this value
        self.yokeWidth=2.5*.0254  #width of yoke, or magnet
        zMax=self.lengthMagnet/2+3*self.boreRadius


        self.numX=41
        self.gridXArr=np.linspace(-self.boreRadius+100e-6,self.boreRadius-100e-6,num=self.numX)
        self.gridYArr=self.gridXArr.copy()
        self.numZ=50
        self.gridZArr=np.linspace(0,zMax-100e-6,num=self.numZ) #to prevent nan at the end
        self.dx=10e-6  #step size for calculating derivatives

        self.gridCoordArr=None  #an array of coordinates at and around each point in the grad that the magnetic field
        #will be evaluated at. The array is shape (6n,3), where n is the number of grid points. An entry goes  as
        #[BVecxl,BVech,BVecyl,BVecyh,BVeczl,BVeczh] where xl stands for x low for example and xh x high
        self.gridBVecArr=None  #similiar to the above except it holds field values from comsol instead of coordinates.



        self.sphereList=None  #list of spheres in the space
        self.numspheres=None
        assert self.numLayers % 2 == 0
        self.fill_Line_List()
    def build_Lens(self,args):
        #Add or modify spheres to build the magnet
        if self.sphereList is None:
            self.sphereList=[]
            phiArr=np.asarray([-30.0,0.0,30.0])*np.pi/180
            psiArr=np.asarray([60,180,300])*np.pi/180
            for i in range(self.numLayers // 2):
                for j in range(3): #num magnets
                    sphere=Sphere(radiusInInches=self.magnetSize/.0254)
                    z=2*i*sphere.radius+sphere.radius
                    r=self.boreRadius+sphere.radius
                    phi=phiArr[j]
                    sphere.position_Sphere(r=r,phi=phi,z=z)
                    psi=psiArr[j]
                    theta=np.pi/2
                    sphere.orient(theta,psi)
                    self.sphereList.append(sphere)
        k=0
        for i in range(self.numLayers//2): #layers
            r=args[i]#args[i:(i+1)]
            for j in range(3): #num sphere
                sphere=self.sphereList[k]
                sphere.position_Sphere(r=r+sphere.radius)
                k+=1
    def fill_Line_List(self):
        coordList=[]
        for x in self.gridXArr:
            for y in self.gridYArr:
                for z in self.gridZArr:
                    coordList.append([x-self.dx,y,z])
                    coordList.append([x+self.dx,y,z])
                    coordList.append([x,y-self.dx,z])
                    coordList.append([x,y+self.dx,z])
                    coordList.append([x,y,z]) #because otherwise the first plane's values are nan!
                    coordList.append([x,y,z+self.dx])
        self.gridCoordArr=np.asarray(coordList)

    def generate_Trace_Data(self,parallel=False):
        gridBVecArr=np.zeros(self.gridCoordArr.shape)
        gridCoordArrTemp=self.gridCoordArr.copy()


        if parallel==True:
            def func(sphere):
                return sphere.B_Symmetric(gridCoordArrTemp)
            results=self.helper.parallel_Problem(func,self.sphereList,onlyReturnResults=True)
            for result in results:
                gridBVecArr+=result
        else:
            for sphere in self.sphereList:
                gridBVecArr+=sphere.B_Symmetric(gridCoordArrTemp)
        #now get the derivative

        B0Gradx=(npl.norm(gridBVecArr[1::6],axis=1)-npl.norm(gridBVecArr[::6],axis=1))/(2*self.dx)
        B0Grady=(npl.norm(gridBVecArr[3::6],axis=1)-npl.norm(gridBVecArr[2::6],axis=1))/(2*self.dx)
        B0Gradz=(npl.norm(gridBVecArr[5::6],axis=1)-npl.norm(gridBVecArr[4::6],axis=1))/self.dx #because the z is
        #forward differentiation
        B0=npl.norm(gridBVecArr[4::6],axis=1)
        # # B0=gridBVecArr[5::6][:,0]
        # BPlot=B0.reshape(41,41,50)[:,:,20]
        # BPlot[BPlot>1]=1
        # BPlot[BPlot<-1]=-1
        # plt.imshow(BPlot)
        # plt.show()
        coords=gridCoordArrTemp[::6]  #get every coordinate that was shifted to negative by dx
        coords[:,0]+=self.dx  #add the shift
        self.data=np.column_stack((coords,B0Gradx,B0Grady,B0Gradz,B0))
        # np.savetxt('traceData.txt',self.data)

    @staticmethod
    @numba.njit(numba.float64(numba.float64,numba.float64))
    def fast_Arctan2(y,x):
        phi=np.arctan2(y,x)
        if phi<0:  # confine phi to be between 0 and 2pi
            phi+=2*np.pi
        return phi


    def cost(self,Print=False,parallel=True):
        self.generate_Trace_Data(parallel=parallel)
        compactor=Compactor(h=1e-5)
        compactor.X={"Lo1":.7,"Lm1":None,"rp1":.05,'ap1':.7,'Lsep1':1.5}
        compactor.build_Lattice()
        compactor.lattice.elList[1].data3D=self.data
        compactor.lattice.elList[1].fill_Params(externalDataProvided=True)
        compactor.lattice.end_Lattice(enforceClosedLattice=False, latticeType='injector', surpressWarning=True,trackPotential=True)
        lensOutputLocation=compactor.lattice.elList[1].r2[0]
        lensInputLocation=compactor.lattice.elList[1].r1[0]
        imagePlane,spotDiam=compactor.find_Image_Plane(parallel=parallel)


        magnification=(imagePlane-lensOutputLocation)/lensInputLocation

        spotDiam=np.round(1e3*spotDiam,3) #realistic precision

        z0=1.5354323232323233
        magnification0=0.9695133149678606
        deltaZ=np.abs(z0-imagePlane)
        cost1=0
        if deltaZ>.1:
            cost1+=np.inf
        # cost1=1e5*(z0-imagePlane)**2
        print('wrong I think')
        adjustedSpotSize=(magnification/magnification0)*spotDiam
        cost2=adjustedSpotSize
        cost=cost1+cost2
        if Print==True:
            print(imagePlane,magnification,adjustedSpotSize,cost)
        return cost



    def minimize_Field(self,limits=None):
        bounds = []
        for i in range(self.numLayers//2):
            bounds.append((self.boreRadius*.8,self.boreRadius*1.5))
        def minCost(args):
            # if self.enforce_Geometry(args)==False: #if configuration violates geometry
            #     return np.inf
            self.build_Lens(args)
            cost=self.cost(parallel=False,Print=False)
            return cost

        print('--Beginning minimization--')
        sol=spo.differential_evolution(minCost,bounds,disp=True,mutation=(.5,1.5),popsize=3,polish=False,maxiter=100,workers=16)
        print(sol)
        self.build_Lens(sol.x)
        print('optimized')
        self.cost(Print=True)
        args=np.ones(self.numLayers//2)*self.boreRadius
        self.build_Lens(args)
        print("not optmized")
        self.cost(Print=True)
        print('Percent reduction',int(100*(1-sol.fun/self.cost())), 'cost',sol.fun)
        return sol.fun


optimizer=ShimOptimizer()

#
optimizer.minimize_Field()
#