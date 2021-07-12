import pandas as pd
from injectorAnalysis import Injector
from ParticleTracerLatticeClass import ParticleTracerLattice
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
import globalTest
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
    def __init__(self,angle,radiusInINches=1.0/2):
        #angle: symmetry plane angle. There is a negative and positive one
        #radius: radius in inches
        self.angle=angle #angle of symmetry plane
        self.radius=radiusInINches*.0254 #meters. RADIUS!!!
        M=1.15e6 #magnetization density
        self.m0=M*(4/3)*np.pi*self.radius**3 #dipole moment
        self.r0=np.asarray([1.0,1.0,1.0]) #location of sphere
        self.n=np.asarray([0,0,1]) #orientation
        self.m=self.m0*self.n #vector sphere moment
        self.theta=0 #orientation of dipole. From local z axis
        self.psi=0 #orientation of dipole. in local xy plane
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
    def B_Symmetric(self,r,planeSymmetry=True):
        #a single magnet actually represents 12 magnet
        #r: array of N position vectors to get field at. Shape (N,3)
        #planeSymmetry: Wether to exploit z symmetry or not
        arr=np.zeros(r.shape)
        arr+=self.B(r)
        arr+=self.B_Symetry(r,"clockwise",factors=1,flipSphere=True)
        arr+=self.B_Symetry(r,"clockwise",factors=2,flipSphere=False)
        arr+=self.B_Symetry(r,"clockwise",factors=3,flipSphere=True)
        arr+=self.B_Symetry(r,"clockwise",factors=4,flipSphere=False)
        arr+=self.B_Symetry(r,"clockwise",factors=5,flipSphere=True)


        if planeSymmetry==True:
            arr+=self.B_Symetry(r,"clockwise",factors=0,flipSphere=False,reflection=True)
            arr+=self.B_Symetry(r,"clockwise",factors=1,flipSphere=True,reflection=True)
            arr+=self.B_Symetry(r,"clockwise",factors=2,flipSphere=False,reflection=True)
            arr+=self.B_Symetry(r,"clockwise",factors=3,flipSphere=True,reflection=True)
            arr+=self.B_Symetry(r,"clockwise",factors=4,flipSphere=False,reflection=True)
            arr+=self.B_Symetry(r,"clockwise",factors=5,flipSphere=True,reflection=True)

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
    def B_Symetry(self, r, orientation, factors=1,flipSphere=False,reflection=False):
        #return the magnetic field of a mirrored dipole reflected across the symmetry boundaries. There are two
        #orientation: String of "clockwise" or "counterclockwise" for orientation
        #factors: how many planes of symmetry to to reflect by. there are 6 total
        #fliSphere: wether to model the sphere as having the opposite orientation
        #reflection: reflect the sphere across the z=0 plane
        phi0=np.arctan2(self.r0[1],self.r0[0])
        #choose the correct reflection angle.
        if orientation=='clockwise': #mirror across the clockwise plane
            # phiSym=2*(-self.angle-phi0)+phi0
            phiSym=phi0+(-2*self.angle)*factors
            # m=np.tan(-self.angle) #slope of symmetry plane
            deltaPhi=-2*self.angle*factors
        elif orientation=='counterclockwise':#mirror across the counterclockwise plane
            # phiSym=2*(self.angle-phi0)+phi0
            phiSym=phi0+2*self.angle*factors
            # m=np.tan(self.angle) #slope of symmetry plane
            deltaPhi=2*self.angle*factors
        else:
            raise Exception('Improper orientation')
        xSym = npl.norm(self.r0[:2]) * np.cos(phiSym)
        ySym = npl.norm(self.r0[:2]) * np.sin(phiSym)
        if reflection==True:
            rSym=np.asarray([xSym,ySym,-self.r0[2]]) #reflection across z=0 plane
        else:
            rSym=np.asarray([xSym,ySym,self.r0[2]])


        mSym = self.m.copy()
        #reflect the dipole moment across the symmetry plane. This is like method of images
        # M=np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2) #reflection matrix
        # mSym[:2]=M@self.m[:2]


        #rotate the dipole moment.
        MRot=np.array([[np.cos(deltaPhi),-np.sin(deltaPhi)],[np.sin(deltaPhi),np.cos(deltaPhi)]])
        mSym[:2]=MRot@mSym[:2]
        if flipSphere==True:
            mSym=-mSym
        BVecArr=self.B_NUMBA(r,rSym,mSym)
        return BVecArr


class ShimOptimizer:
    def __init__(self,file,reuseData=True):
        #file: comsol text file of half the fields in space.
        self.lengthMagnet=.1524
        self.boreRadius=.05
        self.vacuumTubeFrac=None #fraction of the bore that the vacuum tube takes up
        self.yokeWidth=2.5*.0254  #width of yoke, or magnet
        self.maxAngle=30*np.pi/180  #todo: consistent usage
        self.minAngle=-30*np.pi/180
        self.data=np.asarray(pd.read_csv(file, delim_whitespace=True, header=None))

        self.zMin=0
        zMax=self.data[:,2].max()
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

        self.centerMagnet=None #Wether to place a magnet at the center or not, boolean
        self.sphereList=None  #list of spheres in the space
        self.numspheres=None

        if reuseData==True:
            pass
            # self.gridCoordArr=np.loadtxt('gridCoordData.txt')
            # self.gridBVecArr=np.loadtxt('gridBVecData.txt')
        else:
            self.fill_Line_List()
            # np.savetxt('gridCoordData.txt',self.gridCoordArr)
            # np.savetxt('gridBVecData.txt',self.gridBVecArr)

    def fill_Line_List(self):
        funcBx,funcBy,funcBz=make_Interp_Functions(self.data)
        coordList=[]
        BVecList=[]

        def BVec(x,y,z):
            if np.sqrt(x**2+y**2)>self.boreRadius:
                return np.asarray([np.nan,np.nan,np.nan])
            zSign=1
            if True:#0<=theta<=self.maxAngle or 2*np.pi-self.maxAngle<=theta<=2*np.pi:
                M=np.asarray([[1,0],[0,1]])
                x0=x
                y0=y
            Bx0=funcBx((x0,y0,z))
            By0=funcBy((x0,y0,z))
            Bx=M[0,0]*Bx0+M[0,1]*By0
            By=M[1,0]*Bx0+M[1,1]*By0
            Bz=zSign*funcBz((x0,y0,z))
            return np.asarray([Bx,By,Bz])
        for x in self.gridXArr:
            for y in self.gridYArr:
                for z in self.gridZArr:
                    BVecList.append(BVec(x-self.dx,y,z))
                    BVecList.append(BVec(x+self.dx,y,z))
                    BVecList.append(BVec(x,y-self.dx,z))
                    BVecList.append(BVec(x,y+self.dx,z))
                    BVecList.append(BVec(x,y,z))
                    BVecList.append(BVec(x,y,z+self.dx))

                    coordList.append([x-self.dx,y,z])
                    coordList.append([x+self.dx,y,z])
                    coordList.append([x,y-self.dx,z])
                    coordList.append([x,y+self.dx,z])
                    coordList.append([x,y,z]) #because otherwise the first plane's values are nan!
                    coordList.append([x,y,z+self.dx])
        self.gridCoordArr=np.asarray(coordList)
        self.gridBVecArr=np.asarray(BVecList)

    def generate_Trace_Data(self,noSpheres=False):
        gridBVecArrTemp=self.gridBVecArr.copy()
        gridCoordArrTemp=self.gridCoordArr.copy()
        if noSpheres==False:
            i=0
            for sphere in self.sphereList:
                if i==0 and self.centerMagnet==True: #for the first sphere at center plane of magnet
                    gridBVecArrTemp+=sphere.B_Symmetric(gridCoordArrTemp,planeSymmetry=False) #special
                    #case without using z symmetry
                else:
                    gridBVecArrTemp+=sphere.B_Symmetric(gridCoordArrTemp)
                i+=1
        #now get the derivative
        B0Gradx=(npl.norm(gridBVecArrTemp[1::6],axis=1)-npl.norm(gridBVecArrTemp[::6],axis=1))/(2*self.dx)
        B0Grady=(npl.norm(gridBVecArrTemp[3::6],axis=1)-npl.norm(gridBVecArrTemp[2::6],axis=1))/(2*self.dx)
        B0Gradz=(npl.norm(gridBVecArrTemp[5::6],axis=1)-npl.norm(gridBVecArrTemp[4::6],axis=1))/self.dx #because the z is
        #forward differentiation
        B0=npl.norm(gridBVecArrTemp[4::6],axis=1)
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

    def cost(self,noSpheres=False,Print=False,parallel=False,numParticles=150):
        self.generate_Trace_Data(noSpheres=noSpheres)
        

        injector=Injector(h=1e-5) 
        injector.X={"Lo1":.6,"Lm1":None,"rp1":.05,'ap1':self.vacuumTubeFrac,'Lsep1':1.2}
        injector.lattice = ParticleTracerLattice(injector.v0Nominal)
        injector.lattice.add_Drift(injector.X["Lo1"]-injector.X['rp1']*injector.fringeFrac,ap=.05) #need to update this after adding the first lens
        # injector.lattice.add_Lens_Ideal(injector.X["Lm2"],-1*injector.X['Bp2'],injector.X['rp1'])
        injector.lattice.add_Lens_Sim_With_Caps(None,None,injector.fringeFrac,None)#"optimizerData_Full.txt"
        injector.lattice.add_Drift(injector.X['Lsep1'],ap=.25) #need to update this after adding the first lens
        injector.lattice.elList[1].data3D=self.data
        injector.lattice.elList[1].fill_Params(externalDataProvided=True)
        injector.lattice.end_Lattice(enforceClosedLattice=False, latticeType='injector', surpressWarning=True,trackPotential=True)
        lensOutputLocation=injector.lattice.elList[1].r2[0]+3*injector.lattice.elList[1].rp
        lensInputLocation=injector.lattice.elList[1].r1[0]+3*injector.lattice.elList[1].rp

        imagePlane,spotDiam=injector.find_Image_Plane(parallel=parallel,numPhiParticlesMax=numParticles)

        magnification=(imagePlane-lensOutputLocation)/lensInputLocation

        spotDiam=1e3*spotDiam #convert to mm

        z0=1.6589469879518073
        magnification0=1.012251105472415
        #cost0: 0.0031558448904623086
        deltaZ=1e2*(imagePlane-z0) #convert to cm
        if np.abs(deltaZ)<2: #limit to 2 cm
            cost1=0
        else: #punish for being outside
            cost1=np.inf
        adjustedSpotDiam=(magnification0/magnification)*spotDiam #mm^2
        adjustedSpotArea=np.pi*(adjustedSpotDiam/2)**2 #mm^2
        cost2=adjustedSpotArea
        cost=cost1+cost2
        if Print==True:
            print(imagePlane,magnification,adjustedSpotArea,cost1,cost2)
        return cost

    def add_Spheres(self,numSpheres,args=None):
        self.numspheres = numSpheres
        self.sphereList = []
        for i in range(self.numspheres):
            self.sphereList.append(Sphere(self.maxAngle))
        if args is not None:
            self.update_spheres(args)
    def minimize_Field(self,numSpheres=None,centerMagnet=True):
        self.centerMagnet=centerMagnet
        if numSpheres is not None:
            self.add_Spheres(numSpheres)
        self.vacuumTubeFrac=.8
        bounds = []

        limits = [(self.boreRadius*self.vacuumTubeFrac, self.boreRadius + self.yokeWidth), (self.minAngle, self.maxAngle),
                  (0.0, self.lengthMagnet/2+10e-2),(0, np.pi), (0, 2 * np.pi),(0.0,1.5)]

        for i in range(self.numspheres):
            if self.centerMagnet==True and i==0:
                maxDiam=.5#self.boreRadius*self.vacuumTubeFrac/.0254 #convert to inches
                limitCenter=[(self.boreRadius*self.vacuumTubeFrac,self.boreRadius),(self.minAngle, self.maxAngle)
                    ,(0, np.pi), (0, 2 * np.pi),(0.0,maxDiam)] #r,phi,psi,theta,magnetsize
                bounds.extend(limitCenter)
            else:
                bounds.extend(limits)



        print('--Beginning minimization--')
        globalTest.object=self
        globalTest.limits=bounds
        sol=globalTest.solve()
        # sol=spo.differential_evolution(minCost,bounds,disp=False,popsize=32*4,polish=False,mutation=mutation,maxiter=1000,workers=-1)
        print(sol)
        self.update_spheres(sol.x)
        print('optimized')
        self.cost(Print=True)
        print("not optmized")
        self.cost(Print=True,noSpheres=True)
        print('Percent reduction',int(100*(1-sol.fun/self.cost(noSpheres=True))), 'cost',sol.fun)
        return sol
    def unpack_Parameters(self,args,i):
        #unpack the parameteres from the differential evolution algorith'm test arguments for the ith sphere
        diamArr=np.asarray([.25,.5,.75,1.0,1.5])
        if self.centerMagnet==True and i==0:
            r,phi,theta,psi,diameter=args[5*i:5*(i+1)]
            z=0
        elif self.centerMagnet==True and i!=0:  #if using center magnet need to resepct that first limits have less
            #entries because z is set
            r,phi,z,theta,psi,diameter=args[6*i-1:6*(i+1)-1]
        else:
            r,phi,z,theta,psi,diameter=args[6*i:6*(i+1)]
        radius=diamArr[np.argmin(np.abs(diamArr-diameter))]/2 #in inches
        return r,phi,z,theta,psi,radius

    def enforce_Geometry(self,args):
        #enforce geometric constraints of spheres and magnet. Geometry here
        #is cylinder with a cylinderical cutout inside. return True if okay, False is geometry is violated.
        #because of symmetry the sphere is allowed to be as close to the symmetry plane as it like, even right on the
        #middle
        #args: configuraton arguments
        for i in range(self.numspheres):
            sphere=self.sphereList[i]
            r,phi,z,theta,psi,radius=self.unpack_Parameters(args,i)
            sphere.update_Size(radius)
            #check if sphere is above or below top of yoke
            if z-sphere.radius>self.lengthMagnet/2: #bottom sphere is above top of magnet and can be anywhere:
                result= True
            elif z>self.lengthMagnet/2 and z-sphere.radius<self.lengthMagnet/2: #sphere is in the intermediate
                #region where it can kind of roll off the edge
                if self.boreRadius<r<self.boreRadius+self.yokeWidth: #sphere is within the yoke radially
                    #so bottom edge is clipping top of magnet
                    result= False
                elif r+sphere.radius<self.boreRadius or r-sphere.radius>self.boreRadius+self.yokeWidth:
                    #sphere is definitely clear of magnet radially
                    result=True
                else: #sphere is in intermediate region where it can kind of roll off the edge. Test by finding distance
                    #from edge to center of sphere
                    deltaR=r-self.boreRadius
                    deltaZ=z-self.lengthMagnet/2
                    seperation=np.sqrt(deltaZ**2+deltaR**2)
                    if seperation< sphere.radius:
                        result= False #no logic here for now
                    else:
                        result=True
            elif (z<self.lengthMagnet/2 and z-sphere.radius>0) or self.centerMagnet==True: #center below top of magnet
                # and above zero, or the magnet is right at zero
                #check if sphere is inside the yoke radially
                if self.boreRadius<r+sphere.radius and r-sphere.radius<self.boreRadius+self.yokeWidth: #check if inside
                    #yoke
                    result= False
                elif r-sphere.radius<self.vacuumTubeFrac*self.boreRadius:#check if inside vacuum tube
                    result= False
                else:
                    result= True
            else:  #if anywhere else
                result= False
            if result==False: #if invalid geometry for any sphere return False
                return False
        return True #if all tests passed, return True

    def update_spheres(self,args):
        for i in range(self.numspheres):
            sphere=self.sphereList[i]
            r,phi,z,theta,psi,radius=self.unpack_Parameters(args,i)
            x=r*np.cos(phi)
            y=r*np.sin(phi)
            sphere.r0=np.asarray([x,y,z])
            sphere.orient(theta,psi)
            sphere.update_Size(radius)

optimizer=ShimOptimizer('optimizerData.txt',reuseData=False)

for num in [1,1,1,2,2,2]:
    print('-------------------'+str(num)+'---------------------------')
    sol=optimizer.minimize_Field(numSpheres=num,centerMagnet=True)

''' --Beginning minimization--
     fun: 3.172370005356179
 message: 'Optimization terminated successfully.'
    nfev: 7296
     nit: 75
 success: True
       x: array([ 0.0431777 , -0.20296896,  2.45943724,  5.38923083,  0.24807677,
        0.05650959,  0.18321613,  0.09131351,  0.72591997,  2.4571787 ,
        1.00916552])
optimized
1.677621686746988 1.0433911734983958 3.172370005356179 0 3.172370005356179
not optmized
1.6589469879518073 1.012251105472415 8.429857902962405 0 8.429857902962405
Percent reduction 62 cost 3.172370005356179
'''