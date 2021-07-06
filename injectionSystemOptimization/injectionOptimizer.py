import numba
import sys
import scipy.interpolate as spi
import warnings
import time
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
import numpy.linalg as npl
# from storageRingOptimization import elementPT
# from storageRingOptimization.ParticleTracer import ParticleTracer
from storageRingOptimization.ParticleClass import  Swarm,Particle
# import storageRingOptimization.elementPT
from storageRingOptimization.particleTracerLattice import ParticleTracerLattice
from storageRingOptimization.SwarmTracer import SwarmTracer
from shapely.geometry import LineString
# from profilehooks import profile

class ApetureOptimizer:
    def __init__(self,Li=.2,apDiam=.005,h=5e-6,Lsep=.2,Lo=.1,fringeFrac=1.5):
        self.lattice=None
        self.swarmInitial=None
        self.h=None #timestep
        self.v0Nominal=200.0
        self.apDiam=apDiam
        self.h=h
        self.X={'Lm2':None,'rp1':None,'sigma':None,'Li':Li,'Lm1':None,'Lo':Lo,'apDiam':apDiam,'Lsep':Lsep,
                'fringeFrac':fringeFrac} #dictionairy of lattice parameters
        self.X0=self.X.copy() #initial values of lattice parameters
    def initialize_Observed_Swarm(self,numParticles=None,fileName='phaseSpaceCloud.dat',dv=0.0,
                                  magFact=1.0,aperture=5e-3):
        #load a swarm from a file.
        #numParticles: Number of particles to load from the file. if None load all
        #fileName: Name of the file with the particle data in the format (y,z,px,py,pz)
        #dv: extra longitudinal velocity to add
        #magFact: factor to modify the  magnification by artificially. Larger corresponds to smaller image size
        #aperture: Transverse aperture in meters
        fileName='../phaseSpaceGenerationAndExtraction/'+fileName
        self.swarmInitial=Swarm()
        data= np.loadtxt(fileName)
        np.random.shuffle(data)
        if numParticles is None:
            numParticles=np.inf
        numParticlesAdded=0
        for coords in data:
            x,y,z,px,py,pz=coords
            # px=-px #need to flip the coords!!
            px+=dv
            q=np.asarray([1e-10,y/magFact,z/magFact]) #add a tiny offset in the x direction so the particle starts in the element,
            #otherwise the algorithm spends extra time checking to see if the particle is exactly on the edge
            if np.sqrt(q[1]**2+q[2]**2)<aperture:
                numParticlesAdded+=1
                p=np.asarray([px,py*magFact,pz*magFact,])
                self.swarmInitial.add_Particle(qi=q,pi=p)
            if numParticles<=numParticlesAdded:
                break# leave the for loop

    def initialize_Ideal_Point_Swarm(self,numParticles=1000,maxAng=.05,v0=None):
        np.random.seed(42)
        if v0 is None:
            v0=self.v0Nominal
        self.swarmInitial = Swarm()
        x=np.zeros(numParticles)
        y=x.copy()
        z=x.copy()
        x+=1e-10 #to scoot the particle along into the first element to save time
        # [-0.13000223  0.05427037  0.81334243]
        px=v0
        py=(np.random.rand(numParticles)-.5)*200*np.tan(maxAng)
        pz=(np.random.rand(numParticles)-.5)*200*np.tan(maxAng)
        for i in range(numParticles):
            q=np.asarray([x[i],y[i],z[i]])
            p=np.asarray([px,py[i],pz[i]])
            self.swarmInitial.add_Particle(qi=q,pi=p)
        np.random.seed(int(time.time()))



    def get_RMS_Width(self,swarm,axis='y'):
        #Get the RMS width of the beam. Swarm is assumed to have the same x value for all particles. RMS is of the y axis
        temp=[]
        for particle in swarm:
            q=particle.q
            if axis=='y':
                temp.append(q[1])
            elif axis=='z':
                temp.append(q[2])
            else:
                raise Exception('no proper axis provided')
        rms=np.std(np.asarray(temp))
        return rms
    @staticmethod
    @numba.njit(numba.float64(numba.float64[:,:],numba.float64))
    def get_Frac_Width(qArr,fraction):
        #get the width of the beam that captures fraction of the atoms
        #fraction: fraction of the atoms in that width
        if not 0<fraction<1.0:
            raise Exception('fraction must be number between 0 and 1')
        numParticles=qArr.shape[0]
        if numParticles<100:
            print('Low number of particles will result in larger error with this method')
        rArr=np.sqrt(qArr[:,1]**2+qArr[:,2]**2)
        rArrSorted=np.sort(rArr)
        cutoffIndex=int(fraction*numParticles) #the 100*fraction% particle. There is some minor rounding error
        width=2*rArrSorted[cutoffIndex]
        return width
    def get_Max_Displacement(self,swarm,axis='radial'):
        #get the maximum displacement of a particle in the given axis
        #axis: Which axis to find the maximum along. If both, then do the max radial value. options are 'both','y','z'
        temp=[]
        for particle in swarm:
            q=particle.q
            if axis=='radial':
                val=np.sqrt(q[1]**2+q[2]**2)
            elif axis=='y':
                val=q[1]
            elif axis=='z':
                val=q[2]
            else:
                raise Exception('invalid axis provided')
            temp.append(val)
        return max(temp)
    def trace_Through_Bumper(self,parallel=False):
        swarmTracer = SwarmTracer(self.lattice)

        swarm = swarmTracer.trace_Swarm_Through_Lattice(self.swarmInitial, self.h, 1.0, fastMode=False,parallel=parallel)
        #now 'unclip' particles that got clipped at the last element. The last element represents free space so any
        #clipping there is not part of the model now and is considered not clipped cause I assume all that is captured.
        #Also go through and assign interpolated functions
        for particle in swarm:
            if particle.currentElIndex+1==len(self.lattice.elList): #in the last element, so it survived
                particle.clipped=False
            particle.yInterp=spi.interp1d(particle.qArr[:,0],particle.qArr[:,1])
            particle.zInterp=spi.interp1d(particle.qArr[:,0],particle.qArr[:,2])
        return swarm
    def characterize_Output(self,swarm):
        #calculate the mean tilt and offset of the output beam. Offset is the offset at the lens output measured in meter.
        # theta is relative to x axis and in radians
        #swarm: The traced swarm
        #returns an array as [thetaMean,offsetMean]
        lastEl=self.lattice.elList[-1]
        thetaList=[] #list of tilt values
        offsetList=[] #list of offset values
        for particle in swarm:
            if particle.clipped==False:
                pArr=particle.pArr
                qArr=particle.qArr
                thetaList.append(np.arctan(pArr[-1,1]/pArr[-1,0]))
                dx=lastEl.r1[0]-qArr[-1,0]
                offsetList.append(thetaList[-1]*dx+qArr[-1,1])
        thetaMean=np.mean(np.asarray(thetaList))
        offsetMean=np.mean(np.asarray(offsetList))
        return thetaMean,offsetMean
    def project_Swarm_Through_Apeture(self, swarm, theta, offset,copySwarm=True):
        #clip particles in the swarm that clip the lens. This is done by placing a perpindicular line
        #at a distance Li from the end of the last lens along the line given by theta and offset
        #swarm: The swarm to test. Needs to have been traced already
        #Li: Image distance. This is the distance from the last lens to the apeture
        #apSize: the diameter of the apeture
        #theta: the tilt of the outgoing beam
        #offset: the displacement of the outgoing beam from y=0 at the last element
        if copySwarm==True:
            swarmNew=swarm.copy()
        else:
            swarmNew=swarm
        #All the below coordinates are in the x,y plane, ie the plane parallel to the ground
        #get m and b for the nominal trajectory
        mTraj=np.tan(theta)
        #location of the nominal trajectory at the last lens
        x0=(self.lattice.elList[-1].r1[0])
        y0=offset
        #get m and b for the apeture line
        mAp=-1/mTraj
        xAp=x0+np.cos(theta)*self.X['Li']
        yAp=y0+np.sin(theta)*self.X['Li']
        bAp=yAp+mAp*xAp

        for particle in swarmNew:
            if particle.clipped==False:
                mAtom=particle.pArr[-1,1]/particle.pArr[-1,0] #slope of atom trajectory
                bAtom=particle.qArr[-1,1]-mAtom*particle.qArr[-1,0] #b of atom in y=mx+b
                xAtom=(bAp-bAtom)/(mAtom+mAp)
                yAtom=mAtom*(bAp-bAtom)/(mAtom+mAp)+bAtom
                deltay=np.sqrt((xAtom-xAp)**2+(yAtom-yAp)**2) #only in the y direction though

                #now consider the z component!
                mzAtom=particle.pArr[-1,2]/particle.pArr[-1,0] #slope of atom trajectory
                bzAtom=particle.qArr[-1,2]-mzAtom*particle.qArr[-1,0] #b of atom in y=mx+b
                zAtom=mzAtom*xAtom+bzAtom
                deltaz=zAtom

                delta=np.sqrt(deltay**2+deltaz**2)
                if delta>self.apDiam/2:
                    particle.clipped=True
        return swarmNew




    def get_Spot_Size(self,swarm,L,fraction=.95):
        #swarm: swarm to use to calculate spot size
        #L: distance from focus to calculate spot size at. This is in the trajectory reference frame so bend through
        #the bumper
        qList=[]
        for particle in swarm:
            if particle.q[0]>L: #only particles that have survived
                y=particle.yInterp(L)
                z=particle.zInterp(L)
                qList.append([L,y,z])
        if len(qList)==0:
            raise Exception('No particles are available to caculate the spot size')
        spotSize=self.get_Frac_Width(np.asarray(qList),fraction=fraction)
        return spotSize

    def build_Lattice(self):
        #sigma here moves the element upwards
        self.lattice = ParticleTracerLattice(self.v0Nominal)
        self.lattice.add_Drift(None) #need to update this after adding the first lens
        self.lattice.add_Lens_Sim_With_Caps('lens2D_Injection_Short.txt','lens3D_Injection.txt',self.X['fringeFrac'],None)
        self.X['rp1']=self.lattice.elList[1].rp
        self.X['Lm1']=2*self.lattice.elList[1].Lcap0
        Lo0=self.X['Lo']-self.X['rp1']*self.X['fringeFrac']
        self.lattice.elList[0].set_Length(Lo0) #set the drift element length correctly
        self.lattice.add_Drift(self.X['Lsep'])
        self.lattice.add_Bump_Lens_Sim_With_Caps('lens2D_Injection_Short.txt','lens3D_Injection_Short.txt'
                                                 ,self.X['fringeFrac'],self.X['Lm2'],self.X['sigma'],rp=self.X['rp2']
                                                 ,ap=.8*self.X['rp2'])
        self.lattice.add_Drift(self.X['Li']+.05,ap=2*.2*self.X['Li'])
        self.lattice.end_Lattice(enforceClosedLattice=False, latticeType='injector', surpressWarning=True,trackPotential=True)
        self.lattice.elList[3].BpFact=.9
    def make_Apeture_Objects(self,theta,offset):
        #create shapely object to represent the apeture for plotting purposes
        apWidthInPlot=.05 #the width of the apeture for plotting purposes. This is the distance from the beginning
        #of one side of the apeture to the end
        x0 = (self.lattice.elList[-1].r1[0])
        y0 = offset
        xAp = x0 + np.cos(theta) * self.X['Li']
        yAp = y0 + np.sin(theta) * self.X['Li']
        #make upper portion of apeture
        xApUp1=xAp+(self.apDiam/2)*np.cos(theta+np.pi/2)
        yApUp1=yAp+(self.apDiam/2)*np.sin(theta+np.pi/2)
        xApUp2 = xAp + (self.apDiam/2+apWidthInPlot) * np.cos(theta + np.pi / 2)
        yApUp2 = yAp + (self.apDiam/2+apWidthInPlot) * np.sin(theta + np.pi / 2)
        #make lower shape
        xApLow1 = xAp -(self.apDiam/2) * np.cos(theta + np.pi / 2)
        yApLow1 = yAp -(self.apDiam/2) * np.sin(theta + np.pi / 2)
        xApLow2 = xAp -(self.apDiam/2 + apWidthInPlot) * np.cos(theta + np.pi / 2)
        yApLow2 = yAp -(self.apDiam/2 + apWidthInPlot) * np.sin(theta + np.pi / 2)

        upperShape=LineString([(xApUp1,yApUp1),(xApUp2,yApUp2)])
        lowerShape=LineString([(xApLow1,yApLow1),(xApLow2,yApLow2)])
        return [upperShape,lowerShape]

    def collimation_Factor_First_Lens(self,swarm):
        #this function return a factor descibing how collimated the beam is in the first len. It is simply the gradient
        #of the rms at the lens output divided by the value of what the gradient would be if there was no lens present.
        #if zero perfect collimation, if one then zero collimation, if greater than one, divergin. I also return np.inf
        #if the output is a focus, because there is collimation at both mins and maxes of the rms profile, but mins
        #have high divergence
        yList=[]
        xLensInlet=self.lattice.elList[1].r1[0]
        xLensOutlet=self.lattice.elList[1].r2[0]
        xArr=np.linspace(xLensInlet,xLensOutlet,num=1000)
        myList=[]#slope before the lens, for the projected gradient of rms
        byList=[] #intercept before the lens, for the projected gradient of rms
        for particle in swarm:
            if particle.qArr[-1][0]>xLensOutlet: #only test particles that survived past the lens
                yList.append(particle.yInterp(xArr))
                myList.append(particle.pArr[0][1]/particle.pArr[0][0]) #slope at the origin
                byList.append(particle.qArr[0][1]) #evaluated at the origin, so x=0
        byArr=np.asarray(byList)
        myArr=np.asarray(myList)

        rmsProj=np.sqrt(np.mean((myArr*xLensOutlet+byArr)**2)) #theoretical rms at lens outlet
        rmsSlopeProj=np.mean((myArr*xLensOutlet+byArr)*myArr)/rmsProj
        yArr=np.asarray(yList)
        y_rmsArr=np.sqrt(np.mean(yArr**2,axis=0))
        y_rmsSlopeArr=np.gradient(y_rmsArr,xArr) #remember, the slope always stops changing at the end of the sim element
        #because the fields fall off!
        rmsSlopeActual=y_rmsSlopeArr[-1]
        collimationFactor=np.abs(rmsSlopeActual/rmsSlopeProj)
        height=y_rmsArr[0] #to use the prominence feature. Otherwise find peaks will find teeny peaks that aren't
        #real
        # plt.plot(y_rmsArr)
        # plt.show()
        numPeaks=sps.find_peaks(y_rmsArr,prominence=.1*height)[0].shape[0]
        numTrough=sps.find_peaks(-y_rmsArr,prominence=.1*height)[0].shape[0]
        #if there are no peaks, then the output can't be a focus. If there is at least one, then if there is one or more
        #troughs then the output could be a focus. If there is a peak, the outgoin rms value cannot be below
        #the average of the peak and first rms value. this keeps the output away from the focus
        if numPeaks>=1:
            if numTrough>=1:
                return np.inf
            else:
                minVal=y_rmsArr[0] #to keep the output being near a focus focus
                if y_rmsArr[-1]<minVal:
                    return np.inf
                else:
                    return collimationFactor
        else:
            return collimationFactor

    def get_Image_Distance(self,swarm,thetaMean,offsetMean):
        #this uses the analytic method of finding the focus that minimizes the RMS of a cloud of particles.
        #Simple algebra. the swarm here is the swarm leaving the output of the second lens, and so there is a mean
        #tilt and offset for the cloud. These values are used to correct for that.
        xAfter = self.lattice.elList[3].r2[0] #end of second lens
        bList=[]
        mList=[]
        xArr=np.linspace(xAfter-.1,xAfter+.5,num=1000)
        yList=[]
        for particle in swarm.particles:
            if particle.qArr[-1][0]>xAfter: #only consider particles that have survived through the first lens
                #find m and b in the equation y=m*x+b. I find m and b for each axis right after the first lens
                my=(particle.yInterp(xAfter+1e-6)-particle.yInterp(xAfter))/1e-6 #use forward difference to
                #not include step inside lens where it can be changing, though it probably doesn't matter
                my=my-np.tan(thetaMean) #remove the mean difference
                #mz=(particle.zInterp(xAfter+1e-6)-particle.zInterp(xAfter))/1e-6
                by=(particle.yInterp(xAfter)-offsetMean)-my*xAfter #account for mean difference here as well
                #bz=particle.zInterp(xAfter)-mz*particle.zInterp(xAfter)
                mList.extend([my])#,mz])
                bList.extend([by])#,bz])
                yList.append(my*xArr+by)

                plt.plot(xArr,my*xArr+by,c='black',alpha=.1)
        rmsArr=np.sqrt(np.mean(np.asarray(yList)**2,axis=0))
        mArr=np.asarray(mList)
        bArr=np.asarray(bList)
        xFocus=-np.sum(mArr*bArr)/np.sum(mArr**2) #get absolute position of focus
        Li=xFocus-xAfter #image distance is relative
        print('focus',xFocus)
        print('Li',Li)
        plt.axvline(x=xAfter)
        # plt.gca().set_aspect('equal')
        plt.plot(xArr,rmsArr)
        plt.show()
        return Li
    def test_Tunability_For_Velocity(self,args0,dv=5.0):
        #This method tests the tunability characteristics for deviations if velocity. This is done by finding the knob
        #positions for three velocities, (v0-dv,v0,v0+dv) using newton's method and seeing how much this deviates
        #from the actual value.
        #args0: The coordinates to test at
        #dv: max deviation in velocity
        #apDiam: diameter of second aperture
        #returns: the original knob position, as well as the knob position for high and low coupled with the corresponding
        #error before and after adjustement
        # high and low velocities
        self.initialize_Observed_Swarm(numParticles=1000, dv=-dv)
        FLow = self.perturb_Lattice(args0, [0.0, 0.0])
        self.initialize_Observed_Swarm(numParticles=1000,dv=dv) #make cloud with extra longitudinal velocity
        FHigh=self.perturb_Lattice(args0,[0.0,0.0])
        #now for v0
        self.initialize_Observed_Swarm(numParticles=1000)
        F0=self.perturb_Lattice(args0,[0.0,0.0]) #expansion point.
        jacobian=self.get_Tunability_Jacobian(args0)


        #now find the new knob positions using newton's method, and the error in each parameter. This is done by
        #using the values at F0 as target parameters

        deltaXLow = (np.linalg.inv(jacobian) @ (F0[:,None]- FLow[:,None])).T[0] #newton's method. Need arrays to be column
        #vectors, then I turn them back into 1D arrays
        deltaXHigh = (np.linalg.inv(jacobian) @ (F0[:,None]- FHigh[:,None])).T[0]

        X0=np.asarray([args0['sigma'],self.X0['Lsep']])
        XHigh=X0+deltaXHigh
        XLow=X0+deltaXLow

        #now test the results to see if the tweeks in the knobs give a low error. Need to make the new swarm for
        # each velocity
        self.initialize_Observed_Swarm(numParticles=1000,dv=-dv)
        errorLowPost=100*np.linalg.norm((F0-self.perturb_Lattice(args0,deltaXLow))/F0) #percent error magnitude
        errorLowPre=100*np.linalg.norm((F0-FLow)/F0) #percent error magnitude

        self.initialize_Observed_Swarm(numParticles=1000, dv=dv)
        errorHighPost = 100 * np.linalg.norm((F0 - self.perturb_Lattice(args0, deltaXHigh)) / F0)  # percent error magnitude
        errorHighPre = 100 * np.linalg.norm((F0 - FHigh) / F0)  # percent error magnitude


        return X0,[XLow,(errorLowPre,errorLowPost)],[XHigh,(errorHighPre,errorHighPost)]




    def get_Tunability_Jacobian(self,args0,dSigma=1e-3,dSep=1e-3,dLo=1e-3):
        #this method computes the jacobian of the 'knobs' that are available. The object distance (Lo) for the first
        # lens, the transvers position of the second lens (sigma), and the longitudinal seperation between the second
        # and first lens (dSep). The calculation is done with the central difference derivative.
        #args0: The points the jacobian is calculated at
        #: dSigma: The stepsize for the sigma in derivative calculation
        #: dSigma: The stepsize for the sep in derivative calculation
        #: dSigma: The stepsize for the object distance in derivative calculation


        gradTheta=[0.0,0.0] #dtheta/dsigma, dtheta/dsep
        gradOffset=[0.0,0.0] #doffset/dsigma, doffset/dsep
        #gradLoVirtual=[0.0,0.0,0.0] #dLoVirtual/dsigma, dLoVirtual/dsep, dLoVirtual/dLo.
        jacobian=np.asarray([gradTheta,gradOffset])#,gradLoVirtual])
        #fill out the first row
        deltaX1=np.asarray([dSigma,0.0])
        jacobian[:,0]=(self.perturb_Lattice(args0,deltaX1)-self.perturb_Lattice(args0,-deltaX1))/(2*deltaX1[0])
        #fill out the second row
        deltaX2 = np.asarray([0.0,dSep])
        jacobian[:,1]=(self.perturb_Lattice(args0,deltaX2)-self.perturb_Lattice(args0,-deltaX2))/(2*deltaX2[1])
        #fill out the third row. WORK IN PROGRESS
        # deltaX3 = np.asarray([0.0,0.0, dLo])
        # jacobian[:,2] = (self.perturb_Lattice(args0, deltaX3) - self.perturb_Lattice(args0, -deltaX3)) / (2 * deltaX3[2])


        return jacobian

    def perturb_Lattice(self,args0,deltaX):
        #this function returns the paramters of interest for testing tunability, theta, offset and image distance by
        #varying the parameters of interest with a deviation given by deltax. Ensures the total length stay the same
        #args: The pertubration points, This isn't ALL the paramters, just the ones that aren't constant
        #deltaX: the perturbation. deltaX=[deltaSigma,deltaLsep]
        #return the angle with the axis and the b value as y=np.tan(theta)*x+b. Offset is NOT what I want
        self.updateX_With_List(args0)
        self.X['sigma']=args0['sigma']+deltaX[0]
        self.X['Lsep']=self.X0['Lsep']+deltaX[1]
        # self.X['Lo']=args0['Lo']+deltaX[2] #object length for the first lens
        self.build_Lattice()
        swarm = self.trace_Through_Bumper(parallel=True)
        thetaMean, offsetMean = self.characterize_Output(swarm)
        b=offsetMean-np.tan(thetaMean)*self.lattice.elList[-1].r1[0]
        # LoVirtual=self.get_Virtual_Object_Distance(swarm,self.X['Lo'])
        # swarm = self.project_Swarm_Through_Apeture(swarm, thetaMean, offsetMean, copySwarm=True)
        # apetureObjects = self.make_Apeture_Objects(thetaMean, offsetMean)
        # self.lattice.show_Lattice(swarm=swarm,showTraceLines=True,showMarkers=False,extraObjects=apetureObjects)
        return np.asarray([thetaMean, b])


    def updateX_With_List(self,args):
        #unpack args into the lattice paramters dictionary
        Lm2,rp2,sigma=args
        self.X['Lm2']=Lm2
        self.X['rp2']=rp2
        self.X['sigma']=sigma
    def plot_Swarm_In_Bumper(self,args,numParticles=250):
        self.initialize_Observed_Swarm(numParticles=numParticles)
        minAspectRatio=6.0 #the minimum ratio of length to radius of the lens. I've seen quoted in particle acclerator
        #books that 5 is a good value, but there is also a restriction from COMSOL
        Bp1,Bp2,Lm1,Lm2,rp1,rp2,sigma,Lo=args
        aspectRatioLens1=Lm1/rp1
        aspectRatioLens2=Lm2/rp2
        if aspectRatioLens1<minAspectRatio or aspectRatioLens2<minAspectRatio:  #not a valid solution.
            print('aspect ratio violate')
        self.updateX_With_List(args)
        self.build_Lattice()
        swarm=self.trace_Through_Bumper(parallel=True)
        print(self.collimation_Factor_First_Lens(swarm))
        self.lattice.show_Lattice(swarm=swarm,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,
                                          trueAspectRatio=True)

    def optimize_Output(self,numParticles=1000):
        #this method optimizes the injection system for passing through an apeture.
        #Lo: object length of the system
        #Li: Image length, also where the second apeture should be placed.
        #apDiam: diamter of the second apeture, located at the image
        self.initialize_Observed_Swarm(numParticles=numParticles,magFact=2.0)
        bounds=[(.1,.4),(.01,.05),(-.01,-.05)] #Lm2,rp2,sigma
        totalLengthMax=1.2 #maximum length of system
        transmissionTarget=.95 # fraction of suriving atoms through the first lens
        minAspectRatio=6.0 #the minimum ratio of length to radius of the lens. I've seen quoted in particle acclerator
        #books that 5 is a good value, but there is also a restriction from COMSOL
        offsetApetureTarget=.04 #target value for distance from axis at aperture
        transmissionCutoff=.5 #less than this transmission don't do any other analysis
        imageDistanceTarget=self.X['Li']
        def cost_Function(args,returnResults=False,parallel=False):
            #todo: need to improve this alot. Needs to be move to its own function or something
            Lm2,rp2,sigma=args
            aspectRatioLens2 = Lm2 / rp2
            if aspectRatioLens2<minAspectRatio: #not a valid solution.
                return np.inf
            self.updateX_With_List(args)
            self.build_Lattice()

            for el in self.lattice.elList:
                el.fill_Params()
            swarm = self.trace_Through_Bumper(parallel=parallel)
            cost=0
            if swarm.survival_Bool()<=transmissionCutoff: #sometimes  all or nearly all particles are clipped
                cost+=np.inf
            else:
                thetaMean, offsetMean = self.characterize_Output(swarm)
                # LiSwarm=self.get_Image_Distance(swarm,thetaMean,offsetMean) #image distance of swarm. This should line
                #up with the aperture. This is not very robust because this does not behave well in principle
                postApertureSwarm=self.project_Swarm_Through_Apeture(swarm,thetaMean,offsetMean,copySwarm=True) #this
                #swarm is just used to get the injector to respect the location of the aperture.


                #punish if the bump lens is clipping particles
                if swarm.survival_Bool()<transmissionTarget:  #less than this percent% survival punish severaly
                    cost+=5e2*((transmissionTarget/swarm.survival_Bool())**2-1.0)
                #punish for the aperture clipping the swarms.
                cost+=1e2*((swarm.survival_Bool()/postApertureSwarm.survival_Bool())**2-1)
                #0.996 0.898 1.1091314031180401
                #

                #reward for having greater seperation at the aperture. Punish more severely for exceeding the
                #target
                offsetAtAperture = np.abs(offsetMean + thetaMean * self.X['Li'])
                if offsetAtAperture<offsetApetureTarget:  #punish if the beam isn't seperated enough from the helium
                    #beam at the aperture
                    cost+= 5e2*(1.0-(offsetAtAperture/offsetApetureTarget)**2)  # reduce this cost


                #punish severely for exceeding the total length
                totalLength=self.X['Lo']+self.X['Lm1']+Lm2+self.X['Lsep']+self.X['Li'] #total length of the system.
                if totalLength>totalLengthMax:
                    cost+=5e2*((totalLength/totalLengthMax)**2-1.0)

                print('--------------')
                print(args)
                print(swarm.survival_Bool(),postApertureSwarm.survival_Bool(),offsetAtAperture,totalLength,thetaMean,offsetMean,cost)

                #
                # if swarm.survival_Bool()!=0: #sometimes they can all clip on the apeture if that is being used
                #     if offsetAtAperture<offsetApetureTarget: #punish if the beam isn't seperated enough from the helium
                #         #beam at the aperture
                #         cost+= 5e2*(1.0-(offsetAtAperture/offsetApetureTarget)**2)  # reduce this cost
                #     else: #reward
                #         cost+=-10*(offsetAtAperture/offsetApetureTarget-1.0) #encourage more seperation
                #     # cost+=1e3*np.abs(LiSwarm-imageDistanceTarget)/imageDistanceTarget #force the lens to focus at the
                #     #image distance

                # print('---------------------------------------------------')
                # print(cost)

                #     if totalLength>totalLengthMax:

                # else:
                #     cost+=np.inf
            if returnResults==True:
                if swarm.survival_Bool()!=0:
                    return swarm,swarm.survival_Bool(),offsetAtAperture,aspectRatioLens2,thetaMean,offsetMean
                else:
                    raise Exception('no surviving particles!!')
            return cost
        #Bp1,Bp2,Lm1,Lm2,rp1,rp2,sigma,Lo=args
        args=np.array([ 0.32311644,  0.04562323, -0.03038365])
        t=time.time()
        swarm1,survival1,offsetAtApeture1,aspectRatioLens2_1,thetaMean1,offsetMean1=cost_Function(args,returnResults=True,parallel=True)
        print(time.time()-t) #2.65554141998291
        #0.96 0.034196875755942664 0.9924620100000001 -0.07921540321032133 -0.018353795113878395 -127.32676741402396
        apetureObjects1=self.make_Apeture_Objects(thetaMean1,offsetMean1)
        self.lattice.show_Lattice(swarm=swarm1, showTraceLines=True, showMarkers=False, traceLineAlpha=.1,
                                  trueAspectRatio=True,extraObjects=apetureObjects1)
        sys.exit()

        t=time.time()
        sol1=spo.differential_evolution(cost_Function,bounds,workers=-1,polish=False,disp=True,maxiter=100
                                        ,mutation=(.5,1.0))
        print('run time',time.time()-t)
        print(sol1)

        swarm1,survival,offsetApeture,apertureLens2,thetaMean,offsetMean=cost_Function(sol1.x,returnResults=True,parallel=True)


        print('survival',survival,'offset',offsetApeture)
        apetureObjects1=self.make_Apeture_Objects(thetaMean,offsetMean)
        print('----------------------')



        self.lattice.show_Lattice(swarm=swarm1,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,trueAspectRatio=True,
                                  extraObjects=apetureObjects1)








#
# Lo=.1
# apetureOptimizer=ApetureOptimizer(h=1e-5,apDiam=.009,Li=.2,Lo=Lo)
# apetureOptimizer.optimize_Output()