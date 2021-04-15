import scipy.interpolate as spi
import warnings
import time
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt
from storageRingOptimization import elementPT
from storageRingOptimization.ParticleTracer import ParticleTracer
from storageRingOptimization.ParticleClass import  Swarm,Particle
import storageRingOptimization.elementPT
from storageRingOptimization.particleTracerLattice import ParticleTracerLattice
from storageRingOptimization.SwarmTracer import SwarmTracer
from shapely.geometry import LineString

class ApetureOptimizer:
    def __init__(self):
        self.lattice=None
        self.swarmInitial=None
    def initialize_Observed_Swarm(self,numParticles=None,fileName='../storageRingOptimization/phaseSpaceCloud.dat'):
        #load a swarm from a file.
        #numParticles: Number of particles to load from the file. if None load all
        #fileName: Name of the file with the particle data in the format (y,z,px,py,pz)
        self.swarmInitial=Swarm()
        data= np.loadtxt(fileName)
        if numParticles is not None:
            data=data[:numParticles] #restrict the data
        for coords in data:
            x,y,z,px,py,pz=coords
            px=-px #need to flip the coords!!
            q=np.asarray([1e-10,y,z]) #add a tiny offset in the x direction so the particle starts in the element,
            #otherwise the algorithm spends extra time checking to see if the particle is exactly on the edge
            p=np.asarray([px,py,pz])
            self.swarmInitial.add_Particle(qi=q,pi=p)
    def initialize_Ideal_Point_Swarm(self,numParticles=1000,maxAng=.05):
        self.swarmInitial = Swarm()
        x=np.zeros(numParticles)
        y=x.copy()
        z=x.copy()
        x+=1e-10
        px=200.0
        py=(np.random.rand(numParticles)-.5)*200*np.tan(maxAng)
        pz=(np.random.rand(numParticles)-.5)*200*np.tan(maxAng)
        for i in range(numParticles):
            q=np.asarray([x[i],y[i],z[i]])
            p=np.asarray([px,py[i],pz[i]])
            self.swarmInitial.add_Particle(qi=q,pi=p)



    def move_Swarm_Through_Free_Space(self,swarm,L):
        #Move a swarm through free space by simple explicit euler. Distance is along the x axis. Moves TO a position
        #swarm: swarm to move
        #L: the distance to move to, positive or negative, along the x axis
        #Return: A new swarm object
        for particle in swarm:
            q=particle.q
            p=particle.p
            dt=(L-q[0])/p[0]
            q=q+p*dt
            particle.q=q
        return swarm
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
    def get_Frac_Width(self,qArr,fraction=.95):
        #get the width of the beam that captures fraction of the atoms
        #fraction: fraction of the atoms in that width
        if not 0<fraction<1.0:
            raise Exception('fraction must be number between 0 and 1')
        numParticles=qArr.shape[0]
        if numParticles<100:
            warnings.warn('Low number of particles will result in larger error with this method')
        rArr=np.sqrt(qArr[:,1]**2+qArr[:,2]**2)
        rArrSorted=np.sort(rArr)
        cutoffIndex=int(fraction*numParticles) #the 90% particle. There is some minor rounding error
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
    def trace_Through_Bumper(self,h):
        swarmTracer = SwarmTracer(self.lattice)
        swarm = swarmTracer.trace_Swarm_Through_Lattice(self.swarmInitial, h, 1.0, fastMode=False,parallel=True)
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
        #calculate the mean tilt and offset of the output beam. Offset is the offset at the lens output
        #swarm: The traced swarm
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
    def project_Swarm_Through_Apeture(self, swarm, Li, apetureDiam, theta, offset):
        #clip particles in the swarm that clip the lens. This is done by placing a perpindicular line
        #at a distance Li from the end of the last lens along the line given by theta and offset
        #swarm: The swarm to test. Needs to have been traced already
        #Li: Image distance. This is the distance from the last lens to the apeture
        #apSize: the diameter of the apeture
        #theta: the tilt of the outgoing beam
        #offset: the displacement of the outgoing beam from y=0 at the last element
        swarmNew=swarm.copy()

        #All the below coordinates are in the x,y plane, ie the plane parallel to the ground
        #get m and b for the nominal trajectory
        mTraj=np.tan(theta)
        bTraj=offset-mTraj*(self.lattice.elList[-1].r1[0])
        #location of the nominal trajectory at the last lens
        x0=(self.lattice.elList[-1].r1[0])
        y0=offset
        #get m and b for the apeture line
        mAp=-1/mTraj
        xAp=x0+np.cos(theta)*Li
        yAp=y0+np.sin(theta)*Li
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
                deltaz=zAtom*0

                delta=np.sqrt(deltay**2+deltaz**2)
                if delta>apetureDiam/2:
                    particle.clipped=True
        return swarmNew




    def get_Spot_Size(self,swarm,L,fraction=.95):
        #swarm: swarm to use to calculate spot size
        #L: distance from focus to calculate spot size at. This is in the trajectory reference frame so bend through
        #the bumper
        qList=[]
        for particle in swarm:
            if particle.q[0]>L: #only particle that have survived
                y=particle.yInterp(L)
                z=particle.zInterp(L)
                qList.append([L,y,z])
        spotSize=self.get_Frac_Width(np.asarray(qList),fraction=fraction)
        return spotSize
    def update_Lattice(self,args):
        Lm1,Lm2,Bp1,Bp2,rp1,rp2,sigma,LSep,Lo = args
        self.lattice = ParticleTracerLattice(200.0)
        self.lattice.add_Drift(Lo)
        self.lattice.add_Lens_Ideal(Lm1, Bp1, rp1)
        self.lattice.add_Drift(LSep)
        self.lattice.add_Bump_Lens_Ideal(Lm2, Bp2, rp2, sigma, ap=rp2 - .001)
        self.lattice.add_Drift(.5,ap=.5)
        self.lattice.end_Lattice(enforceClosedLattice=False, latticeType='injector', surpressWarning=True)
    def make_Apeture_Objects(self,Li,apDiam,theta,offset):
        #create shapely object to represent the apeture for plotting purposes
        apWidthInPlot=.05 #the width of the apeture for plotting purposes. This is the distance from the beginning
        #of one side of the apeture to the end
        x0 = (self.lattice.elList[-1].r1[0])
        y0 = offset
        xAp = x0 + np.cos(theta) * Li
        yAp = y0 + np.sin(theta) * Li
        #make upper portion of apeture
        xApUp1=xAp+(apDiam/2)*np.cos(theta+np.pi/2)
        yApUp1=yAp+(apDiam/2)*np.sin(theta+np.pi/2)
        xApUp2 = xAp + (apDiam+apWidthInPlot) * np.cos(theta + np.pi / 2)
        yApUp2 = yAp + (apDiam+apWidthInPlot) * np.sin(theta + np.pi / 2)
        #make lower shape
        xApLow1 = xAp -(apDiam/2) * np.cos(theta + np.pi / 2)
        yApLow1 = yAp -(apDiam/2) * np.sin(theta + np.pi / 2)
        xApLow2 = xAp -(apDiam/2 + apWidthInPlot) * np.cos(theta + np.pi / 2)
        yApLow2 = yAp -(apDiam/2 + apWidthInPlot) * np.sin(theta + np.pi / 2)

        upperShape=LineString([(xApUp1,yApUp1),(xApUp2,yApUp2)])
        lowerShape=LineString([(xApLow1,yApLow1),(xApLow2,yApLow2)])
        return [upperShape,lowerShape]
    def optimize_Output(self,Lo=.1,Bp2=1.0,Li=.2):
        bounds=[(.05,.2),(.05,.2),(.01,.4),(.005,.03),(.005,.03),(0.0,-.03),(.025,.1)] #Lm1,Lm2,Bp1,rp1,rp2,sigma,LSep
        def cost_Function(args,printVals=False):
            argsLattice=list(args)
            argsLattice.append(Lo)
            argsLattice.insert(3,Bp2)
            self.update_Lattice(argsLattice)
            for el in self.lattice.elList:
                el.fill_Params()
            swarm = self.trace_Through_Bumper(1e-5)
            if swarm.survival_Bool()==0: #sometimes all particles are clipped
                cost2=np.inf
                cost1=np.inf
            else:

                thetaMean, offsetMean = self.characterize_Output(swarm)
                swarm=self.project_Swarm_Through_Apeture(swarm,Li,.005,thetaMean,offsetMean)
                if swarm.survival_Bool()!=0: #now they can all clip on the apeture
                    offsetAtApeture = np.abs(offsetMean + thetaMean * Li)
                    print(offsetAtApeture,swarm.survival_Bool())
                    cost1 = 1 / offsetAtApeture  # reduce this cost
                    cost2=10*1/swarm.survival_Bool()
                else:
                    cost1=np.inf
                    cost2=np.inf
            if printVals==True:
                print(1/cost1,thetaMean,offsetMean,1/(cost2/10))
                return swarm

            return cost1+cost2
        t=time.time()
        sol=spo.differential_evolution(cost_Function,bounds,workers=-1,polish=False,maxiter=5,disp=True,popsize=4)
        print('run time',time.time()-t)
        print(sol)
        swarm=cost_Function(sol.x,printVals=True)
        self.lattice.show_Lattice(swarm=swarm,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,trueAspectRatio=True)







#
Lo=.1
Bp1=.33
rp1=.011
Lm1=.1
LSep=.05
Bp2=1.0
rp2=3e-2
Lm2=.15
sigma=-.02
lattice=ParticleTracerLattice(200.0)
lattice.add_Drift(Lo)
lattice.add_Lens_Ideal(Lm1,Bp1,rp1)
lattice.add_Drift(LSep)
lattice.add_Bump_Lens_Ideal(Lm2,Bp2,rp2,sigma,ap=rp2-.001)
lattice.add_Drift(.4,ap=.1)
lattice.end_Lattice(enforceClosedLattice=False,latticeType='injector')
# lattice.show_Lattice()

apetureOptimizer=ApetureOptimizer()
# apetureOptimizer.initialize_Ideal_Point_Swarm(numParticles=1000,maxAng=.1)
apetureOptimizer.initialize_Observed_Swarm(numParticles=1000)
apetureOptimizer.lattice=lattice
swarm=apetureOptimizer.trace_Through_Bumper(1e-5)
theta,offset=apetureOptimizer.characterize_Output(swarm)
test=apetureOptimizer.make_Apeture_Objects(.1,.005,theta,offset)
swarm=apetureOptimizer.project_Swarm_Through_Apeture(swarm,.1,.005,theta,offset)

#
# print(apetureOptimizer.characterize_Output(swarm))
# apetureOptimizer.optimize_Output()

lattice.show_Lattice(swarm=swarm,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,trueAspectRatio=True,extraObjects=test)
