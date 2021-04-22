import scipy.misc as spm
import scipy.interpolate as spi
import warnings
import time
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt
# from storageRingOptimization import elementPT
# from storageRingOptimization.ParticleTracer import ParticleTracer
from storageRingOptimization.ParticleClass import  Swarm,Particle
# import storageRingOptimization.elementPT
from storageRingOptimization.particleTracerLattice import ParticleTracerLattice
from storageRingOptimization.SwarmTracer import SwarmTracer
from shapely.geometry import LineString
# from profilehooks import profile

class ApetureOptimizer:
    def __init__(self,Bp1=.9,Bp2=.9,Li=.2,apDiam=.01,h=1e-5,Lsep=.05):
        self.lattice=None
        self.swarmInitial=None
        self.h=None #timestep
        self.v0Nominal=200.0
        self.apDiam=apDiam
        self.X0={'Bp1':Bp1,'Bp2':Bp2,'Li':Li,'apDiam':apDiam,'Lsep':Lsep} #dictionary of lattice paramters that are not
        #varied by the optimizer. This is used in both tunabiliyt testing and the differential evoluiton
        self.X={'rp1':None,'rp2':None,'sigma':None,'Lo':None,'Li':None,'apDiam':None} #the full fictionary of lattice paramters. Need
        #to add the X0 params
        self.X.update(self.X0) #now add the rest
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
            if particle.q[0]>L: #only particle that have survived
                y=particle.yInterp(L)
                z=particle.zInterp(L)
                qList.append([L,y,z])
        spotSize=self.get_Frac_Width(np.asarray(qList),fraction=fraction)
        return spotSize
    def update_Lattice(self):
        self.lattice = ParticleTracerLattice(self.v0Nominal)
        self.lattice.add_Drift(self.X['Lo'])
        self.lattice.add_Lens_Ideal(self.X['Lm1'], self.X['Bp1'], self.X['rp1'], ap=self.X['rp1']*.9)
        self.lattice.add_Drift(self.X['Lsep'])
        self.lattice.add_Bump_Lens_Ideal(self.X['Lm2'], self.X['Bp2'], self.X['rp2'], self.X['sigma'], ap=self.X['rp2']*.9)
        self.lattice.add_Drift(self.X['Li']+.05,ap=2*.1*self.X['Li'])
        self.lattice.end_Lattice(enforceClosedLattice=False, latticeType='injector', surpressWarning=True)
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
    def get_Virtual_Object_Distance(self, swarm, Lo):
        #this method finds the object distance after the swarm has traveled through the first lens. This is done by
        #using the slope of the beam to estimate the object distance with the initial object distance. Only uses y values
        xBefore=self.lattice.elList[1].r1[0]
        xAfter=self.lattice.elList[1].r2[0]
        mBefore=0 #slope
        mAfter=0
        numParticles=0
        for particle in swarm:
            if particle.clipped is False: #only consider particles that havn't clip
                #flip the sign for particles below the y=0 line
                mBefore+=np.sign(particle.yInterp(xAfter))*(particle.yInterp(xBefore+1e-6)-particle.yInterp(xBefore))/1e-6
                mAfter+=np.sign(particle.yInterp(xAfter))*(particle.yInterp(xAfter+1e-6)-particle.yInterp(xAfter))/1e-6
                numParticles+=1
        mBefore=mBefore/numParticles
        mAfter=mAfter/numParticles
        LoVirtual=(mBefore/mAfter)*Lo+(xAfter-xBefore) #the ratio of the angle plus the extra distance traveled in the
        #lens is propotional (I hope, and seems to be) to the virtual object distance
        return LoVirtual
    def test_Tunability(self,args0,dSigma=1e-3,dSep=1e-3,h=1e-5,apDiam=.01):
        #This method varies a few relevant parameters to quantify the tunability of the system. It varies the position
        #of lens1 and lens2 to measure how these affect the image distance, and slope and offset of the output. This
        #is quantified by computing the jacobian and hessian
        #args0: The coordinates to expand around. Lm1,Lm2,rp1,rp2,sigma,Lo
        self.h=h
        self.X['apDiam']=apDiam
        deltaX=np.asarray([dSigma,dSep])
        self.get_Gradients(args0,deltaX)
        # dSigmaArr=np.asarray([-1.0,0.0,1.0])*dSigma
        # dSepArr=np.asarray([-1.0,0.0,1.0])*dSep
        # deltaXArr=np.asarray(np.meshgrid(dSigmaArr,dSepArr)).T.reshape(-1,2)
        # print(deltaXArr)
    def get_Gradients(self,args0,deltaX):
        gradTheta=[0.0,0.0] #dtheta/dsigma, doffset/dsigma
        gradOffset=[0.0,0.0] #dtheta/dsep, doffset/dsep
        jacobian=np.asarray([gradTheta,gradOffset])
        #fill out the jacobian, but need to wrangle later
        jacobian[0]=(self.perturb_Lattice(args0,deltaX)-self.perturb_Lattice(args0,-deltaX))/(2*deltaX)
        jacobian[1]=(self.perturb_Lattice(args0,deltaX)-self.perturb_Lattice(args0,-deltaX))/(2*deltaX)
        #jacobian is not in the right form

        print(jacobian)


    def perturb_Lattice(self,args0,deltaX):
        #this function returns the paramters of interest for testing tunability, theta, offset and image distance by
        #varying the parameters of interest with a deviation given by deltax. Ensures the total length stay the same
        #args: The pertubration points, This isn't ALL the paramters, just the ones that aren't constant
        #deltaX: the perturbation. deltaX=[deltaSigma,deltaLsep]
        self.updateX(args0)
        self.X['Lsep']=self.X0['Lsep']+deltaX[1]
        self.X['Li']=self.X0['Li']-deltaX[1]#preseve total lattice length
        self.X['sigma']=args0[4]+deltaX[0]
        self.update_Lattice()
        swarm = self.trace_Through_Bumper(parallel=True)
        thetaMean, offsetMean = self.characterize_Output(swarm)
        # swarm = self.project_Swarm_Through_Apeture(swarm, thetaMean, offsetMean, copySwarm=True)
        # apetureObjects = self.make_Apeture_Objects(thetaMean, offsetMean)
        # self.lattice.show_Lattice(swarm=swarm,showTraceLines=True,showMarkers=False,extraObjects=apetureObjects)
        return np.asarray([thetaMean, offsetMean])

    def updateX(self,args):
        #unpack args into the lattice paramters dictionary
        self.X['Lm1']=args[0]
        self.X['Lm2']=args[1]
        self.X['rp1']=args[2]
        self.X['rp2']=args[3]
        self.X['sigma']=args[4]
        self.X['Lo']=args[5]
    def optimize_Output(self):
        #this method optimizes the injection system for passing through an apeture.
        #Lo: object length of the system
        #Bp2: Pole face strength of the second lens, ie the shifter
        #Li: Image length, also where the second apeture should be placed.
        #apDiam: diamter of the second apeture, located at the image
        bounds=[(.1,.4),(.1,.4),(.01,.05),(.01,.05),(-.01,-.05),(.075,.3)] #Lm1,Lm2,rp1,rp2,sigma,Lo
        totalLengthCutoff=1.0 #maximum length of system
        virtualObjectCutoff=1.0 # the distance to the virtual object from the second lens
        transmission=.95 # fraction of suriving atoms
        minAspectRatio=5.0 #the minimum ratio of length to radius of the lens
        def cost_Function(args,returnResults=False,parallel=False):
            #todo: need to improve this alot. Needs to be move to its own function or something
            Lm1,Lm2,rp1,rp2,sigma,Lo=args
            self.update_Lattice(args)
            for el in self.lattice.elList:
                el.fill_Params()
            swarm = self.trace_Through_Bumper(parallel=parallel)
            cost=0
            if swarm.survival_Bool()==0: #sometimes  all particles are clipped
                cost+=np.inf
            else:
                swarmPreClipped=swarm #the swarm before clipping at the apeture. This is for testing wether the first
                #lens collimated the beam.
                thetaMean, offsetMean = self.characterize_Output(swarm)
                swarm=self.project_Swarm_Through_Apeture(swarm,thetaMean,offsetMean,copySwarm=True)
                if swarm.survival_Bool()!=0: #sometimes they can all clip on the apeture also
                    LoVirtual = self.get_Virtual_Object_Distance(swarmPreClipped, Lo)
                    offsetAtApeture = np.abs(offsetMean + thetaMean * self.Li)
                    cost+= 10.0 / offsetAtApeture  # reduce this cost
                    totalLength=Lo+Lm1+Lm2+self.Lsep+self.Li #total length of the system.
                    collimaterPhase=Lm1*np.sqrt(2*self.lattice.u0*self.Bp1/(self.v0Nominal**2*rp1**2)) #total phase advance
                    #in the first lens. I use this to prevent multiple oscillations.
                    aspectRatioLens1=Lm1/rp1
                    aspectRatioLens2=Lm2/rp2
                    if swarm.survival_Bool()<transmission: #less than 90% survival punish severaly
                        cost+=5e2*((transmission/swarm.survival_Bool())**2-1.0)
                    if np.abs(LoVirtual)<virtualObjectCutoff: #punish if the object distance is too small
                       cost+=5e2*((virtualObjectCutoff/np.abs(LoVirtual))**2-1.0)
                    if collimaterPhase>np.pi: #the system is in a region with a focus inside the first len, not something
                        #I want
                        cost+=np.inf
                    if totalLength>totalLengthCutoff:
                        cost+=5e2*((totalLength/totalLengthCutoff)**2-1.0)
                    if aspectRatioLens1<minAspectRatio: #punish because the aspect ratio is below the rule of thumb
                        cost+=5e2*((aspectRatioLens1/minAspectRatio)**2-1.0)
                    if aspectRatioLens2<minAspectRatio:
                        cost+=5e2*((aspectRatioLens2/minAspectRatio)**2-1.0)

                    # print(args,swarm.survival_Bool(),offsetAtApeture,LoVirtual,totalLength,thetaMean,offsetMean,cost)
                    print('thetaMean',thetaMean)
                else:
                    cost+=np.inf
            if returnResults==True:
                if swarm.survival_Bool()!=0:
                    return swarm,thetaMean,offsetMean,swarm.survival_Bool(),offsetAtApeture,LoVirtual
                else:
                    return swarm
            return cost
        t=time.time()
        # sol1=spo.differential_evolution(cost_Function,bounds,workers=-1,polish=False,disp=True,maxiter=200,mutation=(.5,1.0))
        # print('run time',time.time()-t)
        # print(sol1)
        args=np.array([ 0.10293192,  0.3442374 ,  0.02746703,  0.04993316, -0.0367223 ,
        0.13033733]) #Lm1,Lm2,rp1,rp2,sigma,Lo
        swarm1,thetaMean,offsetMean,survival,offsetApeture,LoVirtual=cost_Function(args,returnResults=True,parallel=True)


        print('distance',LoVirtual)
        print('survival',survival,'offset',offsetApeture)
        apetureObjects1=self.make_Apeture_Objects(thetaMean,offsetMean)
        print('----------------------')



        # t = time.time()
        # sol2 = spo.differential_evolution(cost_Function, bounds, workers=-1, polish=False, disp=True, maxiter=100,
        #                                   mutation=(.5, 1.0))
        # print('run time', time.time() - t)
        # print(sol2)
        # swarm2, thetaMean, offsetMean, survival, offsetApeture, LoVirtual = cost_Function(sol2.x, returnResults=True)
        # print(time.time() - t)
        # print('distance', LoVirtual)
        # print('survival', survival, 'offset', offsetApeture)
        # apetureObjects2 = self.make_Apeture_Objects(Li, apDiam, thetaMean, offsetMean)


        # self.lattice.show_Lattice(swarm=swarm1,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,trueAspectRatio=True,
        #                           extraObjects=apetureObjects1)
        # self.lattice.show_Lattice(swarm=swarm2,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,trueAspectRatio=True,
        #                           extraObjects=apetureObjects2)







#
# Lo=.1
# Bp1=.33
# rp1=.011
# Lm1=.1
# LSep=.05
# Bp2=1.0
# rp2=3e-2
# Lm2=.15
# sigma=-.02
# lattice=ParticleTracerLattice(200.0)
# lattice.add_Drift(Lo)
# lattice.add_Lens_Ideal(Lm1,Bp1,rp1)
# lattice.add_Drift(LSep)
# lattice.add_Bump_Lens_Ideal(Lm2,Bp2,rp2,sigma,ap=rp2-.001)
# lattice.add_Drift(.4,ap=.1)
# lattice.end_Lattice(enforceClosedLattice=False,latticeType='injector')


apetureOptimizer=ApetureOptimizer()
# apetureOptimizer.initialize_Ideal_Point_Swarm(numParticles=1000,maxAng=.1)
apetureOptimizer.initialize_Observed_Swarm(numParticles=1000)
# apetureOptimizer.lattice=lattice
# swarm=apetureOptimizer.trace_Through_Bumper(1e-5)
# theta,offset=apetureOptimizer.characterize_Output(swarm)
# test=apetureOptimizer.make_Apeture_Objects(.1,.005,theta,offset)
# swarm=apetureOptimizer.project_Swarm_Through_Apeture(swarm,.1,.005,theta,offset)

#
# print(apetureOptimizer.characterize_Output(swarm))
# apetureOptimizer.optimize_Output()
X0 = np.array([0.10293192, 0.3442374, 0.02746703, 0.04993316, -0.0367223,
                 0.13033733])  # Lm1,Lm2,rp1,rp2,sigma,Lo
apetureOptimizer.test_Tunability(X0)
# lattice.show_Lattice(swarm=swarm,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,trueAspectRatio=True,extraObjects=test)
