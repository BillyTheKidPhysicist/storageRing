import matplotlib.pyplot as plt
import time
import numpy as np
import copy
class Swarm:
    #An object that holds a cloud of particles in phase space
    def __init__(self):
        self.particles = [] #list of particles in swarm
    def add_Particle(self, qi=None,pi=None):
        #add an additional particle to phase space
        #qi: spatial coordinates
        #pi: momentum coordinates
        if pi is None:
            pi=np.asarray([-200.0,0.0,0.0])
        if qi is None:
            qi = np.asarray([-1e-10, 0.0, 0.0])

        self.particles.append(Particle(qi, pi))
    def vectorize(self,onlyUnclipped=False):
        #return position and momentum vectors for the particle swarm
        qVec=[]
        pVec=[]
        for particle in self.particles:
            if onlyUnclipped==True:
                if particle.clipped==False:
                    qVec.append(particle.q)
                    pVec.append(particle.p)
            else:
                qVec.append(particle.q)
                pVec.append(particle.p)
        qVec=np.asarray(qVec)
        pVec=np.asarray(pVec)
        return qVec,pVec
    def survival_Rev(self):
        #return average number of revolutions of particles
        revs=0
        for particle in self.particles:
            if particle.clipped is None:
                raise Exception('PARTICLE HAS NOT BEEN TRACED')
            elif particle.revolutions is not None:
                revs+=particle.revolutions

        meanRevs=revs/self.num_Particles()
        return meanRevs
    def longest_Particle_Life_Revolutions(self):
        #return number of revolutions of longest lived particle
        maxList=[]
        for particle in self.particles:
            if particle.revolutions is not None:
                maxList.append(particle.revolutions)
        if len(maxList)==0:
            return 0.0
        else:
            return max(maxList)
    def survival_Bool(self, frac=True):
        #returns fraction of particles that have survived, ie not clipped.
        #frac: if True, return the value as a fraction, the number of surviving particles divided by total particles
        numSurvived = 0.0
        for particle in self.particles:
            if particle.clipped is None:
                raise Exception('PARTICLE HAS NOT BEEN TRACED')
            numSurvived += float(not particle.clipped) #if it has NOT clipped then turn that into a 1.0
        if frac == True:
            return numSurvived / len(self.particles)
        else:
            return numSurvived
    def __iter__(self):
        return (particle for particle in self.particles)
    def copy(self):
        return copy.deepcopy(self)
    def num_Particles(self):
        return len(self.particles)
    def reset(self):
        #reset the swarm.
        for particle in self.particles:
            particle.reset()
class Particle:
    #This object represents a single particle with unit mass. It can track parameters such as position, momentum, and
    #energies, though these are computationally intensive and are not enabled by default. It also tracks where it was
    # clipped if a collision with an apeture occured, the number of revolutions before clipping and other parameters of
    # interest.
    def __init__(self,qi=np.asarray([-1e-10, 0.0, 0.0]),pi=np.asarray([-200.0, 0.0, 0.0])):
        self.q=qi
        self.p=pi
        self.qi=qi.copy()#initial coordinates
        self.pi=pi.copy()#initial coordinates
        self.T=0 #time of particle in simulation
        self.traced=False #recored wether the particle has already been sent throught the particle tracer
        self.v0=np.sqrt(np.sum(pi**2)) #initial speed

        self.force=None #current force on the particle
        self.currentEl=None #which element the particle is ccurently in
        self.currentElIndex=None #Index of the elmenent that the particle is curently in. THis remains unchanged even
        #after the particle leaves the tracing algorithm and such can be used to record where it clipped
        self.cumulativeLength=0 #total length traveled by the particle IN TERMS of lattice elements. It updates after
        #the particle leaves an element by adding that elements length (particle trajectory length that is)
        self.revolutions=0 #revolutions particle makd around lattice. Initially zero
        self.clipped=None #wether particle clipped an apeture
        #these lists track the particles momentum, position etc during the simulation if that feature is enable. Later
        #they are converted into arrays
        self.pList=[] #List of momentum vectors
        self.qList=[] #List of position vector
        self.qoList=[] #List of position in orbit frame vectors
        self.TList=[] #kinetic energy list
        self.VList=[] #potential energy list
        #array versions
        self.pArr=None
        self.qArr=None 
        self.qoArr=None 
        self.TArr=None 
        self.VArr=None 
        self.EArr=None #total energy
        self.yInterp=None #interpolating function as a function of x or s (where s is orbit trajectory analog of x)
        self.zInterp=None #interpolating function as a function of x or s (where s is orbit trajectory analog of x)
    def reset(self):
        #reset the particle
        self.__init__(qi=self.qi,pi=self.pi)
    def log_Params(self,currentEl,qel,pel):
        #this records value like position and momentum
        #qel: element position coordinate
        #pel: momentum position coordinate
        self.q = currentEl.transform_Element_Coords_Into_Lab_Frame(qel)
        self.p = currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(pel)
        self.qList.append(self.q.copy())
        self.pList.append(self.p.copy())
        self.TList.append(np.sum(self.p**2)/2.0)
        if currentEl is not None:
            qel=currentEl.transform_Lab_Coords_Into_Element_Frame(self.q)
            self.qoList.append(currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q, self.cumulativeLength))
            self.VList.append(currentEl.magnetic_Potential(qel))
    def log_Params_In_Drift_Region(self,qEli,pEli,qElf,h,currentEl):
        #log the parameters when the particle has traveled in a straight line inside a drift region
        #qi: initial position
        #pi: initial momentum
        #qf: final position
        #h: stepsize
        T=(qElf[0]-qEli[0])/pEli[0]
        timeSteps=int(T/h)+1
        if timeSteps<=1:
            timeSteps=2
        TArr=np.linspace(0,T,num=timeSteps)
        qElArr=qEli+TArr[:,np.newaxis]*pEli #trick to muliply across rows
        pElArr=np.ones((TArr.shape[0],3))*pEli #to get the right shape
        #the first point is a duplicate because it was already logged by particleTracer.initialize, or right after the
        #particle entered drift element from another element
        qElArr=qElArr[1:]
        pElArr=pElArr[1:]
        qList=[]
        pList=[]
        for i in range(qElArr.shape[0]):
            qList.append(currentEl.transform_Element_Coords_Into_Lab_Frame(qElArr[i]))
            pList.append(currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(pElArr[i]))
        #now fill the lists that log parameters
        self.qList.extend(qList)
        self.pList.extend(pList)
        self.TList.extend(list(np.sum(pElArr**2/2.0,axis=1)))
        self.qoList.extend(qList)
        self.VList.extend([0]*len(qList)) #python list creation trick
    def finished(self,totalLatticeLength=None):
        #finish tracing with the particle, tie up loose ends
        #totalLaticeLength: total length of periodic lattice
        self.traced=True
        self.force=None
        self.qArr=np.asarray(self.qList)
        self.qList = []  # save memory
        self.pArr = np.asarray(self.pList)
        self.pList = []  
        self.qoArr = np.asarray(self.qoList)
        self.qoList = []


        self.TArr = np.asarray(self.TList)
        self.TList = []
        self.VArr = np.asarray(self.VList)
        self.VList = []
        self.EArr=self.TArr+self.VArr
        if self.currentEl is not None: #This option is here so the particle class can be used in situation beside ParticleTracer
            self.currentElIndex=self.currentEl.index
            if totalLatticeLength is not None:
                qo=self.currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q, self.cumulativeLength)
                self.revolutions=qo[0]/totalLatticeLength
            self.currentEl=None # to save memory
    def plot_Energies(self):
        if self.EArr.shape[0]==0:
            raise Exception('PARTICLE HAS NO LOGGED POSITION')
        EArr = self.EArr
        TArr=self.TArr
        VArr=self.VArr
        qoArr=self.qoArr
        plt.close('all')
        plt.plot(qoArr[:,0],EArr-EArr[0],label='E')
        plt.plot(qoArr[:, 0], TArr - TArr[0],label='T')
        plt.plot(qoArr[:, 0], VArr - VArr[0],label='V')
        plt.legend()
        plt.grid()
        plt.show()
    def plot_Orbit_Reference_Frame_Position(self, plotYAxis='y'):
        if plotYAxis!='y' and plotYAxis!='z':
            raise Exception('plotYAxis MUST BE EITHER \'y\' or \'z\'')
        if self.qoArr.shape[0]==0:
            raise Exception('PARTICLE HAS NO LOGGED POSITION')
        qoArr=self.qoArr
        if plotYAxis=='y':
            yPlot=qoArr[:,1]
        else:
            yPlot = qoArr[:, 2]
        plt.close('all')
        plt.plot(qoArr[:,0],yPlot)
        plt.ylabel('Trajectory offset, m')
        plt.xlabel('Trajectory length, m')
        plt.grid()
        plt.show()

    def copy(self):
        return copy.deepcopy(self)