import matplotlib.pyplot as plt
import time
import numpy as np
import copy
class Swarm:
    #An object that holds a cloud of particles in phase space
    def __init__(self):
        self.particles = [] #list of particles in swarm
    def add_Particle(self, qi=np.asarray([-1e-10, 0.0, 0.0]),pi=np.asarray([-200.0, 0.0, 0.0])):
        #add an additional particle to phase space
        #qi: spatial coordinates
        #pi: momentum coordinates
        self.particles.append(Particle(qi, pi))
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
        self.revolutions=None #revolutions particle makd around lattice
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
    def log_Params(self):
        #this records value like position and momentum
        self.qList.append(self.q.copy())
        self.pList.append(self.p.copy())
        self.TList.append(np.sum(self.p**2)/2.0)
        if self.currentEl is not None:
            qel=self.currentEl.transform_Lab_Coords_Into_Element_Frame(self.q)
            self.qoList.append(self.currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q, self.cumulativeLength))
            self.VList.append(self.currentEl.magnetic_Potential(qel))

    def finished(self,totalLatticeLength=None):
        #finish tracing with the particle, tie up loose ends
        #totalLaticeLength: total length of periodic lattice
        self.traced=True
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
    def plot_Position(self,plotYAxis='y'):
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
        plt.show()

    def copy(self):
        return copy.deepcopy(self)