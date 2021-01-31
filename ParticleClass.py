import numpy as np
import copy
class Swarm:
    #An object that holds a cloud of particles in phase space
    def __init__(self):
        self.particles = [] #list of particles in swarm


    def add_Particle(self, qi, pi):
        #add an additional particle to phase space
        #qi: spatial coordinates
        #pi: momentum coordinates
        self.particles.append(Particle(qi, pi))
    def survival(self, frac=True):
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
    def copy(self):
        return copy.deepcopy(self)
    def num_Particles(self):
        return len(self.particles)
class Particle:
    def __init__(self,qi,pi):
        self.q=qi
        self.p=pi
        self.qi=qi#initial coordinates
        self.pi=pi#initial coordinates
        self.m=1.0 #mass is equal to 1 kg. This is not used anywhere
        self.T=0 #time of particle in simulation
        self.v0=np.sqrt(np.sum(pi**2))/self.m #initial speed

        self.force=None #current force on the particle
        self.currentEl=None #which element the particle is ccurently in
        self.currentElIndex=None
        self.cumulativeLength=None
        self.clipped=None #wether particle clipped an apeture
        self.el=None #current element that the particle is in
        self.pList=[] #List of momentum vectors
        self.qList=[] #List of position vector
        self.qoList=[] #List of position in orbit frame vectors
        self.pArr=None #array version
        self.qArr=None #array version
        self.qoArr=None #array version
    def log_Params(self):
        #this records value like position and momentum
        self.qList.append(self.q.copy())
        self.pList.append(self.p.copy())
        if self.currentEl is not None:
            self.qoList.append(self.currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q, self.cumulativeLength))
    def finished(self):
        #finish tracing with the particle, tie up loose ends
        self.qArr=np.asarray(self.qList)
        self.qList = []  # save memory
        self.pArr = np.asarray(self.pList)
        self.pList = []  # save memory
        self.qoArr = np.asarray(self.qoList)
        self.qoList = []  # save memory
        if self.currentEl is not None:
            self.currentElIndex=self.currentEl.index
            self.currentEl=None # to save memory
    def copy(self):
        return copy.deepcopy(self)