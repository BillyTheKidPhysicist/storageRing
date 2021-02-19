import time
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
    def longest_Particle_Life(self):
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
    def copy(self):
        return copy.deepcopy(self)
    def num_Particles(self):
        return len(self.particles)
class Particle:
    def __init__(self,qi=np.asarray([-1e-10, 0.0, 0.0]),pi=np.asarray([-200.0, 0.0, 0.0])):
        self.q=qi
        self.p=pi
        self.qi=qi.copy()#initial coordinates
        self.pi=pi.copy()#initial coordinates
        self.m=1.0 #mass is equal to 1 kg. This is not used anywhere
        self.T=0 #time of particle in simulation
        self.v0=np.sqrt(np.sum(pi**2))/self.m #initial speed

        self.force=None #current force on the particle
        self.currentEl=None #which element the particle is ccurently in
        self.currentElIndex=None
        self.cumulativeLength=0
        self.revolutions=None #revolutions particle makes around lattice
        self.clipped=None #wether particle clipped an apeture
        self.pList=[] #List of momentum vectors
        self.qList=[] #List of position vector
        self.qoList=[] #List of position in orbit frame vectors
        self.TList=[] #kinetic energy list
        self.VList=[] #potential energy list
        self.pArr=None #array version
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

    def copy(self):
        return copy.deepcopy(self)