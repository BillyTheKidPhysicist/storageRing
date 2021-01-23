import numpy.linalg as npl
import numpy as np
class Particle():
    def __init__(self,qi,pi):
        self.q=qi
        self.p=pi
        self.qi=qi#initial coordinates
        self.pi=pi#initial coordinates
        self.m=1.0 #mass is equal to 1 kg. This is not used anywhere
        self.T=0 #time of particle in simulation
        self.v0=npl.norm(pi)/self.m #initial speed

        self.force=None #current force on the particle
        self.currentEl=None #which element the particle is ccurently in
        self.currentElIndex=None
        self.cumulativeLength=None
        self.clipped=True #wether particle clipped an apeture
        self.el=None #current element that the particle is in
        self.pList=[] #List of momentum vectors
        self.qList=[] #List of position vector
        self.qoList=[] #List of position in orbit frame vectors
        self.pArr=None #array version
        self.qArr=None #array version
        self.qoArr=None #array version
    def log_Params(self):
        #this records value like position and momentum
        self.qList.append(self.q)
        self.pList.append(self.p)
        self.qoList.append(self.currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q, self.cumulativeLength))
    def finished(self):
        #finish tracing with the particle, tie up loose ends
        if self.qList is not None:
            self.qArr=np.asarray(self.qList)
            self.qList = None  # save memory
        if self.pList is not None:
            self.pArr = np.asarray(self.pList)
            self.pList = None  # save memory
        if self.qoList is not None:
            self.qoArr = np.asarray(self.qoList)
            self.qoList = None  # save memory
        self.currentElIndex=self.currentEl.index
        self.currentEl=None # to save memory
