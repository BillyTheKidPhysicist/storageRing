import matplotlib.pyplot as plt
import time
import numpy.linalg as npl
import numpy as np
import copy
from math import sqrt
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
            if np.isnan(particle.revolutions)==True:
                raise Exception('Particle revolutions have an issue')
            if particle.revolutions is not None:
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
    def quick_Copy(self): #only copy the initial conditions. For swarms that havn't been traced or been monkeyed
        #with at all
        swarmNew=Swarm()
        for particle in self.particles:
            swarmNew.add_Particle(qi=particle.qi,pi=particle.pi)
        return swarmNew
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
    def __init__(self,qi=np.asarray([0.0, 0.0, 0.0]),pi=np.asarray([-200.0, 0.0, 0.0])):
        self.q=qi #position, lab frame, meters
        self.p=pi #momentu, lab frame, meters*kg/s, where mass=1
        self.qi=qi.copy()#initial position, lab frame, meters
        self.pi=pi.copy()#initial momentu, lab frame, meters*kg/s, where mass=1
        self.T=0 #time of particle in simulation
        self.traced=False #recored wether the particle has already been sent throught the particle tracer
        self.v0=sqrt(pi[0]**2+pi[1]**2+pi[2]**2) #initial speed
        self.color=None #color that can be added to each particle for plotting


        self.force=None #current force on the particle
        self.currentEl=None #which element the particle is ccurently in
        self.currentElIndex=None #Index of the elmenent that the particle is curently in. THis remains unchanged even
        #after the particle leaves the tracing algorithm and such can be used to record where it clipped
        self.cumulativeLength=0 #total length traveled by the particle IN TERMS of lattice elements. It updates after
        #the particle leaves an element by adding that elements length (particle trajectory length that is)
        self.revolutions=0 #revolutions particle makd around lattice. Initially zero
        self.clipped=None #wether particle clipped an apeture
        self.logged=None #wether the particle is loggin parameters such as position and energy. This will typically be
        #false when fastmode is being used in the particle tracer class
        #these lists track the particles momentum, position etc during the simulation if that feature is enable. Later
        #they are converted into arrays
        self._pList=[] #List of momentum vectors
        self._qList=[] #List of position vector
        self._qoList=[] #List of position in orbit frame vectors
        self._TList=[] #kinetic energy list. Each entry contains the element index and corresponding energy
        self._VList=[] #potential energy list. Each entry contains the element index and corresponding energy
        #array versions
        self.pArr=None
        self.p0Arr=None #array of norm of momentum.
        self.qArr=None 
        self.qoArr=None
        self.speedArr=None
        self.TArr=None 
        self.VArr=None 
        self.EArr=None #total energy
        self.elDeltaEDict={} # dictionary to hold energy changes that occur traveling through an element. Entries are
        #element index and list of energy changes for each pass
        self.yInterp=None #interpolating function as a function of x or s (where s is orbit trajectory analog of x)
        self.zInterp=None #interpolating function as a function of x or s (where s is orbit trajectory analog of x)
    def reset(self):
        #reset the particle
        self.__init__(qi=self.qi,pi=self.pi)
    def __str__(self):
        string='------particle-------\n'
        string+='q: '+str(self.q)+'\n'
        string+='p: '+str(self.p)+'\n'
        string+='revolution: '+str(self.revolutions)+'\n'
        return string
    def log_Params(self,currentEl,qel,pel):
        #this records value like position and momentum
        #qel: element position coordinate
        #pel: momentum position coordinate
        # print(qel,self.cumulativeLength)
        q = currentEl.transform_Element_Coords_Into_Lab_Frame(qel)
        p = currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(pel)
        self._qList.append(q.copy())
        self._pList.append(p.copy())
        self._TList.append((currentEl.index,np.sum(p**2)/2.0))
        if currentEl is not None:
            qel=currentEl.transform_Lab_Coords_Into_Element_Frame(q)
            elIndex=currentEl.index
            self._qoList.append(currentEl.transform_Lab_Coords_Into_Global_Orbit_Frame(q, self.cumulativeLength))
            self._VList.append((elIndex,currentEl.magnetic_Potential(qel)))
    def get_Energy(self,currentEl,qel,pel):
        V=currentEl.magnetic_Potential(qel)
        T=npl.norm(pel)
        return T+V

    def log_Params_In_Drift_Region(self,qEli,pEli,qElf,h,driftEl):
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

        if self.currentEl!=driftEl: #the particle has entered the next element before getting here, and since I am
            # recycling that algorithm it has the uninteded side effect of updating the cumulative length,
            # which I do not want yet because it also includes the drift region
            cumulativeLength=self.cumulativeLength-driftEl.Lo
        else:
            cumulativeLength=self.cumulativeLength
        for i in range(qElArr.shape[0]):
            qList.append(driftEl.transform_Element_Coords_Into_Lab_Frame(qElArr[i]))
            self._qoList.append(driftEl.transform_Lab_Coords_Into_Global_Orbit_Frame(qList[-1],cumulativeLength))
            pList.append(driftEl.transform_Element_Frame_Vector_Into_Lab_Frame(pElArr[i]))
        #now fill the lists that log parameters
        self._qList.extend(qList)
        self._pList.extend(pList)
        for pEl in pElArr:
            self._TList.append((driftEl.index,np.sum(pEl**2/2.0)))
            self._VList.append((driftEl.index,0.0)) #python list creation trick
    def fill_Energy_Array_And_Dicts(self):
        self.TArr=np.asarray([entry[1] for entry in self._TList])
        self.VArr=np.asarray([entry[1] for entry in self._VList])
        self.EArr=self.TArr+self.VArr


        elementIndexPrev=self._TList[0][0]
        E_AfterEnteringEl=self.EArr[0]
        for i in range(len(self._TList)):
            if self._TList[i][0]!=elementIndexPrev:
                E_BeforeLeavingEl=self.EArr[i-1]
                deltaE=E_BeforeLeavingEl-E_AfterEnteringEl
                if (str(elementIndexPrev) in self.elDeltaEDict) == False: #need to make a list entry for this element
                    self.elDeltaEDict[str(elementIndexPrev)]=[deltaE]
                else:
                    self.elDeltaEDict[str(elementIndexPrev)].append(deltaE)
                E_AfterEnteringEl=self.EArr[i]
                elementIndexPrev=self._TList[i][0]
        self._TList=[]
        self._VList=[]

    def finished(self,totalLatticeLength=None):
        #finish tracing with the particle, tie up loose ends
        #totalLaticeLength: total length of periodic lattice
        self.traced=True
        self.force=None
        if self.logged==True:
            self.qArr=np.asarray(self._qList)
            self._qList = []  # save memory
            self.pArr = np.asarray(self._pList)
            self.speedArr=npl.norm(self.pArr,axis=1)
            self._pList = []
            self.qoArr = np.asarray(self._qoList)
            self._qoList = []
            if self.pArr.shape[0]!=0:
                self.p0Arr=npl.norm(self.pArr,axis=1)
            self.fill_Energy_Array_And_Dicts()
        if self.currentEl is not None: #This option is here so the particle class can be used in situation beside ParticleTracer
            self.currentElIndex=self.currentEl.index
            if totalLatticeLength is not None:
                qo=self.currentEl.transform_Lab_Coords_Into_Global_Orbit_Frame(self.q, self.cumulativeLength)
                self.revolutions=qo[0]/totalLatticeLength
            self.currentEl=None # to save memory
    def plot_Energies(self,showOnlyTotalEnergy=False):
        if self.EArr.shape[0]==0:
            raise Exception('PARTICLE HAS NO LOGGED POSITION')
        EArr = self.EArr
        TArr=self.TArr
        VArr=self.VArr
        qoArr=self.qoArr
        plt.close('all')
        plt.title('Particle energies vs position. \n Total initial energy is '+str(np.round(EArr[0],1)) +' energy units')
        plt.plot(qoArr[:,0],EArr-EArr[0],label='E')
        if showOnlyTotalEnergy==False:
            plt.plot(qoArr[:, 0], TArr - TArr[0],label='T')
            plt.plot(qoArr[:, 0], VArr - VArr[0],label='V')
        plt.ylabel("Energy")
        plt.xlabel('Meters')
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