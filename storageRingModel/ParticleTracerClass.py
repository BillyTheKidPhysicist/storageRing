import warnings
import math
from typing import Optional
from math import isnan,sqrt
import numba
import numpy.linalg as npl
import numpy as np
from numba.core.errors import NumbaPerformanceWarning
from ParticleClass import Particle
import elementPT
from constants import GRAVITATIONAL_ACCELERATION

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

#TODO: Why does the tracked time differe between fastMode=True and fastMode=False
@numba.njit()
def norm_3D(vec):
    return sqrt(vec[0]**2+vec[1]**2+vec[2]**2)

@numba.njit()
def dot_Prod_3D(veca,vecb):
    return veca[0]*vecb[0]+veca[1]*vecb[1]+veca[2]*vecb[2]

TINY_TIME_STEP=1e-9 #nanosecond time step to move particle from one element to another
@numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
def fast_qNew(q,F,p,h):
    return q+p*h+.5*F*h**2

@numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
def fast_pNew(p,F,F_n,h):
    return p+.5*(F+F_n)*h



@numba.njit()
def _transform_To_Next_Element(q,p,r01,r02,ROutEl1,RInEl2):
    #don't try and condense. Because of rounding, results won't agree with other methods and tests will fail
    q=q.copy()
    p=p.copy()
    q[:2]=ROutEl1@q[:2]
    q+=r01
    q-=r02
    q[:2]=RInEl2@q[:2]
    p[:2]=ROutEl1@p[:2]
    p[:2]=RInEl2@p[:2]
    return q,p
#this class does the work of tracing the particles through the lattice with timestepping algorithms.
#it utilizes fast numba functions that are compiled and saved at the moment that the lattice is passed. If the lattice
#is changed, then the particle tracer needs to be updated.

class ParticleTracer:

    minTimeStepsPerElement: int = 2 #if an element is shorter than this, throw an error

    def __init__(self,PTL):
        #lattice: ParticleTracerLattice object typically
        self.elList = PTL.elList  # list containing the elements in the lattice in order from first to last (order added)
        self.totalLatticeLength=PTL.totalLength

        self.PTL=PTL

        self.tau_Collision=np.inf #collision time constant
        self.T_CollisionLast=0.0 # time since last collision
        self.accelerated=None

        self.T=None #total time elapsed
        self.h=None #step size
        self.energyCorrection =None

        self.elHasChanged=False # to record if the particle has changed to another element in the previous step
        self.E0=None #total initial energy of particle


        self.numRevs=0 #tracking numbre of times particle comes back to wear it started

        self.particle=None #particle object being traced
        self.fastMode=None #wether to use the fast and memory light version that doesn't record parameters of the particle
        self.qEl=None #this is in the element frame
        self.pEl=None #this is in the element frame
        self.currentEl=None
        self.forceLast=None #the last force value. this is used to save computing time by reusing force

        self.fastMode=None #wether to log particle positions
        self.T0=None #total time to trace
        self.logTracker=None
        self.stepsBetweenLogging=None

        self.logPhaseSpaceCoords=False #wether to log lab frame phase space coords at element inputs

    def transform_To_Next_Element(self,q: np.ndarray,p: np.ndarray,nextEll: elementPT.Element)\
            ->tuple[np.ndarray,np.ndarray]:
        el1=self.currentEl
        el2=nextEll
        if el1.shape=='BEND':
            r01 = el1.r0
        elif el1.shape in ('COMBINER_SQUARE','COMBINER_CIRCULAR'):
            r01 = el1.r2
        else:
            r01 = el1.r1
        if el2.shape=='BEND':
            r02 = el2.r0
        elif el2.shape in ('COMBINER_SQUARE','COMBINER_CIRCULAR'):
            r02 = el2.r2
        else:
            r02 = el2.r1
        return _transform_To_Next_Element(q,p,r01,r02,el1.ROut,el2.RIn)

    def initialize(self)-> None:
        # prepare for a single particle to be traced
        self.T=0.0
        if self.particle.clipped is not None:
            self.particle.clipped=False
        LMin=norm_3D(self.particle.pi)*self.h*self.minTimeStepsPerElement
        for el in self.elList:
            if el.Lo<=LMin: #have at least a few steps in each element
                raise Exception('element too short for time steps size')
        if self.particle.qi[0]==0.0:
            raise Exception("a particle appears to be starting with x=0 exactly. This can cause unpredictable "
                            "behaviour")
        self.currentEl = self.which_Element_Lab_Coords(self.particle.qi)
        self.particle.currentEl=self.currentEl
        self.particle.dataLogging= not self.fastMode #if using fast mode, there will NOT be logging
        self.logTracker=0
        if self.logPhaseSpaceCoords:
            self.particle.elPhaseSpaceLog.append((self.particle.qi.copy(),self.particle.pi.copy()))
        if self.currentEl is None:
            self.particle.clipped=True
        else:
            self.particle.clipped=False
            self.qEl = self.currentEl.transform_Lab_Coords_Into_Element_Frame(self.particle.qi)
            self.pEl = self.currentEl.transform_Lab_Frame_Vector_Into_Element_Frame(self.particle.pi)
            self.E0=self.particle.get_Energy(self.currentEl,self.qEl,self.pEl)
        if self.fastMode is False and self.particle.clipped is False:
            self.particle.log_Params(self.currentEl,self.qEl,self.pEl)

    def trace(self,particle: Optional[Particle],h: float,T0: float,fastMode: bool=False,accelerated: bool=False,
              energyCorrection: bool=False,stepsBetweenLogging: int=1, tau_Collision: bool=None,
              logPhaseSpaceCoords:bool=False)-> Particle:
        #trace the particle through the lattice. This is done in lab coordinates. Elements affect a particle by having
        #the particle's position transformed into the element frame and then the force is transformed out. This is obviously
        # not very efficient.
        #qi: initial position coordinates
        #vi: initial velocity coordinates
        #h: timestep
        #T0: total tracing time
        #fastMode: wether to use the performance optimized versoin that doesn't track paramters
        assert 0<h<1e-4 and T0>0.0# reasonable ranges
        if tau_Collision is not None and tau_Collision<=0.0:
            raise Exception('Collision time must be None or nonzero positive number')
        self.tau_Collision=tau_Collision if tau_Collision is not None else np.inf
        self.energyCorrection=energyCorrection
        self.stepsBetweenLogging=stepsBetweenLogging
        self.logPhaseSpaceCoords=logPhaseSpaceCoords
        if particle is None:
            particle=Particle()
        if particle.traced:
            raise Exception('Particle has previously been traced. Tracing a second time is not supported')
        self.particle = particle
        if self.particle.clipped: #some particles may come in clipped so ignore them
            self.particle.finished(self.currentEl,self.qEl,self.pEl,totalLatticeLength=0)
            return self.particle
        self.fastMode=fastMode
        self.h=h
        self.T0=float(T0)
        self.initialize()
        self.accelerated=accelerated
        if self.particle.clipped: #some a particles may be clipped after initializing them because they were about
            # to become clipped
            self.particle.finished(self.currentEl,self.qEl,self.pEl,totalLatticeLength=0)
            return particle
        self.time_Step_Loop()
        self.forceLast=None #reset last force to zero
        self.particle.finished(self.currentEl,self.qEl,self.pEl,totalLatticeLength=self.totalLatticeLength)

        if self.logPhaseSpaceCoords:
            self.particle.elPhaseSpaceLog.append((self.particle.qf,self.particle.pf))


        return self.particle

    def did_Particle_Survive_To_End(self):
        """
        Check if a particle survived to the end of a lattice. This is only intended for lattices that aren't closed.
        This isn't straight forward because the particle tracing stops when the particle is outside the lattice, then
        the previous position is the final position. Thus, it takes a little extra logic to find out wether a particle
        actually survived to the end in an unclosed lattice

        :return: wether particle has survived to end or not
        """

        assert not self.PTL.isClosed
        elLast = self.elList[-1]
        # qEl = elLast.transform_Lab_Coords_Into_Element_Frame(self.qEl)
        # pEl = elLast.transform_Lab_Frame_Vector_Into_Element_Frame(self.pEl)
        if isinstance(elLast, elementPT.LensIdeal):
            timeStepToEnd = (elLast.L - self.qEl[0]) / self.pEl[0]
        elif isinstance(elLast, elementPT.CombinerIdeal):
            timeStepToEnd = self.qEl[0] / -self.pEl[0]
        else:
            print('not implemented, falling back to previous behaviour')
            return self.particle.clipped

        if not 0 <= timeStepToEnd <= self.h:
            clipped = True
        else:
            qElEnd = self.qEl + .99 * timeStepToEnd * self.pEl
            clipped = not elLast.is_Coord_Inside(qElEnd)
        return clipped

    def time_Step_Loop(self)-> None:
        while True:
            if self.T >= self.T0: #if out of time
                self.particle.clipped = False
                break
            if self.fastMode is False or self.currentEl.fastFieldHelper is None: #either recording data at each step
                #or the element does not have the capability to be evaluated with the much faster multi_Step_Verlet
                self.time_Step_Verlet()
                if self.fastMode is False and self.logTracker%self.stepsBetweenLogging==0:
                    if not self.particle.clipped: #nothing to update if particle clipped
                        self.particle.log_Params(self.currentEl,self.qEl,self.pEl)
                self.logTracker+=1
            else:
                self.multi_Step_Verlet()
            if self.particle.clipped:
                break
            self.T+=self.h
            self.particle.T=self.T

        if not self.PTL.isClosed:
            if self.currentEl is self.elList[-1] and self.particle.clipped: #only bother if particle is in last element
                self.particle.clipped=self.did_Particle_Survive_To_End()

    def multi_Step_Verlet(self)-> None:
        results=self._multi_Step_Verlet(self.qEl,self.pEl,self.T,self.T0,self.h,self.currentEl.fastFieldHelper)
        qEl_n,self.qEl[:],self.pEl[:],self.T,particleOutside=results
        qEl_n=np.array(qEl_n)
        self.particle.T=self.T
        if particleOutside:
            self.check_If_Particle_Is_Outside_And_Handle_Edge_Event(qEl_n,self.qEl,self.pEl)

    @staticmethod
    @numba.njit()
    def _multi_Step_Verlet(qEln, pEln, T, T0, h, helper):
        #pylint: disable = E, W, R, C
        # copy the input arrays to prevent modifying them outside the function
        force=helper.force
        x, y, z = qEln
        px, py, pz = pEln
        Fx, Fy, Fz = force(x,y,z)
        Fz=Fz-GRAVITATIONAL_ACCELERATION #simulated mass is 1kg always
        if math.isnan(Fx)or T >= T0:
            particleOutside = True
            qEln,pEln=(x,y,z),(px,py,pz)
            return qEln, qEln, pEln, T, particleOutside
        particleOutside = False
        while True:
            if T >= T0:
                pEl,qEl = (px, py, pz),(x, y, z)
                return qEl, qEl, pEl, T, particleOutside
            x =x+px * h + .5 * Fx * h ** 2
            y =y+py * h + .5 * Fy * h ** 2
            z =z+pz * h + .5 * Fz * h ** 2

            Fx_n, Fy_n, Fz_n = force(x,y,z)
            Fz_n=Fz_n-GRAVITATIONAL_ACCELERATION #simulated mass is 1kg always

            if math.isnan(Fx_n):
                xo = x - (px * h + .5 * Fx * h ** 2)
                yo = y - (py * h + .5 * Fy * h ** 2)
                zo = z - (pz * h + .5 * Fz * h ** 2)
                pEl, qEl,qEl_o = (px, py, pz), (x, y, z),(xo, yo, zo)
                particleOutside = True
                return qEl, qEl_o, pEl, T, particleOutside
            px =px+.5 * (Fx_n + Fx) * h
            py =py+.5 * (Fy_n + Fy) * h
            pz =pz+.5 * (Fz_n + Fz) * h
            Fx, Fy, Fz = Fx_n, Fy_n, Fz_n
            T += h

    @staticmethod
    @numba.njit(numba.float64(numba.float64[:],numba.float64[:],numba.float64))
    def compute_Aperture_Collision_Distance(qi,pi,ap):
        my=pi[1]/pi[0]
        mz=pi[2]/pi[0]
        by=qi[1]-my*qi[0]
        bz=qi[2]-mz*qi[0]
        #now to find out if the particle is outside the drift region. Assume circular aperture. use simple geometry of
        #lines
        if my==0 and mz==0: #no slope then it never clips
            x0=np.inf
        else:
            x0=(np.sqrt((mz*ap)**2+(my*ap)**2+2*by*bz*my*mz-(bz*my)**2-(by*mz)**2)-(by*my+bz*mz))/(my**2+mz**2) #x value
        #that the particle clips the aperture at
        return x0

    def handle_Drift_Region(self)-> None:
        raise NotImplementedError #This does not account for gravitational effects
        driftEl=self.currentEl #to  record the drift element for logging params
        self.particle.currentEl=driftEl
        pi=self.pEl.copy() #position at beginning of drift in drift frame
        qi=self.qEl.copy() #momentum at begining of drift in drift frame
        x0=self.compute_Aperture_Collision_Distance(qi,pi,self.currentEl.ap)
        if x0>driftEl.Lo: #if particle clips the aperture at a x position past the end of the drift element,
            #then it has not clipped
            clipped=False
            xEnd=driftEl.Lo #particle ends at the end of the drift
        else: #otherwise it clips inside the drift region
            xEnd=x0 #particle ends somewhere inside the drift
            clipped=True
            #now place it just at the beginning of the next element
        dt = abs((xEnd - qi[0]) / pi[0])  # time to travel to end coordinate. Use absolute to catch if the momentum
        #is the wrong way
        if self.T+dt>=self.T0: #there is not enough simulation time
            dt=self.T0-self.T #set to the remaining time available
            self.T=self.T0
            self.qEl=qi+pi*dt
        else:
            self.T += dt
            if not clipped:
                qEl=qi+pi*dt
                self.check_If_Particle_Is_Outside_And_Handle_Edge_Event(qEl,qi,pi) #put the particle just on the other
                #side. This is how this is done with other elements, so I am resuing code. really I don't need to do
                #because I know the particle finished
                if self.particle.clipped:
                    self.qEl=qEl
            else:
                self.qEl=qi+pi*dt
                self.pEl=self.pEl
                self.particle.clipped=True
        if not self.fastMode: #need to log the parameters
            qf = qi + pi * dt  # the final position of the particle. Either at the end, or at some place where it
            # clipped, or where time ran out
            self.particle.log_Params_In_Drift_Region(qi, pi, qf, self.h,driftEl)

    def time_Step_Verlet(self)-> None:
        #the velocity verlet time stepping algorithm. This version recycles the force from the previous step when
        #possible
        qEl=self.qEl #q old or q sub n
        pEl=self.pEl #p old or p sub n
        if not self.elHasChanged and self.forceLast is not None: #if the particle is inside the lement it was in
            #last time step, and it's not the first time step, then recycle the force. The particle is starting at the
            #same position it stopped at last time, thus same force
            F=self.forceLast
        else: #the last force is invalid because the particle is at a new position
            F=self.currentEl.force(qEl)
            F[2]=F[2]-GRAVITATIONAL_ACCELERATION #simulated mass is 1kg always
        #a = F # acceleration old or acceleration sub n
        qEl_n=fast_qNew(qEl,F,pEl,self.h)#q new or q sub n+1
        F_n=self.currentEl.force(qEl_n)
        F_n[2] =F_n[2]- GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always
        if isnan(F_n[0]): #particle is outside element if an array of length 1 with np.nan is returned
            self.check_If_Particle_Is_Outside_And_Handle_Edge_Event(qEl_n,qEl,pEl)  #check if element has changed.
            return
        #a_n = F_n  # acceleration new or acceleration sub n+1
        pEl_n=fast_pNew(pEl,F,F_n,self.h)
        # if self.tau_Collision is not None:
        #     probabilityCutoff=self.h/self.tau_Collision
        #     if np.random.rand()<probabilityCutoff:
        #         vHe=self.sample_Helium_Momentum()
        #         pEl_n=np.asarray(vLi_final(pEl_n,vHe))
        #         self.T_CollisionLast = self.T
        self.qEl=qEl_n
        self.pEl=pEl_n
        self.forceLast=F_n #record the force to be recycled
        self.elHasChanged=False

    def check_If_Particle_Is_Outside_And_Handle_Edge_Event(self,qEl_Next: np.ndarray,qEl: np.ndarray,pEl: np.ndarray)\
            -> None:
        #qEl_Next: coordinates that are outside the current element and possibley in the next
        #qEl: coordinates right before this method was called, should still be in the element
        #pEl: momentum for both qEl_Next and qEl

        if self.accelerated:
            if self.energyCorrection:
                pEl[:]+=self.momentum_Correction_At_Bounday(self.E0,qEl, pEl, self.currentEl.fastFieldHelper, 'leaving')
            if self.logPhaseSpaceCoords:
                qElLab = self.currentEl.transform_Element_Coords_Into_Lab_Frame(qEl_Next)  # use the old  element for transform
                pElLab = self.currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(pEl)  # use the old  element for transform
                self.particle.elPhaseSpaceLog.append((qElLab, pElLab))
            nextEl = self.get_Next_Element()
            q_nextEl,p_nextEl=self.transform_To_Next_Element(qEl_Next,pEl,nextEl)
            if not nextEl.is_Coord_Inside(q_nextEl):
                self.particle.clipped=True
            else:
                if self.energyCorrection:
                    p_nextEl[:] +=self.momentum_Correction_At_Bounday(self.E0,q_nextEl, p_nextEl,
                                                                              nextEl.fastFieldHelper, 'entering')
                self.particle.cumulativeLength+=self.currentEl.Lo  # add the previous orbit length
                self.currentEl=nextEl
                self.particle.currentEl=nextEl
                self.qEl=q_nextEl
                self.pEl=p_nextEl
                self.elHasChanged = True
        else:
            el=self.which_Element(qEl_Next)
            if el is None: #if outside the lattice
                self.particle.clipped = True
            elif el is not self.currentEl: #element has changed
                if self.energyCorrection:
                    pEl[:] += self.momentum_Correction_At_Bounday(self.E0,qEl, pEl, self.currentEl.fastFieldHelper, 'leaving')
                nextEl=el
                self.particle.cumulativeLength += self.currentEl.Lo  # add the previous orbit length
                qElLab=self.currentEl.transform_Element_Coords_Into_Lab_Frame(qEl_Next) #use the old  element for transform
                pElLab=self.currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(pEl) #use the old  element for transform
                if self.logPhaseSpaceCoords:
                    self.particle.elPhaseSpaceLog.append((qElLab,pElLab))
                self.currentEl=nextEl
                self.particle.currentEl=nextEl
                self.qEl = self.currentEl.transform_Lab_Coords_Into_Element_Frame(qElLab)  # at the beginning of the next element
                self.pEl = self.currentEl.transform_Lab_Frame_Vector_Into_Element_Frame(pElLab)  # at the beginning of the next
                # element
                self.elHasChanged = True
                if self.energyCorrection:
                    self.pEl[:]+=self.momentum_Correction_At_Bounday(self.E0,self.qEl,self.pEl,nextEl.fastFieldHelper,'entering')
            else:
                raise Exception('Particle is likely in a region of magnetic field which is invalid because its '
                                'interpolation extends into the magnetic material. Particle is also possibly frozen '
                                'because of broken logic that returns it to the same location.')

    @staticmethod
    @numba.njit()
    def momentum_Correction_At_Bounday(E0,qEl: np.ndarray,pEl: np.ndarray,fastFieldHelper,direction: str)-> \
            tuple[float,float,float]:
        #a small momentum correction because the potential doesn't go to zero, nor do i model overlapping potentials
        assert direction in ('entering','leaving')
        Fx,Fy,Fz=fastFieldHelper.force(qEl[0],qEl[1],qEl[2])
        FNorm = np.sqrt(Fx**2+Fy**2+Fz**2)
        if FNorm< 1e-6: #force is too small, and may cause a division error
            return 0.0,0.0,0.0
        else:
            if direction == 'leaving': #go from zero to non zero potential instantly
                ENew = dot_Prod_3D(pEl,pEl) / 2.0  # ideally, no magnetic field right at border
                deltaE = E0 - ENew  # need to lose energy to maintain E0 when the potential turns off
            else: #go from zero to non zero potentially instantly
                deltaE = -np.array(fastFieldHelper.magnetic_Potential(qEl[0],qEl[1],qEl[2]))  # need to lose this energy
            Fx_unit,Fy_unit,Fz_unit = Fx / FNorm,Fy / FNorm,Fz / FNorm
            deltaPNorm = deltaE / (Fx_unit*pEl[0]+Fy_unit*pEl[1]+Fz_unit*pEl[2])
            deltaPx,deltaPy,deltaPz = deltaPNorm * Fx_unit,deltaPNorm * Fy_unit,deltaPNorm * Fz_unit
            return deltaPx,deltaPy,deltaPz

    def which_Element_Lab_Coords(self,qLab: np.ndarray)-> Optional[elementPT.Element]:
        for el in self.elList:
            if el.is_Coord_Inside(el.transform_Lab_Coords_Into_Element_Frame(qLab)):
                return el
        return None

    def get_Next_Element(self)-> elementPT.Element:
        if self.currentEl.index+1>=len(self.elList):
            nextEl=self.elList[0]
        else:
            nextEl=self.elList[self.currentEl.index+1]
        return nextEl

    def which_Element(self,qEl: np.ndarray)-> Optional[elementPT.Element]:
        #find which element the particle is in, but check the next element first ,which save time
        #and will be the case most of the time. Also, recycle the element coordinates for use in force evaluation later
        qElLab=self.currentEl.transform_Element_Coords_Into_Lab_Frame(qEl)
        nextEl=self.get_Next_Element()
        if nextEl.is_Coord_Inside(nextEl.transform_Lab_Coords_Into_Element_Frame(qElLab)): #try the next element
            return nextEl
        else:
            #now instead look everywhere, except the next element we already checked
            for el in self.elList:
                if el is not nextEl: #don't waste rechecking current element or next element
                    if el.is_Coord_Inside(el.transform_Lab_Coords_Into_Element_Frame(qElLab)):
                        return el
            return None

    # def sample_Helium_Momentum(self)-> tuple[float,float,float]:
    #     T_Room = 300.0
    #     sigma = np.sqrt(BOLTZMANN_CONSTANT * T_Room / MASS_HELIUM)
    #     px = np.random.normal(loc=0.0, scale=sigma)
    #     py = np.random.normal(loc=0.0, scale=sigma)
    #     pz = np.random.normal(loc=0.0, scale=sigma)
    #     return px, py, pz
