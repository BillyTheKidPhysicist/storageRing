from ParticleClass import Particle
import numpy.linalg as npl
import numba
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import time
import numpy as np
import math
from math import isnan
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import sys
from shapely.geometry import Polygon,Point
import elementPT

#TODO: Why does the tracked time differe between fastMode=True and fastMode=False


TINY_TIME_STEP=1e-9 #nanosecond time step to move particle from one element to another
@numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
def fast_qNew(q,F,p,h):
    return q+p*h+.5*F*h**2

@numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
def fast_pNew(p,F,F_n,h):
    return p+.5*(F+F_n)*h



@numba.njit(numba.types.UniTuple(numba.float64[:],2)(numba.float64[:],numba.float64[:],numba.float64[:],numba.float64[:]
                                                     ,numba.float64[:,:],numba.float64[:,:]))
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
    def __init__(self,lattice):
        #lattice: ParticleTracerLattice object typically
        self.latticeElementList = lattice.elList  # list containing the elements in the lattice in order from first to last (order added)
        self.totalLatticeLength=lattice.totalLength
        self.numbaMultiStepCache=[]
        self.generate_Multi_Step_Cache()


        self.accelerated=None

        self.T=None #total time elapsed
        self.h=None #step size
        self.minTimeStepsPerElement=3
        self.energyCorrection=None

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
    def transform_To_Next_Element(self,q,p,nextEll):
        el1=self.currentEl
        el2=nextEll
        if el1.type=='BEND':
            r01 = el1.r0
        elif el1.type in ('COMBINER_SQUARE','COMBINER_CIRCULAR'):
            r01 = el1.r2
        else:
            r01 = el1.r1
        if el2.type=='BEND':
            r02 = el2.r0
        elif el2.type in ('COMBINER_SQUARE','COMBINER_CIRCULAR'):
            r02 = el2.r2
        else:
            r02 = el2.r1
        return _transform_To_Next_Element(q,p,r01,r02,el1.ROut,el2.RIn)
    def initialize(self):
        # prepare for a single particle to be traced
        self.T=0.0
        if self.particle.clipped is not None:
            self.particle.clipped=False
        LMin=npl.norm(self.particle.pi)*self.h*self.minTimeStepsPerElement
        for el in self.latticeElementList:
            if el.Lo<=LMin: #have at least a few steps in each element
                raise Exception('element too short for time steps size')
        self.currentEl = self.which_Element_Lab_Coords(self.particle.qi)
        self.particle.currentEl=self.currentEl
        self.particle.dataLogging= not self.fastMode #if using fast mode, there will NOT be logging
        self.logTracker=0
        if self.currentEl is None:
            self.particle.clipped=True
        else:
            self.particle.clipped=False
            self.qEl = self.currentEl.transform_Lab_Coords_Into_Element_Frame(self.particle.qi)
            self.pEl = self.currentEl.transform_Lab_Frame_Vector_Into_Element_Frame(self.particle.pi)
            self.E0=self.particle.get_Energy(self.currentEl,self.qEl,self.pEl)
        if self.fastMode==False and self.particle.clipped == False:
            self.particle.log_Params(self.currentEl,self.qEl,self.pEl)
    def trace(self,particle,h,T0,fastMode=False,accelerated=False,energyCorrection=False,stepsBetweenLogging=1):
        #trace the particle through the lattice. This is done in lab coordinates. Elements affect a particle by having
        #the particle's position transformed into the element frame and then the force is transformed out. This is obviously
        # not very efficient.
        #qi: initial position coordinates
        #vi: initial velocity coordinates
        #h: timestep
        #T0: total tracing time
        #fastMode: wether to use the performance optimized versoin that doesn't track paramters
        assert 0<h<1e-4 and T0>0.0# reasonable ranges
        self.energyCorrection=energyCorrection
        self.stepsBetweenLogging=stepsBetweenLogging
        if particle is None:
            particle=Particle()
        if particle.traced==True:
            raise Exception('Particle has previously been traced. Tracing a second time is not supported')
        self.particle = particle
        if self.particle.clipped==True: #some particles may come in clipped so ignore them
            self.particle.finished(self.currentEl,self.qEl,self.pEl,totalLatticeLength=0)
            return self.particle
        self.fastMode=fastMode
        self.h=h
        self.T0=float(T0)
        self.initialize()
        self.accelerated=accelerated
        if self.particle.clipped==True: #some a particles may be clipped after initializing them because they were about
            # to become clipped
            self.particle.finished(self.currentEl,self.qEl,self.pEl,totalLatticeLength=0)
            return particle
        self.time_Step_Loop()
        self.forceLast=None #reset last force to zero
        self.particle._q = self.currentEl.transform_Element_Coords_Into_Lab_Frame(self.qEl)
        self.particle.p = self.currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(self.pEl)
        self.particle.currentEl=self.currentEl

        self.particle.finished(self.currentEl,self.qEl,self.pEl,totalLatticeLength=self.totalLatticeLength)
        return self.particle
    def time_Step_Loop(self):
        while (True):
            if self.T >= self.T0: #if out of time
                self.particle.clipped = False
                break
            if False:#type(self.currentEl)==elementPT.Drift :
                self.handle_Drift_Region()
                if self.particle.clipped==True:
                    break
            else:
                if self.fastMode==False or self.numbaMultiStepCache[self.currentEl.index] is None: #either recording data at each step
                    #or the element does not have the capability to be evaluated with the much faster multi_Step_Verlet
                    self.time_Step_Verlet()
                    if self.fastMode is False and self.logTracker%self.stepsBetweenLogging==0: #if false, take time to log parameters
                        self.particle.log_Params(self.currentEl,self.qEl,self.pEl)
                    self.logTracker+=1
                else:
                    self.multi_Step_Verlet()
                if self.particle.clipped == True:
                    break
                self.T+=self.h
                self.particle.T=self.T
    def multi_Step_Verlet(self):
        results=self.numbaMultiStepCache[self.currentEl.index](self.qEl,self.pEl,self.T,self.T0,self.h,self.currentEl.fieldFact)
        qEl_n,self.qEl,self.pEl,self.T,particleOutside=results
        if particleOutside is True:
            self.check_If_Particle_Is_Outside_And_Handle_Edge_Event(qEl_n,self.qEl,self.pEl) #it doesn't quite make sense
            #that I'm doing it like this. The outside the element system could be revamped.
    def generate_Multi_Step_Cache(self):
        self.numbaMultiStepCache = []
        for el in self.latticeElementList:
            if el.fast_Numba_Force_Function is None:
                self.numbaMultiStepCache.append(None)
            else:
                func = self.generate_Multi_Step_Verlet(el.fast_Numba_Force_Function)
                self.numbaMultiStepCache.append(func)


    def generate_Multi_Step_Verlet(self,forceFunc):
        func=self._multi_Step_Verlet
        @numba.njit()
        def wrap(qEl,pEl,T,T0,h,forceFact):
            return func(qEl,pEl,T,T0,h,forceFact,forceFunc)
        test=np.empty(3)
        test[:]=np.nan
        wrap(test,test,0.0,0.0,0.0,0.0) #force numba to compile
        return wrap

    @staticmethod
    @numba.njit()
    def _multi_Step_Verlet(qEln, pEln, T, T0, h, forceFact, force):
        # copy the input arrays to prevent modifying them outside the function
        x, y, z = qEln
        px, py, pz = pEln
        Fx, Fy, Fz = force(x,y,z)
        if math.isnan(Fx) == True or T >= T0:
            particleOutside = True
            return qEln, qEln, pEln, T, particleOutside
        Fx, Fy, Fz = Fx * forceFact, Fy * forceFact, Fz * forceFact
        particleOutside = False
        while (True):
            if T >= T0:
                pEl = np.asarray([px, py, pz])
                qEl = np.asarray([x, y, z])
                return qEl, qEl, pEl, T, particleOutside
            x =x+px * h + .5 * Fx * h ** 2
            y =y+py * h + .5 * Fy * h ** 2
            z =z+pz * h + .5 * Fz * h ** 2

            Fx_n, Fy_n, Fz_n = force(x,y,z)
            Fx_n, Fy_n, Fz_n = Fx_n * forceFact, Fy_n * forceFact, Fz_n * forceFact

            if math.isnan(Fx_n) == True:
                qEl = np.asarray([x, y, z])
                pEl = np.asarray([px, py, pz])
                xo = x - (px * h + .5 * Fx * h ** 2)
                yo = y - (py * h + .5 * Fy * h ** 2)
                zo = z - (pz * h + .5 * Fz * h ** 2)
                qEl_o = np.asarray([xo, yo, zo])

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
    def handle_Drift_Region(self):
        #TODO: currently not used, of dubious value. Hopefully to be used soon, otherwise to be discarded
        assert False
        # it is more efficient to explote the fact that there is no field inside the drift region to project the
        #paricle through it rather than timestep if through.
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
            if clipped==False:
                qEl=qi+pi*dt
                self.check_If_Particle_Is_Outside_And_Handle_Edge_Event(qEl+pi*1e-10) #put the particle just on the other
                #side. This is how this is done with other elements, so I am resuing code. really I don't need to do
                #because I know the particle finished
                if self.particle.clipped==True:
                    self.qEl=qEl
            else:
                self.qEl=qi+pi*dt
                self.pEl=self.pEl
                self.particle.clipped=True
        if self.fastMode==False: #need to log the parameters
            qf = qi + pi * dt  # the final position of the particle. Either at the end, or at some place where it
            # clipped, or where time ran out
            self.particle.log_Params_In_Drift_Region(qi, pi, qf, self.h,driftEl)


    def time_Step_Verlet(self):
        #the velocity verlet time stepping algorithm. This version recycles the force from the previous step when
        #possible
        qEl=self.qEl #q old or q sub n
        pEl=self.pEl #p old or p sub n
        if self.elHasChanged==False and self.forceLast is not None: #if the particle is inside the lement it was in
            #last time step, and it's not the first time step, then recycle the force. The particle is starting at the
            #same position it stopped at last time, thus same force
            F=self.forceLast
        else: #the last force is invalid because the particle is at a new position
            F=self.currentEl.force(qEl)
        #a = F # acceleration old or acceleration sub n
        qEl_n=fast_qNew(qEl,F,pEl,self.h)#q new or q sub n+1
        F_n=self.currentEl.force(qEl_n)
        if isnan(F_n[0]) == True: #particle is outside element if an array of length 1 with np.nan is returned
            self.check_If_Particle_Is_Outside_And_Handle_Edge_Event(qEl_n,qEl,pEl)  #check if element has changed.
            return
        #a_n = F_n  # acceleration new or acceleration sub n+1
        pEl_n=fast_pNew(pEl,F,F_n,self.h)
        self.qEl=qEl_n
        self.pEl=pEl_n
        self.forceLast=F_n #record the force to be recycled
        self.elHasChanged = False# if the leapfrog is completed, then the element did not change during the leapfrog
    # @profile
    def check_If_Particle_Is_Outside_And_Handle_Edge_Event(self,qEl_n,qEl,pEl):
        #qEl_n: coordinates that are outside the current element and possibley in the next
        #qEl: coordinates right before this method was called, should still be in the element
        #pEl: momentum for both qEl_n and qEl
        qEl_Scooted=qEl_n+TINY_TIME_STEP*pEl #the particle might land right on an edge, scoot it along
        if self.accelerated==True:
            if self.energyCorrection == True:
                pEl = pEl + self.momentum_Correction_At_Bounday(qEl, pEl, self.currentEl, 'leaving')
            nextEl = self.get_Next_Element()
            q_nextEl,p_nextEl=self.transform_To_Next_Element(qEl_Scooted,pEl,nextEl)
            if nextEl.is_Coord_Inside(q_nextEl)==False:
                self.particle.clipped=True
                return
            else:
                if self.energyCorrection == True:
                    p_nextEl = p_nextEl + self.momentum_Correction_At_Bounday(q_nextEl, p_nextEl,
                                                                              nextEl, 'entering')
                self.particle.cumulativeLength+=self.currentEl.Lo  # add the previous orbit length
                self.currentEl=nextEl
                self.particle.currentEl=nextEl
                self.qEl=q_nextEl
                self.pEl=p_nextEl
                self.elHasChanged = True
                return
        else:
            el=self.which_Element(qEl_Scooted)
            if el is None: #if outside the lattice
                self.particle.clipped = True
                return
            elif el is not self.currentEl: #element has changed
                if self.energyCorrection==True:
                    pEl = pEl + self.momentum_Correction_At_Bounday(qEl, pEl, self.currentEl, 'leaving')
                nextEl=el
                self.particle.cumulativeLength += self.currentEl.Lo  # add the previous orbit length
                qElLab=self.currentEl.transform_Element_Coords_Into_Lab_Frame(qEl_Scooted) #use the old  element for transform
                pElLab=self.currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(pEl) #use the old  element for transform
                self.currentEl=nextEl
                self.particle.currentEl=nextEl
                self.qEl = self.currentEl.transform_Lab_Coords_Into_Element_Frame(qElLab)  # at the beginning of the next element
                self.pEl = self.currentEl.transform_Lab_Frame_Vector_Into_Element_Frame(pElLab)  # at the beginning of the next
                # element
                self.elHasChanged = True
                if self.energyCorrection==True:
                    self.pEl=self.pEl+self.momentum_Correction_At_Bounday(self.qEl,self.pEl,nextEl,'entering')
                return
            else:
                raise Exception('Something is wrong, element should have changed. Particle is possibly frozen because '
                                'of broken logic, or being returned the same location')
    def momentum_Correction_At_Bounday(self,qEl,pEl,el,direction):
        #a small momentum correction because the potential doesn't go to zero, nor do i model overlapping potentials
        assert direction in ('entering','leaving')
        F = el.force(qEl)
        FNorm = np.linalg.norm(F)
        if FNorm< 1e-6: #force is too small, and may cause a division error
            return np.zeros(3)
        else:
            if direction == 'leaving': #go from zero to non zero potential instantly
                ENew = np.sum(pEl ** 2) / 2.0  # ideally, no magnetic field right at border
                deltaE = self.E0 - ENew  # need to lose energy to maintain E0 when the potential turns off
            else: #go from zero to non zero potentially instantly
                deltaE = -el.magnetic_Potential(qEl)  # need to lose this energy
            F_unit = F / np.linalg.norm(F)
            deltaPNorm = deltaE / np.dot(F_unit, pEl)
            deltaP = deltaPNorm * F_unit
            return deltaP
    def which_Element_Lab_Coords(self,qLab):
        for el in self.latticeElementList:
            if el.is_Coord_Inside(el.transform_Lab_Coords_Into_Element_Frame(qLab))==True:
                return el
        return None
    def get_Next_Element(self):
        if self.currentEl.index+1>=len(self.latticeElementList):
            nextEl=self.latticeElementList[0]
        else:
            nextEl=self.latticeElementList[self.currentEl.index+1]
        return nextEl
    def which_Element(self,qEl): #.134
        #find which element the particle is in, but check the next element first ,which save time
        #and will be the case most of the time. Also, recycle the element coordinates for use in force evaluation later
        qElLab=self.currentEl.transform_Element_Coords_Into_Lab_Frame(qEl)
        nextEl=self.get_Next_Element()
        if nextEl.is_Coord_Inside(nextEl.transform_Lab_Coords_Into_Element_Frame(qElLab)) == True: #try the next element
            return nextEl
        else:
            #now instead look everywhere, except the next element we already checked
            for el in self.latticeElementList:
                if el is not nextEl: #don't waste rechecking current element or next element
                    if el.is_Coord_Inside(el.transform_Lab_Coords_Into_Element_Frame(qElLab))==True:
                        return el
            return None