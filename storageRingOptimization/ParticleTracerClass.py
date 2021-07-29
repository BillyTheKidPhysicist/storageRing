import numpy.linalg as npl
import numba
import time
import numpy as np
import math
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import sys
from shapely.geometry import Polygon,Point

import elementPT


def Compute_Bending_Radius_For_Segmented_Bender(L,rp,yokeWidth,numMagnets,angle,space=0.0):
    #ucAng=angle/(2*numMagnets)
    rb=(L+2*space)/(2*np.tan(angle/(2*numMagnets)))+yokeWidth+rp
    #ucAng1=np.arctan((L/2)/(rb-rp-yokeWidth))
    return rb


@numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
def fast_qNew(q,F,p,h):
    return q+p*h+.5*F*h**2

@numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
def fast_pNew(p,F,F_n,h):
    return p+.5*(F+F_n)*h


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

        self.T=None #total time elapsed
        self.h=None #step size


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
        self.test=[]


    def initialize(self):
        # prepare for a single particle to be traced
        self.T=0.0
        if self.particle.clipped is not None:
            self.particle.clipped=False
        dl=self.particle.v0*self.h #approximate stepsize
        for el in self.latticeElementList:
            if dl>el.Lo/3.0: #have at least a few steps in each element
                raise Exception('STEP SIZE TOO LARGE')
        self.currentEl = self.which_Element_Lab_Coords(self.particle.q)
        self.particle.currentEl=self.currentEl
        if self.currentEl is None:
            self.particle.clipped=True
        else:
            self.qEl = self.currentEl.transform_Lab_Coords_Into_Element_Frame(self.particle.q)
            self.pEl = self.currentEl.transform_Lab_Frame_Vector_Into_Element_Frame(self.particle.p)
        if self.fastMode==False:
            self.particle.log_Params(self.currentEl,self.qEl,self.pEl)
    def trace(self,particle,h,T0,fastMode=False):
        #trace the particle through the lattice. This is done in lab coordinates. Elements affect a particle by having
        #the particle's position transformed into the element frame and then the force is transformed out. This is obviously
        # not very efficient.
        #qi: initial position coordinates
        #vi: initial velocity coordinates
        #h: timestep
        #T0: total tracing time
        #fastMode: wether to use the performance optimized versoin that doesn't track paramters
        if particle.traced==True:
            raise Exception('Particle has previously been traced. Tracing a second time is not supported')
        self.particle = particle
        if self.particle.clipped==True: #some particles may come in clipped so ignore them
            self.particle.finished(totalLatticeLength=0)
            return self.particle
        self.fastMode=fastMode
        self.h=h
        self.T0=float(T0)
        self.initialize()
        if self.particle.clipped==True: #some a particles may be clipped after initializing them because they were about
            # to become clipped
            self.particle.finished(totalLatticeLength=0)
            return particle
        self.time_Step_Loop()
        self.forceLast=None #reset last force to zero
        self.particle.q = self.currentEl.transform_Element_Coords_Into_Lab_Frame(self.qEl)
        self.particle.p = self.currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(self.pEl)
        self.particle.currentEl=self.currentEl
        self.particle.finished(totalLatticeLength=self.totalLatticeLength)
        return self.particle
    def update(self):
        #call after changing some parameter of the lattice to reflect the change here.
        #todo: implement
        pass
    def time_Step_Loop(self):
        while (True):
            if self.T >= self.T0: #if out of time
                self.particle.clipped = False
                break
            if type(self.currentEl)==elementPT.Drift :
                self.handle_Drift_Region()
                if self.particle.clipped==True:
                    break
            else:
                if self.fastMode==False or self.numbaMultiStepCache[self.currentEl.index] is None: #either recording data at each step
                    #or the element does not have the capability to be evaluated with the much faster multi_Step_Verlet
                    self.time_Step_Verlet()
                    if self.fastMode is False: #if false, take time to log parameters
                        self.particle.log_Params(self.currentEl,self.qEl,self.pEl)
                else:
                    self.multi_Step_Verlet()
                if self.particle.clipped == True:
                    break
                self.T+=self.h
                self.particle.T=self.T
    def multi_Step_Verlet(self):
        results=self.numbaMultiStepCache[self.currentEl.index](self.qEl,self.pEl,self.T,self.T0,self.h,self.currentEl.BpFact)
        qEl_n,self.qEl,self.pEl,self.T,particleOutside=results
        if particleOutside is True:
            self.check_If_Particle_Is_Outside_And_Handle_Edge_Event(qEl_n) #it doesn't quite make sense
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
        @numba.jit()
        def wrap(qEl,pEl,T,T0,h,forceFact):
            return func(qEl,pEl,T,T0,h,forceFact,forceFunc)
        test=np.empty(3)
        test[:]=np.nan
        wrap(test,test,0.0,0.0,0.0,0.0) #force numba to compile
        return wrap

    @staticmethod
    @numba.njit()
    def _multi_Step_Verlet(qEl,pEl,T,T0,h,forceFact,force):
        F=force(qEl)*forceFact
        if math.isnan(F[0]) == True:
            particleOutside = True
            return qEl, qEl, pEl, T, particleOutside
        particleOutside=False
        while(True):
            if T>=T0:
                break
            qEl_n = qEl+pEl*h+.5*F*h**2  # q new or q sub n+1
            F_n = force(qEl_n)*forceFact
            if math.isnan(F_n[0])==True:
                particleOutside=True
                return qEl_n,qEl,pEl,T,particleOutside
            pEl_n = pEl+.5*(F+F_n)*h
            qEl = qEl_n
            pEl = pEl_n
            F = F_n  # record the force to be recycled
            T += h
        return qEl_n,qEl,pEl,T,particleOutside
    def handle_Drift_Region(self):
        # it is more efficient to explote the fact that there is no field inside the drift region to project the
        #paricle through it rather than timestep if through.
        driftEl=self.currentEl #to  record the drift element for logging params
        self.particle.currentEl=driftEl
        pi=self.pEl.copy() #position at beginning of drift in drift frame
        qi=self.qEl.copy() #momentum at begining of drift in drift frame
        pf=pi
        my=pi[1]/pi[0]
        mz=pi[2]/pi[0]
        by=qi[1]-my*qi[0]
        bz=qi[2]-mz*qi[0]
        #now to find out if the particle is outside the drift region. Assume circular aperture. use simple geometry of
        #lines
        r0=self.currentEl.ap #aperture, radial
        if my==0 and mz==0: #no slope then it never clips
            x0=np.inf
        else:
            x0=(np.sqrt((mz*r0)**2+(my*r0)**2+2*by*bz*my*mz-(bz*my)**2-(by*mz)**2)-(by*my+bz*mz))/(my**2+mz**2) #x value
        #that the particle clips the aperture at
        if x0>driftEl.Lo: #if particle clips the aperture at a x position past the end of the drift element,
            #then it has not clipped
            clipped=False
            xEnd=driftEl.Lo #particle ends at the end of the drift
        else: #otherwise it clips inside the drift region
            xEnd=x0 #particle ends somewhere inside the drift
            clipped=True
            #now place it just at the beginning of the next element
        dt = (xEnd - qi[0]) / pi[0]  # time to travel to end coordinate
        if self.T+dt>self.T0: #there is not enough simulation time
            dt=self.T0-self.T #set to the remaining time available
            self.T=self.T0
            self.qEl=qi+pi*dt
        else:
            if clipped==False:
                qEl=qi+pi*dt
                el=self.which_Element(qEl+1e-10) #put the particle just on the other side
                exitLoop=self.check_If_Element_Has_Changed_And_Handle_Edge_Event(el) #reuse the code here. This steps alos logs
                #the time!!
                if exitLoop==True:
                    self.particle.currentEl=self.currentEl
                    if el is None:
                        self.qEl=qEl
            else:
                self.T += dt
                self.qEl=qi+pi*dt
                self.pEl=pf
                self.particle.clipped=True
        if self.fastMode==False: #need to log the parameters
            qf = qi + pi * dt  # the final position of the particle. Either at the end, or at some place where it
            # clipped, or where time ran out
            self.particle.log_Params_In_Drift_Region(qi, pi, qf, self.h,driftEl)
    def handle_Element_Edge(self):
        # This method uses the correct timestep to put the particle just on the other side of the end of the element
        # using explicit euler.
        el=self.currentEl
        q=el.transform_Element_Coords_Into_Lab_Frame(self.qEl)
        p=el.transform_Element_Frame_Vector_Into_Lab_Frame(self.pEl)

        r=el.r2-q

        rt=np.abs(np.sum(el.ne*r[:2])) #perindicular position  to element's end
        pt=np.abs(np.sum(el.ne*p[:2]))#perpindicular momentum to surface of element's end
        h=rt/pt
        self.T += h-self.h #to account for the smaller step size here
        q=q+p*h
        eps=1e-9 #tiny step to put the particle on the other side
        n_p=p/np.sum(np.sqrt(p**2)) #normalized vector of particle velocity direction

        q=q+n_p*eps
        #now the particle is in the next element!
        return q,p






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
        # el= self.which_Element(qEl_n) # todo: a more efficient algorithm here will make up to a 17% difference. Perhaps
        # #not always checking if the particle is inside the element by looking at how far away it is from and edge and
        # #calculating when I should check again the soonest
        F_n=self.currentEl.force(qEl_n)
        if np.isnan(F_n[0]) == True: #particle is outside element if an array of length 1 with np.nan is returned
            self.check_If_Particle_Is_Outside_And_Handle_Edge_Event(qEl_n)  #check if element has changed.
            return
        #a_n = F_n  # acceleration new or acceleration sub n+1
        pEl_n=fast_pNew(pEl,F,F_n,self.h)
        self.qEl=qEl_n
        self.pEl=pEl_n
        self.forceLast=F_n #record the force to be recycled
        self.elHasChanged = False# if the leapfrog is completed, then the element did not change during the leapfrog

    def check_If_Particle_Is_Outside_And_Handle_Edge_Event(self,qEl_n):
        #this method checks if the element that the particle is in, or being evaluated, has changed. If it has
        #changed then that needs to be recorded and the particle carefully walked up to the edge of the element
        #This returns True if the particle has been walked to the next element with a special algorithm, or is when
        #using this algorithm the particle is now outside the lattice. It also return True if the provided element is
        #None. Most of the time this return false and the leapfrog algorithm continues
        #el: The element to check against

        #todo: there are some issues here with element edges
        el=self.which_Element(qEl_n)
        if el is None: #if outside the lattice
            self.particle.clipped = True
        elif el is not self.currentEl: #element has changed
            self.particle.cumulativeLength += self.currentEl.Lo  # add the previous orbit length
            qElLab=self.currentEl.transform_Element_Coords_Into_Lab_Frame(qEl_n)
            pElLab=self.currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(self.pEl)
            self.currentEl=el
            self.qEl = self.currentEl.transform_Lab_Coords_Into_Element_Frame(qElLab)  # at the beginning of the next element
            self.pEl = self.currentEl.transform_Lab_Frame_Vector_Into_Element_Frame(pElLab)  # at the beginning of the next
            # element
            self.elHasChanged = True
    def which_Element_Lab_Coords(self,qLab):
        for el in self.latticeElementList:
            if el.is_Coord_Inside(el.transform_Lab_Coords_Into_Element_Frame(qLab))==True:
                return el
        return None
    def which_Element(self,qEl): #.134
        #find which element the particle is in, but check the current element first to see if it's there ,which save time
        #and will be the case most of the time. Also, recycle the element coordinates for use in force evaluation later
        isInside=self.currentEl.is_Coord_Inside(qEl)
        if isInside==True: #if the particle is defintely inside the current element, then we found it! Otherwise, go on to search
            #with shapely
            return self.currentEl
        else: #if not defintely inside current element, search everywhere.
            #first, start with the element that follows the one that the particle is currently in to save time because
            #it's likely there
            qElLab=self.currentEl.transform_Element_Coords_Into_Lab_Frame(qEl)
            if self.currentEl.index+1>=len(self.latticeElementList):
                nextEl=self.latticeElementList[0]
            else:
                nextEl=self.latticeElementList[self.currentEl.index+1]
            if nextEl.is_Coord_Inside(nextEl.transform_Lab_Coords_Into_Element_Frame(qElLab)) == True:
                return nextEl

            for el in self.latticeElementList:
                if el is not self.currentEl and el is not nextEl: #don't waste rechecking current element or next element
                    if el.is_Coord_Inside(el.transform_Lab_Coords_Into_Element_Frame(qElLab))==True:
                        return el
            return None
