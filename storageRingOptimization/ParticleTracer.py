import numpy.linalg as npl
import numba
import time
import numpy as np
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import sys
from shapely.geometry import Polygon,Point


def Compute_Bending_Radius_For_Segmented_Bender(L,rp,yokeWidth,numMagnets,angle,space=0.0):
    #ucAng=angle/(2*numMagnets)
    rb=(L+2*space)/(2*np.tan(angle/(2*numMagnets)))+yokeWidth+rp
    #ucAng1=np.arctan((L/2)/(rb-rp-yokeWidth))

    return rb



#this class does the work of tracing the particles through the lattice with timestepping algorithms

class ParticleTracer:
    def __init__(self,latticeObject):
        self.latticeElementList = latticeObject.elList  # list containing the elements in the lattice in order from first to last (order added)
        self.totalLatticeLength=latticeObject.totalLength

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
        self.test=[]


    def initialize(self):
        # prepare for a single particle to be traced
        self.T=0
        if self.particle.clipped is not None:
            self.particle.clipped=False
        dl=self.particle.v0*self.h #approximate stepsize
        for el in self.latticeElementList:
            if dl>el.Lo/10.0:
                raise Exception('STEP SIZE TOO LARGE')
        self.currentEl = self.which_Element_Slow(self.particle.q)
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

        self.initialize()
        if self.particle.clipped==True: #some a particles may be clipped after initializing them because they were about
            # to become clipped
            self.particle.finished(totalLatticeLength=0)
            return particle
        while(True):
            if self.T>T0:
                self.particle.clipped=False
                break
            self.time_Step_Verlet()
            if self.particle.clipped==True:
                break
            if fastMode==False:
                self.particle.log_Params(self.currentEl,self.qEl,self.pEl)
            self.T += self.h
            self.particle.T=self.T
            # self.test.append(npl.norm(self.forceLast))
        self.particle.q = self.currentEl.transform_Element_Coords_Into_Lab_Frame(self.qEl)
        self.particle.p = self.currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(self.pEl)
        self.particle.currentEl=self.currentEl
        self.particle.finished(totalLatticeLength=self.totalLatticeLength)

        return self.particle

    def handle_Element_Edge(self):
        # This method calculates the correct timestep to put the particle just on the other side of the end of the element
        # using velocity verlet. I had intended this to use the force right at the end of the element, but that didn't
        # work right. For now it simply uses explicit euler more or less
        #This returns the new position and momentum in lab frame
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




    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
    def fast_qNew(q,F,p,h):
        return q+p*h+.5*F*h**2

    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
    def fast_pNew(p,F,F_n,h):
        return p+.5*(F+F_n)*h
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
        qEl_n=self.fast_qNew(qEl,F,pEl,self.h)#q new or q sub n+1
        el= self.which_Element(qEl_n) # todo: a more efficient algorithm here will make up to a 17% difference. Perhaps
        #not always checking if the particle is inside the element by looking at how far away it is from and edge and
        #calculating when I should check again the soonest
        exitLoop=self.check_Which_Element_And_Handle_Edge_Event(el)  #check if element has changed.
        if exitLoop==True:
            self.elHasChanged = True
            return
        F_n=self.currentEl.force(qEl_n)

        #a_n = F_n  # acceleration new or acceleration sub n+1
        pEl_n=self.fast_pNew(pEl,F,F_n,self.h)
        self.qEl=qEl_n
        self.pEl=pEl_n
        self.forceLast=F_n #record the force to be recycled
        self.elHasChanged = False# if the leapfrog is completed, then the element did not change during the leapfrog

    def check_Which_Element_And_Handle_Edge_Event(self, el):
        #this method checks if the element that the particle is in, or being evaluated, has changed. If it has
        #changed then that needs to be recorded and the particle carefully walked up to the edge of the element
        #This returns True if the particle has been walked to the next element with a special algorithm, or is when
        #using this algorithm the particle is now outside the lattice. It also return True if the provided element is
        #None. Most of the time this return false and the leapfrog algorithm continues
        exitLoop=False
        if el is None: #if the particle is outside the lattice, the simulation is over
            self.particle.clipped = True
            return True
        elif el is not self.currentEl:
            qLab,pLab=self.handle_Element_Edge()
            #it's possible that the particle is now outside the lattice, so check which element it's in
            if self.which_Element_Slow(qLab) is None:
                exitLoop=True
                self.particle.clipped=True
                return exitLoop
            self.particle.cumulativeLength += self.currentEl.Lo #add the previous orbit length
            self.currentEl = el
            self.qEl=self.currentEl.transform_Lab_Coords_Into_Element_Frame(qLab)
            self.pEl=self.currentEl.transform_Lab_Frame_Vector_Into_Element_Frame(pLab)
            exitLoop=True
        return exitLoop

    def which_Element(self,qel):
        #find which element the particle is in, but check the current element first to see if it's there ,which save time
        #and will be the case most of the time. Also, recycle the element coordinates for use in force evaluation later
        isInside=self.currentEl.is_Coord_Inside(qel)
        if isInside==True: #if the particle is defintely inside the current element, then we found it! Otherwise, go on to search
            #with shapely
            return self.currentEl
        else: #if not defintely inside current element, search everywhere and more carefully (and slowly) with shapely
            q=self.currentEl.transform_Element_Coords_Into_Lab_Frame(qel)
            el = self.which_Element_Slow(q)
            return el


    def which_Element_Shapely(self,q):
        # Use shapely to find where the element is. If the object is exaclty on the edge, this will return None. This
        #is slower than using simple geometry and should be avoided. It does not account for vertical (z) apetures
        point = Point([q[0], q[1]])
        for el in self.latticeElementList:
            if el.SO.contains(point) == True:
                return el  # return the element the particle is in
        return None #if no element found, or particle exactly on an edge
    def which_Element_Slow(self,q):
        #find which element the particle is in. First try with shapely. If that fails, maybe the particle landed right on
        #or between two element. So try scooting the particle on a tiny bit and try again.
        el=self.which_Element_Shapely(q)
        if el is not None:
            qel = el.transform_Lab_Coords_Into_Element_Frame(q)
            isInside = el.is_Coord_Inside(qel)
            if isInside==True:
                return el

        #try scooting the particle along a tiny amount in case it landed in between elements, which is very rare
        #but can happen. First compute roughly the center of the ring.
        #add up all the beginnings and end of elements and average them
        center=np.zeros(2)
        for el in self.latticeElementList: #find the geometric center of the lattice. This won't be it exactly, but should be
            #qualitatively correct
            center+=el.r1[:-1]+el.r2[:-1]
        center=center/(2 * len(self.latticeElementList))

        #relative position vector
        r=center-q[:-1]

        #now rotate this and add the difference to our original vector. rotate by a small amount
        dphi=-1e-9 #need a clock wise rotation. 1 nanoradian
        R=np.array([[np.cos(dphi),-np.sin(dphi)],[np.sin(dphi),np.cos(dphi)]])
        dr=R@(q[:-1]-r)-(q[:-1]-r)
        #add that to the position and try once more
        qNew=q.copy()
        qNew[:-1]=qNew[:-1]+dr
        el=self.which_Element_Shapely(qNew)

        if el is not None:
            qel = el.transform_Lab_Coords_Into_Element_Frame(q)
            isInside = el.is_Coord_Inside(qel)
            if isInside == True:
                return el
            else:
                return None
        else:
            return None