import numpy.linalg as npl
import numba
import time
import numpy as np
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import sys
from shapely.geometry import Polygon,Point
import pathos as pa
#from ParticleClass import Particle
#from profilehooks import profile,coverage

#todo: compute energy

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
        self.test=[]


    def initialize(self):
        # prepare for a single particle to be traced
        self.T=0
        self.particle.cumulativeLength=0
        if self.particle.clipped is not None:
            self.particle.clipped=False
        self.particle.force=None
        dl=self.particle.v0*self.h #approximate stepsize
        for el in self.latticeElementList:
            if dl>el.Lo/10.0:
                raise Exception('STEP SIZE TOO LARGE')
        self.particle.currentEl = self.which_Element_Slow(self.particle.q)
        if self.particle.currentEl is None:
            self.particle.clipped=True
        if self.fastMode==False:
            self.particle.log_Params()

    def trace(self,particle,h,T0,fastMode=False):
        #TODO: HOW TO DEAL WITH PARTICLE BEING TRACED MULTIPLE TIMES. IS THAT SOMETHING I EVEN WANT TO DO EVER?
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
            #self.test.append(npl.norm(self.particle.force))
            if self.particle.clipped==True:
                break
            if fastMode==False:
                self.particle.log_Params()
            self.T += self.h
            self.particle.T=self.T

        self.particle.finished(totalLatticeLength=self.totalLatticeLength)

        return self.particle

    def handle_Element_Edge(self):
        # This method calculates the correct timestep to put the particle just on the other side of the end of the element
        # using velocity verlet. I had intended this to use the force right at the end of the element, but that didn't
        # work right. For now it simply uses explicit euler more or less
        el,q,p=self.particle.currentEl, self.particle.q, self.particle.p
        r=el.r2-q

        rt=np.abs(np.sum(el.ne*r[:2])) #perindicular position  to element's end
        pt=np.abs(np.sum(el.ne*p[:2]))#perpindicular momentum to surface of element's end
        h=rt/pt
        self.T += h-self.h #to account for the smaller step size here
        q=q+p*h
        eps=1e-9 #tiny step to put the particle on the other side
        n_p=p/np.sum(np.sqrt(p**2)) #normalized vector of particle velocity direction

        q=q+n_p*eps
        self.particle.q=q
        self.particle.p=p
        #TODO: FIX THIS, IT'S BROKEN. DOESN'T WORK WITH FORCE :(

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
        q=self.particle.q #q old or q sub n
        p=self.particle.p #p old or p sub n
        if self.elHasChanged==False and self.particle.force is not None: #if the particle is inside the lement it was in
            #last time step, and it's not the first time step, then recycle the force. The particle is starting at the
            #same position it stopped at last time, thus same force
            F=self.particle.force
        else: #the last force is invalid because the particle is at a new position
            F=self.force(q)

        #a = F # acceleration old or acceleration sub n
        q_n=self.fast_qNew(q,F,p,self.h)#q new or q sub n+1
        el, qel = self.which_Element(q_n) # todo: a more efficient algorithm here will make up to a 17% difference. Perhaps
        #not always checking if the particle is inside the element by looking at how far away it is from and edge and
        #calculating when I should check again the soonest
        exitLoop=self.check_Element_And_Handle_Edge_Event(el)  #check if element has changed.
        if exitLoop==True:
            self.elHasChanged = True
            return
        F_n=self.force(q_n,el=el,qel=qel)

        #a_n = F_n  # acceleration new or acceleration sub n+1
        p_n=self.fast_pNew(p,F,F_n,self.h)
        self.particle.q=q_n
        self.particle.p=p_n
        self.particle.force=F_n #record the force to be recycled
        self.elHasChanged = False# if the leapfrog is completed, then the element did not change during the leapfrog

    def check_Element_And_Handle_Edge_Event(self,el):
        #todo: change this to something that makes more sense
        #this method checks if the element that the particle is in, or being evaluated, has changed. If it has
        #changed then that needs to be recorded and the particle carefully walked up to the edge of the element
        #This returns True if the particle has been walked to the next element with a special algorithm, or is when
        #using this algorithm the particle is now outside the lattice. It also return True if the provided element is
        #None. Most of the time this return false and the leapfrog algorithm continues
        exitLoop=False
        if el is None: #if the particle is outside the lattice, the simulation is over
            self.particle.clipped = True
            return True
        elif el is not self.particle.currentEl:
            self.handle_Element_Edge()
            #it's possible that the particle is now outside the lattice, so check which element it's in
            if self.which_Element_Slow(self.particle.q) is None:
                exitLoop=True
                self.particle.clipped=True
                return exitLoop
            self.particle.cumulativeLength += self.particle.currentEl.Lo #add the previous orbit length
            self.particle.currentEl = el
            exitLoop=True

        return exitLoop


    #@profile
    def force(self,q,qel=None,el=None):
        #calculate force. The force from the element is in the element coordinates, and the particle's coordinates
        #must be in the element frame
        #q: particle's coordinates in lab frame
        if el is None:
            el=self.which_Element_Slow(q) #find element the particle is in
            if el is None: #if particle is outside!
                return None
        if qel is None:
            qel = el.transform_Lab_Coords_Into_Element_Frame(q)
        Fel=el.force(qel) #force in element frame
        FLab=el.transform_Element_Frame_Vector_To_Lab_Frame(Fel) #force in lab frame
        return FLab


    def which_Element(self,q,return_qel=True):
        #find which element the particle is in, but check the current element first to see if it's there ,which save time
        #and will be the case most of the time. Also, recycle the element coordinates for use in force evaluation later
        qel = self.particle.currentEl.transform_Lab_Coords_Into_Element_Frame(q)
        isInside=self.particle.currentEl.is_Coord_Inside(qel)
        if isInside==True: #if the particle is defintely inside the current element, then we found it! Otherwise, go on to search
            #with shapely
            if return_qel==True:
                return self.particle.currentEl,qel
            else:
                return self.particle.currentEl
        else: #if not defintely inside current element, search everywhere and more carefully (and slowly) with shapely
            el = self.which_Element_Slow(q)
            if el is None: #if there is no element, then there are also no corresponding coordinates
                qel=None
            else:
                qel = el.transform_Lab_Coords_Into_Element_Frame(q)
            if return_qel == True:
                return el,qel
            else:
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