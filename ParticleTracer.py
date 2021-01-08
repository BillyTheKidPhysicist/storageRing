import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from shapely.geometry import Polygon,Point
import pathos as pa
import numpy.linalg as npl


def Compute_Bending_Radius_For_Segmented_Bender(L,rp,yokeWidth,numMagnets,angle,space=0.0):
    #ucAng=angle/(2*numMagnets)
    rb=(L+2*space)/(2*np.tan(angle/(2*numMagnets)))+yokeWidth+rp
    #ucAng1=np.arctan((L/2)/(rb-rp-yokeWidth))

    return rb



#this class does the work of tracing the particles through the lattice with timestepping algorithms
class ParticleTracer:
    def __init__(self,latticeObject):
        self.lattice = latticeObject.elList  # list containing the elements in the lattice in order from first to last (order added)
        self.test=latticeObject
        self.m_Actual = 1.1648E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009994E-24 # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.m=1 #adjusted value of mass. 1 is equal to li7 mass
        self.q=np.zeros(3) #contains the particles current position coordinates, m
        self.p=np.zeros(3) #contains the particles current momentum. m is set to 1 so this is the same
            #as velocity. Li7 mass*meters/s
        self.qoList=[] #coordinates in orbit frame [s,x,y] where s is along orbit
        self.poList=[] #momentum coordinates in orbit frame [s,x,y] where s is along orbit
        self.qList=[] #coordinates in labs frame,[x,y,z] position,m
        self.pList=[] #coordinates in labs frame,[vx,vy,v] velocity,m/s

        self.cumulativeLength=0 #cumulative length of previous elements. This accounts for revolutions, it doesn't reset each
            #time
        self.deltaT=0 #time spent in current element by particle. Resets to zero after entering new element
        self.T=0 #total time elapsed
        self.h=None #current step size. This changes near boundaries
        self.h0=None # initial step size.
        self.particleOutside=False #if the particle has stepped outside the chamber

        self.currentEl=None #to keep track of the element the particle is currently in
        self.elHasChanged=False # to record if the particle has changed to another element in the previous step
        self.ForceLast=None #vector to hold onto last force value used to save resources. Each verlet computes the force
            #twice, and the last value of the previous step will be the first value of the next step (barring any
            #element changes)
        self.v0=None #the particles total speed.
        self.vs=None #particle speed along orbit
        self.E0=None #total initial energy of particle

        self.VList=[] #list of potential energy
        self.TList=[] #list of kinetic energy
        self.EList=[] #too keep track of total energy. This can tell me if the simulation is behaving
            #This won't always be on




    def reset(self):
        #reset parameters related to the particle to do another simulation.
        self.qList=[]
        self.pList=[]
        self.qoList=[]
        self.poList = []
        self.TList=[]
        self.VList=[]
        self.EList=[]
        self.particleOutside=False
        self.cumulativeLength=0
        self.currentEl=None
        self.T=0
        self.deltaT=0
        self.ForceLast=None
    def initialize(self):
        # prepare for a single particle to be traced
        self.reset() #reset params
        self.vs = np.abs(self.p[0] / self.m)
        dl=self.vs*self.h #approximate stepsize
        for el in self.lattice:
            if dl*10>el.L:
                raise Exception('STEP SIZE TOO LARGE')


        self.currentEl = self.which_Element_Slow(self.q)
        if self.currentEl is None:
            raise Exception('Particle\'s initial position is outside vacuum' )

        self.qList.append(self.q)
        self.pList.append(self.p)
        self.qoList.append(self.currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q,self.cumulativeLength))
        self.poList.append(self.currentEl.transform_Lab_Momentum_Into_Orbit_Frame(self.q,self.p))

        #temp = self.get_Energies(self.q, self.p, self.currentEl)
        #self.VList.append(temp[0])
        #self.TList.append(temp[1])
        #self.EList.append(sum(temp))

    def trace(self,qi,pi,h,T0):
        #trace the particle through the lattice. This is done in lab coordinates. Elements affect a particle by having
        #the particle's position transformed into the element frame and then the force is transformed out. This is obviously
        # not very efficient.
        #qi: initial position coordinates
        #pi: initial momentum coordinates
        #h: timestep
        #T0: total tracing time
        #method: the integration method to use. Now it's either velocity verlet or implicit trapezoid

        self.q=qi.copy().astype(float)
        self.p=pi.copy().astype(float)*self.m
        self.h=h
        self.h0=h
        self.v0 = npl.norm(pi)/self.m
        self.initialize()
        self.deltaT=0
        while(True):
            if self.T+self.h>T0:
                break
            self.time_Step_Verlet()
            if self.particleOutside==True:
                break
            self.qList.append(self.q)
            self.pList.append(self.p)
            self.qoList.append(self.currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q,self.cumulativeLength))
            #self.poList.append(self.currentEl.transform_Lab_Momentum_Into_Orbit_Frame(self.q, self.p))
            #self.test.append(npl.norm(self.p))
            self.T += self.h
            #temp=self.get_Energies(self.q,self.p,self.currentEl)
            #self.VList.append(temp[0])
            #self.TList.append(temp[1])
            #self.EList.append(sum(temp))
        #print(npl.norm(self.p))
        qArr=np.asarray(self.qList)
        pArr=np.asarray(self.pList)
        qoArr=np.asarray(self.qoList)
        poArr=np.asarray(self.poList)
        return qArr,pArr,qoArr,poArr,self.particleOutside

    def handle_Element_Edge(self, el, q, p):
        # This method calculates the correct timestep to put the particle just on the other side of the end of the element
        # using velocity verlet. I had intended this to use the force right at the end of the element, but that didn't
        # work right. For now it simply uses explicit euler more or less
        #

        import numpy as np #todo: wtf is happening here, why do I need this

        r=el.r2-q[:2]

        rt=np.abs(np.sum(el.ne*r[:2])) #perindicular position  to element's end
        pt=np.abs(np.sum(el.ne*p[:2]))#npl.norm(el.ne*p[:2]) #perpindicular momentum to surface of element's end
        h=rt/(pt/self.m)
        self.T += h-self.h #to account for the smaller step size here
        q=q+(self.p/self.m)*h

        eps=1e-9 #tiny step to put the particle on the other side
        np=p/npl.norm(p) #normalized vector of particle velocity direction

        q=q+np*eps

        return q,p
        #TODO: FIX THIS, IT'S BROKEN. DOESN'T WORK WITH FORCE :(
        ##precision loss is killing me here! Usually Force in the direction of the input to the next element is basically
        ##zero, thus if that is the case, I need to substitute a really small (but not hyper-super small) number for force
        ##to get the correct answer
        #if 2*Ft*self.m*rt <1e-6*pt**2:
        #    h=rt/(pt/self.m) #assuming no force
        #else:
        #    import numpy as np
        #    h = (np.sqrt(2 * Ft * self.m * rt + pt ** 2) - pt) / Ft #assuming force
        #h = rt / (pt / self.m)
        #q=q+h*p/self.m+.5*(F1/self.m)*h**2
        #F2=self.force(q,el=el)
        #p=p+.5*(F1+F2)*h
        #eps=1e-9 #tiny step to put the particle on the other side
        #np=p/npl.norm(p) #normalized vector of particle direction
        #q=q+np*eps
        #return q,p
    def time_Step_Verlet(self):
        #the velocity verlet time stepping algorithm. This version recycles the force from the previous step when
        #possible
        q=self.q #q old or q sub n
        p=self.p #p old or p sub n
        if self.elHasChanged==False and self.ForceLast is not None: #if the particle is inside the lement it was in
            #last time step, and it's not the first time step, then recycle the force. The particle is starting at the
            #same position it stopped at last time, thus same force
            F=self.ForceLast
        else: #the last force is invalid because the particle is at a new position
            F=self.force(q)
        a = F / self.m  # acceleration old or acceleration sub n
        q_n=q+(p/self.m)*self.h+.5*a*self.h**2 #q new or q sub n+1
        el, qel = self.which_Element(q_n)

        exit=self.check_Element_And_Handle_Edge_Event(el)
        if exit==True:
            self.elHasChanged = True
            return
        F_n=self.force(q_n,el=el,qel=qel)

        a_n = F_n / self.m  # acceleration new or acceleration sub n+1
        p_n=p+self.m*.5*(a+a_n)*self.h

        self.q=q_n
        self.p=p_n
        self.ForceLast=F_n #record the force to be recycled
        self.elHasChanged = False# if the leapfrog is completed, then the element did not change during the leapfrog

    def check_Element_And_Handle_Edge_Event(self,el):
        #this method checks if the element that the particle is in, or being evaluated, has changed. If it has
        #changed then that needs to be recorded and the particle carefully walked up to the edge of the element
        #This returns True if the particle has been walked to the next element with a special algorithm, or is when
        #using this algorithm the particle is now outside the lattice. It also return True if the provided element is
        #None. Most of the time this return false and the leapfrog algorithm continues
        exit=False
        if el is None: #if the particle is outside the lattice, the simulation is over
            self.particleOutside = True
            return True
        elif el is not self.currentEl:
            self.q, self.p = self.handle_Element_Edge(self.currentEl, self.q, self.p)
            #it's possible that the particle is now outside the lattice, so check which element it's in
            if self.which_Element_Slow(self.q) is None:
                exit=True
                self.particleOutside=True
                return exit
            self.cumulativeLength += self.currentEl.Lo #add the previous orbit length
            self.currentEl = el
            exit=True
        return exit

    def get_Energies(self,q,p,el):
        #get the energy of the particle at the given point and momentum
        PE =el.get_Potential_Energy(q)
        KE =npl.norm(p)**2/(2*self.m)
        return  PE,KE

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
        qel = self.currentEl.transform_Lab_Coords_Into_Element_Frame(q)
        isInside=self.currentEl.is_Coord_Inside(qel) #this returns True or None or false
        if isInside==True: #if the particle is defintely inside the current element, then we found it! Otherwise, go on to search
            #with shapely
            if return_qel==True:
                return self.currentEl,qel
            else:
                return self.currentEl
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
        #is slower than using simple geometry and should be avoided
        point = Point([q[0], q[1]])
        for el in self.lattice:
            if el.SO.contains(point) == True:
                if np.abs(q[2]) > el.ap:  # element clips in the z direction
                    return None
                else:
                    # now check to make sure the particle isn't exactly on and edge
                    return el  # return the element the particle is in
        return None #if no element found, or particle exactly on an edge
    def which_Element_Slow(self,q):
        #find which element the particle is in. First try with shapely. If that fails, maybe the particle landed right on
        #or between two element. So try scooting the particle on a tiny bit and try again.
        el=self.which_Element_Shapely(q)
        if el is not None: #if particle is definitely inside an element.
            #now test for the z clipping
            if np.abs(q[2])>el.ap:
                return None
            else:
                return el
        #try scooting the particle along a tiny amount in case it landed in between elements, which is very rare
        #but can happen. First compute roughly the center of the ring.
        #add up all the beginnings and end of elements and average them
        center=np.zeros(2)
        for el in self.lattice:
            center+=el.r1+el.r2
        center=center/(2*len(self.lattice))

        #relative position vector
        r=center-q[:-1]

        #now rotate this and add the difference to our original vector. rotate by a small amount
        dphi=-1e-9 #need a clock wise rotation. 1 nanooradian
        R=np.array([[np.cos(dphi),-np.sin(dphi)],[np.sin(dphi),np.cos(dphi)]])
        dr=R@(q[:2]-r)-(q[:2]-r)
        #add that to the position and try once more
        qNew=q.copy()
        qNew[:-1]=qNew[:-1]+dr
        el=self.which_Element_Shapely(qNew)
        if el is not None:
            # now test for the z clipping
            if np.abs(q[2]) > el.ap:
                return None
            else:
                return el
        else:
            return None
    def adjust_Energy(self):
        #when the particle traverses an element boundary, energy needs to be conserved. This would be irrelevant if I
        #included fringe field effects
        #TODO: BROKEN
        raise Exception('something doenst make sense')
        el=self.currentEl
        E=sum(self.get_Energies())
        deltaE=E-self.EList[-1]
        #Basically solve .5*m*(v+dv)dot(v+dv)+PE+deltaPE=E0 then add dv to v
        ndotv= el.nb[0]*self.p[0]/self.m+el.nb[1]*self.p[1]/self.m#the normal to the element inlet dotted into the particle velocity
        deltav=-(np.sqrt(ndotv**2-2*deltaE/self.m)+ndotv)

        self.p[0]+=deltav*el.nb[0]*self.m
        self.p[1]+=deltav*el.nb[1]*self.m


    def multi_Trace(self, argsList):
        # trace out multiple trajectories
        # argsList: list of tuples for each particle, where each tuple is (qi,vi,timestep,totaltime).
        def wrap(args): #helps to use a wrapper function to unpack arguments. this is for a single particle.
            # args is a tuple.
            qi = args[0] #initial position, 3D, m
            vi = args[1] #initial velocity, 3D, m
            h=args[2] #timestep, seconds
            T=args[3] #total time, seconds
            q, p, qo, po, particleOutside = self.trace(qi, vi, h,T)
            argsReturn=(qo,particleOutside) #return orbital coords and wether particle clipped an apeture
            return argsReturn

        pool = pa.pools.ProcessPool(nodes=pa.helpers.cpu_count()) #pool object to distribute work load
        jobs = [] #list of jobs. jobs are distributed across processors
        results = [] #list to hold results of particle tracing.
        for arg in argsList:
            jobs.append(pool.apipe(wrap, arg)) #create job for each argument in arglist, ie for each particle
        for job in jobs:
            results.append(job.get()) #get the results. wait for the result in order given
        return results
