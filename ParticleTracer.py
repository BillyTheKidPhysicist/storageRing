import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import cProfile
import sys
from shapely.geometry import Polygon,Point
from pathos.pools import ProcessPool
import pathos as pa
import scipy.interpolate as spi
import numpy.linalg as npl
import sympy as sym
from elementPT import Element

def Compute_Bending_Radius_For_Segmented_Bender(L,rp,yokeWidth,numMagnets,angle,space=0.0):
    #ucAng=angle/(2*numMagnets)
    rb=(L+2*space)/(2*np.tan(angle/(2*numMagnets)))+yokeWidth+rp
    #ucAng1=np.arctan((L/2)/(rb-rp-yokeWidth))

    return rb

class particleTracer:
    def __init__(self,v0Nominal):
        self.v0Nominal = v0Nominal  # Design particle speed
        self.m_Actual = 1.1648E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009994E-24 # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.m=1 #adjusted value of mass. 1 is equal to li7 mass
        self.u0=self.u0_Actual/self.m_Actual #Adjusted value of bohr magneton, about equal to 800
        self.kb = 1.38064852E-23  # boltzman constant, SI
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
        self.benderIndices=[] #list that holds index values of benders. First bender is the first one that the particle sees
            #if it started from beginning of the lattice. Remember that lattice cannot begin with a bender
        self.combinerIndex=None #the index in the lattice where the combiner is
        self.totalLength=None #total length of lattice, m

        self.vacuumTube=None #holds the vacuum tube object
        self.lattice=[] #to hold all the lattice elements

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


        self.test=[]

    def add_Lens_Ideal(self,L,Bp,rp,ap=None):
        #Add element to the lattice. see elementPT.py for more details on specific element
        #L: Length of lens, m
        #Bp: field strength at pole face of lens, T
        #rp: bore radius of element, m
        #ap: size of apeture. If none then a fraction of the bore radius. Can't be bigger than bore radius. unitless
        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        args=[Bp,rp,L,ap]
        el=Element(args,'LENS_IDEAL',self) #create a lens element object
        el.index = len(self.lattice) #where the element is in the lattice
        self.lattice.append(el) #add element to the list holding lattice elements in order

    def add_Drift(self,L,ap=.03):
        #Add element to the lattice. see elementPT.py for more details on specific element
        #L: length of drift element, m
        #ap:apeture. Default value of 3 cm radius, unitless
        args=[L,ap]
        el=Element(args,'DRIFT',self)#create a drift element object
        el.index = len(self.lattice) #where the element is in the lattice
        self.lattice.append(el) #add element to the list holding lattice elements in order
    def add_Lens_Sim_Transverse(self,file,L,rp,ap=None):
        #Add element to the lattice. see elementPT.py for more details on specific element
        # file: string filename of file containing field data. rows contains points and values
        # L: Hard edge length of segment magnet in a segment
        # rp: bore radius of segment magnet
        apFrac = .9  # apeture fraction
        if ap is None:  # set the apeture as fraction of bore radius to account for tube thickness
            ap = apFrac * rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        args=[file,L,rp,ap]
        el=Element(args,"LENS_SIM_TRANSVERSE",self)
        self.lattice.append(el)
    def add_Lens_Sim_With_Fringe_Fields(self,fileTransverse,fileFringe,L,rp,ap=None,edgeFact=3):
        #Add element to the lattice. see elementPT.py for more details on specific element
        #fileTransverse: File containing transverse data for the lens
        #fileFringe: File containing grid data to model fringe fields
        # L: Hard edge length of segment magnet in a segment
        # rp: bore radius of segment magnet
        #ap: apeture of bore, typically the vacuum tube
        #edgeFact: the length of the fringe field simulating cap as a multiple of the boreradius
        if L/(2*edgeFact*rp)<1:
            raise Exception('LENS IS TOO SHORT')
        apFrac = .9  # apeture fraction
        if ap is None:  # set the apeture as fraction of bore radius to account for tube thickness
            ap = apFrac * rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        LInnerLens=L-2*edgeFact*rp #length of inner segment is smaller than total lenght because of fringe field ends
        args=[fileTransverse,LInnerLens,rp,ap]
        lens = Element(args, "LENS_SIM_TRANSVERSE", self)

        cap1Args=[fileFringe,rp,ap,'INLET',edgeFact]
        cap2Args = [fileFringe, rp, ap, 'OUTLET',edgeFact]
        cap1 = Element(cap1Args, 'LENS_SIM_CAP', self)
        cap2 = Element(cap2Args, 'LENS_SIM_CAP', self)
        self.lattice.extend([cap1, lens, cap2])



    def add_Bender_Sim_Segmented(self,file,L,rp,rb,extraspace,yokeWidth,numMagnets,ap=None):
        #Add element to the lattice. see elementPT.py for more details on specific element
        #file: string filename of file containing field data. MUST BE DATA ON A GRID. Data is organized linearly though.
        #data must be exported exactly like it is in comsol or it will break. Field data is for 1 unit cell
        #L: Hard edge length of segment magnet in a segment
        #rp: bore radius of segment magnet
        #rb: bending radius
        #extraSpace: extra space added to each end of segment. Space between two segment will be twice this value
        #yokeWidth: width of magnets comprising the yoke. Basically, the height of each magnet in each layer
        #numMagnets: number of magnet
        #ap: apeture of vacuum. Default is .9*rp
        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        args=[file,L,rp,rb,extraspace,yokeWidth,numMagnets,ap]
        el=Element(args,'BENDER_SIM_SEGMENTED',self)
        self.lattice.append(el)
    def add_Bender_Sim_Segmented_With_End_Cap(self,fileBend,fileCap,L,Lcap,rp,rb,extraspace,yokeWidth,numMagnets,ap=None):
        #Add element to the lattice. see elementPT.py for more details on specific element
        #Lcap: Length of element on the end/input of bender
        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        benderArgs=[fileBend,L,rp,rb,extraspace,yokeWidth,numMagnets,ap]


        bender=Element(benderArgs,'BENDER_SIM_SEGMENTED',self)
        cap1Args=[fileCap,Lcap,bender.rOffset,rp,ap,'INLET']
        cap2Args = [fileCap, Lcap, bender.rOffset, rp, ap, 'OUTLET']
        cap1 = Element(cap1Args, 'BENDER_SIM_SEGMENTED_CAP', self)
        cap2 = Element(cap2Args, 'BENDER_SIM_SEGMENTED_CAP', self)

        self.lattice.extend([cap1,bender,cap2])
    def add_Bender_Ideal_Segmented(self,L,Bp,rb,rp,yokeWidth,numMagnets,ap=None,space=None):
        #Add element to the lattice. see elementPT.py for more details on specific element
        #L: Length of individual magnet.
        #Bp: Field strength at pole face
        # rb: nominal bending radius of element's centerline. Actual radius is larger because particle 'rides' a little
            #outside this, m
        #rp: Bore radius of element
        #yokeWidth: width of the yoke, but also refers to the width of the magnetic material
        #numMagnet: number of magnets in segmented bender
        #ap: apeture of magnet, ie the inner radius of the vacuum tube
        #space: extra space from magnet holder in the direction of the length of the magnet. This effectively add length
            #to the magnet. total length will be changed by TWICE this value, the space is on each end

        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        args=[L,Bp,rb,rp,yokeWidth,numMagnets,ap,space]
        el=Element(args,"BENDER_IDEAL_SEGMENTED",self)
        el.index = len(self.lattice)  # where the element is in the lattice
        self.lattice.append(el)
    def add_Bender_Ideal(self,ang,Bp,rb,rp,ap=None):
        #Add element to the lattice. see elementPT.py for more details on specific element
        #ang: Bending angle of bender, radians
        #rb: nominal bending radius of element's centerline. Actual radius is larger because particle 'rides' a little
        # outside this, m
        #Bp: field strength at pole face of lens, T
        #rp: bore radius of element, m
        #ap: size of apeture. If none then a fraction of the bore radius. Can't be bigger than bore radius, unitless

        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        args=[Bp,rb,rp,ang,ap]
        el=Element(args,'BENDER_IDEAL',self) #create a bender element object
        el.index = len(self.lattice) #where the element is in the lattice
        self.benderIndices.append(el.index)
        self.lattice.append(el) #add element to the list holding lattice elements in order
    def add_Combiner_Ideal(self,La,Lb,ang,offset,c1=1,c2=20,ap=.01):
        # Add element to the lattice. see elementPT.py for more details on specific element
        #add combiner (stern gerlacht) element to lattice
        #La: input length of combiner. The bent portion outside of combiner
        #Lb:  length of vacuum tube through the combiner. The segment that contains the field
        #ang: angle that particle enters the combiner at
        # offset: particle enters inner section with some offset
        #c1: dipole component of combiner
        #c2: quadrupole component of bender
        #check to see if inlet length is too short. The minimum length is a function of apeture and angle
        minLa=ap*np.sin(ang)
        if La<minLa:
            raise Exception('INLET LENGTH IS SHORTER THAN MINIMUM')
        args=[La,Lb,ang,offset,ap,c1,c2]
        el=Element(args,'COMBINER_IDEAL',self) #create a combiner element object
        el.index = len(self.lattice) #where the element is in the lattice
        self.combinerIndex=el.index
        self.lattice.append(el) #add element to the list holding lattice elements in order


    def reset(self):
        #reset parameters related to the particle to do another simulation. Doesn't change anything about the lattice though.
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
        self.reset() #reset params
        #prepare for a particle to be traced
        self.vs = np.abs(self.p[0] / self.m)
        dl=self.vs*self.h #approximate stepsize
        for el in self.lattice:
            if dl*10>el.L:
                raise Exception('STEP SIZE TOO LARGE')


        self.currentEl = self.which_Element(self.q)
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
            self.test.append(npl.norm(self.p))
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
        #when the particle has stepped over into the next element in time_Step_Trapezoid this method is called.
        #This method calculates the correct timestep to put the particle just on the other side of the end of the element
        #using velocity verlet
        #print(q[0])
        r=el.r2-q[:2]
        rt=npl.norm(el.nb*r[:2]) #perindicular position  to element's end
        pt=npl.norm(el.nb*p[:2]) #perpindicular momentum to surface of element's end


        F1=self.force(q,el=el)
        Ft=npl.norm(el.nb*F1[:2])
        #print(rt,pt)

        h=rt/(pt/self.m)
        self.T += h-self.h #to account for the smaller step size here
        q=q+(self.p/self.m)*h

        eps=1e-9 #tiny step to put the particle on the other side
        np=p/npl.norm(p) #normalized vector of particle direction
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
        q=self.q #q old or q sub n
        p=self.p #p old or p sub n
        if self.elHasChanged==False and self.ForceLast is not None:
            F=self.ForceLast
        else:
            F=self.force(q)
        a = F / self.m  # acceleration old or acceleration sub n
        q_n=q+(p/self.m)*self.h+.5*a*self.h**2 #q new or q sub n+1
        el, qel = self.which_Element_Wrapper(q_n)
        exit=self.check_Element_And_Handle_Edge_Event(el)
        if exit==True:
            self.elHasChanged = True
            return
        else:
            self.elHasChanged=False

        F_n=self.force(q_n,el=el,qel=qel)
        a_n = F_n / self.m  # acceleration new or acceleration sub n+1
        p_n=p+self.m*.5*(a+a_n)*self.h

        self.q=q_n
        self.p=p_n
        self.ForceLast=F_n #record the force to be recycled


    def check_Element_And_Handle_Edge_Event(self,el):
        #this method checks if the element that the particle is in, or being evaluated, has changed. If it has
        #changed then that needs to be recorded and the particle carefully walked up to the edge of the element
        #This returns True if the particle has been walked to the edge or is outside the element becuase the element
        #is none type, which occurs if the particle is found to be outside the lattice when evaluating force
        exit=False
        if el is None:
            self.particleOutside = True
            exit=True
        elif el is not self.currentEl:
            self.q, self.p = self.handle_Element_Edge(self.currentEl, self.q, self.p)
            self.cumulativeLength += self.currentEl.Lo #add the previous orbit length
            self.currentEl = el
            exit=True
        return exit

    def get_Energies(self,q,p,el):
        PE =el.get_Potential_Energy(q)
        KE =npl.norm(p)**2/(2*self.m)

        return  PE,KE


    def force(self,q,qel=None,el=None):
        #calculate force. The force from the element is in the element coordinates, and the particle's coordinates
        #must be in the element frame
        #q: particle's coordinates in lab frame
        if el is None:
            el=self.which_Element(q) #find element the particle is in
        if qel is None:
            qel = el.transform_Lab_Coords_Into_Element_Frame(q)
        Fel=el.force(qel) #force in element frame
        FLab=el.transform_Element_Frame_Vector_To_Lab_Frame(Fel) #force in lab frame
        #self.test.append(npl.norm(FLab))
        #if self.q[0]!=q[0]:
        #    self.test.append(npl.norm(FLab))
        #if el.type=="BENDER_SIM_SEGMENTED":
        #    print(el.type,qel[:2],qel[0]+el.rb)
        #else:
        #    print(el.type, qel[:2])

        return FLab


    def which_Element_Wrapper(self,q,return_qel=True):
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
            el = self.which_Element(q)
            if el is None: #if there is no element, then there are also no corresponding coordinates
                qel=None
            else:
                qel = el.transform_Lab_Coords_Into_Element_Frame(q)
            if return_qel == True:
                return el,qel
            else:
                return el


    def which_Element_Shapely(self,q):
        # Use shapely to find where the element is. If the object is exaclty on the edge, it should catch that
        point = Point([q[0], q[1]])
        for el in self.lattice:
            if el.SO.contains(point) == True:
                if np.abs(q[2]) > el.ap:  # element clips in the z direction
                    return None
                else:
                    # now check to make sure the particle isn't exactly on and edge
                    return el  # return the element the particle is in
        return None #if no element found
    def which_Element(self,q):
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
        raise Exception('something doenst make sense')
        el=self.currentEl
        E=sum(self.get_Energies())
        deltaE=E-self.EList[-1]
        #Basically solve .5*m*(v+dv)dot(v+dv)+PE+deltaPE=E0 then add dv to v
        ndotv= el.nb[0]*self.p[0]/self.m+el.nb[1]*self.p[1]/self.m#the normal to the element inlet dotted into the particle velocity
        deltav=-(np.sqrt(ndotv**2-2*deltaE/self.m)+ndotv)

        self.p[0]+=deltav*el.nb[0]*self.m
        self.p[1]+=deltav*el.nb[1]*self.m
    def end_Lattice(self):

        self.catch_Errors()
        self.set_Element_Coordinates()
        self.make_Geometry()

        self.totalLength=0
        #for el in self.lattice: #total length of particle's orbit in an element
        #    self.totalLength+=el.Lo

    def solve_Triangle_Problem(self,phi1,L1,L2,offset,r1,r2):
        #the triangle problem refers to two circles and a kinked section connected by their tangets. This is the situation
        #with the combiner and the two benders.
        #phi1: bending angle of the combiner, or the amount of 'kink'
        #L1: length of the section after the kink to the bender
        #L2: length of the section before the kink to the bender
        #offset: offset of input trajectory to y=0 and input plane at field section of vacuum.
        #r1: radius of circle after the combiner
        #r2: radius of circle before the combiner
        #note that L1 and L2 INCLUDE the sections in the combiner. This function will likely be used by another function
        #that sorts that all out.


        L1=L1-offset/np.tan(phi1)
        L2=L2-offset*np.sin(phi1)+offset/np.sin(phi1)


        L3=np.sqrt((L1-np.sin(phi1)*r2+L2*np.cos(phi1))**2+(L2*np.sin(phi1)-r2*(1-np.cos(phi1))+(r2-r1))**2)
        phi2=np.pi*1.5-np.arctan(L2/r2)-np.arccos((L3**2+L2**2-L1**2)/(2*L3*np.sqrt(L2**2+r2**2))) #bending radius of bender
        #after combiner
        phi3=np.pi*1.5-np.arctan(L1/r1)-np.arccos((L3**2+L1**2-L2**2)/(2*L3*np.sqrt(L1**2+r1**2)))#bending radius of bender
        #before combiner
        return phi3,phi2,L3

    def make_Geometry(self):
        #construct the shapely objects used to plot the lattice and to determine if particles are inside of the lattice.
        #it could be changed to only use the shapely objects for plotting, but it would take some clever algorithm I think
        #and I am in a crunch kind of.
        #----
        #all of these take some thinking to visualize what's happening.
        benderPoints=500 #how many points to represent the bender with along each curve
        for el in self.lattice:
            L=el.L
            xb=el.r1[0]
            yb=el.r1[1]
            xe=el.r2[0]
            ye=el.r2[1]
            ap=el.ap
            theta=el.theta
            if el.type=='DRIFT' or el.type=='LENS_IDEAL' or el.type=="BENDER_SIM_SEGMENTED_CAP" or el.type=="LENS_SIM_TRANSVERSE"\
                    or el.type=='LENS_SIM_CAP':
                q1=np.asarray([xb-np.sin(theta)*ap,yb+ap*np.cos(theta)]) #top left when theta=0
                q2=np.asarray([xe-np.sin(theta)*ap,ye+ap*np.cos(theta)]) #top right when theta=0
                q3=np.asarray([xe+np.sin(theta)*ap,ye-ap*np.cos(theta)]) #bottom right when theta=0
                q4=np.asarray([xb+np.sin(theta)*ap,yb-ap*np.cos(theta)]) #bottom left when theta=0
                el.SO=Polygon([q1,q2,q3,q4])
            elif el.type=='BENDER_IDEAL'or el.type=='BENDER_IDEAL_SEGMENTED' or el.type=='BENDER_SIM_SEGMENTED':
                phiArr=np.linspace(0,-el.ang,num=benderPoints)+theta+np.pi/2 #angles swept out
                xInner=(el.rb-ap)*np.cos(phiArr)+el.r0[0] #x values for inner bend
                yInner=(el.rb-ap)*np.sin(phiArr)+el.r0[1] #y values for inner bend
                xOuter=np.flip((el.rb+ap)*np.cos(phiArr)+el.r0[0]) #x values for outer bend
                yOuter=np.flip((el.rb+ap)*np.sin(phiArr)+el.r0[1]) #y values for outer bend
                x=np.append(xInner,xOuter) #list of x values in order
                y=np.append(yInner,yOuter) #list of y values in order
                el.SO=Polygon(np.column_stack((x,y))) #shape the coordinates and make the object
            elif el.type=='COMBINER_IDEAL':
                q1=np.asarray([0,ap]) #top left when theta=0
                q2=np.asarray([el.Lb,ap]) #top middle when theta=0
                q3=np.asarray([el.Lb+(el.La-el.ap*np.sin(el.ang))*np.cos(el.ang),ap+(el.La-el.ap*np.sin(el.ang))*np.sin(el.ang)]) #top right when theta=0
                q4=np.asarray([el.Lb+(el.La+el.ap*np.sin(el.ang))*np.cos(el.ang),-ap+(el.La+el.ap*np.sin(el.ang))*np.sin(el.ang)]) #bottom right when theta=0
                q5=np.asarray([el.Lb,-ap]) #bottom middle when theta=0
                q6 = np.asarray([0, -ap])  # bottom left when theta=0
                points=[q1,q2,q3,q4,q5,q6]
                for i in range(len(points)):
                    points[i]=el.ROut@points[i]+el.r2
                el.SO=Polygon(points)
            else:
                raise Exception('No correct element provided')
    def catch_Errors(self):
        #catch any preliminary errors. Alot of error handling happens in other methods. This is a catch all for other
        #kinds
        if self.lattice[0].type=='BENDER_IDEAL': #first element can't be a bending element
            raise Exception('FIRST ELEMENT CANT BE A BENDER')
        if self.lattice[0].type=='COMBINER': #first element can't be a bending element
            raise Exception('FIRST ELEMENT CANT BE A COMBINER')
    def set_Element_Coordinates(self):
        #each element has a coordinate for beginning and for end, as well as a value describing it's rotation where
        #0 degrees is to the east and 180 degrees to the west. Each element also has a normal vector for the input
        #and output planes. The first element's beginning is at 0,0 with a -180 degree angle and each following element 
        # builds upon that. The final element's ending coordinates must match the beginning elements beginning coordinates
        i=0
        rbo=np.asarray([0,0]) #beginning of first element, vector. Always 0,0
        reo=None #end of last element, vector. This is not the geometri beginning but where the orbit enters the element,
        #it could be the geometric point though

        for el in self.lattice: #loop through elements in the lattice
            if i==0: #if the element is the first in the lattice
                xb=0.0#set beginning coords
                yb=0.0#set beginning coords
                el.theta=np.pi #first element is straight. It can't be a bender
                xe=el.L*np.cos(el.theta) #set ending coords
                ye=el.L*np.sin(el.theta) #set ending coords
                el.nb=-np.asarray([np.cos(el.theta),np.sin(el.theta)]) #normal vector to input
                el.ne=-el.nb
            else: #if element is not the first
                xb=self.lattice[i-1].r2[0]#set beginning coordinates to end of last
                yb=self.lattice[i-1].r2[1]#set beginning coordinates to end of last
                prevEl = self.lattice[i - 1]
                #set end coordinates
                if el.type=='DRIFT' or el.type=='LENS_IDEAL' or el.type=="LENS_SIM_TRANSVERSE" or el.type=='LENS_SIM_CAP':
                    if prevEl.type=='COMBINER_IDEAL':
                        el.theta = prevEl.theta-np.pi  # set the angle that the element is tilted relative to its
                        # reference frame. This is based on the previous element
                    else:
                        el.theta=prevEl.theta-prevEl.ang
                    xe=xb+el.L*np.cos(el.theta)
                    ye=yb+el.L*np.sin(el.theta)
                    el.nb = -np.asarray([np.cos(el.theta), np.sin(el.theta)])  # normal vector to input
                    el.ne = -el.nb #normal vector to end
                    if prevEl.type=='BENDER_IDEAL' or prevEl.type=='BENDER_IDEAL_SEGMENTED' or prevEl.type=='BENDER_SIM_SEGMENTED'\
                            or prevEl.type=="BENDER_SIM_SEGMENTED_CAP" or prevEl.type=='LENS_SIM_CAP':
                        n=np.zeros(2)
                        n[0]=-prevEl.ne[1]
                        n[1]=prevEl.ne[0]
                        dr=n*prevEl.rOffset
                        xb+=dr[0]
                        yb+=dr[1]
                        xe+=dr[0]
                        ye+=dr[1]

                    el.r0=np.asarray([(xb+xe)/2,(yb+ye)/2]) #center of lens or drift is midpoint of line connecting beginning and end
                elif el.type=="BENDER_SIM_SEGMENTED_CAP":
                    el.theta=prevEl.theta-prevEl.ang
                    xe=xb+el.L*np.cos(el.theta)
                    ye=yb+el.L*np.sin(el.theta)
                    el.nb = -np.asarray([np.cos(el.theta), np.sin(el.theta)])  # normal vector to input
                    el.ne = -el.nb  # normal vector to end

                    if prevEl.type == "BENDER_SIM_SEGMENTED":
                        pass
                    else:
                        xb += el.rOffset * np.sin(el.theta)
                        yb += el.rOffset * (-np.cos(el.theta))
                        xe += el.rOffset * np.sin(el.theta)
                        ye += el.rOffset * (-np.cos(el.theta))
                elif el.type=='BENDER_IDEAL' or el.type=='BENDER_IDEAL_SEGMENTED' or el.type=='BENDER_SIM_SEGMENTED':
                    if prevEl.type=='COMBINER_IDEAL':
                        el.theta = prevEl.theta-np.pi  # set the angle that the element is tilted relative to its
                        # reference frame. This is based on the previous element
                    else:
                        el.theta=prevEl.theta-prevEl.ang
                    if prevEl.type=="BENDER_SIM_SEGMENTED_CAP":
                        rOffset=0
                    else:
                        rOffset=el.rOffset
                    #the bender can be tilted so this is tricky. This is a rotation about a point that is
                    #not the origin. First I need to find that point.
                    xc=xb+np.sin(el.theta)*el.rb #without including trajectory offset
                    yc=yb-np.cos(el.theta)*el.rb
                    #now use the point to rotate around
                    phi=-el.ang #bending angle. Need to keep in mind that clockwise is negative
                    #and counterclockwise positive. So I add a negative sign here to fix that up
                    xe=np.cos(phi)*xb-np.sin(phi)*yb-xc*np.cos(phi)+yc*np.sin(phi)+xc
                    ye=np.sin(phi)*xb+np.cos(phi)*yb-xc*np.sin(phi)-yc*np.cos(phi)+yc
                    #accomodate for bending trajectory
                    xb+=rOffset*np.sin(el.theta)
                    yb+=rOffset*(-np.cos(el.theta))
                    xe+=rOffset*np.sin(el.theta)
                    ye+=rOffset*(-np.cos(el.theta))
                    el.nb=np.asarray([np.cos(el.theta-np.pi), np.sin(el.theta-np.pi)])  # normal vector to input
                    el.ne=np.asarray([np.cos(el.theta-np.pi+(np.pi-el.ang)), np.sin(el.theta-np.pi+(np.pi-el.ang))])  # normal vector to output
                    el.r0=np.asarray([xb+el.rb*np.sin(el.theta),yb-el.rb*np.cos(el.theta)]) #coordinates of center of bender
                elif el.type=='COMBINER_IDEAL':
                    el.theta=2*np.pi-el.ang-(np.pi-prevEl.theta)# Tilt the combiner down by el.ang so y=0 is perpindicular
                        #to the input. Rotate it 1 whole revolution, then back it off by the difference. Need to subtract
                        #np.pi because the combiner's input is not at the origin, but faces 'east'
                    el.theta=el.theta-2*np.pi*(el.theta//(2*np.pi)) #the above logic can cause the element to have to rotate
                        #more than 360 deg
                    #to find location of output coords use vector that connects input and output in element frame
                    #and rotate it. Input is where nominal trajectory particle enters
                    drx=-(el.Lb+(el.La-el.inputOffset*np.sin(el.ang))*np.cos(el.ang))
                    dry=-(el.inputOffset+(el.La-el.inputOffset*np.sin(el.ang))*np.sin(el.ang))

                    #drx = -(el.Lb+el.La *np.cos(el.ang))
                    #dry = -(el.La*np.sin(el.ang))



                    el.r1El=np.asarray([0,0])
                    el.r2El=-np.asarray([drx,dry])
                    dr=np.asarray([drx,dry]) #position vector between input and output of element in element frame
                    R = np.asarray([[np.cos(el.theta), -np.sin(el.theta)], [np.sin(el.theta), np.cos(el.theta)]])
                    dr=R@dr #rotate to lab frame
                    xe,ye=xb+dr[0],yb+dr[1]
                    el.ne=-np.asarray([np.cos(el.theta),np.sin(el.theta)]) #output normal vector
                    el.nb=np.asarray([np.cos(el.theta+el.ang),np.sin(el.theta+el.ang)]) #input normal vector

                else:
                    raise Exception('No correct element name provided')
            #need to make rotation matrices for element
            if el.type=='BENDER_IDEAL' or el.type=='BENDER_IDEAL_SEGMENTED' or el.type=='BENDER_SIM_SEGMENTED':
                rot = (el.theta - el.ang + np.pi / 2)
            elif el.type=='LENS_IDEAL' or el.type=='DRIFT' or el.type=='COMBINER_IDEAL' or el.type=="LENS_SIM_TRANSVERSE":
                rot = el.theta
            elif el.type=="BENDER_SIM_SEGMENTED_CAP" or el.type=='LENS_SIM_CAP':
                if el.position=='INLET':
                    rot = el.theta
                elif el.position=='OUTLET':
                    rot = el.theta+np.pi
                else:
                    raise Exception('NOT IMPLEMENTED')
                #print('here',rot)
            else:
                raise Exception('No correct element name provided')
            el.ROut = np.asarray([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]) #the rotation matrix for
                #rotating out of the element reference frame
            el.RIn = np.asarray([[np.cos(-rot), -np.sin(-rot)], [np.sin(-rot), np.cos(-rot)]]) #the rotation matrix for
                #rotating into the element reference frame
            #clean up tiny numbers. There are some numbers like 1e-16 that should be zero
            el.ne=np.round(el.ne,10)
            el.nb=np.round(el.nb,10)


            el.r1=np.asarray([xb,yb])#position vector of beginning of element
            el.r2=np.asarray([xe,ye])#position vector of ending of element
            if i==len(self.lattice)-1: #if the last element then set the end of the element correctly
                reo = el.r2.copy()
                if el.type=='BENDER_IDEAL' or el.type=='COMBINER_IDEAL'or el.type=='BENDER_IDEAL_SEGMENTED' or el.type=='BENDER_SIM_SEGMENTED'\
                        or el.type=='BENDER_SIM_SEGMENTED_CAP':
                    reo[1]-=el.rOffset
            i+=1
        #check that the last point matchs the first point within a small number.
        #need to account for offset.
        deltax=np.abs(rbo[0]-reo[0])
        deltay=np.abs(rbo[1]-reo[1])
        smallNum=1e-10
        if deltax>smallNum or deltay>smallNum:
            raise Exception('ENDING POINTS DOES NOT MEET WITH BEGINNING POINT. LATTICE IS NOT CLOSED')


    def compute_Output_Angle_And_Offset(self,L,ap,c1,c2,v0,h=1e-6):
        #this computes the output angle and offset for a combiner magnet
        #L: length of combiner magnet
        #c1: dipole moment
        #c2: quadrupole moment
        #ap: apeture of combiner in x axis, half gap
        #v0: nominal particle speed
        args=[L,ap,c1,c2]
        el=Element(args,'COMBINER',self) #create a combiner element object
        q = np.asarray([0, 0, 0])
        p=self.m*np.asarray([v0,0,0])
        while True:
            F=self.force(q,coordsxy=q,el=el)
            a=F/self.m
            q_n=q+(p/self.m)*h+.5*a*h**2
            F_n=self.force(q_n,coordsxy=q_n,el=el)
            a_n = F_n / self.m  # acceleration new or acceleration sub n+1
            p_n=p+self.m*.5*(a+a_n)*h
            if q_n[0]>L:
                dr=L-q[0]
                dt=dr/(p[0]/self.m)
                q_n=q+(p/self.m)*dt+.5*a*dt**2
                F_n=self.force(q_n,coordsxy=q_n,el=el)
                a_n = F_n / self.m  # acceleration new or acceleration sub n+1
                p_n=p+self.m*.5*(a+a_n)*dt
                q=q_n
                p=p_n
                break
            q=q_n
            p=p_n
        outputAngle = np.arctan2(p[1], p[0])
        outputOffset = q[1]
        return outputAngle,outputOffset
    def show_Lattice(self,particleCoords=None):
        #plot the lattice using shapely
        #particleCoords: Array or list holding particle coordinate such as [x,y]
        plt.close('all')
        for el in self.lattice:
            plt.plot(*el.SO.exterior.xy,c='black')
        if particleCoords is not None:
            if particleCoords.shape[0]==3: #if the 3d value is provided trim it to 2D
                particleCoords=particleCoords[:2]
            #plot the particle as both a dot and a X
            plt.scatter(*particleCoords,marker='x',s=1000,c='r')
            plt.scatter(*particleCoords, marker='o', s=50, c='r')
        plt.grid()
        plt.gca().set_aspect('equal')
        plt.show()

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
