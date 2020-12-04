import time
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import sys
from shapely.geometry import Polygon,Point
import scipy.interpolate as spi
import numpy.linalg as npl
import scipy.optimize as spo
from shapely.ops import unary_union
import sympy as sym
from shapely.affinity import translate, rotate




class Element:
    #Class to represent the lattice element such as a drift/lens/bender/combiner.
    #each element type has its own reference frame, which I will described below.
    #Lens and Drift: Input is centered at origin and points to the 'west' with the ouput pointing towards the 'east'
    #Bender: output is facing south and aligned with y=0. The center of the bender is at the origin. Input is at some
        #positive angle relative to the output. A pi/2 bend would have the input aligned with x=0 for example
    #combiner: the output is at the origin, and the input is towards the east, but pointing a bit up at north. Note that
        #the input/beginning is considered to be at the origin. This doesn't really make sense and sould be changed
        #TODO: SWITCH THIS
    def __init__(self,args,type,PT):
        self.args=args
        self.type=type #type of element. Options are 'BENDER', 'DRIFT', 'LENS', 'COMBINER'
        self.PT = PT  # particle tracer object
        self.Bp=None #field strength at bore of element, T
        self.c1=None #dipole component of combiner, T
        self.c2=None #quadrupole component of combiner, T/m
        self.rp=None #bore of element, m
        self.L=None #length of element, m
        self.rb=None #'bending' radius of magnet. actual bending radius of atoms is slightly different cause they
                #ride on the outside edge, m
        self.r0=None #center of element (for bender this is at bending radius center),vector, m
        self.ro=None #bending radius of orbit, m. Includes trajectory offset
        self.ang=0 #Angle that the particles are bent, either bender or combiner. this is the change in  angle in
            # polar coordinates
        self.r1=None #position vector of beginning of element in lab frame, m
        self.r2=None #position vector of ending of element in lab frame, m
        self.r1El=None #position vector of beginning of element in element frame, m
        self.r2El=None #position vector of ending of element in element frame, m


        self.ne=None #normal vector to end of element
        self.nb=None #normal vector to beginning of element
        self.theta=None #angle from horizontal of element. zero degrees is to the right in polar coordinates
        self.ap=None #size of apeture. For now the same in both dimensions and vacuum tubes are square
        self.SO=None #shapely object used to find if particle is inside
        self.index=None #the index of the element in the lattice
        self.K=None #spring constant for magnets
        self.rOffset=0 #the offset of the particle trajectory in a bending magnet
        self.ROut=None #rotation matrix so values don't need to be calculated over and over. This is the rotation
            #matrix OUT of the element frame
        self.RIn=None #rotation matrix so values don't need to be calculated over and over. This is the rotation
            #matrix IN to the element frame
        self.inputOffset=None #for the combiner. Incoming particles enter the combiner with an offset relative to its
            #geometric center. A positive value is more corresponds to moved up the y axis in the combiner's regerence
            #frame.
        self.LFunc=None #for the combiner. The length along the trajector that the particle has traveled. This length
            #is referring to the nominal trajectory, not the actual distance of the particle
        self.distFunc=None #The transerse distance from the nominal trajectory of the particle.
        self.cFact=None #factor in the function y=c*x**2. This is used for finding the trajectory of the particle
            #in the combiner.
        self.trajLength=None #total length of trajectory, m. This is for combiner because it is not trivial like
            #benders or lens or drifts
        
        self.unpack_Args_And_Fill_Params()

    def unpack_Args_And_Fill_Params(self):
        if self.type=='LENS':
            self.Bp=self.args[0]
            self.rp=self.args[1]
            self.L=self.args[2]
            self.ap=self.args[3]
            self.K =(2*self.Bp*self.PT.u0_Actual/self.rp**2)/self.PT.m_Actual  # reduced force
        elif self.type=='DRIFT':
            self.L=self.args[0]
            self.ap=self.args[1]
        elif self.type=='BENDER':
            self.Bp=self.args[0]
            self.rb=self.args[1]
            self.rp=self.args[2]
            self.ang=self.args[3]
            self.ap=self.args[4]
            self.K=(2*self.Bp*self.PT.u0_Actual/self.rp**2)/self.PT.m_Actual #reduced force
            self.rOffset = np.sqrt(self.rb**2/4+self.PT.m*self.PT.v0Nominal**2/self.K)-self.rb/2 #this method does not
                #account for reduced speed in the bender from energy conservation
            #self.rOffset=np.sqrt(self.rb**2/16+self.PT.m*self.PT.v0Nominal**2/(2*self.K))-self.rb/4 #this acounts for reduced
                #energy
            self.ro=self.rb+self.rOffset
            self.L = self.ang * self.ro
        elif self.type=='COMBINER':
            self.L=self.args[0]
            self.ap=self.args[1]
            self.c1=self.args[2]
            self.c2=self.args[3]
            if self.inputOffset is not None:
                #if self.inputOffset is none, then this feature is not being used. This will happen so that the combiner element
                #can be used to predict what the trajectory looks like inside before final use.
                #solve for LFunc and distFunc. These equation are very big so I use sympy to handle them
                x1,x,y1,c=sym.symbols('x1 x y1 c',real=True,positive=True,nonzero=True)
                dist=sym.sqrt((x1-x)**2+(y1-self.cFact*x**2)**2)
                func=x1-x+2*self.cFact*x*(y1-self.cFact*x**2)
                x0=sym.simplify(sym.solve(func,x)[0]) #REMEMBER, THE CORRECT ANSWER CAN CHANGE POSITION WITH DIFFERENT
                    #INPUT
                #NEED TO NAME FUNCTIONS DIFFERENTLY EACH TIME. LAMBDA EVALUATES A FUNCTION IN IT'S LAST STATE, SO MULTIPLE
                #TEMPORARY FUNCTIONS WITH THE SAME NAME INTERFERE WITH EACH OTHER
                tempFunc1=sym.lambdify([x1,y1],dist.subs(x,x0))
                self.distFunc=lambda x1,y1: np.real(tempFunc1(x1+0J,y1)) #need input to be complex to avoid error on
                    #roots of negative numbers. There is a tiny imaginary component from numerical imprecision, so I take
                    #only the real
                tempFunc2=sym.lambdify([x1,y1],sym.integrate(sym.sqrt(1+(2*self.cFact*x)**2),(x,0,x0)))
                self.LFunc=lambda x1,y1: np.real(tempFunc2(x1+0J,y1))

                self.trajLength=sym.integrate(sym.sqrt(1+(2*self.cFact*x)**2),(x,0,self.L)).subs(x,self.L)

        else:
            raise Exception('No proper element name provided')
    def transform_Lab_Coords_Into_Orbit_Frame(self,q,cumulativeLength):
        q = self.transform_Lab_Coords_Into_Element_Frame(q)
        qo=self.transform_Element_Coords_Into_Orbit_Frame(q)
        qo[0]=qo[0]+cumulativeLength
        return qo
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        # q: particle coords in x and y plane. numpy array
        # self: element object to transform to
        # get the coordinates of the particle in the element's frame. See element class for description
        qNew = q.copy()  # CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        if self.type == 'DRIFT' or self.type == 'LENS':
            qNew[0] = qNew[0] - self.r1[0]
            qNew[1] = qNew[1] - self.r1[1]
        elif self.type == 'BENDER':
            qNew[:2] = qNew[:2] - self.r0
        elif self.type == 'COMBINER':
            qNew[:2] = qNew[:2] - self.r2
        qNewx = qNew[0]
        qNewy = qNew[1]
        qNew[0] = qNewx * self.RIn[0, 0] + qNewy * self.RIn[0, 1]
        qNew[1] = qNewx * self.RIn[1, 0] + qNewy * self.RIn[1, 1]
        return qNew
    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        #This returns the nominal orbit in the element's reference frame.
        #q: particles position in the element's reference frame
        #The description for each element is given below.
        qo=q.copy()
        if self.type == 'LENS' or self.type == 'DRIFT':
           pass
        elif self.type == 'BENDER':
            qo = q.copy()
            phi = self.ang-np.arctan2(q[1]+1e-10,q[0])  # angle swept out by particle in trajectory. This is zero
            # when the particle first enters
            ds=self.ro*phi
            qos=ds
            qox=np.sqrt(q[0]**2+q[1]**2)-self.ro
            qo[0]=qos
            qo[1]=qox
        elif self.type=='COMBINER':
            if qo[0]>self.L:
                dr=self.r2El-qo[:2]
                rot=np.asarray([[np.cos(-self.ang),-np.sin(-self.ang)],[np.sin(-self.ang),np.cos(-self.ang)]])
                qo[:2]=rot@dr
            else:
                qo[0]=self.trajLength-self.LFunc(q[0],q[1])+np.sin(self.ang)*(self.ap-self.inputOffset)
                qo[1]=self.distFunc(q[0],q[1]) #TODO: FUCKING FIX THIS....
        return qo
    def force(self,q):
        #force at point q in element frame
        F = np.zeros(3)  # force vector starts out as zero
        if self.type == 'DRIFT': #no force from drift region
            pass
        elif self.type == 'BENDER':
            r = np.sqrt(q[0] ** 2 + q[1] ** 2)  # radius in x y frame
            F0 = -self.K * (r - self.rb)  # force in x y plane
            phi = np.arctan2(q[1], q[0])
            F[0] = np.cos(phi) * F0
            F[1] = np.sin(phi) * F0
            F[2] = -self.K * q[2]
        elif self.type == 'LENS':
            # note: for the perfect lens, in it's frame, there is never force in the x direction
            F[1] = -self.K * q[1]
            F[2] = -self.K * q[2]
        elif self.type == 'COMBINER':
            if q[0] < self.L:
                B0 = np.sqrt((self.c2 * q[2]) ** 2 + (self.c1 + self.c2 * q[1]) ** 2)
                F[1] = self.PT.u0 * self.c2 * (self.c1 + self.c2 * q[1]) / B0
                F[2] = self.PT.u0 * self.c2 ** 2 * q[2] / B0
        return F
    def transform_Vector_Out_Of_Element_Frame(self, vec):
        #rotation matrix is 3x3 to account for z axis
        vecx=vec[0]
        vecy=vec[1]
        vec[0]=vecx*self.ROut[0,0]+vecy*self.ROut[0,1]
        vec[1]=vecx*self.ROut[1,0]+vecy*self.ROut[1,1]
        return vec

    def transform_Lab_Momentum_Into_Orbit_Frame(self, q, p):
        #TODO: CONSOLIDATE THIS WITH GET_POSITION
        pNew = p.copy()
        pNew[0] = p[0] * self.RIn[0, 0] + p[1] * self.RIn[0, 1]
        pNew[1] = p[0] * self.RIn[1, 0] + p[1] * self.RIn[1, 1]
        if self.type=='BENDER': #need to use a change of vectors from cartesian to polar for bender
            q=self.transform_Lab_Coords_Into_Element_Frame(q)
            pNew=p.copy()
            sDot=(q[0]*pNew[1]-q[1]*pNew[0])/np.sqrt((q[0]**2+q[1]**2))
            rDot=(q[0]*pNew[0]+q[1]*pNew[1])/np.sqrt((q[0]**2+q[1]**2))
            po=np.asarray([sDot,rDot,pNew[2]])
            return po
        elif self.type=='LENS' or self.type == 'DRIFT':
            return pNew
        if self.type=='COMBINER':
            raise Exception('NOT YET IMPLEMENTED')



class particleTracer:
    def __init__(self,v0Nominal):
        self.v0Nominal = v0Nominal  # Design particle speed
        self.m_Actual = 1.1648E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009994E-24 # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.m=1 #adjusted value of mass. 1 is equal to li7 mass
        self.u0=self.u0_Actual/self.m_Actual #Adjusted value of permeability of free space. About equal to 800

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
        self.elHasChanged=False # to record if the particle has changed to another element


        self.v0=None #the particles total speed.
        self.vs=None #particle speed along orbit
        self.E0=None #total initial energy of particle

        self.VList=[] #list of potential energy
        self.TList=[] #list of kinetic energy
        self.EList=[] #too keep track of total energy. This can tell me if the simulation is behaving
            #This won't always be on
    def add_Lens(self,L,Bp,rp,ap=None):
        #add a lens to the lattice
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
        el=Element(args,'LENS',self) #create a lens element object
        el.index = len(self.lattice) #where the element is in the lattice
        self.lattice.append(el) #add element to the list holding lattice elements in order

    def add_Drift(self,L,ap=.03):
        #add drift region to the lattice
        #L: length of drift element, m
        #ap:apeture. Default value of 3 cm radius, unitless
        args=[L,ap]
        el=Element(args,'DRIFT',self)#create a drift element object
        el.index = len(self.lattice) #where the element is in the lattice
        self.lattice.append(el) #add element to the list holding lattice elements in order
    def add_Bender(self,ang,rb,Bp,rp,ap=None):
        #add bender to lattice
        #ang: Bending angle of bender, radians
        #rb: nominal bending radius of bender. Actual radius is larger because particle 'rides' a little outside this, m
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
        el=Element(args,'BENDER',self) #create a bender element object
        el.index = len(self.lattice) #where the element is in the lattice
        self.benderIndices.append(el.index)
        self.lattice.append(el) #add element to the list holding lattice elements in order
    def add_Combiner(self,L=.2,c1=1,c2=20,ap=.015):
        #THIS IS CURRENTLY INCOMPLETE. WHAT NEEDS TO BE FINISHED IS A GOOD WAY OF FITTING THE COMBINER INTO THE LATTICE,
        #BECAUSE LATTICE NEEDS TO CHANGE FOR IT. ALSO, REFERENCING THE PARTICLE'S NOMINAL TRAJECTORY
        #TODO: UPDATE COMBINER LENGTH AND ANGLE
        #add combiner (stern gerlacht) element to lattice
        #L: length of combiner
        #c1: dipole component of combiner
        #c2: quadrupole component of bender
        raise Exception('currently broken!!')
        args=[L,ap,c1,c2]
        el=Element(args,'COMBINER',self) #create a combiner element object
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
    def initialize(self):
        #prepare for a particle to be traced
        self.vs = np.abs(self.p[0] / self.m)
        dl=self.vs*self.h #approximate stepsize
        for el in self.lattice:
            if dl*10>el.L:
                raise Exception('STEP SIZE TOO SMALL')

        self.reset()
        self.currentEl = self.which_Element(self.q)
        if self.currentEl is None:
            raise Exception('Particle\'s initial position is outside vacuum' )

        self.qList.append(self.q)
        self.pList.append(self.p)
        self.qoList.append(self.currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q,self.cumulativeLength))
        self.poList.append(self.currentEl.transform_Lab_Momentum_Into_Orbit_Frame(self.q,self.p))

        self.E0=sum(self.get_Energies())
        self.EList.append(self.E0)

    def trace(self,qi,pi,h,T0,method='verlet'):
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
            self.adapt_Time_Step(self.currentEl)
            self.time_Step(method=method)
            if self.particleOutside==True:
                break
            self.qList.append(self.q)
            self.pList.append(self.p)
            self.qoList.append(self.currentEl.transform_Lab_Coords_Into_Orbit_Frame(self.q,self.cumulativeLength))
            self.poList.append(self.currentEl.transform_Lab_Momentum_Into_Orbit_Frame(self.q, self.p))
            #temp=self.get_Energies() #sum the potential and kinetic energy
            #self.VList.append(temp[0])
            #self.TList.append(temp[1])
            #self.EList.append(sum(temp))
            self.T+=self.h
        qArr=np.asarray(self.qList)
        pArr=np.asarray(self.pList)
        qoArr=np.asarray(self.qoList)
        poArr=np.asarray(self.poList)


        return qArr,pArr,qoArr,poArr,self.particleOutside
    def time_Step_Trapezoid(self):
        #This time stepping solves an implicit equation. In order to find the solution, a function's root
        #needs to be found.
        el, coordsxy = self.which_Element_Wrapper(self.q)
        if el is None:
            self.particleOutside=True
            return

        F1 = self.force(self.q)
        p0=self.p+self.h*F1/self.m
        tol=1e-8
        iMax=10
        for i in range(iMax):
            q2=self.q+.5*self.h*(self.p+p0)
            el,coordsxy=self.which_Element_Wrapper(q2)

            exit = self.check_Element_And_Handle_Edge_Event(el)
            if exit == True:
                return
            F2=self.force(q2,el=el,coordsxy=coordsxy)

            psi = self.p + .5 * self.h * ( F1+F2 ) - p0

            phi = self.get_Jacobian_Inv(q2, el=el)
            p0 = p0 - (phi @ psi)
            if np.all(np.abs(psi)<tol):
                self.q = q2
                self.p = p0
                break

    def handle_Element_Edge(self, el, q, p):
        #when the particle has stepped over into the next element in time_Step_Trapezoid this method is called.
        #This method calculates the correct timestep to put the particle just on the other side of the end of the element
        #using velocity verlet
        #TODO: SHOULD WE BE USING EL.NB
        r=el.r2-q[:2]
        rt=npl.norm(el.nb*r[:2]) #perindicular position vector to element's end
        pt=npl.norm(el.nb*p[:2]) #perpindicular momentum vector to surface of element's end
        F1=self.force(q,el=el)
        Ft=npl.norm(el.nb*F1[:2])

        #precision loss is killing me here! Usually Force in the direction of the input to the next element is basically
        #zero, thus if that is the case, I need to substitute a really small (but not hyper-super small) number for force
        #to get the correct answer
        if 2*Ft*self.m*rt <1e-6*pt**2:
            h=rt/(pt/self.m) #assuming no force
        else:
            import numpy as np
            h = (np.sqrt(2 * Ft * self.m * rt + pt ** 2) - pt) / Ft #assuming force

        q=q+h*p/self.m+.5*(F1/self.m)*h**2
        F2=self.force(q,el=el)
        p=p+.5*(F1+F2)*h
        eps=1e-12 #tiny step to put the particle on the other side
        np=p/npl.norm(p) #normalized vector of particle direction
        q=q+np*eps
        return q,p

    def time_Step_Verlet(self):
        q=self.q #q old or q sub n
        p=self.p #p old or p sub n
        F=self.force(q)

        a = F / self.m  # acceleration old or acceleration sub n
        q_n=q+(p/self.m)*self.h+.5*a*self.h**2 #q new or q sub n+1
        el, coordsxy = self.which_Element_Wrapper(q_n)
        exit=self.check_Element_And_Handle_Edge_Event(el)
        if exit==True:
            return

        F_n=self.force(q_n,el=el,coordsxy=coordsxy)
        a_n = F_n / self.m  # acceleration new or acceleration sub n+1
        p_n=p+self.m*.5*(a+a_n)*self.h

        self.q=q_n
        self.p=p_n

    def time_Step(self,method='verlet'):
        if method=='verlet':
            self.time_Step_Verlet()
        elif method=='trapezoid':
            self.time_Step_Trapezoid()
        else:
            raise Exception('No proper method propvided')


    def check_Element_And_Handle_Edge_Event(self,el):
        #this method checks if the element that the particle is in, or being evaluated, has changed. If it has
        #changed then that needs to be recorded and the particle carefully walked up to the edge of the element
        #This returns True if the particle has been walked to the edge or is outside the element becuase the element
        #is none type, which occurs if the particle is found to be outside the lattice when evaluating force
        exit=False
        if el is None:
            self.particleOutside = True
            exit=True
        if el is not self.currentEl:

            self.q, self.p = self.handle_Element_Edge(self.currentEl, self.q, self.p)
            self.cumulativeLength += self.currentEl.L
            self.currentEl = el
            exit=True
            #self.adjust_Energy()
        return exit



    def get_Jacobian_Inv(self,q,el=None):
        if el is None:
            el=self.which_Element_Wrapper(q) #find element the particle is in, and the coords in
            #the element frame as well
        if el.type=='LENS':
            temp=-1/(1+self.h**2*el.K/4)
            Jinv=np.asarray([[-1,0,0],[1,temp,0],[0,0,temp]])
            return Jinv
        if el.type=='DRIFT':
            return np.eye(3)
        if el.type=='BENDER':
            coordsxy = self.transform_Coords_To_Element_Frame(q, el)
            theta=np.arctan2(coordsxy[1],coordsxy[0])
            k=el.K
            h=self.h
            #J00=-h*k*np.cos(theta)**2/2
            #J01=h*k*np.sin(2*theta)/4
            #J10=J01
            #J11=-h*k*np.sin(theta)**2/2
            J00=-(h**2*k*np.cos(theta)**2/8+1)
            J11=-(h**2*k*np.sin(theta)**2/8+1)
            J01=h**2*k*np.sin(2*theta)/8
            J10=J01
            J=np.asarray([[J00,J01,0],[J10,J11,0],[0,0,-h**2*k/4-1]])
            Jinv=npl.inv(J)
            return Jinv

    def get_Energies(self):
        PE =0
        KE =0
        if self.currentEl.type == 'LENS':
            qxy = self.currentEl.transform_Lab_Coords_Into_Element_Frame(self.q)[:2]
            r = np.sqrt(qxy[1] ** 2 + self.q[2] ** 2)
            B = self.currentEl.Bp * r ** 2 / self.currentEl.rp ** 2
            PE = self.u0 * B
            KE = np.sum(self.p ** 2) / (2 * self.m)
        elif self.currentEl.type == 'DRIFT':
            PE = 0
            KE = np.sum(self.p ** 2) / (2 * self.m)
        elif self.currentEl.type == 'BENDER':
            qxy = self.currentEl.transform_Lab_Coords_Into_Element_Frame(self.q[:-1])  # only x and y coords
            r = np.sqrt(qxy[0] ** 2 + qxy[1] ** 2) - self.currentEl.rb
            B = self.currentEl.Bp * r ** 2 / self.currentEl.rp ** 2
            PE = self.u0 * B
            KE = np.sum(self.p ** 2) / (2 * self.m)
        return PE,KE

    def force(self,q,coordsxy=None,el=None):
        #todo: remove redundant q and coordsxy
        if el is None:
            el=self.which_Element(q) #find element the particle is in
        if coordsxy is None:
            coordsxy = el.transform_Lab_Coords_Into_Element_Frame(q)
        F=el.force(coordsxy)
        return el.transform_Vector_Out_Of_Element_Frame(F)

    def distance_From_End(self,coordsxy,el):
        #determine the distance along the orbit that the particle is from the end of the element
        if el.type=='BENDER':
            s=el.rb*np.arctan2(coordsxy[1],coordsxy[0])
            return s
        elif el.type=='LENS' or el.type=='DRIFT':
            return el.L-coordsxy[0]
        pass
    def which_Element_Wrapper(self,q):
        #TODO: USING SHAPELY IS PRETTY SLOW. ITS WOULD BE MUCH FASTER TO USE THE COORDS IN THE ELEMENT FRAME AND
        #CHECK EACH ELEMENT THAT WAY. DOUBLE CHECK THOUGH
        el = self.which_Element(q)
        if el is None:
            return None, None
        else:
            coordsxy = self.currentEl.transform_Lab_Coords_Into_Element_Frame(q)
            if np.abs(q[-1])>el.ap:
                return None,None
            else:
                return el, coordsxy
    def adapt_Time_Step(self,el):
        if el.type=='DRIFT':
            self.h=10e-3/self.v0 #step size of 1cm

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
        el=self.which_Element_Shapely(q)
        if el is not None: #if particle is definitely inside an element.
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

        return el
    def adjust_Energy(self):
        #when the particle traverses an element boundary, energy needs to be conserved. This would be irrelevant if I
        #included fringe field effects
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
    def solve_Triangle_Problem(self,phi1,L1,L2,r1,r2):
        #the triangle problem refers to two circles and a kinked section connected by their tangets. This is the situation
        #with the combiner and the two benders.
        #phi1: bending angle of the combiner, or the amount of 'kink'
        #L1: length of the section after the kink to the bender
        #L2: length of the section before the kink to the bender
        #r1: radius of circle after the combiner
        #r2: radius of circle before the combiner
        #note that L1 and L2 INCLUDE the sections in the combiner. This function will likely be used by another function
        #that sorts that all out.

        L3=sym.sqrt((L1-sym.sin(phi1)*r2+L2*sym.cos(phi1))**2+(L2*sym.sin(phi1)-r2*(1-sym.cos(phi1))+(r2-r1))**2)
        phi2=sym.pi*1.5-sym.atan(L1/r1)-sym.acos((L3**2+L1**2-L2**2)/(2*L3*sym.sqrt(L1**2+r1**2)))
        phi3=sym.pi*1.5-sym.atan(L2/r2)-sym.acos((L3**2+L2**2-L1**2)/(2*L3*sym.sqrt(L2**2+r2**2)))
        return phi2,phi3,L3

    def make_Geometry(self):
        #construct the shapely objects used to plot the lattice and to determine if particles are inside of the lattice.
        #it could be changed to only use the shapely objects for plotting, but it would take some clever algorithm I think
        #and I am in a crunch kind of.
        #----
        #all of these take some thinking to visualize what's happening.
        benderPoints=50 #how many points to represent the bender with along each curve
        for el in self.lattice:
            L=el.L
            xb=el.r1[0]
            yb=el.r1[1]
            xe=el.r2[0]
            ye=el.r2[1]
            ap=el.ap
            theta=el.theta
            if el.type=='DRIFT' or el.type=='LENS':
                q1=np.asarray([xb-np.sin(theta)*ap,yb+ap*np.cos(theta)]) #top left when theta=0
                q2=np.asarray([xe-np.sin(theta)*ap,ye+ap*np.cos(theta)]) #top right when theta=0
                q3=np.asarray([xe+np.sin(theta)*ap,ye-ap*np.cos(theta)]) #bottom right when theta=0
                q4=np.asarray([xb+np.sin(theta)*ap,yb-ap*np.cos(theta)]) #bottom left when theta=0
                el.SO=Polygon([q1,q2,q3,q4])
            elif el.type=='BENDER':
                phiArr=np.linspace(0,-el.ang,num=benderPoints)+theta+np.pi/2 #angles swept out
                xInner=(el.rb-ap)*np.cos(phiArr)+el.r0[0] #x values for inner bend
                yInner=(el.rb-ap)*np.sin(phiArr)+el.r0[1] #y values for inner bend
                xOuter=np.flip((el.rb+ap)*np.cos(phiArr)+el.r0[0]) #x values for outer bend
                yOuter=np.flip((el.rb+ap)*np.sin(phiArr)+el.r0[1]) #y values for outer bend
                x=np.append(xInner,xOuter) #list of x values in order
                y=np.append(yInner,yOuter) #list of y values in order
                el.SO=Polygon(np.column_stack((x,y))) #shape the coordinates and make the object
            elif el.type=='COMBINER':
                q1=np.asarray([0,ap]) #top left when theta=0
                q2=np.asarray([L,ap]) #top right when theta=0
                q3=np.asarray([np.cos(el.ang)*np.sin(el.ang)*2*ap+L, np.sin(el.ang)**2*2*ap-ap])  # bottom right when theta=0
                q4=np.asarray([L,-ap]) #bottom middle when theta=0
                q5=np.asarray([0,-ap]) #bottom left when theta=0
                points=[q1,q2,q3,q4,q5]
                for i in range(len(points)):
                    points[i]=el.ROut@points[i]+el.r2
                el.SO=Polygon(points)
            else:
                raise Exception('No correct element provided')
    def catch_Errors(self):
        #catch any preliminary errors. Alot of error handling happens in other methods. This is a catch all for other
        #kinds
        if self.lattice[0].type=='BENDER': #first element can't be a bending element
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
            else:
                xb=self.lattice[i-1].r2[0]#set beginning coordinates to end of last
                yb=self.lattice[i-1].r2[1]#set beginning coordinates to end of last
                prevEl = self.lattice[i - 1]
                #set end coordinates
                if el.type=='DRIFT' or el.type=='LENS':
                    if prevEl.type=='COMBINER':
                        el.theta = prevEl.theta-np.pi  # set the angle that the element is tilted relative to its
                        # reference frame. This is based on the previous element
                    else:
                        el.theta=prevEl.theta-prevEl.ang
                    xe=xb+el.L*np.cos(el.theta)
                    ye=yb+el.L*np.sin(el.theta)
                    el.nb = -np.asarray([np.cos(el.theta), np.sin(el.theta)])  # normal vector to input
                    el.ne = -el.nb #normal vector to end
                    if prevEl.type=='BENDER':
                        n=np.zeros(2)
                        n[0]=-prevEl.ne[1]
                        n[1]=prevEl.ne[0]
                        dr=n*prevEl.rOffset
                        xb+=dr[0]
                        yb+=dr[1]
                        xe+=dr[0]
                        ye+=dr[1]
                    el.r0=np.asarray([(xb+xe)/2,(yb+ye)/2]) #center of lens or drift is midpoint of line connecting beginning and end
                elif el.type=='BENDER':
                    if prevEl.type=='COMBINER':
                        el.theta = prevEl.theta-np.pi  # set the angle that the element is tilted relative to its
                        # reference frame. This is based on the previous element
                    else:
                        el.theta=prevEl.theta-prevEl.ang
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
                    xb+=el.rOffset*np.sin(el.theta)
                    yb+=el.rOffset*(-np.cos(el.theta))
                    xe+=el.rOffset*np.sin(el.theta)
                    ye+=el.rOffset*(-np.cos(el.theta))
                    el.nb=np.asarray([np.cos(el.theta-np.pi), np.sin(el.theta-np.pi)])  # normal vector to input
                    el.ne=np.asarray([np.cos(el.theta-np.pi+(np.pi-el.ang)), np.sin(el.theta-np.pi+(np.pi-el.ang))])  # normal vector to output
                    el.r0=np.asarray([xb+el.rb*np.sin(el.theta),yb-el.rb*np.cos(el.theta)]) #coordinates of center of bender
                elif el.type=='COMBINER':
                    el.theta=2*np.pi-el.ang-(np.pi-prevEl.theta)# Tilt the combiner down by el.ang so y=0 is perpindicular
                        #to the input. Rotate it 1 whole revolution, then back it off by the difference. Need to subtract
                        #np.pi because the combiner's input is not at the origin, but faces 'east'
                    el.theta=el.theta-2*np.pi*(el.theta//(2*np.pi)) #the above logic can cause the element to have to rotate
                        #more than 360 deg

                    #to find location of output coords use vector that connects input and output in element frame
                    #and rotate it. Input is where nominal trajectory particle enters
                    drx=-(el.L+np.cos(el.ang)*np.sin(el.ang)*(el.ap-el.inputOffset))
                    dry=-(el.inputOffset+np.sin(el.ang)**2*(el.ap-el.inputOffset))

                    el.r1El=np.asarray([0,0])
                    el.r2El=-np.asarray([drx,dry])
                    dr=np.asarray([drx,dry]) #position vector between input and output of element.
                    R = np.asarray([[np.cos(el.theta), -np.sin(el.theta)], [np.sin(el.theta), np.cos(el.theta)]])
                    dr=R@dr
                    xe,ye=xb+dr[0],yb+dr[1]
                    el.ne=-np.asarray([np.cos(el.theta),np.sin(el.theta)])
                    el.nb=np.asarray([np.cos(el.theta+el.ang),np.sin(el.theta+el.ang)])
                    #print(el.ne,el.nb)
                else:
                    raise Exception('No correct element name provided')
            #need to make rotation matrices for bender, lens and drift now. Already made for combiner
            if el.type=='BENDER':
                rot = (el.theta - el.ang + np.pi / 2)
            elif el.type=='LENS' or el.type=='DRIFT' or el.type=='COMBINER':
                rot = el.theta
            else:
                raise Exception('No correct element name provided')
            el.ROut = np.asarray([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]) #the rotation matrix for
                #rotating out of the element reference frame
            el.RIn = np.asarray([[np.cos(-rot), -np.sin(-rot)], [np.sin(-rot), np.cos(-rot)]]) #the rotation matrix for
                #rotating into the element reference frame
            #clean up tiny numbers. There are some numbers like 1e-16 that should be zero
            el.ne=np.round(el.ne,10)
            el.nb=np.round(el.nb,10)


            el.r1=np.round(np.asarray([xb,yb]),12) #position vector of beginning of element
            el.r2=np.round(np.asarray([xe,ye]),12) #position vector of ending of element
            if i==len(self.lattice)-1: #if the last element then set the end of the element correctly
                reo = el.r2.copy()
                if el.type=='BENDER' or el.type=='COMBINER':
                    reo[0]+=el.rOffset * np.sin(el.theta)
                    reo[1]+=el.rOffset*(-np.cos(el.theta))
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
        for el in self.lattice:
            plt.plot(*el.SO.exterior.xy)
        if particleCoords is not None:
            if particleCoords.shape[0]==3: #if the 3d value is provided trim it to 2D
                particleCoords=particleCoords[:2]
            #plot the particle as both a dot and a X
            plt.scatter(*particleCoords,marker='x',s=1000,c='r')
            plt.scatter(*particleCoords, marker='o', s=50, c='r')
        plt.grid()
        #plt.gca().set_aspect('equal')
        plt.show()



# test=particleTracer(200)
# test.add_Lens(1,1,.02)
# test.add_Bender(np.pi,1,1,.01)
# test.add_Lens(1,1,.02)
# test.add_Bender(np.pi,1,1,.01)
# test.end_Lattice()
# q0=np.asarray([-1e-10,1e-3,0.0])
# v0=np.asarray([-200.0,0,0])
# q, p, qo, po, particleOutside = test.trace(q0, v0, 1e-6, 1/200,method='verlet')
# print(particleOutside)
# plt.plot(qo[:,0],qo[:,1])
# plt.show()
# test.show_Lattice(particleCoords=q[-1])