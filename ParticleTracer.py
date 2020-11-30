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

#TODO: MAKE IMPLICIT
#TODO: CLEAN UP REDUNDANT FUNCTION CALLS
#TODO: DEAL WITH BOUNDARIES BETTER


class Element:
    #Class to represent the lattice element such as a drift/lens/bender/combiner.
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
        self.ro=None #bending radius of orbit, m
        self.ang=None #bending angle of bender or combiner, radians
        self.r1=None #position vector of beginning of element, m
        self.r2=None #position vector of ending of element, m
        self.ne=None #normal vector to end of element
        self.nb=None #normal vector to beginning of element
        self.theta=None #angle from horizontal of element. zero degrees is to the right in polar coordinates
        self.ap=None #size of apeture. For now the same in both dimensions and vacuum tubes are square
        self.SO=None #shapely object used to find if particle is inside
        self.index=None #the index of the element in the lattice
        self.K=None #spring constant for magnets
        self.rOffset=None #the offset of the particle trajectory in a bending magnet
        self.ROut=None #rotation matrix so values don't need to be calculated over and over. This is the rotation
            #matrix OUT of the element frame
        self.RIn=None #rotation matrix so values don't need to be calculated over and over. This is the rotation
            #matrix IN to the element frame
        self.inputAng=None #the combiner's input is tilted, this that value. 0 is the input pointing to the east,
            #pi is pointing to the west, pi/2 is pointing north


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
            self.ang=self.args[4]
            self.rOffset=0 #yes, a combiner doesn't have an offset in it's trajectory, but this is so I can piggyback off
                #all the bender logic
            self.inputAng=.01

        else:
            raise Exception('No proper element name provided')
class particleTracer:
    def __init__(self):
        self.m_Actual = 1.1648E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009994E-24 # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.m=1 #adjusted value of mass. 1 is equal to li7 mass
        self.u0=self.u0_Actual/self.m_Actual #Adjusted value of permeability of free space. About equal to 800

        self.kb = .38064852E-23  # boltzman constant, SI
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
        self.timeAdapted=False #wether the time step has been shrunk because nearness to a new element

        self.v0=None #the particles total speed. TODO: MAKE CHANGE WITH HARD EDGE MODEL
        self.vs=None #particle speed along orbit
        self.v0Nominal=200 #Design particle speed
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
        #TODO: UPDATE COMBINER LENGTH AND ANGLE
        #add combiner (stern gerlacht) element to lattice
        #L: length of combiner
        #c1: dipole component of combiner
        #c2: quadrupole component of bender
        args=[L,ap,c1,c2]
        args.append(ang)
        el=Element(args,'COMBINER',self) #create a combiner element object
        el.index = len(self.lattice) #where the element is in the lattice
        self.combinerIndex=el.index
        self.lattice.append(el) #add element to the list holding lattice elements in order


    def compute_Angle_For_Combiner(self,args):
        #angle needs to be calculated to respect that the inner edge needs to be L for the combiner
        L=args[0]
        c2=args[2]
        r=50.0/c2
        return L/r
    def reset(self):
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
    def initialize_For_Particle(self):
        #prepare for a particle to be traced
        self.vs = np.abs(self.p[0] / self.m)
        dl=self.vs*self.h #approximate stepsize
        for el in self.lattice:
            if dl*10>el.L:
                raise Exception('STEP SIZE TOO SMALL')

        self.reset()
        self.currentEl = self.which_Element(self.q)

        self.qList.append(self.q)
        self.pList.append(self.p)
        self.qoList.append(self.get_Coords_In_Orbit_Frame(self.q,self.currentEl))
        self.poList.append(self.get_Momentum_In_Orbit_Frame(self.q,self.p))


        self.E0=sum(self.get_Energies())
        self.EList.append(self.E0)

    def trace(self,qi,vi,h,T0,method='verlet'):

        self.q=qi.copy().astype(float)
        self.p=vi.copy().astype(float)*self.m
        self.h=h
        self.h0=h
        self.v0 = npl.norm(vi)
        self.initialize_For_Particle()

        if self.currentEl is None:
            raise Exception('Particle\'s initial position is outside vacuum' )



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
            self.qoList.append(self.get_Coords_In_Orbit_Frame(self.q,self.currentEl))
            self.poList.append(self.get_Momentum_In_Orbit_Frame(self.q,self.p))
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
        r=el.r2-q[:2]
        rt=npl.norm(el.nb*r[:2])
        pt=npl.norm(el.nb*p[:2]) #tangential momentum vector to surface of element's end
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
        eps=1e-9 #tiny step to put the particle on the other side
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
        elif method=='RK4':
            self.time_Step_RK4()
        else:
            raise Exception('No proper method propvided')
    def time_Step_RK4(self):
        #this method uses the runge-kutta 4th order method, or RK4
        h=self.h
        q=self.q

        #k1
        q1=q
        el1, coordsxy = self.which_Element_Wrapper(q1)
        exit=self.check_Element_And_Handle_Edge_Event(el1)
        if exit==True:
            return
        k1v=h*self.force(q1,el=el1,coordsxy=coordsxy)
        k1x=h*(self.p/self.m)


        #k2
        q2=q+k1x/2
        el2, coordsxy = self.which_Element_Wrapper(q2)
        exit=self.check_Element_And_Handle_Edge_Event(el2)
        if exit==True:
            return
        k2v=h*self.force(q2, el=el2, coordsxy=coordsxy)
        k2x=h*(self.p/self.m + k1v/2)


        #k3
        q3=q+k2x/2
        el3, coordsxy = self.which_Element_Wrapper(q3)
        exit=self.check_Element_And_Handle_Edge_Event(el3)
        if exit==True:
            return
        k3v=h*self.force(q3, el=el3, coordsxy=coordsxy)
        k3x=h*(self.p/self.m + k2v/2)

        #k4
        q4=q+h*k3x
        el4,coordsxy = self.which_Element_Wrapper(q2)
        exit=self.check_Element_And_Handle_Edge_Event(el4)
        if exit==True:
            return
        k4v=h*self.force(q4, el=el4, coordsxy=coordsxy)
        k4x=h*(self.p/self.m+k3v)


        self.q=self.q+(k1x+2*k2x+2*k3x+k4x)/6
        self.p=self.p+self.m*(k1v+2*k2v+2*k3v+k4v)/6

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

    def is_Particle_Inside(self,q):
        #this could be done with which_Element, but this is faster
        if self.currentEl is None:
            return False
        if np.abs(q[2])>self.currentEl.ap: #check z axis
            self.particleOutside=True
            return False
        coordsxy = self.transform_Coords_To_Element_Frame(q[:-1], self.currentEl)
        if self.currentEl.type=='LENS' or self.currentEl.type=='DRIFT':
            if np.abs(coordsxy[1])>self.currentEl.ap or coordsxy[0]>self.currentEl.L:
                self.particleOutside=True
                return False

        elif self.currentEl.type=='BENDER':
            r=np.sqrt(np.sum(coordsxy**2))
            deltar=r-self.currentEl.rb
            if np.abs(deltar)>self.currentEl.ap:
                self.particleOutside=True
                return False
        return True

    def get_Coords_In_Orbit_Frame(self,q,el):
        #need to rotate coordinate system to align with the element
        coordsxy = self.transform_Coords_To_Element_Frame(q[:-1], el)
        if el.type=='LENS' or el.type=='DRIFT':
            qos=self.cumulativeLength+coordsxy[0]
            qox=coordsxy[1]
        elif el.type=='BENDER':
            phi=el.ang-np.arctan2(coordsxy[1]+1e-10,coordsxy[0]) #angle swept out by particle in trajectory. This is zero
                #when the particle first enters
            ds=el.ro*phi
            qos=self.cumulativeLength+ds
            qox=np.sqrt(coordsxy[0]**2+coordsxy[1]**2)-el.ro
        elif self.type=='COMBINER':
            pass

        else:
            raise Exception('No correct element name provided')
        qoy = q[2]
        qo=np.asarray([qos, qox, qoy])
        return qo

    def get_Momentum_In_Orbit_Frame(self,q,p):
        #TODO: CONSOLIDATE THIS WITH GET_POSITION
        el=self.currentEl
        coordsxy = self.transform_Coords_To_Element_Frame(q[:-1], self.currentEl)
        pNew = p.copy()
        pNew[0] = p[0] * el.RIn[0, 0] + p[1] * el.RIn[0, 1]
        pNew[1] = p[0] * el.RIn[1, 0] + p[1] * el.RIn[1, 1]
        if el.type=='BENDER':
            pNew=p.copy()
            sDot=(coordsxy[0]*pNew[1]-coordsxy[1]*pNew[0])/np.sqrt((coordsxy[0]**2+coordsxy[1]**2))
            rDot=(coordsxy[0]*pNew[0]+coordsxy[1]*pNew[1])/np.sqrt((coordsxy[0]**2+coordsxy[1]**2))
            po=np.asarray([sDot,rDot,pNew[2]])
            return po
        elif el.type=='LENS' or el.type == 'DRIFT':
            pNew = p.copy()
            return pNew
    def loop_Check(self):
        z=self.q[2]
        if z>2:
            return False
        else:
            return True
    def get_Energies(self):
        PE =0
        KE =0
        if self.currentEl.type == 'LENS':
            qxy = self.transform_Coords_To_Element_Frame(self.q[:-1], self.currentEl)
            r = np.sqrt(qxy[1] ** 2 + self.q[2] ** 2)
            B = self.currentEl.Bp * r ** 2 / self.currentEl.rp ** 2
            PE = self.u0 * B
            KE = np.sum(self.p ** 2) / (2 * self.m)
        elif self.currentEl.type == 'DRIFT':
            PE = 0
            KE = np.sum(self.p ** 2) / (2 * self.m)
        elif self.currentEl.type == 'BENDER':
            qxy = self.transform_Coords_To_Element_Frame(self.q[:-1], self.currentEl)  # only x and y coords
            r = np.sqrt(qxy[0] ** 2 + qxy[1] ** 2) - self.currentEl.rb
            B = self.currentEl.Bp * r ** 2 / self.currentEl.rp ** 2
            PE = self.u0 * B
            KE = np.sum(self.p ** 2) / (2 * self.m)
        return PE,KE
    def force(self,q,coordsxy=None,el=None):
        if el is None:
            el=self.which_Element(q) #find element the particle is in
        if coordsxy is None:
            coordsxy = self.transform_Coords_To_Element_Frame(q,el)
        F = np.zeros(3) #force vector starts out as zero
        if el.type == 'DRIFT':
            pass
        elif el.type == 'BENDER':
            r=np.sqrt(coordsxy[0]**2+coordsxy[1]**2) #radius in x y frame
            F0=-el.K*(r-el.rb) #force in x y plane
            phi=np.arctan2(coordsxy[1],coordsxy[0])
            F[0]=np.cos(phi)*F0
            F[1]=np.sin(phi)*F0
            F[2]=-el.K*q[2]
            F=self.transform_Force_Out_Of_Element_Frame(F,el)
        elif el.type=='LENS':
            #note: for the perfect lens, in it's frame, there is never force in the x direction
            F[1] =-el.K*coordsxy[1]
            F[2] =-el.K*q[2]
            F = self.transform_Force_Out_Of_Element_Frame(F, el)
        else:
            raise Exception('No correct element name provided')
        return F
    def transform_Force_Out_Of_Element_Frame(self,F,el):
        #rotation matrix is 3x3 to account for z axis
        Fx=F[0]
        Fy=F[1]
        F[0]=Fx*el.ROut[0,0]+Fy*el.ROut[0,1]
        F[1]=Fx*el.ROut[1,0]+Fy*el.ROut[1,1]
        return F
    def transform_Coords_To_Element_Frame(self, q, el):
        #q: particle coords in x and y plane. numpy array
        #el: element object to transform to
        #get the coordinates of the particle in the element's frame
        #For lens and drift the beginning is at the origin and the output is point at theta=0
        #For bender the input is facing south and is parallel to and coincident with the x axis
        qNew=q.copy() #only use x and y. CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        if el.type=='DRIFT' or el.type=='LENS':
            qNew[0]=qNew[0]-el.r1[0]
            qNew[1]=qNew[1]-el.r1[1]
        elif el.type=='BENDER':
            qNew[:2]=qNew[:2]-el.r0
        elif el.type=='COMBINER':
            sys.exit()
            pass
        else:
            raise Exception('No correct element name provided')
        qNewx=qNew[0]
        qNewy = qNew[1]
        qNew[0] = qNewx*el.RIn[0,0]+qNewy*el.RIn[0,1]
        qNew[1] = qNewx*el.RIn[1,0]+qNewy*el.RIn[1,1]
        return qNew
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
            coordsxy = self.transform_Coords_To_Element_Frame(q, el)
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
        #TODO: REPLACE THIS WITH MORE ROBUST GEOMETRIC ARGUMENT. SHAPELY HAS TINY GAPS BETWEEN ELEMENTS!!!!!

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
        dphi=-1e-6 #need a clock wise rotation. 1 microradian
        R=np.array([[np.cos(dphi),np.sin(dphi)],[np.sin(dphi),np.cos(dphi)]])
        dr=R@r-r
        #add that to the position and try once more
        qNew=q.copy()
        qNew[:-1]=qNew[:-1]+dr
        el=self.which_Element_Shapely(qNew)
        return el
    def adjust_Energy(self):
        #when the particel traverses an element boundary, energy needs to be conserved. This would be irrelevant if I
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
    def solve_Monge_Problem(self,r1,r2,r3,L2,L3,phi1):
        #courtesy of the QFT nerd Tyler Guglielmo
        #this is the problem of 3 circles connected by their tangents. there are 3 lengths, L1 L2 L3, and 3 angles phi1,
        #phi2,phi3. Here phi1 is the bending angle of the combiner and L2 and L3 the length of elements leading into it.
        #phi2 and phi3 are bending angles of the benders and L1 the length between them. Similarly, r1 is bending angle
        # of the combiner and r2 and r3 of the benders
        def errorFunc(x):
            phi2,phi3,L1=x
            s1=L1+r2*np.tan(phi2/2)+r3*np.tan(phi3/2)
            s2=L2+r1*np.tan(phi1/2)+r3*np.tan(phi3/2)
            s3=L3+r1*np.tan(phi1/2)+r2*np.tan(phi2/2)

            errors1=s2**2+s3**2-2*s2*s3*np.cos(np.pi-phi1)-s1**2
            errors2=s1**2+s3**2-2*s1*s3*np.cos(np.pi-phi2)-s2**2
            errors3=s1**2+s2**2-2*s1*s2*np.cos(np.pi-phi3)-s3**2
            error=np.sqrt(errors1**2+errors2**2+errors3**2)
            return error#errors1,errors2,errors3
        num = 50
        x1 = np.linspace(np.pi - phi1, np.pi, num=num)
        x2 = np.linspace(np.pi - phi1, np.pi, num=num)
        x3 = np.linspace((L2 + L3), (L2 + L3), num=num ** 2)
        temp = np.asarray(np.meshgrid(x1, x2))
        temp = temp.T.reshape(num ** 2, 2)
        temp = np.column_stack((temp, x3))
        vals = []
        for arg in temp:
            vals.append(errorFunc(arg))
        vals = np.asarray(vals)
        x0 = temp[np.argmin(vals)]
        res = spo.minimize(errorFunc, x0, method='Nelder-Mead',options={'xtol':1e-26,'fatol':1e-26,'maxiter':512})
        return res.x


    def make_Geometry(self):
        #construct the shapely objects used to plot the lattice and to determine if particles are inside of the lattice.
        #it could be changed to only use the shapely objects for plotting, but it would take some clever algorithm I think
        #and I am in a crunch kind of.
        #----
        #all of these take some thinking to visualize what's happening.
        benderPoints=50 #how many points to represent the bender with along each curve
        for el in self.lattice:
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
                pass
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

                #if previous element was a bender then this changes the next element's angle.
                #if the previous element or next element is a bender, then there is a shift so the particle
                #rides in the orbit correctly
                prevEl = self.lattice[i - 1]
                if prevEl.type=='BENDER' or prevEl.type=='COMBINER':
                    theta=prevEl.theta-prevEl.ang
                elif prevEl.type=='LENS' or prevEl.type=='DRIFT':
                    theta=prevEl.theta
                else:
                    raise Exception('incorrect element name provided')
                el.theta=theta
                #set end coordinates
                if el.type=='DRIFT' or el.type=='LENS':
                    xe=xb+el.L*np.cos(theta)
                    ye=yb+el.L*np.sin(theta)
                    el.nb = -np.asarray([np.cos(el.theta), np.sin(el.theta)])  # normal vector to input
                    el.ne = -el.nb #normal vector to end
                    if prevEl.type=='BENDER' or prevEl.type=='COMBINER':
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
                    #the bender can be tilted so this is tricky. This is a rotation about a point that is
                    #not the origin. First I need to find that point.
                    xc=xb+np.sin(theta)*el.rb #without including trajectory offset
                    yc=yb-np.cos(theta)*el.rb
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
                    el.r0=np.asarray([xb+el.rb*np.sin(theta),yb-el.rb*np.cos(theta)]) #coordinates of center of bender
                elif el.type=='COMBINER':
                    pass
                else:
                    raise Exception('No correct element name provided')
            if el.type=='BENDER' or el.type=='COMBINER':
                rot = (el.theta - el.ang + np.pi / 2)
            elif el.type=='LENS' or el.type=='DRIFT':
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
        plt.show()

test=particleTracer()




test.add_Lens(1,1,.01)
#test.add_Combiner()
test.add_Bender(np.pi,1,1,.01)
test.add_Lens(1,1,.01)
test.add_Bender(np.pi,1,1,.01)
test.end_Lattice()

q0=np.asarray([0.0,1e-3,0.0])
v0=np.asarray([-200.0,0,0])
q, p, qo, po, particleOutside = test.trace(q0, v0, 1e-5, 4.1/ 200,method='verlet')
print(qo)
plt.plot(qo[:,0],qo[:,1])
plt.show()
test.show_Lattice(particleCoords=q[-1])