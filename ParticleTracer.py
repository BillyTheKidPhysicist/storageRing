import time
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import sys
from shapely.geometry import Polygon,Point
import scipy.interpolate as spi
import numpy.linalg as npl
from shapely.affinity import translate, rotate




class Element:
    def __init__(self,args,type,PT):
        self.PT=PT #particle tracer object
        self.type=type
        self.Bp=None #field strength at tip of element, T
        self.rp=None #bore of element, m
        self.L=None #length of element, m
        self.rb=None #'bending' radius of magnet. actual bending radius of atoms is slightly different cause they
                #ride on the outside edge, m
        self.r0=None #center of element (for bender this is at bending radius center),[x,y], m
        self.ro=None #bending radius of orbit.
        self.ang=None #bending angle of bender, radians
        self.xb=None #x coordinate of center (transversally) of beginning of element
        self.xe=None #end of element
        self.yb = None  # y coordinate of center (transversally) of beginning of element
        self.ye = None  # end of element
        self.ne=None #normal vector to end of element
        self.nb=None #normal vector to beginning of element
        self.theta=None #angle from horizontal of element. zero degrees is to the right in polar coordinates
        self.ap=None #size of apeture. For now the same in both dimensions and vacuum tubes are square
        self.SO=None #shapely object used to find if particle is inside
        self.index=None #the index of the element in the lattice
        self.K=None #spring constant for magnets
        self.rOffset=None #the offset of the particle trajectory in a bending magnet

        self.unpack_Args_And_Fill_Params(args)


    def unpack_Args_And_Fill_Params(self,args):
        if self.type=='LENS':
            self.Bp=args[0]
            self.rp=args[1]
            self.L=args[2]
            self.ap=args[3]
            self.K = (2 * self.Bp * self.PT.u0_Actual / self.rp ** 2) / self.PT.m_Actual  # reduced force
        if self.type=='DRIFT':
            self.L=args[0]
            self.ap=args[1]
        if self.type=='BENDER':
            self.Bp=args[0]
            self.rb=args[1]
            self.rp=args[2]
            self.ang=args[3]
            self.ap=args[4]
            self.K=(2*self.Bp*self.PT.u0_Actual/self.rp**2)/self.PT.m_Actual #reduced force
            self.rOffset = np.sqrt(self.rb**2/4+self.PT.m*self.PT.v0Nominal**2/self.K)-self.rb/2 #this method does not
                #account for reduced speed in the bender from energy conservation
            #self.rOffset=np.sqrt(self.rb**2/16+self.PT.m*self.PT.v0Nominal**2/(2*self.K))-self.rb/4 #this acounts for reduced
                #energy
            self.ro=self.rb+self.rOffset
            self.L = self.ang * self.ro

class particleTracer:
    def __init__(self):
        self.m_Actual = 1.1648E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009994E-24 # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.m=1 #adjusted value. 1 is equal to li7 mass
        self.u0=self.u0_Actual/self.m_Actual

        self.kb = .38064852E-23  # boltzman constant, SI
        self.q=np.zeros(3) #contains the particles current position coordinates
        self.p=np.zeros(3) #contains the particles current momentum. m is set to 1 so this is the same
            #as velocity
        self.qoList=[] #coordinates in orbit frame [s,x,y] where s is along orbit
        self.poList=[] #momentum coordinates in orbit frame [s,x,y] where s is along orbit
        self.qList=[] #coordinates in labs frame,[x,y,z] position,m
        self.pList=[] #coordinates in labs frame,[vx,vy,v] velocity,m/s

        self.cumulativeLength=0 #cumulative length of previous elements. This accounts for revolutions, it doesn't reset each
            #time
        self.deltaT=0 #time spent in current element by particle. Resets to zero after entering each element
        self.T=0 #total time elapsed
        self.h=None #current step size. This changes near boundaries
        self.h0=None # initial step size.
        self.particleOutside=False #if the particle has stepped outside the chamber

        self.vacuumTube=None #holds the vacuum tube object
        self.lattice=[] #to hold all the lattice elements

        self.currentEl=None #to keep track of the element the particle is currently in
        self.elHasChanged=False # to record if the particle has changed to another element
        self.timeAdapted=False #wether the time step has been shrunk because nearness to a new element

        self.v0=None #the particles total speed. TODO: MAKE CHANGE WITH HARD EDGE MODEL
        self.v0Nominal=200 #Design particle speed
        self.E0=None #total initial energy of particle

        self.VList=[] #list of potential energy
        self.TList=[] #list of kinetic energy
        self.EList=[] #too keep track of total energy. This can tell me if the simulation is behaving
            #This won't always be on
    def add_Lens(self,L,Bp,rp,ap=None):
        if ap is None:
            ap=.9*rp
        args=[Bp,rp,L,ap]
        el=Element(args,'LENS',self)
        el.index = len(self.lattice)
        self.lattice.append(el)


    def add_Drift(self,L,ap=.03):
        #ap: apeture. Default value of 3 cm radius
        args=[L,ap]
        el=Element(args,'DRIFT',self)
        el.index = len(self.lattice)
        self.lattice.append(el)
    def add_Bender(self,ang,rb,Bp,rp,ap=None):
        if ap is None:
            ap=.9*rp
        args=[Bp,rb,rp,ang,ap]
        el=Element(args,'BENDER',self)
        el.index = len(self.lattice)
        self.lattice.append(el)
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

    def trace(self,qi,vi,h,T0):
        self.reset()  # reset paremters between runs
        self.currentEl = self.which_Element(qi)
        self.q=qi.copy()
        self.p=vi.copy()*self.m
        self.qList.append(self.q)
        self.pList.append(self.p)
        self.qoList.append(self.get_Coords_In_Orbit_Frame(self.q))
        self.poList.append(self.get_Momentum_In_Orbit_Frame(self.q,self.p))
        if self.currentEl is None:
            raise Exception('Particle\'s initial position is outside vacuum' )
        self.v0=np.sqrt(np.sum(vi**2))
        self.h=h
        self.h0=h
        self.E0=sum(self.get_Energies())
        self.EList.append(self.E0)
        self.deltaT=self.set_Initial_T()

        while(True):
            if self.T+self.h>T0:
                break

            self.time_Step()
            if self.particleOutside==True:
                break

            self.qList.append(self.q)
            self.pList.append(self.p)
            self.qoList.append(self.get_Coords_In_Orbit_Frame(self.q))
            self.poList.append(self.get_Momentum_In_Orbit_Frame(self.q,self.p))
            temp=self.get_Energies() #sum the potential and kinetic energy
            self.VList.append(temp[0])
            self.TList.append(temp[1])
            self.EList.append(sum(temp))
            self.T+=self.h
        qArr=np.asarray(self.qList)
        pArr=np.asarray(self.pList)
        qoArr=np.asarray(self.qoList)
        poArr=np.asarray(self.poList)

        return qArr,pArr,qoArr,poArr,self.particleOutside
    def set_Initial_T(self):
        return 0
    def time_Step(self):

        q=self.q #q old or q sub n
        p=self.p #p old or p sub n

        F=self.force(q)


        a = F / self.m  # acceleration old or acceleration sub n

        q_n=q+(p/self.m)*self.h+.5*a*self.h**2 #q new or q sub n+1

        F_n,coordsxy=self.force(q_n,returnCoords=True)
        if self.is_Particle_Inside(q_n) == False: #must come after F_n. #TODO: THIS CAN BE IMPROVED. REDUDANCY
            return

        a_n = F_n / self.m  # acceleration new or acceleration sub n+1
        p_n=p+self.m*.5*(a+a_n)*self.h


        self.q=q_n
        self.p=p_n
        self.deltaT+=self.h #update time spent in element. This resets to zero everytime element changes
        self.adapt_Time_Step(coordsxy) #adapt time step
        if self.elHasChanged==True:
            self.elHasChanged=False
            self.timeAdapted=False
            self.h=self.h0
            #self.adjust_Energy()


    def is_Particle_Inside(self,q):
        #this could be done with which_Element, but this is faster
        if self.currentEl is None:
            return False
        if np.abs(q[-1])>self.currentEl.ap: #check z axis
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

    def get_Coords_In_Orbit_Frame(self,q):
        #need to rotate coordinate system to align with the element
        coordsxy = self.transform_Coords_To_Element_Frame(q[:-1], self.currentEl)
        if self.currentEl.type=='LENS' or self.currentEl.type=='DRIFT':
            qos=self.cumulativeLength+coordsxy[0]
            qox=coordsxy[1]
        elif self.currentEl.type=='BENDER':
            phi=self.currentEl.ang-np.arctan2(coordsxy[1]+1e-10,coordsxy[0])

            ds=self.currentEl.ro*phi

            qos=self.cumulativeLength+ds
            qox=np.sqrt(coordsxy[0]**2+coordsxy[1]**2)-self.currentEl.rb-self.currentEl.rOffset
        qoy = q[2]
        qo=np.asarray([qos, qox, qoy])


        return qo

    def get_Momentum_In_Orbit_Frame(self,q,p):
        #TODO: CONSOLIDATE THIS WITH GET_POSITION
        el=self.currentEl
        coordsxy = self.transform_Coords_To_Element_Frame(q[:-1], self.currentEl)
        if el.type=='BENDER':
            pNew=p.copy()
            rot = -(el.theta - el.ang + np.pi / 2)
            pNew[0] = p[0]* np.cos(rot) + p[1]* (-np.sin(rot))
            pNew[1] = p[0]* np.sin(rot) + p[1]* np.cos(rot)
            sDot=(coordsxy[0]*pNew[1]-coordsxy[1]*pNew[0])/np.sqrt((coordsxy[0]**2+coordsxy[1]**2))
            rDot=(coordsxy[0]*pNew[0]+coordsxy[1]*pNew[1])/np.sqrt((coordsxy[0]**2+coordsxy[1]**2))
            po=np.asarray([sDot,rDot,pNew[2]])
            return po
        elif el.type=='LENS' or el.type == 'DRIFT':
            pNew = p.copy()
            rot = (el.theta)
            pNew[0] = p[0]* np.cos(rot) + p[1]* (-np.sin(rot))
            pNew[1] = p[0]* np.sin(rot) + p[1]* np.cos(rot)
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



    def force(self,q,returnCoords=False):
        el,coordsxy=self.which_Element_Wrapper(q,returnCoords=True) #find element the particle is in, and the coords in
            #the element frame as well
        if el is None: #the particle is outside the lattice! Note that this isn't a perfect way to determine this because
            #which_Element_Wrper only check when the particle is near the element
            if returnCoords==True:
                return None,None
            else:
                return None
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

            #sys.exit()
        elif el.type=='LENS':
            #note: for the perfect lens, in it's frame, there is never force in the x direction
            F[1] =-el.K*coordsxy[1]
            F[2] =-el.K*q[2]
            F = self.transform_Force_Out_Of_Element_Frame(F, el)
        else:
            sys.exit()

        if returnCoords==True:
            return F,coordsxy
        else:
            return F
    def transform_Force_Out_Of_Element_Frame(self,F,el):
        #rotation matrix is 3x3 to account for z axis
        if el.type=='BENDER':
            rot=(el.theta-el.ang+np.pi/2)
        if el.type=='LENS':
            rot=el.theta
        Fx=F[0]
        Fy=F[1]
        F[0]=Fx*np.cos(rot)+Fy*(-np.sin(rot))
        F[1]=Fx*np.sin(rot)+Fy*np.cos(rot)
        return F

    def transform_Coords_To_Element_Frame(self, q, el):
        #q: particle coords in x and y plane. numpy array
        #el: element object to transform to
        #get the coordinates of the particle in the element's frame where the input is facing towards the 'west'.
        #note that this would mean that theta is 0 deg. Also, this is only in the xy plane
        qNew=q.copy() #only use x and y. CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        if el.type=='DRIFT' or el.type=='LENS':
            qNew[0]=qNew[0]-el.xb
            qNew[1]=qNew[1]-el.yb
            rot = -el.theta
        elif el.type=='BENDER':
            rot=-(el.theta-el.ang+np.pi/2)
            qNew=qNew-el.r0

        qNewx=qNew[0]
        qNewy = qNew[1]
        qNew[0] = qNewx*np.cos(rot)+qNewy*(-np.sin(rot))
        qNew[1] = qNewx*np.sin(rot)+qNewy*(np.cos(rot))
        return qNew
    def distance_From_End(self,coordsxy,el):
        #determine the distance along the orbit that the particle is from the end of the element
        if el.type=='BENDER':
            s=el.rb*np.arctan2(coordsxy[1],coordsxy[0])
            return s
        elif el.type=='LENS' or el.type=='DRIFT':
            return el.L-coordsxy[0]
        pass
    def which_Element_Wrapper(self,q,returnCoords=False):
        #this is used to determine wether it's worthwile to call which_Element which is kind of expensive to call
        #dist=np.sqrt((self.q[0]-self.currentEl.xe)**2+(self.q[1]-self.currentEl.ye)**2) #distance to end of element
        T_el=self.currentEl.L/self.v0 #total time to traverse element assuming all speed is along length. This won't
            #work if the particles velocity increases with time! TODO: ROBUST METHOD WITH PARTICLES CURRENT SPEED
        if self.deltaT<T_el-self.h0*150: #if particle is far enough away that it isn't worth checking
            coordsxy = self.transform_Coords_To_Element_Frame(q[:-1], self.currentEl)  # only x and y coords
            el=self.currentEl
        else: #if close enough, try more precise check
            el=self.which_Element(q)
            if el is not self.currentEl: #if element has changed
                #the order here is very important!
                self.cumulativeLength += self.currentEl.L  #1, length up to the beginning of each element
                self.currentEl = el #2, update element
                self.deltaT = 0
                self.elHasChanged=True #the particle is now in a new element
            if el is None:
                self.particleOutside=True
                coordsxy=None
            else:
                coordsxy = self.transform_Coords_To_Element_Frame(q[:-1], self.currentEl)  # only x and y coords
        if returnCoords==True:
            return el, coordsxy
        else:
            return el
    def adapt_Time_Step(self,coordsxy):
        if self.timeAdapted==False:
            T_el=self.currentEl.L/self.v0 #total time to traverse element assuming all speed is along length. This won't
                #work if the particles velocity increases with time! TODO: ROBUST METHOD WITH PARTICLES CURRENT SPEED
            if self.deltaT<T_el-self.h0*150: #if particle is far enough away that it isn't worth checking
                pass
            else:
                deltas=self.distance_From_End(coordsxy,self.currentEl)
                if deltas<1.5*self.v0*self.h0: #if very near, decrease time step
                    self.h=self.h0/100
                    self.timeAdapted=True
    def which_Element(self,q):
        #TODO: PROPER METHOD OF DEALING WITH PARTICLE EXACTLY ON EDGE
        point=Point([q[0],q[1]])
        for el in self.lattice:
            if el.SO.contains(point)==True:
                if np.abs(q[2])>el.ap: #element clips in the z direction
                    return None
                else:
                    return el #return the element the particle is in
        return None #if the particle is inside no elements
    def adjust_Energy(self):
        #when the particel traverses an element boundary, energy needs to be conserved
        el=self.currentEl
        E=sum(self.get_Energies())
        deltaE=E-self.EList[-1]
        #Basically solve .5*m*(v+dv)dot(v+dv)+PE+deltaPE=E0 then add dv to v
        ndotv= el.nb[0]*self.p[0]/self.m+el.nb[1]*self.p[1]/self.m#the normal to the element inlet dotted into the particle velocity
        deltav=-(np.sqrt(ndotv**2-2*deltaE/self.m)+ndotv)

        #sys.exit()
        self.p[0]+=deltav*el.nb[0]*self.m
        self.p[1]+=deltav*el.nb[1]*self.m
    def end_Lattice(self):
        self.catch_Errors()
        self.set_Element_Coordinates()
        self.make_Geometry()
    def make_Geometry(self):
        #all of these take some thinking to visualize what's happening
        benderPoints=50
        for el in self.lattice:
            xb=el.xb
            yb=el.yb
            xe=el.xe
            ye=el.ye
            ap=el.ap
            theta=el.theta
            if el.type=='DRIFT' or el.type=='LENS':
                el.r0=np.asarray([(xb+xe)/2,(yb+ye)/2]) #center of lens or drift is midpoint of line connecting beginning and end
                q1=[xb-np.sin(theta)*ap,yb+ap*np.cos(theta)] #top left when theta=0
                q2=[xe-np.sin(theta)*ap,ye+ap*np.cos(theta)] #top right when theta=0
                q3=[xe-np.sin(theta)*ap,ye-ap*np.cos(theta)] #bottom right when theta=0
                q4=[xb-np.sin(theta)*ap,yb-ap*np.cos(theta)] #bottom left when theta=0
                el.SO=Polygon([q1,q2,q3,q4])
            if el.type=='BENDER':
                el.r0=np.asarray([el.xb+el.rb*np.sin(theta),el.yb-el.rb*np.cos(theta)]) #coordinates of center of bender
                phiArr=np.linspace(0,-el.ang,num=benderPoints)+theta+np.pi/2 #angles swept out
                xInner=(el.rb-ap)*np.cos(phiArr)+el.r0[0]
                yInner=(el.rb-ap)*np.sin(phiArr)+el.r0[1]
                xOuter=np.flip((el.rb+ap)*np.cos(phiArr)+el.r0[0])
                yOuter=np.flip((el.rb+ap)*np.sin(phiArr)+el.r0[1])
                x=np.append(xInner,xOuter)
                y=np.append(yInner,yOuter)
                el.SO=Polygon(np.column_stack((x,y)))
    def catch_Errors(self):
        if self.lattice[0].type=='BENDER':
            raise Exception('FIRST ELEMENT CANT BE A BENDER')
    def set_Element_Coordinates(self):
        #each element has
        i=0
        for el in self.lattice:
            if i==0:
                if el.type=='BENDER':
                    raise Exception('first element cant be bender')
                el.xb=0.0#set beginning coords
                el.yb=0.0#set beginning coords
                el.theta=np.pi #first element is straight. It can't be a bender
                el.xe=el.L*np.cos(el.theta) #set ending coords
                el.ye=el.L*np.sin(el.theta) #set ending coords
                el.nb=-np.asarray([np.cos(el.theta),np.sin(el.theta)]) #normal vector to input
                el.ne=-el.nb
            else:
                el.xb=self.lattice[i-1].xe#set beginning coordinates to end of last 
                el.yb=self.lattice[i-1].ye#set beginning coordinates to end of last

                #if previous element was a bender then this changes the next element's angle.
                #if the previous element or next element is a bender, then there is a shift so the particle
                #rides in the orbit correctly
                prevEl = self.lattice[i - 1]
                if prevEl.type=='BENDER':
                    theta=-prevEl.theta+prevEl.ang
                elif prevEl.type=='LENS' or prevEl.type=='DRIFT':
                    theta=prevEl.theta
                el.theta=theta

                #set end coordinates
                if el.type=='DRIFT' or el.type=='LENS':
                    el.xe=el.xb+el.L*np.cos(theta)
                    el.ye=el.yb+el.L*np.sin(theta)
                    el.nb = -np.asarray([np.cos(el.theta), np.sin(el.theta)])  # normal vector to input
                    el.ne = -el.nb #normal vector to end
                    if prevEl.type=='BENDER':
                        el.xb+=prevEl.rOffset*np.sin(prevEl.theta)
                        el.yb+=prevEl.rOffset*(-np.cos(prevEl.theta))
                        el.xe+=prevEl.rOffset*np.sin(prevEl.theta)
                        el.ye+=prevEl.rOffset*(-np.cos(prevEl.theta))


                elif el.type=='BENDER':
                    #the bender can be tilted so this is tricky. This is a rotation about a point that is
                    #not the origin. First I need to find that point.
                    xc=el.xb-np.sin(theta)*el.rb
                    yc=el.yb-np.cos(theta)*el.rb
                    #now use the point to rotate around
                    phi=-el.ang #bending angle. Need to keep in mind that clockwise is negative
                    #and counterclockwise positive. So I add a negative sign here to fix that up
                    el.xe=np.cos(phi)*el.xb-np.sin(phi)*el.yb-xc*np.cos(phi)+yc*np.sin(phi)+xc
                    el.ye=np.sin(phi)*el.xb+np.cos(phi)*el.yb-xc*np.sin(phi)-yc*np.cos(phi)+yc
                    #accomodate for bending trajectory
                    el.xb+=el.rOffset*np.sin(el.theta)
                    el.yb+=el.rOffset*(-np.cos(el.theta))
                    el.xe+=el.rOffset*np.sin(el.theta)
                    el.ye+=el.rOffset*(-np.cos(el.theta))

                    el.nb = -np.asarray([np.cos(el.theta), np.sin(el.theta)])  # normal vector to input
                    el.ne = el.nb #normal vector to end

            #clean up tiny numbers. There are some numbers like 1e-16 for numerical accuracy
            el.xb=np.round(el.xb,10)
            el.yb=np.round(el.yb,10)
            el.xe=np.round(el.xe,10)
            el.ye=np.round(el.ye,10)
            el.ne=np.round(el.ne,10)
            el.nb=np.round(el.nb,10)
            i+=1
        #check that the last point matchs the first point within a small number.
        #need to account for offset.
        deltax=np.abs(self.lattice[0].xb-self.lattice[-1].xe)+self.lattice[-1].rOffset*np.sin(self.lattice[-1].theta)
        deltay=np.abs(self.lattice[0].yb-self.lattice[-1].ye)+self.lattice[-1].rOffset*(-np.cos(self.lattice[-1].theta))
        smallNum=1e-10
        if deltax>smallNum or deltay>smallNum:
            raise Exception('ENDING POINTS DOES NOT MEET WITH BEGINNING POINT. LATTICE IS NOT CLOSED')
    def show_Lattice(self,particleCoords=None):
        for el in self.lattice:
            plt.plot(*el.SO.exterior.xy)
        if particleCoords is not None:
            plt.scatter(*particleCoords[:-1],marker='x',s=1000,c='r')
            plt.scatter(*particleCoords[:-1], marker='o', s=50, c='r')
        plt.grid()
        plt.show()

test=particleTracer()
Ld=.094
#test.add_Lens(np.pi,1,.01)
test.add_Drift(Ld)
test.add_Bender(np.pi,1,1,.01)
test.add_Drift(Ld)
#test.add_Lens(np.pi,1,.01)
test.add_Bender(np.pi,1,1,.01)
test.end_Lattice()

xi=2.5e-3
v0=np.asarray([-200.0,0,0])
el=test.lattice[1]
L=1*Ld+1*np.pi*(1+test.lattice[1].rOffset)
Lt=L
dt=5e-7

#
q0=np.asarray([-1E-10,xi,0])
q,p,qo,po,particleOutside=test.trace(q0,v0,dt,1.05*Lt/200)
print(particleOutside)
func1=spi.interp1d(qo[:,0],qo[:,1],kind='quadratic')
func11=spi.interp1d(qo[:,0],po[:,1],kind='quadratic')


q0=np.asarray([-1E-10,-xi,0])
q,p,qo,po,particleOutside=test.trace(q0,v0,dt,1.05*Lt/200)
#test.show_Lattice(particleCoords=q[-1])
print(particleOutside)
func2=spi.interp1d(qo[:,0],qo[:,1],kind='quadratic')
func22=spi.interp1d(qo[:,0],po[:,1],kind='quadratic')


print(1e6*func1(Lt),-1e6*func2(Lt),(1e6*func1(Lt)-1e6*func2(Lt))/2) #332.3062292540387
print(1e3*func11(Lt),-1e3*func22(Lt),(1e3*func11(Lt)-1e3*func22(Lt))/2) #332.3062292540387



#a=1e6*func1(Lt)-1e6*func11(Lt)*Ld/po[-1,0]
#b=1e6*func2(Lt)-1e6*func22(Lt)*Ld/po[-1,0]
#print(a,b,(a-b)/2)
# +:-198.3161555885482
# -:198.31711977089498

#+: -198.31757256461063
#-: 198.31783446125363


#print(1e6*qo[-1][1])#,qo[-1,0])#,-1e3*p[-1,1],qo[-1,0])
#test.show_Lattice(particleCoords=q[-1])
#test.EList
#plt.plot(test.EList)
#plt.show()
#
#

#print(particleOutside)
#plt.plot(qo[:,0],np.sqrt(np.sum(p**2,axis=1)))
xTest=np.linspace(1e-3,Lt,num=100000)
##
##`plt.plot(xTest,func(xTest))
#print(np.trapz(po[:,0],dx=1e-7))
plt.plot(xTest,func11(xTest))
plt.plot(xTest,func22(xTest))
plt.grid()
plt.show()

plt.plot(xTest,func1(xTest))
plt.plot(xTest,func2(xTest))
plt.grid()
plt.show()
