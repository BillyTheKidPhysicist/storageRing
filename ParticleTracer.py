import time
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import sys
from shapely.geometry import Polygon,Point
from shapely.affinity import translate, rotate


class Element:
    def __init__(self,args,type,PT):
        self.PT=PT #particle tracer object
        self.type=type
        self.Bp=None #field strength at tip of element, T
        self.rp=None #bore of element, m
        self.L=None #length of element, m
        self.rb=None #bending radius, m
        self.r0=None #center of element (for bender this is at bending radius center), m
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
            self.L=self.ang*self.rb
            self.K=(2*self.Bp*self.PT.u0_Actual/self.rp**2)/self.PT.m_Actual #reduced force
            self.rOffset = self.PT.m * self.PT.v0Nominal ** 2 * self.rp ** 2 / (2 * self.Bp * self.PT.u0)
class particleTracer:
    def __init__(self):
        self.m_Actual = 1.16503E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009E-24  # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.m=1 #adjusted value
        self.u0=self.u0_Actual/self.m_Actual

        self.kb = .38064852E-23  # boltzman constant, SI
        self.q=np.zeros(3) #contains the particles current position coordinates
        self.p=np.zeros(3) #contains the particles current momentum. m is set to 1 so this is the same
            #as velocity
        self.qoList=[] #coordinates in orbit frame [s,x,y] where s is along orbit
        self.qList=[] #coordinates in labs frame,[x,y,z] position,m
        self.pList=[] #coordinates in labs frame,[vx,vy,v] velocity,m/s

        self.cumulativeLength=0 #cumulative length of previous elements. This accounts for revolutions, it doesn't reset each
            #time
        self.h=None #current step size. This changes near boundaries
        self.h0=None # initial step size.
        self.particleOutside=False #if the particle has stepped outside the chamber

        self.vacuumTube=None #holds the vacuum tube object
        self.lattice=[] #to hold all the lattice elements

        self.currentEl=None #to keep track of the element the particle is currently in
        self.elHasChanged=False # to record if the particle has changed to another element

        self.v0=None #the particles total speed. TODO: MAKE CHANGE WITH HARD EDGE MODEL
        self.v0Nominal=200 #Design particle speed
        self.E0=None #total initial energy of particle

        self.VList=[] #list of potential energy
        self.TList=[] #list of kinetic energy
        self.EList=[] #too keep track of total energy. This can tell me if the simulation is behaving
            #This won't always be on
    def add_Lens(self,Bp,rp,L,ap=None):
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
    def add_Bender(self,Bp,rb,rp,ang,ap=None):
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
        self.TList=[]
        self.VList=[]
        self.EList=[]
        self.particleOutside=False
        self.cumulativeLength=0

    def trace(self,qi,vi,h,T0):
        self.reset() #reset paremters between runs
        self.q[0]=qi[0]
        self.q[1]=qi[1]
        self.q[2]=qi[2]
        self.p[0]=self.m*vi[0]
        self.p[1]=self.m*vi[1]
        self.p[2]=self.m*vi[2]
        self.currentEl=self.which_Element(qi)
        if self.currentEl is None:
            raise Exception('Particle\'s initial position is outside vacuum' )
        self.v0=np.sqrt(np.sum(vi**2))
        self.h=h
        self.E0=sum(self.get_Energies())
        self.EList.append(self.E0)
        loop=True
        iMax=int(T0/h)
        i=0
        while(loop==True):
            self.qList.append(self.q)
            self.pList.append(self.p)
            self.time_Step()
            i += 1
            if self.particleOutside==True or i==iMax:
                loop=False
                break
            temp=self.get_Energies() #sum the potential and kinetic energy
            self.VList.append(temp[0])
            self.TList.append(temp[1])
            self.EList.append(sum(temp))
        qArr=np.asarray(self.qList)
        pArr=np.asarray(self.pList)
        qoArr=np.asarray(self.qoList)

        return qArr,pArr,qoArr,self.particleOutside
    def time_Step(self):
        q=self.q #q old or q sub n
        p=self.p #p old or p sub n
        F=self.force(q)
        #if F is None: #force returns None if particle is outside. Only does this when near the boundary of an element
        #    #though
        #    self.particleOutside=True
        #    return
        a = F / self.m  # acceleration old or acceleration sub n
        q_n=q+(p/self.m)*self.h+.5*a*self.h**2 #q new or q sub n+1
        if self.is_Particle_Inside(q_n)==False:
            return
        F_n=self.force(q_n)
        #if F_n is None: #force returns None if particle is outside, but not always!
        #    self.particleOutside=True
        #    return
        a_n = F_n / self.m  # acceleration new or acceleration sub n+1
        p_n=p+self.m*.5*(a+a_n)*self.h
        self.q=q_n
        self.p=p_n
        self.update_Coords_In_Orbit_Frame(self.q) #convert coordinates in lab frame to orbit frame and save
        if self.elHasChanged==True:
            self.adjust_Energy()
            self.elHasChanged=False
    def is_Particle_Inside(self,q):
        #this could be done with which_Element, but this is faster
        if np.abs(q[-1])>self.currentEl.ap:
            self.particleOutside=True
            return False
        coordsxy = self.transform_To_Element_Frame(q[:-1],self.currentEl)
        if self.currentEl.type=='LENS' or self.currentEl.type=='DRIFT':
            if coordsxy[1]>self.currentEl.ap or coordsxy[0]>self.currentEl.L*1.1:
                self.particleOutside=True
                return False

        elif self.currentEl.type=='BENDER':
            r=np.sqrt(np.sum(coordsxy**2))
            deltar=r-self.currentEl.rb
            if np.abs(deltar)>self.currentEl.ap:
                self.particleOutside=True
                return False
        return True
    def update_Coords_In_Orbit_Frame(self,q):
        #need to rotate coordinate system to align with the element
        coordsxy = self.transform_To_Element_Frame(q[:-1], self.currentEl)
        if self.currentEl.type=='LENS' or self.currentEl.type=='DRIFT':
            qos=self.cumulativeLength+coordsxy[0]
            qox=coordsxy[1]
        elif self.currentEl.type=='BENDER':
            phi=self.currentEl.ang-np.arctan2(coordsxy[1],coordsxy[0])
            ds=self.currentEl.rb*phi
            qos=self.cumulativeLength+ds
            qox=np.sqrt(coordsxy[0]**2+coordsxy[1]**2)-self.currentEl.rb-self.currentEl.rOffset
        qoy = q[2]
        self.qoList.append(np.asarray([qos, qox, qoy]))


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
            qxy = self.transform_To_Element_Frame(self.q[:-1], self.currentEl)
            r = np.sqrt(qxy[1] ** 2 + self.q[2] ** 2)
            B = self.currentEl.Bp * r ** 2 / self.currentEl.rp ** 2
            PE = self.u0 * B
            KE = np.sum(self.p ** 2) / (2 * self.m)
        elif self.currentEl.type == 'DRIFT':
            PE = 0
            KE = np.sum(self.p ** 2) / (2 * self.m)
        elif self.currentEl.type == 'BENDER':
            qxy = self.transform_To_Element_Frame(self.q[:-1], self.currentEl)  # only x and y coords
            r = np.sqrt(qxy[0] ** 2 + qxy[1] ** 2) - self.currentEl.rb
            B = self.currentEl.Bp * r ** 2 / self.currentEl.rp ** 2
            PE = self.u0 * B
            KE = np.sum(self.p ** 2) / (2 * self.m)
        return PE,KE



    def force(self,q):
        el,coordsxy=self.which_Element_Wrapper(q,returnCoords=True) #find element the particle is in, and the coords in
            #the element frame as well
        if el is None: #the particle is outside the lattice! Note that this isn't a reliable way to determine this because
            #which_Element_Wrper only check when the particle is near the element
            return None
        F = np.zeros(3) #force vector starts out as zero
        if el.type == 'DRIFT':
            return F #empty force vector
        elif el.type == 'BENDER':

            r=np.sqrt(coordsxy[0]**2+coordsxy[1]**2) #radius in x y frame
            F0=-el.K*(r-el.rb) #force in x y plane
            phi=np.arctan2(coordsxy[1],coordsxy[0])
            F[0]=np.cos(phi)*F0
            F[1]=np.sin(phi)*F0
            F[2]=-el.K*q[2]
            F=self.transform_Force_Out_Of_Element_Frame(F,el)
            return F
        elif el.type=='LENS':
            #note: for the perfect lens, in it's frame, there is never force in the x direction
            F[1] =-el.K*coordsxy[1]
            F[2] =-el.K*q[2]
            F = self.transform_Force_Out_Of_Element_Frame(F, el)
            return F
        else:
            sys.exit()
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

    def transform_To_Element_Frame(self,q,el):
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
        elif el.type=='LENS' or el.type=='DRFIT':
            return el.L-coordsxy[0]
        pass

    def which_Element_Wrapper(self,q,returnCoords=False):
        #this is used to determine wether it's worthwile to call which_Element which is kind of expensive to call
        dist=np.sqrt((self.q[0]-self.currentEl.xe)**2+(self.q[1]-self.currentEl.ye)**2) #distance to end of element
        coordsxy = self.transform_To_Element_Frame(q[:-1], self.currentEl)  # only x and y coords
        checkDist=self.currentEl.ap*2

        if dist>checkDist: #if particle is far enough away that it isn't worth checking
            el=self.currentEl
        else: #if close enough, try more precise check
            el=self.which_Element(q)

            if el is not self.currentEl:
                self.cumulativeLength+=self.currentEl.L
                self.currentEl=el
                self.elHasChanged=True #the particle is now in a new element
                coordsxy = self.transform_To_Element_Frame(q[:-1], el)  # only x and y coords
        if returnCoords==True:
            return el, coordsxy
        else:
            return el


    def which_Element(self,q):
        point=Point([q[0]+np.pi*1e-10,q[1]+np.pi*1e-10]) #if the particle is right on the border
            #it returns false for all. This 'ensures' that doesn't happen
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
        plt.show()

test=particleTracer()
test.add_Lens(1,.02,1)
test.add_Bender(2,1,.03,np.pi)
test.add_Lens(1,.02,1)
test.add_Bender(2,1,.03,np.pi)
test.end_Lattice()
q0=np.asarray([-.01,1e-3,0])
v0=np.asarray([-200,0,0])
###def testRun():
###    test.trace(q0, v0, 5300, 10e-6)
###testRun()
###cProfile.run('testRun()')
###t=time.time()
##steps=5000
q,p,qo,particleOutside=test.trace(q0,v0,1e-5,.0051)
plt.plot(qo[:,0],qo[:,1])
plt.grid()
#plt.show()
#plt.plot(qo[:,0],test.EList)
#plt.show()
test.show_Lattice(particleCoords=q[-1])
#plt.plot(qo[:,0],np.sqrt(np.sum(p**2,axis=1)))
#plt.grid()
#plt.show()


