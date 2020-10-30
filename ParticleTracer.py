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
        self.theta=None #angle from horizontal of element. zero degrees is to the right in polar coordinates
        self.ap=None #size of apeture. For now the same in both dimensions and vacuum tubes are square
        self.SO=None #shapely object used to find if particle is inside
        self.index=None #the index of the element in the lattice
        self.K=None #spring constant for magnets

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
            self.L=2*np.pi*self.rb
            self.K=(2*self.Bp*self.PT.u0_Actual/self.rp**2)/self.PT.m_Actual #reduced force
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
        self.h=None
        self.revs=None
        self.particleOutside=False #if the particle has stepped outside the chamber

        self.vacuumTube=None #holds the vacuum tube object
        self.lattice=[] #to hold all the lattice elements

        self.currentEl=None #to keep track of the element the particle is currently in
        self.currentElDeltaT=0 #to keep track of the total distance traveled by the particle since it entered the element
            #these needs to be at least equal to the element length to begin asking shapely what element the particle is
            #in
        self.v0=None #the particles total speed. TODO: MAKE CHANGE WITH HARD EDGE MODEL

        self.EList=[] #too keep track of total energy. This can tell me if the simulation is behaving #TODO: MAKE THIS WORK
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
    def trace(self,qi,vi,revs,h):
        self.q[0]=qi[0]
        self.q[1]=qi[1]
        self.q[2]=qi[2]
        self.p[0]=self.m*vi[0]
        self.p[1]=self.m*vi[1]
        self.p[2]=self.m*vi[2]
        self.revs=revs
        self.currentEl=self.which_Element(qi)
        self.v0=np.sqrt(np.sum(vi**2))
        self.h=h

        qList=[]
        pList=[]
        loop=True
        i=0
        while(loop==True):
            qList.append(self.q)
            pList.append(self.p)
            self.time_Step()
            if self.particleOutside==True:
                loop=False
            i+=1
            if i==self.revs:
                loop=False
        qArr=np.asarray(qList)
        pArr=np.asarray(pList)
        return qArr,pArr
    def time_Step(self):
        q=self.q #q old or q sub n
        p=self.p #p old or p sub n
        a=self.force(q)/self.m #acceleration old or acceleration sub n
        q_n=q+(p/self.m)*self.h+.5*a*self.h**2 #q new or q sub n+1
        if self.is_Particle_Inside_Chamber(q_n)==False:
            self.particleOutside=True
            return
        a_n=self.force(q_n)/self.m #acceleration new or acceleration sub n+1
        p_n=p+self.m*.5*(a+a_n)*self.h
        self.q=q_n
        self.p=p_n
    def is_Particle_Inside_Chamber(self,q):
        return True
    def loop_Check(self):
        z=self.q[2]
        if z>2:
            return False
        else:
            return True
    def force(self,q):
        el=self.which_Element_Wrapper(q)
        if el is None: #the particle is outside the lattice!
            return None
        F = np.zeros(3) #force vector starts out as zero
        if el.type == 'DRIFT':
            return F #empty force vector
        elif el.type == 'BENDER':
            coords=self.transform_To_Element_Frame(q,el)
            r=np.sqrt(np.sum(coords**2)) #radius in x y frame
            F0=-el.K*(r-el.rb) #force in x y plane
            phi=np.arctan2(coords[1],coords[0])
            F[0]=np.cos(phi)*F0
            F[1]=np.sin(phi)*F0
            F[2]=-el.K*q[2]
            F=self.transform_Force_Out_Of_Element_Frame(F,el)
            return F
        elif el.type=='LENS':
            coords = self.transform_To_Element_Frame(q, el) #x and y plane only
            #note: for the perfect lens, in it's frame, there is never force in the x direction
            F[1] =-el.K*coords[1]
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

    def transform_To_Element_Frame(self,point,el):
        #get the coordinates of the particle in the element's frame where the input is facing towards the 'west'.
        #note that this would mean that theta is 0 deg. Also, this is only in the xy plane
        point=point[:-1].copy() #only use x and y. CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!!
        newPoint = np.empty(2)
        if el.type=='DRIFT' or el.type=='LENS':
            point[0]=point[0]-el.xb
            point[1]=point[1]-el.yb
            rot = -el.theta
            #rotMat=np.asarray([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
            #point=rotMat@point[:,np.newaxis]
        elif el.type=='BENDER':
            rot=-(el.theta-el.ang+np.pi/2)
            point=point-el.r0
            #newPoint[0] = point[0] * np.cos(rot) + point[1] * (-np.sin(rot))
            #newPoint[0] = point[0] * np.sin(rot) + point[1] * (np.cos(rot))
            #rotMat=np.asarray([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
            #point=rotMat@point[:,np.newaxis]
        newPoint[0] = point[0] * np.cos(rot) + point[1] * (-np.sin(rot))
        newPoint[1] = point[0] * np.sin(rot) + point[1] * (np.cos(rot))
        return newPoint

    def transform_To_Element_Frame1(self, Point, el):
        # get the coordinates of the particle in the element's frame where the input is facing towards the 'west'.
        # note that this would mean that theta is 0 deg. Also, this is only in the xy plane
        point = Point[:-1].copy()  # only use x and y. CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT OUTSIDE OF METHOD!!!!
        newPoint = np.empty(2)
        if el.type == 'DRIFT' or el.type == 'LENS':
            point[0] = point[0] - el.xb
            point[1] = point[1] - el.yb
            rot = -el.theta
            # rotMat=np.asarray([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
            # point=rotMat@point[:,np.newaxis]
        elif el.type == 'BENDER':
            rot = -(el.theta - el.ang + np.pi / 2)
            point = point - el.r0
            # newPoint[0] = point[0] * np.cos(rot) + point[1] * (-np.sin(rot))
            # newPoint[0] = point[0] * np.sin(rot) + point[1] * (np.cos(rot))
            # rotMat=np.asarray([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
            # point=rotMat@point[:,np.newaxis]
        newPoint[0] = point[0] * np.cos(rot) + point[1] * (-np.sin(rot))
        newPoint[1] = point[0] * np.sin(rot) + point[1] * (np.cos(rot))
        return newPoint

    def which_Element_Wrapper(self,q):
        #this is used to determine wether it's worthwile to call which_Element which is kind of expensive to call
        dist=np.sqrt((self.q[0]-self.currentEl.xe)**2+(self.q[1]-self.currentEl.ye)**2) #distance to end of element
        stepSize=self.h*self.v0
        if dist>2*stepSize: #if particle is far enough away that it isn't worth checking
            return self.currentEl
        else:
            el=self.which_Element(q)
            if el is not self.currentEl:
                self.currentElDeltaT=0
                self.currentEl=el
            return el
    def which_Element(self,q):
        point=Point([q[0]+np.pi*1e-10,q[1]+np.pi*1e-10]) #if the particle is right on the border
            #it returns false for all. This 'ensures' that doens't happen
        for el in self.lattice:
            if el.SO.contains(point)==True:
                if np.abs(q[2])>el.ap: #element clips in the z direction
                    return None
                else:
                    return el #return the element the particle is in
        return None #if the particle is inside no elements
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
                el.xb=0.0#set beginning coords
                el.yb=0.0#set beginning coords
                el.theta=np.pi #first element is straight. It can't be a bender
                el.xe=el.L*np.cos(el.theta) #set ending coords
                el.ye=el.L*np.sin(el.theta) #set ending coords
            else:
                el.xb=self.lattice[i-1].xe#set beginning coordinates to end of last 
                el.yb=self.lattice[i-1].ye#set beginning coordinates to end of last

                #if previous element was a bender then this changes the next element's angle
                theta=None

                if self.lattice[i-1].type=='BENDER':
                    theta=-self.lattice[i-1].theta+self.lattice[i-1].ang
                else:
                    theta=self.lattice[i-1].theta
                el.theta=theta
                #set end coordinates
                if el.type=='DRIFT' or el.type=='LENS':
                    el.xe=el.xb+el.L*np.cos(theta)
                    el.ye=el.yb+el.L*np.sin(theta)
                elif el.type=='BENDER':
                    #the bender can be tilted so this is tricky. This is a rotation about a point that is
                    #not the orignin. First I need to find that point.
                    xc=el.xb-np.sin(theta)*el.rb
                    yc=el.yb-np.cos(theta)*el.rb
                    #now use the point to rotate around
                    phi=-el.ang #bending angle. Need to keep in mind that clockwise is negative
                    #and counterclockwise positive. So I add a negative sign here to fix that up
                    el.xe=np.cos(phi)*el.xb-np.sin(phi)*el.yb-xc*np.cos(phi)+yc*np.sin(phi)+xc
                    el.ye=np.sin(phi)*el.xb+np.cos(phi)*el.yb-xc*np.sin(phi)-yc*np.cos(phi)+yc

            #clean up tiny numbers. There are some numbers like 1e-16 for numerical accuracy
            el.xb=np.round(el.xb,10)
            el.yb=np.round(el.yb,10)
            el.xe=np.round(el.xe,10)
            el.ye=np.round(el.ye,10)
            i+=1
        #check that the last point matchs the first point within a small number.
        deltax=np.abs(self.lattice[0].xb-self.lattice[-1].xe)
        deltay=np.abs(self.lattice[0].yb-self.lattice[-1].ye)
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
q0=np.asarray([-.01,.00,0])
v0=np.asarray([-200,0,0])
#def testRun():
#    test.trace(q0, v0, 5300, 10e-6)
#testRun()
#cProfile.run('testRun()')
t=time.time()
q,p=test.trace(q0,v0,5300,1e-5)
print(time.time()-t)
test.show_Lattice(particleCoords=q[-1])
plt.plot(q[:,0],q[:,1])
#print(q[-1])
plt.show()

