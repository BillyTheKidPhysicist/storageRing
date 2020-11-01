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
            self.L=self.ang*self.rb
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
        self.qoList=[] #coordinates in orbit frame

        self.cumulativeLength=0 #cumulative length of previous elements. This accounts for revolutions, it doesn't reset each
            #time
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

        self.EList=[] #too keep track of total energy. This can tell me if the simulation is behaving
            #This won't always be on #TODO: MAKE THIS WORK
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
            self.update_Energy_List()
            if self.particleOutside==True:
                loop=False
            i+=1
            if i==self.revs:
                loop=False
        qArr=np.asarray(qList)
        pArr=np.asarray(pList)
        qoArr=np.asarray(self.qoList)
        return qArr,pArr,qoArr
    def time_Step(self):
        q=self.q #q old or q sub n
        p=self.p #p old or p sub n
        F=self.force(q)
        if F is None: #force returns None if particle is outside
            self.particleOutside=True
            return
        a = F / self.m  # acceleration old or acceleration sub n
        q_n=q+(p/self.m)*self.h+.5*a*self.h**2 #q new or q sub n+1
        F_n=self.force(q_n)
        if F_n is None: #force returns None if particle is outside
            self.particleOutside=True
            return
        a_n = F_n / self.m  # acceleration new or acceleration sub n+1
        p_n=p+self.m*.5*(a+a_n)*self.h
        self.q=q_n
        self.p=p_n
        self.update_Coords_In_Orbit_Frame(self.q)
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
            qox=np.sqrt(coordsxy[0]**2+coordsxy[1]**2)-self.currentEl.rb
        qoy = q[2]

        self.qoList.append(np.asarray([qos, qox, qoy]))


    def loop_Check(self):
        z=self.q[2]
        if z>2:
            return False
        else:
            return True
    def update_Energy_List(self):
        PE=None
        KE=None
        if self.currentEl.type=='LENS':
            qxy=self.transform_To_Element_Frame(self.q[:-1],self.currentEl)
            r=np.sqrt(qxy[1]**2+self.q[2]**2)
            B=self.currentEl.Bp*r**2/self.currentEl.rp**2
            PE=self.u0*B
            KE=np.sum(self.p**2)/(2*self.m)
        elif self.currentEl.type=='DRIFT':
            PE=0
            KE=np.sum(self.p**2)/(2*self.m)
        elif self.currentEl.type=='BENDER':
            qxy=self.transform_To_Element_Frame(self.q[:-1],self.currentEl) #only x and y coords
            r=np.sqrt(qxy[0]**2+qxy[1]**2) -self.currentEl.rb
            B=self.currentEl.Bp*r**2/self.currentEl.rp**2
            PE=self.u0*B
            KE=np.sum(self.p**2)/(2*self.m)
        #print(self.currentEl.type,PE+KE,B)
        self.EList.append(PE + KE)

    def force(self,q):
        el=self.which_Element_Wrapper(q)
        if el is None: #the particle is outside the lattice!
            return None
        F = np.zeros(3) #force vector starts out as zero
        if el.type == 'DRIFT':
            return F #empty force vector
        elif el.type == 'BENDER':
            coordsxy=self.transform_To_Element_Frame(q[:-1],el) #only x and y coords
            r=np.sqrt(coordsxy[0]**2+coordsxy[1]**2) #radius in x y frame
            F0=-el.K*(r-el.rb) #force in x y plane
            phi=np.arctan2(coordsxy[1],coordsxy[0])
            F[0]=np.cos(phi)*F0
            F[1]=np.sin(phi)*F0
            F[2]=-el.K*q[2]
            F=self.transform_Force_Out_Of_Element_Frame(F,el)
            return F
        elif el.type=='LENS':
            coordsxy = self.transform_To_Element_Frame(q[:-1], el) #x and y plane only
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
        #get the coordinates of the particle in the element's frame where the input is facing towards the 'west'.
        #note that this would mean that theta is 0 deg. Also, this is only in the xy plane
        qNew=q.copy() #only use x and y. CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        if el.type=='DRIFT' or el.type=='LENS':
            qNew[0]=qNew[0]-el.xb
            qNew[1]=qNew[1]-el.yb
            rot = -el.theta
            #rotMat=np.asarray([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
            #qNew=rotMat@qNew[:,np.newaxis]
        elif el.type=='BENDER':
            rot=-(el.theta-el.ang+np.pi/2)
            qNew=qNew-el.r0
            #newPoint[0] = qNew[0] * np.cos(rot) + qNew[1] * (-np.sin(rot))
            #newPoint[0] = qNew[0] * np.sin(rot) + qNew[1] * (np.cos(rot))
            #rotMat=np.asarray([[np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)]])
            #qNew=rotMat@qNew[:,np.newaxis]
        qNewx=qNew[0]
        qNewy = qNew[1]
        qNew[0] = qNewx*np.cos(rot)+qNewy*(-np.sin(rot))
        qNew[1] = qNewx*np.sin(rot)+qNewy*(np.cos(rot))
        return qNew

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
                self.cumulativeLength+=self.currentEl.L
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
q0=np.asarray([-.01,0.01,0])
v0=np.asarray([-200,0,0])
#def testRun():
#    test.trace(q0, v0, 5300, 10e-6)
#testRun()
#cProfile.run('testRun()')
t=time.time()
steps=2000
q,p,qo=test.trace(q0,v0,steps,1e-4)
#print(time.time()-t)
test.show_Lattice(particleCoords=q[-1])
tArr=np.linspace(0,steps*test.h,num=steps)
#plt.plot(tArr,test.EList)
#plt.plot(tArr,test.EList)
plt.plot(qo[:,0],qo[:,1])
EArr=np.asarray(test.EList)
#print(100*(np.max(EArr)-np.min(EArr))/EArr[0])
#print(q[-1])
plt.show()
