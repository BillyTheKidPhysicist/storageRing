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
from ParticleTracer import ParticleTracer

def Compute_Bending_Radius_For_Segmented_Bender(L,rp,yokeWidth,numMagnets,angle,space=0.0):
    #ucAng=angle/(2*numMagnets)
    rb=(L+2*space)/(2*np.tan(angle/(2*numMagnets)))+yokeWidth+rp
    #ucAng1=np.arctan((L/2)/(rb-rp-yokeWidth))

    return rb

class ParticleTracerLattice:
    def __init__(self,v0Nominal):
        self.v0Nominal = v0Nominal  # Design particle speed
        self.m_Actual = 1.1648E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009994E-24 # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.m=1 #adjusted value of mass. 1 is equal to li7 mass
        self.u0=self.u0_Actual/self.m_Actual #Adjusted value of bohr magneton, about equal to 800
        self.kb = 1.38064852E-23  # boltzman constant, SI

        self.benderIndices=[] #list that holds index values of benders. First bender is the first one that the particle sees
            #if it started from beginning of the lattice. Remember that lattice cannot begin with a bender
        self.combinerIndex=None #the index in the lattice where the combiner is
        self.totalLength=None #total length of lattice, m

        self.elList=[] #to hold all the lattice elements


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
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Drift(self,L,ap=.03):
        #Add element to the lattice. see elementPT.py for more details on specific element
        #L: length of drift element, m
        #ap:apeture. Default value of 3 cm radius, unitless
        args=[L,ap]
        el=Element(args,'DRIFT',self)#create a drift element object
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
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
        self.elList.append(el)
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
        self.elList.extend([cap1, lens, cap2])



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
        self.elList.append(el)
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

        self.elList.extend([cap1,bender,cap2])
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
        el.index = len(self.elList)  # where the element is in the lattice
        self.elList.append(el)
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
        el.index = len(self.elList) #where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el) #add element to the list holding lattice elements in order
    def add_Combiner_Ideal(self,Lb=.2,ang=None,offset=None,c1=1,c2=20,ap=.015):
        # Add element to the lattice. see elementPT.py for more details on specific element
        #add combiner (stern gerlacht) element to lattice
        #La: input length of combiner. The bent portion outside of combiner
        #Lb:  length of vacuum tube through the combiner. The segment that contains the field. This is the length of the
        #field
        #ang: angle that particle enters the combiner at
        # offset: particle enters inner section with some offset
        #c1: dipole component of combiner
        #c2: quadrupole component of bender
        #check to see if inlet length is too short. The minimum length is a function of apeture and angle
        #minLa=ap*np.sin(ang)
        #if La<minLa:
        #    raise Exception('INLET LENGTH IS SHORTER THAN MINIMUM')
        args=[Lb,ang,offset,ap,c1,c2]
        el=Element(args,'COMBINER_IDEAL',self) #create a combiner element object
        el.index = len(self.elList) #where the element is in the lattice
        self.combinerIndex=el.index
        self.elList.append(el) #add element to the list holding lattice elements in order


    def end_Lattice(self):

        self.catch_Errors()
        self.constrain_Lattice()
        self.set_Element_Coordinates()
        self.make_Geometry()

        self.totalLength=0
        #for el in self.elList: #total length of particle's orbit in an element
        #    self.totalLength+=el.Lo
    def constrain_Lattice(self):
        if self.combinerIndex is not None:
            combinerEl=self.elList[self.combinerIndex]

            inputAng,inputOffset=self.compute_Input_Angle_And_Offset(combinerEl) #TODO: PUT THIS INTO ELEMENT CLASS?
            combinerEl.ang=inputAng
            combinerEl.inputOffset=inputOffset
            combinerEl.fill_Params_And_Functions()

            phi2,phi3,L3=self.solve_Combiner_Constraints()

            lensIndex = 4 #temporary!!
            self.elList[self.benderIndices[0]].ang=phi2
            self.elList[self.benderIndices[1]].ang=phi3
            self.elList[lensIndex].L=L3

        for el in self.elList:
            el.fill_Params_And_Functions()
    def solve_Combiner_Constraints(self):
        #this solves for the constraint coming from two benders and a combiner. The bending angle of each bender is computed
        #as well as the distance between the two on the segment without the combiner

        #find the distance from the kink in the combiner from the bender before it, and the distance to the bender after
        # (clockwise)
        L1=self.elList[self.combinerIndex].La #from previous bender to kink
        L2=self.elList[self.combinerIndex].Lb #from kink to next bender
        for i in range(self.combinerIndex-1,-1,-1): #assumes that the combiner will be before the first bender in the
            #lattice element list
            L1+=self.elList[i].L
        for i in range(self.combinerIndex+1,len(self.elList)+1):
            if self.elList[i].ang!=0: #if some kind of bender (ideal,segmented,sim etc) then the end has been reached
                break
            L2 += self.elList[i].L
        r1=self.elList[self.benderIndices[0]].rb
        r2=self.elList[self.benderIndices[1]].rb
        inputAng=self.elList[self.combinerIndex].ang
        inputOffset=self.elList[self.combinerIndex].inputOffset
        phi2,phi3,L3=self.solve_Triangle_Problem(inputAng,inputOffset,L1,L2,r1,r2)
        print(inputOffset)
        return phi2,phi3,L3


    def solve_Triangle_Problem(self,inputAng,inputOffset,L1,L2,r1,r2):
        #the triangle problem refers to two circles and a kinked section connected by their tangets. This is the situation
        #with the combiner and the two benders.
        #phi1: bending angle of the combiner, or the amount of 'kink'
        #L1: length of the section before the kink to the bender
        #L2: length of the section after the kink to the bender
        #offset: offset of input trajectory to y=0 and input plane at field section of vacuum.
        #r1: radius of circle after the combiner
        #r2: radius of circle before the combiner
        #note that L1 and L2 INCLUDE the sections in the combiner. This function will likely be used by another function
        #that sorts that all out.
        L1 +=- inputOffset * np.sin(inputAng) + inputOffset / np.sin(inputAng)
        L2+=-inputOffset/np.tan(inputAng)


        L3=np.sqrt((L1-np.sin(inputAng)*r2+L2*np.cos(inputAng))**2+(L2*np.sin(inputAng)-r2*(1-np.cos(inputAng))+(r2-r1))**2)
        phi2=np.pi*1.5-np.arctan(L2/r2)-np.arccos((L3**2+L2**2-L1**2)/(2*L3*np.sqrt(L2**2+r2**2))) #bending radius of bender
        #after combiner
        phi3=np.pi*1.5-np.arctan(L1/r1)-np.arccos((L3**2+L1**2-L2**2)/(2*L3*np.sqrt(L1**2+r1**2)))#bending radius of bender
        #before combiner
        return phi2,phi3,L3

    def make_Geometry(self):
        #construct the shapely objects used to plot the lattice and to determine if particles are inside of the lattice.
        #it could be changed to only use the shapely objects for plotting, but it would take some clever algorithm I think
        #and I am in a crunch kind of.
        #----
        #all of these take some thinking to visualize what's happening.
        benderPoints=500 #how many points to represent the bender with along each curve
        for el in self.elList:
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
        #kinds. This class is not meant to have tons of error handling, so user must be cautious
        if self.elList[0].type=='BENDER_IDEAL' or self.elList[0].type=="BENDER_IDEAL_SEGMENTED"\
                or self.elList[0].type=="BENDER_SIM_SEGMENTED": #first element can't be a bending element
            raise Exception('FIRST ELEMENT CANT BE A BENDER')
        if self.elList[0].type=='COMBINER': #first element can't be a combiner element
            raise Exception('FIRST ELEMENT CANT BE A COMBINER')
        if self.combinerIndex is not None:
            if self.combinerIndex>self.benderIndices[0]:
                raise Exception('COMBINER MUST BE BEFORE FIRST BENDER')
    def set_Element_Coordinates(self):
        #each element has a coordinate for beginning and for end, as well as a value describing it's rotation where
        #0 degrees is to the east and 180 degrees to the west. Each element also has a normal vector for the input
        #and output planes. The first element's beginning is at 0,0 with a -180 degree angle and each following element 
        # builds upon that. The final element's ending coordinates must match the beginning elements beginning coordinates
        i=0
        rbo=np.asarray([0,0]) #beginning of first element, vector. Always 0,0
        reo=None #end of last element, vector. This is not the geometri beginning but where the orbit enters the element,
        #it could be the geometric point though

        for el in self.elList: #loop through elements in the lattice
            if i==0: #if the element is the first in the lattice
                xb=0.0#set beginning coords
                yb=0.0#set beginning coords
                el.theta=np.pi #first element is straight. It can't be a bender
                xe=el.L*np.cos(el.theta) #set ending coords
                ye=el.L*np.sin(el.theta) #set ending coords
                el.nb=-np.asarray([np.cos(el.theta),np.sin(el.theta)]) #normal vector to input
                el.ne=-el.nb
            else: #if element is not the first
                xb=self.elList[i-1].r2[0]#set beginning coordinates to end of last
                yb=self.elList[i-1].r2[1]#set beginning coordinates to end of last
                prevEl = self.elList[i - 1]
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
            if i==len(self.elList)-1: #if the last element then set the end of the element correctly
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

        #if deltax>smallNum or deltay>smallNum:
        #    raise Exception('ENDING POINTS DOES NOT MEET WITH BEGINNING POINT. LATTICE IS NOT CLOSED')


    def compute_Input_Angle_And_Offset(self,el,h=1e-6):
        #this computes the output angle and offset for a combiner magnet
        #L: length of combiner magnet
        #c1: dipole moment
        #c2: quadrupole moment
        #ap: apeture of combiner in x axis, half gap
        #v0: nominal particle speed
        q = np.asarray([0, 0, 0])
        p=self.m*np.asarray([self.v0Nominal,0,0])
        #xList=[]
        #yList=[]
        while True:
            F=el.force(q)
            a=F/self.m
            q_n=q+(p/self.m)*h+.5*a*h**2
            F_n=el.force(q_n)
            a_n = F_n / self.m  # acceleration new or acceleration sub n+1
            p_n=p+self.m*.5*(a+a_n)*h
            if q_n[0]>el.Lb: #if overshot, go back and walk up to the edge assuming no force
                dr=el.Lb-q[0]
                dt=dr/(p[0]/self.m)
                q=q+(p/self.m)*dt
                break
            #xList.append(q[0])
            #yList.append(q[1])
            q=q_n
            p=p_n
        outputAngle = np.arctan2(p[1], p[0])
        outputOffset = q[1]
        return outputAngle,outputOffset
    def show_Lattice(self,particleCoords=None):
        #plot the lattice using shapely
        #particleCoords: Array or list holding particle coordinate such as [x,y]
        plt.close('all')
        for el in self.elList:
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

lattice=ParticleTracerLattice(200)
particleTracer=ParticleTracer(lattice)

Llens=.25

lattice.add_Lens_Ideal(Llens,1,.01)
lattice.add_Combiner_Ideal()
lattice.add_Lens_Ideal(Llens,1,.01)
lattice.add_Bender_Ideal(None,1,1,.01)
lattice.add_Lens_Ideal(None,1,.01)
lattice.add_Bender_Ideal(None,1,1,.01)
lattice.end_Lattice()
#lattice.show_Lattice()


q0=np.asarray([-1e-10,1e-3,0])
v0=np.asarray([-200.0,0,0])

Lt=500e-2

dt=10e-6
q, p, qo, po, particleOutside = particleTracer.trace(q0, v0,dt, Lt/200)
print(particleOutside)
plt.plot(qo[:,0],qo[:,1])
plt.show()

lattice.show_Lattice(particleCoords=q[-1])