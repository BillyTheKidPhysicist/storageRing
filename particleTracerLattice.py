import time
import numpy as np
import matplotlib.pyplot as plt
import pathos as pa
import sys
from shapely.geometry import Polygon,Point
import numpy.linalg as npl
from elementPT import LensIdeal,BenderIdeal,CombinerIdeal,BenderIdealSegmentedWithCap,BenderIdealSegmented,Drift \
    ,BenderSimSegmentedWithCap,LensSimWithCaps,CombinerSim
from ParticleTracer import ParticleTracer
import scipy.optimize as spo
from profilehooks import profile
import copy


class ParticleTracerLattice:
    def __init__(self,v0Nominal):
        self.v0Nominal = v0Nominal  # Design particle speed
        self.m_Actual = 1.1648E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009994E-24 # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.u0=self.u0_Actual/self.m_Actual #Adjusted value of bohr magneton, about equal to 800
        self.kb = 1.38064852E-23  # boltzman constant, SI

        self.benderIndices=[] #list that holds index values of benders. First bender is the first one that the particle sees
        #if it started from beginning of the lattice. Remember that lattice cannot begin with a bender
        self.combinerIndex=None #the index in the lattice where the combiner is
        self.totalLength=None #total length of lattice, m

        self.bender1=None #bender element object
        self.bender2=None #bender element object
        self.combiner=None #combiner element object

        self.elList=[] #to hold all the lattice elements
    @staticmethod
    def compute_Bending_Radius_For_Segmented_Bender(L, rp, yokeWidth, numMagnets, angle, space):
        # ucAng=angle/(2*numMagnets)
        rb = (L + 2 * space) / (2 * np.tan(angle / (2 * numMagnets))) + yokeWidth + rp
        ucAng1=np.arctan(((L+2*space)/2)/(rb-rp-yokeWidth))
        return rb

    def add_Combiner_Sim(self,file):
        #file: name of the file that contains the simulation data from comsol. must be in a very specific format

        el = CombinerSim(self,file)
        el.index = len(self.elList) #where the element is in the lattice
        self.combiner=el
        self.combinerIndex=el.index
        self.elList.append(el) #add element to the list holding lattice elements in order
    def add_Lens_Sim_With_Caps(self, file2D, file3D, L, ap=None):
        el=LensSimWithCaps(self, file2D, file3D, L, ap)
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
    def add_Lens_Ideal(self,L,Bp,rp,ap=None):
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
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
        el=LensIdeal(self, L, Bp, rp, ap=ap) #create a lens element object
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Drift(self,L,ap=.03):
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #L: length of drift element, m
        #ap:apeture. Default value of 3 cm radius, unitless
        el=Drift(self,L,ap)#create a drift element object
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
    def add_Bender_Ideal_Segmented_With_Cap(self,numMagnets,Lm,Lcap,Bp,rp,rb,yokeWidth,space,ap=None):
        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        el=BenderIdealSegmentedWithCap(self,numMagnets,Lm,Lcap,Bp,rp,rb,yokeWidth,space,ap)
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)
    def add_Bender_Sim_Segmented_With_End_Cap(self,fileSeg,fileCap,fileInternalFringe,Lm,numMagnets,rb,extraSpace,yokeWidth,ap=None):
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #Lcap: Length of element on the end/input of bender
        el = BenderSimSegmentedWithCap(self, fileSeg,fileCap,fileInternalFringe,Lm,numMagnets,rb,extraSpace,yokeWidth,ap)
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)
    def add_Bender_Ideal_Segmented(self,numMagnets,Lm,Bp,rp,rb,yokeWidth,space,ap=None):
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #L: hard edge Length of individual magnet.
        #Bp: Field strength at pole face
        # rb: nominal bending radius of element's centerline. Actual radius is larger because particle 'rides' a little
        #outside this, m
        #rp: Bore radius of element
        #yokeWidth: width of the yoke, but also refers to the width of the magnetic material
        #numMagnet: number of magnets in segmented bender
        #space: extra space from magnet holder in the direction of the length of the magnet. This effectively add length
        #to the magnet. total length will be changed by TWICE this value, the space is on each end
        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')

        el=BenderIdealSegmented(self,numMagnets,Lm,Bp,rp,rb,yokeWidth,space,ap)
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)
    def add_Bender_Ideal(self,ang,Bp,rb,rp,ap=None):
        #TODO: remove keyword args
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
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
        el=BenderIdeal(self, ang, Bp, rp, rb, ap) #create a bender element object
        el.index = len(self.elList) #where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el) #add element to the list holding lattice elements in order
    def add_Combiner_Ideal(self,Lm=.2,c1=1,c2=20,ap=.015):
        # Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #add combiner (stern gerlacht) element to lattice
        #La: input length of combiner. The bent portion outside of combiner
        #Lm:  hard edge length of the magnet, which is the same as the vacuum tube
        #ang: angle that particle enters the combiner at
        # offset: particle enters inner section with some offset
        #c1: dipole component of combiner
        #c2: quadrupole component of bender
        #check to see if inlet length is too short. The minimum length is a function of apeture and angle
        #minLa=ap*np.sin(ang)
        #if La<minLa:
        #    raise Exception('INLET LENGTH IS SHORTER THAN MINIMUM')

        el=CombinerIdeal(self, Lm, c1, c2, ap) #create a combiner element object
        el.index = len(self.elList) #where the element is in the lattice
        self.combiner = el
        self.combinerIndex=el.index
        self.elList.append(el) #add element to the list holding lattice elements in order


    def end_Lattice(self,constrain=True,enforceClosedLattice=True):
        #TODO: document which parameters mean what when using constrain!
        #call various functions to construct the lattice geometry, the shapely objects for visualization and finding particle
        # positions, catch errors, enforce constraints and fill parameter(s)
        # constrain: wether to constrain the lattice by varying parameters. typically to accomodate the presence of the
        #combiner magnet
        # enfroceClosedLattice: Wether to throw an error when the lattice end point does not coincide with beginning point

        if len(self.benderIndices) ==2:
            self.bender1=self.elList[self.benderIndices[0]]   #save to use later
            self.bender2 = self.elList[self.benderIndices[1]] #save to use later

        self.catch_Errors()
        if constrain==True:
            self.constrain_Lattice()
        self.set_Element_Coordinates(enforceClosedLattice=enforceClosedLattice)
        self.make_Geometry()
        self.totalLength=0
        for el in self.elList: #total length of particle's orbit in an element
            self.totalLength+=el.Lo
    def constrain_Lattice(self):
        #enforce constraints on the lattice. this comes from the presence of the combiner for now because the total
        #angle must be 2pi around the lattice, and the combiner has some bending angle already. Additionally, the lengths
        #between bending segments must be set in this manner as well

        if self.combinerIndex is not None:
            params=self.solve_Combiner_Constraints()
            if self.bender1.segmented==True:
                r1, r2, numMags1,numMags2,L3=params
                self.bender1.rb=r1
                self.bender2.rb=r2
                self.bender1.numMagnets=numMags1
                self.bender2.numMagnets=numMags2
            else:
                phi1,phi2,L3=params
                self.bender1.ang = phi1
                self.bender2.ang = phi2
            #update benders

            lens1Index=4
            #Lfringe=4*self.elList[lens1Index].edgeFact*self.elList[lens1Index].rp
            #L=L3-Lfringe

            L=L3-self.bender1.Lcap*2
            self.elList[lens1Index].set_Length(L)

            self.bender1.fill_Params()
            self.bender2.fill_Params()
            self.elList[lens1Index].fill_Params()
            #if L<0:
            #    raise Exception("L is less than zero")
        #for el in self.elList:
        #    el.fill_Params_And_Functions()
    def solve_Combiner_Constraints(self):
        #this solves for the constraint coming from two benders and a combiner. The bending angle of each bender is computed
        #as well as the distance between the two on the segment without the combiner. For a segmented bender, this solves
        #an implicit equation because the number of segments must be an integer. For continuously extruded benders
        #it is an explicit equation, courtesy of Jamie Hall. This is mostly a wrapper that handles the overhead, the actual
        #equations are contained in functions

        #find the distance from the kink in the combiner from the bender before it, and the distance to the bender after
        # (clockwise)
        L1=self.elList[self.combinerIndex].Lb+self.bender1.Lcap #from kink to next bender
        L2=self.elList[self.combinerIndex].La+self.bender1.Lcap #from previous bender to kink
        inputAng = self.elList[self.combinerIndex].ang
        inputOffset = self.elList[self.combinerIndex].inputOffset
        for i in range(self.combinerIndex+1,len(self.elList)+1):
            if self.elList[i].type=='BEND': #if some kind of bender (ideal,segmented,sim etc) then the end has been reached
                break
            L1 += self.elList[i].L
        for i in range(self.combinerIndex-1,-1,-1): #assumes that the combiner will be before the first bender in the
            #lattice element list
            L2+=self.elList[i].L
        if self.bender1.segmented==True: #both bender1 and bender2 are same type
            params=self.solve_Implicit_Segmented_Triangle_Problem(inputAng,inputOffset, L1, L2)
        else:
            params=self.solve_Triangle_Problem(inputAng, inputOffset, L1, L2, self.bender1.ro, self.bender2.ro)
        return params

    def solve_Implicit_Segmented_Triangle_Problem( self, inputAng, inputOffset, L1, L2,tol=1e-9):
        #this method solves the solve_Triangle_Problem subject to the constraint that the benders are made of segments
        #of magnets rather than one continuous extrusion. This confines the solution to a limited number of configurations.
        # This is done by creating a cost function that goes to zero when for a given configuration the integer number
        #of segments are required.
        #inputAng: input angle to combiner
        #inputOffset: input offset to combiner.
        # L1: length of the section after the kink to the next bender
        # L2: length of the section from first bender to kink
        # tol: acceptable tolerance on cost function for a solution
        Lseg = self.bender1.Lseg #length of magnet, including spacing between neighbors
        yokeWidth = self.bender1.yokeWidth #width of magnets and holding structure radially
        rp = self.bender1.rp #bore radius, same for both benders
        D = rp + yokeWidth
        r10 = self.bender1.rb #bending radius without including offset, ie center of bender
        r20 = self.bender2.rb #bending radius without including offset, ie center of bender
        def cost(args,returnParams=False):
            r1,r2 = args
            r1Offset=self.bender1.rOffsetFunc(r1) #must include offset in geometry calculation
            r2Offset=self.bender2.rOffsetFunc(r2) #must include offset in geometry calculation
            r1+=r1Offset
            r2+=r2Offset
            phi1,phi2,L3=self.solve_Triangle_Problem(inputAng,inputOffset,L1,L2,r1,r2)
            r1-=r1Offset #remove offset for final step of finding number of magnets
            r2-=r2Offset #remove offset for final step of finding number of magnets
            UCAng1=np.arctan(.5*Lseg/(r1-D))
            UCAng2 = np.arctan(.5 * Lseg / (r2 - D))
            N1=.5*phi1/UCAng1 #number of magnets, as a float
            N2=.5*phi2/UCAng2 #number of magnets, as a float
            if returnParams==True:
                return [r1,r2,N1,N2,L3]
            else:
                cost1=np.round(N1)-N1
                cost2=np.round(N2)-N2
                cost=np.sqrt(cost1**2+cost2**2)
                return cost
        ranges=[(.98*r10,1.02*r10),(.98*r20,1.02*r20)]
        x0=np.asarray([r10,r20])
        ##print(spo.approx_fprime(x0,cost,1e-10))
        #x=x0
        #eps0=1e-4
        #eps=eps0
        #num=100
        #epsArr=np.empty(num)
        #costArr=np.zeros(4)
        #cost0=cost(x0)
        #print(cost(x), x)
        #dxArr = np.asarray([[eps0, eps0], [-eps0, eps0], [eps0, -eps0], [-eps0, -eps0]])
        #costPrev=0
        #fact=1.0
        #i0=10
        #for i in range(num):
        #    for j in range(4):
        #        costArr[j]=cost(x+dxArr[j])
        #    dxArr = np.asarray([[eps, eps], [-eps, eps], [eps, -eps], [-eps, -eps]])
        #    x+=dxArr[np.argmin(costArr)]
        #    eps = fact*eps0 * np.sqrt(cost(x) / cost0)
        #    epsArr[i] = eps
        #    print(eps, np.min(costArr))
        #    if i>i0:
        #        checkArr=epsArr[i-5:i]
        #        if np.std(checkArr)<np.mean(checkArr)/100:
        #            fact=fact*.1
        #            i0=i0+10
        #            print('here')
        #print(cost(x),x)
        #sys.exit()

        #sol=spo.minimize(cost,x0,bounds=ranges,method='TNC')#,'xtol':1e-12})#,options={'ftol':1e-12})
        #sol=spo.differential_evolution(cost,ranges)
        #print(sol)
        #print(sol)
        #print(cost(sol.x,returnParams=True))
        #sys.exit()
        #print(sol.x[0],sol.x[1])
        #print(sol.x[0],sol.x[1])
        #roffset 0.0033327012609907503 0.0033327012609907503

        x=[0.9845681497497591,0.988114259813233 ]
        params = cost(x, returnParams=True)
        params[2]=int(np.round(params[2]))
        params[3]=int(np.round(params[3]))
        #print(params)
        #print(params[0],params[1],params,cost(x))
        #sys.exit()
        #print(cost(x))
        #sys.exit()
        #if cost(params[:2])>tol:
        #    raise Exception('FAILED TO SOLVE IMPLICIT TRIANGLE PROBLEM TO REQUESTED ACCURACY')
        return params
    @staticmethod
    #@njit()
    def solve_Triangle_Problem(inputAng,inputOffset,L1,L2,r1,r2):
        #the triangle problem refers to two circles and a kinked section connected by their tangets. This is the situation
        #with the combiner and the two benders. The geometry of the combiner requires an additional step because the orbit
        #does not go straight through the combiner input, but is shifted a small amount by an offset. This, combined with
        #the angle, is used to make the new equivalent orbit from which L1 and L2 are calculated
        #phi1: bending angle of the combiner, or the amount of 'kink'
        #L1: length of the section after the kink to the next bender
        #L2: length of the section from first bender to kink
        #offset: offset of input trajectory to y=0 and input plane at field section of vacuum.
        #r1: radius of circle after the combiner
        #r2: radius of circle before the combiner
        #note that L1 and L2 INCLUDE the sections in the combiner.

        L1 += -inputOffset / np.tan(inputAng)
        L2 +=- inputOffset * np.sin(inputAng) + inputOffset / np.sin(inputAng)


        theta3 = np.pi - inputAng
        a = theta3 - np.arctan(r1 / L1) - np.arctan(r2 / L2)
        B = np.sqrt(L1 ** 2 + r1 ** 2)
        C = np.sqrt(L2 ** 2 + r2 ** 2)
        A = np.sqrt(B ** 2 + C ** 2 - 2 * B * C * np.cos(a))
        L3 = np.sqrt(A ** 2 - (r2 - r1) ** 2)
        c = np.arccos(-(C ** 2 - B ** 2 - A ** 2) / (2 * A * B))
        alpha = np.pi / 2 - np.arctan(r1 / L1)
        tau = np.arctan((r2 - r1) / L3)
        theta1 = 2 * np.pi - np.pi / 2 - c - alpha - tau
        theta2 = 2 * np.pi - inputAng - theta1
        return theta1,theta2,L3

    def make_Geometry(self):
        #construct the shapely objects used to plot the lattice and to determine if particles are inside of the lattice.
        #Ideally I would never need to use this to find which particle the elment is in because it's slower
        #----------
        #all of these take some thinking to visualize what's happening.
        benderPoints=250 #how many points to represent the bender with along each curve
        for el in self.elList:
            xb=el.r1[0]
            yb=el.r1[1]
            xe=el.r2[0]
            ye=el.r2[1]
            ap=el.ap
            theta=el.theta
            if el.type=='STRAIGHT':
                q1=np.asarray([xb-np.sin(theta)*ap,yb+ap*np.cos(theta)]) #top left when theta=0
                q2=np.asarray([xe-np.sin(theta)*ap,ye+ap*np.cos(theta)]) #top right when theta=0
                q3=np.asarray([xe+np.sin(theta)*ap,ye-ap*np.cos(theta)]) #bottom right when theta=0
                q4=np.asarray([xb+np.sin(theta)*ap,yb-ap*np.cos(theta)]) #bottom left when theta=0
                el.SO=Polygon([q1,q2,q3,q4])
            elif el.type=='BEND':
                phiArr=np.linspace(0,-el.ang,num=benderPoints)+theta+np.pi/2 #angles swept out
                r0=el.r0.copy()
                xInner=(el.rb-ap)*np.cos(phiArr)+r0[0] #x values for inner bend
                yInner=(el.rb-ap)*np.sin(phiArr)+r0[1] #y values for inner bend
                xOuter=np.flip((el.rb+ap)*np.cos(phiArr)+r0[0]) #x values for outer bend
                yOuter=np.flip((el.rb+ap)*np.sin(phiArr)+r0[1]) #y values for outer bend
                if el.cap==True:
                    xInner=np.append(xInner[0]+el.nb[0]*el.Lcap,xInner)
                    yInner = np.append(yInner[0] + el.nb[1] * el.Lcap, yInner)
                    xInner=np.append(xInner,xInner[-1]+el.ne[0]*el.Lcap)
                    yInner = np.append(yInner, yInner[-1] + el.ne[1] * el.Lcap)

                    xOuter=np.append(xOuter,xOuter[-1]+el.nb[0]*el.Lcap)
                    yOuter = np.append(yOuter,yOuter[-1] + el.nb[1] * el.Lcap)
                    xOuter=np.append(xOuter[0]+el.ne[0]*el.Lcap,xOuter)
                    yOuter = np.append(yOuter[0] + el.ne[1] * el.Lcap,yOuter)
                x=np.append(xInner,xOuter) #list of x values in order
                y=np.append(yInner,yOuter) #list of y values in order

                el.SO=Polygon(np.column_stack((x,y))) #shape the coordinates and make the object
            elif el.type=='COMBINER':
                q1=np.asarray([0,ap]) #top left when theta=0
                q2=np.asarray([el.Lb,ap]) #top middle when theta=0
                q3=np.asarray([el.Lb+(el.La-el.ap*np.sin(el.ang))*np.cos(el.ang),ap+(el.La-el.ap*np.sin(el.ang))*np.sin(el.ang)]) #top right when theta=0
                q4=np.asarray([el.Lb+(el.La+el.ap*np.sin(el.ang))*np.cos(el.ang),-ap+(el.La+el.ap*np.sin(el.ang))*np.sin(el.ang)]) #bottom right when theta=0
                q5=np.asarray([el.Lb,-ap]) #bottom middle when theta=0
                q6 = np.asarray([0, -ap])  # bottom left when theta=0
                points=[q1,q2,q3,q4,q5,q6]
                for i in range(len(points)):
                    points[i]=el.ROut@points[i]+el.r2[:2]
                el.SO=Polygon(points)
            else:
                raise Exception('No correct element provided')
    def catch_Errors(self):
        #catch any preliminary errors. Alot of error handling happens in other methods. This is a catch all for other
        #kinds. This class is not meant to have tons of error handling, so user must be cautious
        if self.elList[0].type=='BEND': #first element can't be a bending element
            raise Exception('FIRST ELEMENT CANT BE A BENDER')
        if self.elList[0].type=='COMBINER': #first element can't be a combiner element
            raise Exception('FIRST ELEMENT CANT BE A COMBINER')
        if len(self.benderIndices)==2: #if there are two benders they must be the same. There could be more benders, but
            #that is not dealth with here
            if type(self.bender1)!=type(self.bender2):
                raise Exception('BOTH BENDERS MUST BE THE SAME KIND')
        if self.combinerIndex is not None: #if combiner is present, must be before first bender
            if len(self.benderIndices)!=2:
                raise Exception('THERE MUST BE TWO BENDERS WITH THE COMBINER')
            if self.combinerIndex>self.benderIndices[0]:
                raise Exception('COMBINER MUST BE BEFORE FIRST BENDER')
            if self.bender1.segmented==True:
                if (self.bender1.Lseg!=self.bender2.Lseg) or (self.bender1.yokeWidth!=self.bender2.yokeWidth):
                    raise Exception('SEGMENT LENGTHS AND YOKEWIDTHS MUST BE EQUAL BETWEEN BENDERS')


    def set_Element_Coordinates(self,enforceClosedLattice=True):
        #each element has a coordinate for beginning and for end, as well as a value describing it's rotation where
        #0 degrees is to the east and 180 degrees to the west. Each element also has a normal vector for the input
        #and output planes. The first element's beginning is at 0,0 with a -180 degree angle and each following element 
        # builds upon that. The final element's ending coordinates must match the beginning elements beginning coordinates
        # enfroceClosedLattice: Wether to throw an error when the lattice end point does not coincide with beginning point
        i=0 #too keep track of the current element
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
                if el.type=='STRAIGHT':
                    if prevEl.type=='COMBINER':
                        el.theta = prevEl.theta-np.pi  # set the angle that the element is tilted relative to its
                        # reference frame. This is based on the previous element
                    else:
                        el.theta=prevEl.theta-prevEl.ang
                    xe=xb+el.L*np.cos(el.theta)
                    ye=yb+el.L*np.sin(el.theta)
                    el.nb = -np.asarray([np.cos(el.theta), np.sin(el.theta)])  # normal vector to input
                    el.ne = -el.nb #normal vector to end
                    if prevEl.type=='BEND' or prevEl.type=='LENS_SIM_CAP':
                        n=np.zeros(2)
                        n[0]=-prevEl.ne[1]
                        n[1]=prevEl.ne[0]
                        dr=n*prevEl.rOffset
                        xb+=dr[0]
                        yb+=dr[1]
                        xe+=dr[0]
                        ye+=dr[1]

                    el.r0=np.asarray([(xb+xe)/2,(yb+ye)/2,0]) #center of lens or drift is midpoint of line connecting beginning and end
                elif el.type=='BEND':
                    if prevEl.type=='COMBINER':
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
                    #TODO: INCLUDE TRAJECTORY OFFSET STRAIGHT AWAY?
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
                    el.r0=np.asarray([xb+el.rb*np.sin(el.theta),yb-el.rb*np.cos(el.theta),0]) #coordinates of center of bending

                    #section, even with caps
                    if el.cap==True:
                        xe+=-el.nb[0]*el.Lcap+el.ne[0]*el.Lcap
                        ye+=-el.nb[1] * el.Lcap+el.ne[1]*el.Lcap
                        el.r0[0]+=-el.nb[0]*el.Lcap
                        el.r0[1]+=-el.nb[1]*el.Lcap
                elif el.type=='COMBINER':
                    el.theta=2*np.pi-el.ang-(np.pi-prevEl.theta)# Tilt the combiner down by el.ang so y=0 is perpindicular
                    #to the input. Rotate it 1 whole revolution, then back it off by the difference. Need to subtract
                    #np.pi becaus the combiner's input is not at the origin, but faces 'east'
                    el.theta=el.theta-2*np.pi*(el.theta//(2*np.pi)) #the above logic can cause the element to have to rotate
                    #more than 360 deg
                    #to find location of output coords use vector that connects input and output in element frame
                    #and rotate it. Input is where nominal trajectory particle enters
                    drx=-(el.Lb+(el.La-el.inputOffset*np.sin(el.ang))*np.cos(el.ang))
                    dry=-(el.inputOffset+(el.La-el.inputOffset*np.sin(el.ang))*np.sin(el.ang))

                    #drx = -(el.Lb+el.La *np.cos(el.ang))
                    #dry = -(el.La*np.sin(el.ang))


                    dr=np.asarray([drx,dry]) #position vector between input and output of element in element frame
                    R = np.asarray([[np.cos(el.theta), -np.sin(el.theta)], [np.sin(el.theta), np.cos(el.theta)]])
                    dr=R@dr #rotate to lab frame
                    xe,ye=xb+dr[0],yb+dr[1]
                    el.ne=-np.asarray([np.cos(el.theta),np.sin(el.theta)]) #output normal vector
                    el.nb=np.asarray([np.cos(el.theta+el.ang),np.sin(el.theta+el.ang)]) #input normal vector

                else:
                    raise Exception('No correct element name provided')
            #need to make rotation matrices for element
            if el.type=='BEND':
                rot = (el.theta - el.ang + np.pi / 2)
            elif el.type=='STRAIGHT' or el.type=='COMBINER':
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


            el.r1=np.asarray([xb,yb,0])#position vector of beginning of element
            el.r2=np.asarray([xe,ye,0])#position vector of ending of element

            if i==len(self.elList)-1: #if the last element then set the end of the element correctly
                reo = el.r2.copy()
                if el.type=='BEND' or el.type=='COMBINER':
                    reo[1]-=el.rOffset
            i+=1
        #check that the last point matchs the first point within a small number.
        #need to account for offset.
        deltax=np.abs(rbo[0]-reo[0])
        deltay=np.abs(rbo[1]-reo[1])
        smallNum=1e-10
        #print(deltax,deltay)
        if deltax > smallNum or deltay > smallNum:
            closed=False
        else:
            closed=True
        if enforceClosedLattice==True and closed==False:
            print(deltax, deltay)
            raise Exception('ENDING POINTS DOES NOT MEET WITH BEGINNING POINT. LATTICE IS NOT CLOSED')
        elif enforceClosedLattice==False and closed==False:
            import warnings
            print(deltax, deltay)
            warnings.warn('ENDING POINTS DOES NOT MEET WITH BEGINNING POINT. LATTICE IS NOT CLOSED')
    def compute_Input_Angle_And_Offset(self,el,h=1e-6):
        #this computes the output angle and offset for a combiner magnet. These values need to be computed numerically
        #for numerical fields. For consistency, this is also used on the analytic hard edge combiner.
        q = np.asarray([0, 0, 0])
        p=np.asarray([self.v0Nominal,0,0])
        #xList=[]
        #yList=[]
        while True:
            F=el.force(q)
            a=F
            q_n=q+p*h+.5*a*h**2
            F_n=el.force(q_n)
            a_n = F_n  # acceleration new or acceleration sub n+1
            p_n=p+.5*(a+a_n)*h
            if q_n[0]>el.Lb: #if overshot, go back and walk up to the edge assuming no force
                dr=el.Lb-q[0]
                dt=dr/p[0]
                q=q+p*dt
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
    def measure_Phase_Space_Survival(self,qMax,vtMax,vlMax,Lt,h=1e-5,parallel=False):
        #this method populates and tests initial conditions in 5 deg phase space, there is always one particle at the origin
        #qMax: maximum distance in position phase space
        #vMax: maximum distance in velocity phase space
        #vlMax: maximum distance in velocity space along the longitudinal direction
        #num: number of point along each dimension in phase space. must be even so point doesn't end up at origin
        #todo: reuse argslist!!!
        #if num<=0:
        #    raise Exception('NUM MUST BE NOT ZERO and POSITIVE')
        qArr=np.linspace(-qMax,qMax,num=2)
        vtArr=np.linspace(-vtMax,vtMax,num=2)
        vlArr=np.linspace(-vlMax,vlMax,num=2)
        tempArr = np.asarray(np.meshgrid(qArr,qArr, vtArr, vtArr,vlArr)).T.reshape(-1, 5) #qx,qy,vx,vy
        argList = []
        for arg in tempArr:
            if arg[1]>0: #if the particle is in the upper half
                qi = np.asarray([-1e-10, arg[0], arg[1]])
                pi = np.asarray([-200+arg[4], arg[2], arg[3]])
                argList.append((qi, pi, h, Lt / 200, True)) #position, velocity, timestep,total time, fastmode
        elIndexArr=np.zeros(len(argList),dtype=int) #array of element indices where the particle was last
        survivalArr=np.zeros(len(argList))
        if parallel==True:
            particleTracer=ParticleTracer(self)
            results=particleTracer.multi_Trace(argList)
            for i in range(survivalArr.shape[0]):
                survivalArr[i]=results[i][0]/self.totalLength
                elIndexArr[i]=results[i][2]
        else:
            i=0
            for arg in argList:
                qi=arg[0]
                pi=arg[1]
                particleTracer = ParticleTracer(self)
                Lo,temp,currentElIndex=particleTracer.trace(qi,pi,h,Lt/200,fastMode=True)
                survivalArr[i]=(Lo/self.totalLength)
                elIndexArr[i]=currentElIndex
                #q, p, qo, po, particleOutside,elIndex = particleTracer.trace(qi, pi, h, Lt / 200)
                #self.show_Lattice(particleCoords=q[-1])
                i+=1
        return survivalArr,elIndexArr
    def inject_Into_Combiner(self,args,qi,pi,h=1e-6):
        #this models the phenomena of the particles originating at the focus of the big lens and traveling through
        #a injecting lens, then through the combiner. This is done in two steps. This is all done in the lab frame
        pass
    def trace_Through_Injector(self,argList,Lo,Li,Bp=.25,rp=.03):
        #models particles traveling through an injecting element
        #for now a thin lens
        #qi: intial particle coords
        #vi: initial velocity coords
        #Lo: object distance for injector
        #f: focal length
        K=2*self.u0*Bp/(rp**2*self.v0Nominal**2)

        #now find the magnet length that gives Li. Need to parametarize each entry of the transfer matrix
        CFunc= lambda x : np.cos(np.sqrt(K)*x)
        SFunc = lambda x: np.sin(np.sqrt(K)*x)/np.sqrt(K)
        CdFunc= lambda x: -np.sqrt(K)*np.sin(np.sqrt(K)*x)
        SdFunc=lambda x :np.cos(np.sqrt(K)*x)
        LiFunc= lambda x: -(CFunc(x)*Lo+SFunc(x))/(CdFunc(x)+SdFunc(x))
        minFunc=lambda x: (LiFunc(x)-Li)**2
        sol=spo.minimize_scalar(minFunc,method='bounded',bounds=(1e-3,.25))
        Lm=sol.x

        MLens=np.asarray([[CFunc(Lm),SFunc(Lm)],[CdFunc(Lm),SdFunc(Lm)]])
        MLo=np.asarray([[1,Lo],[1,0]])
        MLi=np.asarray([[1,Li],[1,0]])
        MTot=MLi@MLens@MLo
        for arg in argList:
            q=arg[0]
            p=arg[1]
            qNew=q.copy()
            pNew=p.copy()
            qNew[0]=MTot[0,0]*q[0]+MTot[0,1]*p[0]
            pNew[0]=MTot[1,0]*q[0]+MTot[1,1]*p[0]

            pNew[1] = MTot[0, 0] * q[1] + MTot[0, 1] * p[1]
            qNew[1] = MTot[1, 0] * q[1] + MTot[1, 1] * p[1]


def main():

    lattice = ParticleTracerLattice(200.0)
    fileBend1='benderSeg1.txt'
    fileBend2 = 'benderSeg2.txt'
    fileBender1Fringe='benderFringeCap1.txt'
    fileBenderInternalFringe1='benderFringeInternal1.txt'
    fileBender2Fringe='benderFringeCap2.txt'
    fileBenderInternalFringe2='benderFringeInternal2.txt'
    file2DLens='lens2D.txt'
    file3DLens='lens3D.txt'
    fileCombiner='combinerData.txt'
    yokeWidth=.0254*5/8
    numMagnets=110
    extraSpace=1e-3
    Lm=.0254
    rp=.0125
    Llens1=.3
    rb=1.0




    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
    lattice.add_Combiner_Sim(fileCombiner)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend1,fileBender1Fringe,fileBenderInternalFringe1,Lm,None,rb,extraSpace,yokeWidth)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, None)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend2,fileBender2Fringe, fileBenderInternalFringe2, Lm, None, rb, extraSpace,yokeWidth)
    lattice.end_Lattice()
    #lattice.show_Lattice()


    lattice.elList[2].forceFact = .175
    lattice.elList[4].forceFact = .175

    q0=np.asarray([-1e-10,-1e-3,1e-3])
    v0=np.asarray([-201.0,-1.0,-1.0])
    h=10e-6
    Lt=100
    particleTracer = ParticleTracer(lattice)

    #q, p, qo, po, particleOutside,elIndex = particleTracer.trace(q0, v0, h, Lt / 200)
    #print(qo[-1]) #[ 3.19866564e+01  0.00000000e+00 -1.64801837e-04]
    #sys.exit()
    #lattice.show_Lattice(particleCoords=q[-1])
    #plt.plot(qo[:,1])
    #plt.show()


    particleTracer = ParticleTracer(lattice)
    tList=[]
    for j in range(25):
        t=time.time()
        for i in range(10):
            particle=particleTracer.trace(q0, v0, 1e-5, Lt / 200,fastMode=True)
        tList.append((time.time()-t)/10.0)
    tArr=np.asarray(tList)
    print(np.mean(tArr),np.std(tArr)) #0.2599016189575195 0.006449517871561657
    #print(particle.cumulativeLength,particle.p,particle.pi)
    #plt.plot(particle.qoArr[:,0],particle.qoArr[:,1])
    #plt.show()
    #print(particle.lengthTraveled,particle.clipped) #(23.943571330550235, True, 1)
    sys.exit()
    def func(arg, parallel=True):
        X, newLattice = arg
        F1, F2 = X
        newLattice.elList[2].forceFact = F1
        newLattice.elList[4].forceFact = F2

        h = 10e-6
        Lt = 10 * lattice.totalLength
        qMax = 1e-3
        vtMax = 1.0
        vlMax = 1.0

        q0=np.asarray([-1e-10,0,0])
        v0=np.asarray([-200.0,0,0])
        #particleTracer=ParticleTracer(lattice)
        temp1=None
        for i in range(5):
            temp1,temp2,temp3=particleTracer.trace(q0, v0, h, Lt / 200,fastMode=True)
        survival = temp1/lattice.totalLength#qo[-1, 0] / lattice.totalLength
        #temp1Arr,temp2Arr = newLattice.measure_Phase_Space_Survival(qMax, vtMax, vlMax, Lt, h=h, parallel=parallel)
        #survival = np.mean(temp1Arr)
        return survival#,np.bincount(temp2Arr), X
    print(func([[.1666,.23333333],lattice]))# (3.440503668588188, array([0, 5, 9, 0, 0, 2], dtype=int64), [0.1666, 0.16666])
    sys.exit()
    # print('starting')
    # X=[0.19183673,0.44897959]
    # func(X,parallel=False)

    num = 4
    F1Arr = np.linspace(.1, 1, num=num)
    F2Arr = np.linspace(.1, 1, num=num)
    argsArr = np.asarray(np.meshgrid(F1Arr, F2Arr)).T.reshape(-1, 2)
    survivalList = []

    #t=time.time()
    #argsList = []
    #for arg in argsArr:
    #    argsList.append([arg, copy.deepcopy(lattice)])
    #print(time.time()-t)
    #sys.exit()
    pool = pa.pools.ProcessPool(nodes=pa.helpers.cpu_count())  # pool object to distribute work load
    jobs = []  # list of jobs. jobs are distributed across processors
    results = []  # list to hold results of particle tracing.
    print('starting')
    t = time.time()
    for arg in argsArr:
        jobs.append(pool.apipe(func, arg))  # create job for each argument in arglist, ie for each particle
    for job in jobs:
        results.append(job.get())  # get the results. wait for the result in order given
    print(time.time() - t)  # 795

    survivalList = []
    XList = []
    for result in results:
        survivalList.append(result[0])
        XList.append(result[1])
    survivalArr = np.asarray(survivalList)
    XArr = np.asarray(XList)
    # print(survivalArr)
    print(np.max(survivalArr))  # 44.42
    print(XArr[np.argmax(survivalArr)][0], XArr[np.argmax(survivalArr)][1])


if __name__=='__main__':
    pass
    main()