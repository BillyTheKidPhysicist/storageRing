import numpy as np
import matplotlib.pyplot as plt
import numba
from shapely.geometry import Polygon
import scipy.interpolate as spi
import numpy.linalg as npl
from elementPT import LensIdeal,BenderIdeal,CombinerIdeal,BenderIdealSegmentedWithCap,BenderIdealSegmented,Drift \
    ,HalbachBenderSimSegmentedWithCap,HalbachLensSim,CombinerSim,BumpsLensIdeal,BumpsLensSimWithCaps
class SimpleLocalMinimizer:
    #for solving the implicit geometry problem. I found that the scipy solvers left something to be desired because
    #they operated under theh assumption that it is either local or global minimized. I want to from a starting point
    #find the minimum in a system that has no global minimum, but is rather more like a lumpy mattress. I would prefer
    #to have used a vanilla newton methods or gradient descent, but from scipy I was not able to select the appropriate
    #step size. Assumes the cost function goes to zero.
    def __init__(self,function):
        self.function=function
    def gradient(self,q0, dq):
        #find gradeint of the function
        x, y = q0
        dfx = (self.function(np.asarray([x + dq, y]))- self.function(np.asarray([x - dq, y]))) / (2*dq)
        dfy = (self.function(np.asarray([x, y + dq]))- self.function(np.asarray([x, y - dq]))) / (2*dq)
        return np.asarray([dfx, dfy])

    def estimate_Step_Size_Gradient(self,q0, fracReduction, eps=1e-10):
        #simple method that uses the gradient to make a tiny step in the direction of reduction of the cost function,
        #then uses that information to figure out how big the step should be to get to zero. Basically newton's method
        #without matrix inversion, and assuming that the cost function goes to zero.
        funcVal0 = self.function(q0)
        grad = self.gradient(q0, eps)
        grad = grad / npl.norm(grad)
        dqEps = -grad * eps  # tiny step size
        funcValNew = self.function(q0 - dqEps)  # look backwareds for the slope
        dFracFuncVal = -(funcValNew - funcVal0) / funcVal0
        dq = dqEps * fracReduction / np.abs(dFracFuncVal)
        return dq
    def solve(self,X0,tol=1e-10,maxIter=100):
        X = X0.copy()
        error = self.function(X0)
        i = 0
        while (error > tol):
            dq = self.estimate_Step_Size_Gradient(X, .5)
            X = X + dq
            error = self.function(X)
            i += 1
            if i > maxIter:
                break
        return X,error

class ParticleTracerLattice:
    def __init__(self,v0Nominal,latticeType='storageRing'):
        if latticeType!='storageRing' and latticeType!='injector':
            raise Exception('invalid lattice type provided')
        self.latticeType=latticeType#options are 'storageRing' or 'injector'. If storageRing, the geometry is the the first element's
        #input at the origin and succeeding elements in a counterclockwise fashion. If injector, then first element's input
        #is also at the origin, but seceeding elements follow along the positive x axis
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
        self.linearElementToConstraint=None #element whos length will be changed when the lattice is constrained to
        # satisfy geometry. Can only be one

        self.elList=[] #to hold all the lattice elements
    def find_Optimal_Offset_Factor(self,rp,rb,Lm,parallel=False):
        #How far exactly to offset the bending segment from linear segments is exact for an ideal bender, but for an
        #imperfect segmented bender it needs to be optimized.
        from ParticleClass import Particle
        from ParticleTracerClass import ParticleTracer
        from ParaWell import ParaWell
        numMagnetsHalfBend=int(np.pi*rb/Lm)
        #todo: this should be self I think
        PTL_Ring=ParticleTracerLattice(200.0,latticeType='injector')
        PTL_Ring.add_Drift(.05)
        PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rp,numMagnetsHalfBend,rb,rOffsetFact=1.0)
        PTL_Ring.end_Lattice(enforceClosedLattice=False,constrain=False)

        def errorFunc(rOffsetFact):
            PTL_Ring.elList[1].update_rOffset_Fact(rOffsetFact)
            PTL_Ring.build_Lattice()
            particle=Particle()
            particleTracer=ParticleTracer(PTL_Ring)
            particle=particleTracer.trace(particle,1e-5,1.0,fastMode=False)
            error=np.std(1e6*particle.qoArr[:,1])
            return error

        rOffsetFactArr=np.linspace(.5,1.5,12)
        if parallel==True:
            helper=ParaWell()
            vals=np.asarray(helper.parallel_Problem(errorFunc,rOffsetFactArr,onlyReturnResults=True))
        else:
            vals=np.asarray([errorFunc(rOffsetFact) for rOffsetFact in rOffsetFactArr])
        fit=spi.RBFInterpolator(rOffsetFactArr[:,np.newaxis],vals)
        rOffsetFactArrDense=np.linspace(rOffsetFactArr[0],rOffsetFactArr[-1],10000)
        newVals=fit(rOffsetFactArrDense[:,np.newaxis])
        # plt.plot(rOffsetFactArr,vals,marker='x')
        # plt.plot(rOffsetFactArrDense,newVals)
        # plt.show()
        return rOffsetFactArrDense[np.argmin(newVals)]  #optimal rOffsetFactor
    def set_Constrained_Linear_Element(self,el):
        if self.linearElementToConstraint is not None:
            raise Exception('There can only be one linear constrained element')
        else:
            self.linearElementToConstraint=el
    def add_Combiner_Sim(self,file,sizeScale=1.0):
        #file: name of the file that contains the simulation data from comsol. must be in a very specific format
        el = CombinerSim(self,file,self.latticeType,sizeScale=sizeScale)
        el.index = len(self.elList) #where the element is in the lattice
        self.combiner=el
        self.combinerIndex=el.index
        self.elList.append(el) #add element to the list holding lattice elements in order
    def add_Halbach_Lens_Sim(self,rp,Lm,apFrac=.8,constrain=False):
        el=HalbachLensSim(self, rp,Lm,apFrac)
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
        if constrain==True: self.set_Constrained_Linear_Element(el)
    def add_Bump_Lens_Sim_With_Caps(self, file2D, file3D,fringeFrac, L,sigma, ap=None,rp=None,constrain=False):
        el=BumpsLensSimWithCaps(self, file2D, file3D, fringeFrac,L, rp,ap,sigma)
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
        if constrain==True: self.set_Constrained_Linear_Element(el)
    def add_Lens_Ideal(self,L,Bp,rp,ap=None,constrain=False):
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
        el=LensIdeal(self, L, Bp, rp, ap) #create a lens element object
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
        if constrain==True: self.set_Constrained_Linear_Element(el)
    def add_Bump_Lens_Ideal(self,L,Bp,rp,sigma,ap=None):
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #L: Length of lens, m
        #Bp: field strength at pole face of lens, T
        #rp: bore radius of element, m
        #ap: size of apeture. If none then a fraction of the bore radius. Can't be bigger than bore radius. unitless
        #sigma: The transverse translation of the element that gives rise to the bumping of the beam
        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        el=BumpsLensIdeal(self, L, Bp, rp,sigma, ap) #create a lens element object
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Drift(self,L,ap=.03):
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #L: length of drift element, m
        #ap:apeture. Default value of 3 cm radius, unitless
        el=Drift(self,L,ap)#create a drift element object
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
    def add_Bender_Ideal_Segmented_With_Cap(self,numMagnets,Lm,Lcap,Bp,rp,rb,yokeWidth,space,rOffsetFact=1.0,ap=None):
        apFrac=.9 #apeture fraction
        if ap is None:#set the apeture as fraction of bore radius to account for tube thickness
            ap=apFrac*rp
        else:
            if ap > rp:
                raise Exception('Apeture cant be bigger than bore radius')
        el=BenderIdealSegmentedWithCap(self,numMagnets,Lm,Lcap,Bp,rp,rb,yokeWidth,space,rOffsetFact,ap)
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)
    def add_Halbach_Bender_Sim_Segmented_With_End_Cap(self,Lm,rp,numMagnets,rb,extraSpace=0.0,rOffsetFact=1.0,apFrac=None):
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #Lcap: Length of element on the end/input of bender
        #rOffsetFact: factor to multply the theoretical offset by to minimize oscillations in the bending segment.
        #modeling shows that ~.675 is ideal
        el = HalbachBenderSimSegmentedWithCap(self, Lm,rp,numMagnets,rb,extraSpace,rOffsetFact,apFrac)
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)
    def add_Bender_Ideal_Segmented(self,numMagnets,Lm,Bp,rp,rb,yokeWidth,space,rOffsetFact,ap=None):
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

        el=BenderIdealSegmented(self,numMagnets,Lm,Bp,rp,rb,yokeWidth,space,rOffsetFact,ap)
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)
    def add_Bender_Ideal(self,ang,Bp,rb,rp,ap=None):
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
    def add_Combiner_Ideal(self,Lm=.2,c1=1,c2=20,ap=.015,sizeScale=1.0):
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

        el=CombinerIdeal(self, Lm, c1, c2, ap,self.latticeType,sizeScale) #create a combiner element object
        el.index = len(self.elList) #where the element is in the lattice
        self.combiner = el
        self.combinerIndex=el.index
        self.elList.append(el) #add element to the list holding lattice elements in order


    def end_Lattice(self,constrain=False,enforceClosedLattice=True,buildLattice=True,
                    surpressWarning=False):
        #TODO: THIS WHOLE THING IS WACK, especially the contraint part
        #TODO: REALLY NEED TO CLEAN UP ERROR CATCHING
        #TODO: document which parameters mean what when using constrain!
        #TODO: THIS ALL NEEDS TO MAKE MORE SENSE
        #call various functions to construct the lattice geometry, the shapely objects for visualization and finding particle
        # positions, catch errors, enforce constraints and fill parameter(s)
        # constrain: wether to constrain the lattice by varying parameters. typically to accomodate the presence of the
        #combiner magnet
        # enfroceClosedLattice: Wether to throw an error when the lattice end point does not coincide with beginning point
        #track potential. Wether to have enable the element function that returns magnetic pontential at a given point.
        #there is a cost to keeping this enabled because of pickling time
        #latticeType: Wether lattice is 'storageRing' type or 'injector' type.

        if len(self.benderIndices) ==2:
            self.bender1=self.elList[self.benderIndices[0]]   #save to use later
            self.bender2=self.elList[self.benderIndices[1]] #save to use later
        # self.catch_Errors(constrain,buildLattice)
        if constrain==True:
            self.constrain_Lattice()
        if buildLattice==True:
            self.build_Lattice(enforceClosedLattice=enforceClosedLattice,surpressWarning=surpressWarning)
            self.totalLength=0
            for el in self.elList: #total length of particle's orbit in an element
                self.totalLength+=el.Lo
    def build_Lattice(self,enforceClosedLattice=True,surpressWarning=False):
        self.set_Element_Coordinates(enforceClosedLattice,surpressWarning)
        self.make_Geometry()
    def constrain_Lattice(self):
        #enforce constraints on the lattice. this comes from the presence of the combiner for now because the total
        #angle must be 2pi around the lattice, and the combiner has some bending angle already. Additionally, the lengths
        #between bending segments must be set in this manner as well
        params=self.solve_Combiner_Constraints()
        assert (isinstance(self.linearElementToConstraint,HalbachLensSim) or
               isinstance(self.linearElementToConstraint,LensIdeal)) and \
               self.linearElementToConstraint is not None
        if self.bender1.segmented==True:
            rb1, rb2, numMagnets1, numMagnets2, L3=params
            self.bender1.rb=rb1
            self.bender1.numMagnets=numMagnets1
            self.bender2.rb=rb2
            self.bender2.numMagnets=numMagnets2
            self.linearElementToConstraint.set_Length(L3)
            self.bender1.fill_Params_Post_Constrained()
            self.bender2.fill_Params_Post_Constrained()
        else:
            phi1,phi2,L3=params
            self.bender1.ang = phi1
            self.bender2.ang = phi2
            #update benders
            # Lfringe=4*self.elList[lens1Index].edgeFact*self.elList[lens1Index].rp
            self.linearElementToConstraint.set_Length(L3)
            self.bender1.fill_Params()
            self.bender2.fill_Params()
            self.linearElementToConstraint.fill_Params()
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
        # need to account for length of caps in bender
        params[-1]-=(self.bender1.Lcap+self.bender2.Lcap) #remove spurious length from straight section
        return params

    def solve_Implicit_Segmented_Triangle_Problem( self, inputAng, inputOffset, L1, L2,tol=1e-10):
        #this method solves the solve_Triangle_Problem subject to the constraint that the benders are made of segments
        #of magnets rather than one continuous extrusion. This confines the solution to a limited number of configurations.
        # This is done by creating a cost function that goes to zero when for a given configuration the integer number
        #of segments are required.
        #inputAng: input angle to combiner
        #inputOffset: input offset to combiner.
        # L1: length of the section after the kink to the next bender
        # L2: length of the section from first bender to kink
        # tol: acceptable tolerance on cost function for a solution

        if (self.bender1.Lseg != self.bender2.Lseg) or (self.bender1.yokeWidth != self.bender2.yokeWidth):
            raise Exception('SEGMENT LENGTHS AND YOKEWIDTHS MUST BE EQUAL BETWEEN BENDERS')
        if self.bender1.segmented != self.bender2.segmented:
            raise Exception('BENDER MUST BOTH BE SEGMENTED OR BOTH NOT')

        Lseg = self.bender1.Lseg #length of magnet, including spacing between neighbors
        yokeWidth = self.bender1.yokeWidth #width of magnets and holding structure radially
        rp = self.bender1.rp #bore radius, same for both benders
        D = rp + yokeWidth
        r10 = self.bender1.rb #nominal bending radius without including offset, ie center of bender
        r20 = self.bender2.rb #nominal bending radius without including offset, ie center of bender
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
                cost3=1#np.sqrt((r1-r10)**2+(r2-r20)**2)
                cost=np.sqrt(cost1**2+cost2**2)*cost3
                return cost
        X,error=SimpleLocalMinimizer(cost).solve(np.asarray([r10,r20]))
        params = cost(X, returnParams=True)
        params[2]=int(np.round(params[2]))
        params[3]=int(np.round(params[3]))
        if cost(params[:2])>tol:
            print(inputAng, inputOffset, L1, L2,r10,r20)
            raise Exception('FAILED TO SOLVE IMPLICIT TRIANGLE PROBLEM TO REQUESTED ACCURACY')
        return params

    @staticmethod
    @numba.njit(numba.float64[:](numba.float64,numba.float64,numba.float64,numba.float64,numba.float64,numba.float64))
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
        params=np.asarray([theta1,theta2,L3])
        return params
    def make_Geometry(self):
        #todo: refactor this whole thing
        #construct the shapely objects used to plot the lattice and to determine if particles are inside of the lattice.
        #Ideally I would never need to use this to find which particle the elment is in because it's slower.
        #This is all done in the xy plane.
        #----------
        #all of these take some thinking to visualize what's happening.
        benderPoints=250 #how many points to represent the bender with along each curve
        for el in self.elList:
            xb=el.r1[0]
            yb=el.r1[1]
            xe=el.r2[0]
            ye=el.r2[1]
            halfWidth=el.ap
            theta=el.theta
            if el.type=='STRAIGHT':
                q1Inner=np.asarray([xb-np.sin(theta)*halfWidth,yb+halfWidth*np.cos(theta)]) #top left when theta=0
                q2Inner=np.asarray([xe-np.sin(theta)*halfWidth,ye+halfWidth*np.cos(theta)]) #top right when theta=0
                q3Inner=np.asarray([xe+np.sin(theta)*halfWidth,ye-halfWidth*np.cos(theta)]) #bottom right when theta=0
                q4Inner=np.asarray([xb+np.sin(theta)*halfWidth,yb-halfWidth*np.cos(theta)]) #bottom left when theta=0
                pointsInner=[q1Inner,q2Inner,q3Inner,q4Inner]
                if el.outerHalfWidth is None:
                    pointsOuter=pointsInner.copy()
                else:
                    width=el.outerHalfWidth
                    if False:#el.fringeFrac is not None:
                        pass
                    else:
                        q1Outer=np.asarray([xb-np.sin(theta)*width,yb+width*np.cos(theta)])  #top left when theta=0
                        q2Outer=np.asarray([xe-np.sin(theta)*width,ye+width*np.cos(theta)])  #top right when theta=0
                        q3Outer=np.asarray([xe+np.sin(theta)*width,ye-width*np.cos(theta)])  #bottom right when theta=0
                        q4Outer=np.asarray([xb+np.sin(theta)*width,yb-width*np.cos(theta)])  #bottom left when theta=0
                        pointsOuter=[q1Outer,q2Outer,q3Outer,q4Outer]
                    
            elif el.type=='BEND':
                phiArr=np.linspace(0,-el.ang,num=benderPoints)+theta+np.pi/2 #angles swept out
                r0=el.r0.copy()
                xInner=(el.rb-halfWidth)*np.cos(phiArr)+r0[0] #x values for inner bend
                yInner=(el.rb-halfWidth)*np.sin(phiArr)+r0[1] #y values for inner bend
                xOuter=np.flip((el.rb+halfWidth)*np.cos(phiArr)+r0[0]) #x values for outer bend
                yOuter=np.flip((el.rb+halfWidth)*np.sin(phiArr)+r0[1]) #y values for outer bend
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
                pointsInner=np.column_stack((x,y)) #shape the coordinates and make the object
                if el.outerHalfWidth is None:
                    pointsOuter=pointsInner.copy()
            elif el.type=='COMBINER':
                apR=el.apR #the 'right' apeture. here this confusingly means when looking in the yz plane, ie the place
                #that the particle would look into as it revolves in the lattice
                apL=el.apL
                q1Inner=np.asarray([0,apR]) #top left ( in standard xy plane) when theta=0
                q2Inner=np.asarray([el.Lb,apR]) #top middle when theta=0
                q3Inner=np.asarray([el.Lb+(el.La-el.apR*np.sin(el.ang))*np.cos(el.ang),apR+(el.La-el.apR*np.sin(el.ang))*np.sin(el.ang)]) #top right when theta=0
                q4Inner=np.asarray([el.Lb+(el.La+el.apL*np.sin(el.ang))*np.cos(el.ang),-apL+(el.La+el.apL*np.sin(el.ang))*np.sin(el.ang)]) #bottom right when theta=0
                q5Inner=np.asarray([el.Lb,-apL]) #bottom middle when theta=0
                q6Inner = np.asarray([0, -apL])  # bottom left when theta=0
                pointsInner=[q1Inner,q2Inner,q3Inner,q4Inner,q5Inner,q6Inner]
                for i in range(len(pointsInner)):
                    pointsInner[i]=el.ROut@pointsInner[i]+el.r2[:2]
                if el.outerHalfWidth is None:
                    pointsOuter=pointsInner.copy()
            else:
                raise Exception('No correct element provided')
            el.SO=Polygon(pointsInner)
            el.SO_Outer=Polygon(pointsOuter)
    def catch_Errors(self,constrain,builLattice):
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
        if constrain==True:
            if type(self.bender1)!=type(self.bender2): #for continuous benders
                raise Exception('BENDERS BOTH MUST BE SAME TYPE')
            if self.combinerIndex is None:
                raise  Exception('THERE MUST BE A COMBINER')
            if len(self.benderIndices) != 2:
                raise Exception('THERE MUST BE TWO BENDERS')
            if self.combinerIndex>self.benderIndices[0]:
                raise Exception('COMBINER MUST BE BEFORE FIRST BENDER')

            if self.bender1.segmented==True:
                if (self.bender1.Lseg != self.bender2.Lseg) or (self.bender1.yokeWidth != self.bender2.yokeWidth):
                    raise Exception('SEGMENT LENGTHS AND YOKEWIDTHS MUST BE EQUAL BETWEEN BENDERS')
            if self.bender1.segmented != self.bender2.segmented:
                raise Exception('BENDER MUST BOTH BE SEGMENTED OR BOTH NOT')
        if builLattice==True and constrain==False:
            if self.bender1.ang is None or self.bender2.ang is None:
                raise Exception('BENDER ANGLE IS NOT SPECIFIED')
            if self.bender1.numMagnets is None and self.bender2.numMagnets is None:
                raise Exception('BENDER NUMBER OF MAGNETS IS NOT SPECIFIED')
            if type(self.bender1.numMagnets)!=type(self.bender2.numMagnets): #for segmented benders
                raise Exception('BENDERS BOTH MUST HAVE NUMBER OF MAGNETS SPECIFIED OR BE None')
    def set_Element_Coordinates(self,enforceClosedLattice,surpressWarning):
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
                if el.sigma is not None:
                    raise Exception('First element cannot be bump element')
                if self.latticeType=='storageRing' or self.latticeType=='injector':
                    el.theta=np.pi #first element is straight. It can't be a bender
                else:
                    el.theta=0.0
                xe=el.L*np.cos(el.theta) #set ending coords
                ye=el.L*np.sin(el.theta) #set ending coords
                el.nb=-np.asarray([np.cos(el.theta),np.sin(el.theta)]) #normal vector to input
                el.ne=-el.nb
            else: #if element is not the first
                xb=self.elList[i-1].r2[0]#set beginning coordinates to end of last
                yb=self.elList[i-1].r2[1]#set beginning coordinates to end of last
                prevEl = self.elList[i - 1]
                if el.sigma is not None:  # a bump element, need to ad transverse displacement
                    yb += el.sigma
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
                    el.theta = 2 * np.pi - el.ang - (
                                np.pi - prevEl.theta)  # Tilt the combiner down by el.ang so y=0 is perpindicular
                    # to the input. Rotate it 1 whole revolution, then back it off by the difference. Need to subtract
                    # np.pi because the combiner's input is not at the origin, but faces 'east'
                    el.theta = el.theta - 2 * np.pi * (el.theta // (2 * np.pi))  # the above logic can cause the element
                    # to have to rotate more than 360 deg
                    # to find location of output coords use vector that connects input and output in element frame
                    # and rotate it. Input is where nominal trajectory particle enters
                    drx = -(el.Lb + (el.La - el.inputOffset * np.sin(el.ang)) * np.cos(el.ang))
                    dry = -(el.inputOffset + (el.La - el.inputOffset * np.sin(el.ang)) * np.sin(el.ang))



                    el.r1El = np.asarray([0, 0])
                    el.r2El = -np.asarray([drx, dry])
                    dr = np.asarray([drx, dry])  # position vector between input and output of element in element frame
                    R = np.asarray([[np.cos(el.theta), -np.sin(el.theta)], [np.sin(el.theta), np.cos(el.theta)]])
                    dr = R @ dr  # rotate to lab frame
                    xe, ye = xb + dr[0], yb + dr[1]
                    el.ne = -np.asarray([np.cos(el.theta), np.sin(el.theta)])  # output normal vector
                    el.nb = np.asarray([np.cos(el.theta + el.ang), np.sin(el.theta + el.ang)])  # input normal vector


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
                if el.type=='BEND':
                    reo[1]-=el.rOffset
            i+=1
        #check that the last point matchs the first point within a small number.
        #need to account for offset.
        deltax=np.abs(rbo[0]-reo[0])
        deltay=np.abs(rbo[1]-reo[1])
        smallNum=1e-10
        if deltax > smallNum or deltay > smallNum:
            closed=False
        else:
            closed=True
        if enforceClosedLattice==True and closed==False and self.latticeType!='injector':
            print(deltax, deltay)
            raise Exception('ENDING POINTS DOES NOT MEET WITH BEGINNING POINT. LATTICE IS NOT CLOSED')
        elif enforceClosedLattice==False and closed==False and surpressWarning==False and self.latticeType!='injector':
            import warnings
            print('vector between ending and beginning',deltax, deltay)
            warnings.warn('ENDING POINTS DOES NOT MEET WITH BEGINNING POINT. LATTICE IS NOT CLOSED')
    def get_Element_Before_And_After(self,elCenter):
        if (elCenter.index==len(self.elList)-1 or elCenter.index==0) and self.latticeType=='injector':
            raise Exception('Element cannot be first or last if lattice is injector type')
        elBeforeIndex=elCenter.index-1 if elCenter.index!=0 else len(self.elList)-1
        elAfterIndex=elCenter.index+1 if elCenter.index<len(self.elList)-1 else 0
        elBefore=self.elList[elBeforeIndex]
        elAfter=self.elList[elAfterIndex]
        return elBefore,elAfter
    def show_Lattice(self,particleCoords=None,particle=None,swarm=None, showRelativeSurvival=True,showTraceLines=False,
                     showMarkers=True,traceLineAlpha=1.0,trueAspectRatio=True,extraObjects=None):
        #plot the lattice using shapely. if user provides particleCoords plot that on the graph. If users provides particle
        #or swarm then plot the last position of the particle/particles. If particles have not been traced, ie no
        #revolutions, then the x marker is not shown
        #particleCoords: Array or list holding particle coordinate such as [x,y]
        #particle: particle object
        #swarm: swarm of particles to plot.
        #showRelativeSurvival: when plotting swarm indicate relative particle survival by varying size of marker
        #showMarkers: Wether to plot a marker at the position of the particle
        #traceLineAlpha: Darkness of the trace line
        #trueAspectRatio: Wether to plot the width and height to respect the actual width and height of the plot dimensions
        # it can make things hard to see
        #extraObjects: List of shapely objects to add to the plot. Used for adding things like apetures. Limited
        #functionality right now
        plt.close('all')
        def plot_Particle(particle,xMarkerSize=1000):
            if particle.color is None: #use default plotting behaviour
                if particle.clipped==True:
                    color='red'
                elif particle.clipped==False:
                    color='green'
                else: #if None
                    color='blue'
            else: #else use the specified color
                color=particle.color
            if showMarkers==True:
                plt.scatter(*particle.q[:2], marker='x', s=xMarkerSize, c=color)
                plt.scatter(*particle.q[:2], marker='o', s=10, c=color)
            if showTraceLines==True:
                if particle.qArr.shape[0]!=0:
                    plt.plot(particle.qArr[:,0],particle.qArr[:,1],c=color,alpha=traceLineAlpha)

        for el in self.elList:
            plt.plot(*el.SO.exterior.xy,c='black')
        if particleCoords is not None: #plot from the provided particle coordinate
            if particleCoords.shape[0]==3: #if the 3d value is provided trim it to 2D
                particleCoords=particleCoords[:2]
            #plot the particle as both a dot and a X
            if showMarkers==True:
                plt.scatter(*particleCoords,marker='x',s=1000,c='r')
                plt.scatter(*particleCoords, marker='o', s=50, c='r')
        elif particle is not None: #instead plot from provided particle
            plot_Particle(particle)
        if swarm is not None:
            maxRevs=swarm.longest_Particle_Life_Revolutions()
            if maxRevs==0.0: #if it hasn't been traced
                maxRevs=1.0
            for particle in swarm:
                revs=particle.revolutions
                if revs is None:
                    revs=0
                if showRelativeSurvival==True:
                    plot_Particle(particle,xMarkerSize=1000*revs/maxRevs)
                else:
                    plot_Particle(particle)

        if extraObjects is not None: #plot shapely objects that the used passed through. SO far this has limited
            # functionality
            for object in extraObjects:
                plt.plot(*object.coords.xy,linewidth=1,c='black')


        plt.grid()
        if trueAspectRatio==True:
            plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.show()