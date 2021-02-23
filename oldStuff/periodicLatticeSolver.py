import sympy as sym
import sympy.utilities.autowrap as symWrap
import numpy as np
from oldStuff.elementPLS import Element
import pathos as pa
import functools


class VariableObject:
    # class used to hold information about the variables made by the user when they call PLS.Variable().
    def __init__(self):
        self.varMin=None
        self.varMax=None
        self.varInit=None #initial value of variable
        self.symbol=None
        self.sympyObject=None
        self.elIndex=None #index of use of variables. Can be overwritten if reused.
        self.type=None # to hold wether the optic is assigned to lattice, injector or both
class Particle:
    def __init__(self,xi,xdi,deltaV,PLS):
        self.xi=xi #particle transverse position at focus
        self.xdi=xdi #particle transverse velocity at focus
        self.deltaV=deltaV #difference in longitudinal speed
        self.delta = self.deltaV / PLS.v0
        self.clippedx=[False,False] #wether the particle clips the x apeture. first entry is for left (outer) apeture and
                #second is for right (inner) apeture. delta greater/less than zero corresponds to outer/inner side
        self.clippedy=None #wether paticle clips in the y plane
class Injector:
    #this class creates an object that models the injection system which as of now (august 31,2020) would be composed of
    #the large permanent magnet collector and an electromagnet to shape the beam. The magnet's length, object distance
    #can drive parameters. In addition, there is a parameter that adds an offset to the object distance for mode matching
    #optimization. The optimal configuration may not be two lens whos focal length exactly meet at the image or object distance.
    def __init__(self,PLS):
        self.PLS=PLS
        self.M=None #to hold transfer matrix
        self.MFunc=None #to hold numpy function for matrix
        self.MVelCor=None #matrix that represent first order correction to velocity
        self.el=None #to hold element object
        self.Li=None #to hold sympy expression for image length of injector
        self.LiFunc=None #to hold the function that returns the image distance
        self.epsFunc=None #function that return value of emittance. First argument must be beta
        self.epsSampleFunc=None #A with xi and xdi as input paramters
        self.rp=None #To hold the value of the radius of the injector bore
        self.rpFunc=None #function that return the radius of the shaper
        self.sympyVarList=[] #list of sympy variables used in injector
        self.Lo = sym.symbols('L0', real=True, positive=True,nonzero=True)
        self.Lm=sym.symbols('Lm',real=True,positive=True,nonzero=True)
        self.sOffset=sym.symbols('s_offset',real=True) #this is for mode matching. it is the offset from the focus of the
            #collector magnet. I believe it is advantageous to sometimes not use the image of the collector as the
            #object of the shaper, but rather offset it. Positive values correspond to shifting the shaper
            #away from the nozzle
        self.Bp=.45
        self.sympyVarList=[self.Lo,self.Lm,self.sOffset]
        maxLen=2.5

        #----see above comments for better descriptions of below bounds
        self.LoMin=.1 #minimum distance from collector focus to face of shaper, ie object distance
        self.LoMax=.2 #maximum distance from collector focus to face of shaper, ie object distance
        self.LiMin=1.1 #min distance from back of shaper to focus of shaper, ie image distance
        self.LiMax=maxLen #max distance from back of shaper to focus of shaper, ie image distance
        self.LmMin=.01 #min length of magnet
        self.LmMax=.25 #max length of magnet
        self.LtMin=self.LoMin+self.LiMin+self.LmMin #min length of total system
        self.LtMax=maxLen #max length of total system
        self.sOffsetMax=.05 #max offset of mode matching. positive is away from the nozzle
        self.sOffsetMin=-.05 #min offset of mode matching. negative is towards from the nozzle


        #---these three paremeters, thetaMax, riMax, and sigma are used to construct a realistic lens. Thetamax is the
        #maximum expected input anlge. ri is the maximumu expected offset from axis at the point of the focus of the
        # collector. sigma is the fraction of the apeture to use. bigger thetaMax, riMax and smaller sigma are more difficult to
        #achieve. smaller sigma increases field quality because particle will be further from the poles
        self.thetaMax=.07 #maximum expected angle that the shaper could collect
        self.riMax = .002  # safe offset for object transverse displacement
        self.sigma=.9 #the fraction of the bore that will be used for particle trajectories.

        #I exploit symmetry here. Looking at eps notice that -x,xd and x,-xd have the same value.
        #Notice that delta has a symmetry also because the env is +/-(sqrt(beta*eps)+delta*eta)
        self.numParticles = 0
        self.numParticlesx=0
        self.numParticlesy=0
        self.particles=[]
        
        #maximum values for the population of particles to use in the injector
        xMax=.003 #maximum offset in x (and y) plane, mm
        xdMax=10 #maximum transverse velocity in the x (and y) plane, m/s
        deltaVMax=7.5 #maximum longitudinal velocity difference from the nominal 200 m/s
        x=np.linspace(-xMax,xMax,num=6)
        xd=np.linspace(0,xdMax,num=6)
        v0=np.linspace(0,deltaVMax,num=6) #total longitudinal velocity
        temp=np.meshgrid(x,xd,v0)
        self.xiArr=temp[0].flatten()
        self.xdiArr = temp[1].flatten()
        self.deltaVArr = temp[2].flatten()
        argsArr=np.transpose(np.row_stack((self.xiArr,self.xdiArr,self.deltaVArr)))# (3 x number of particles) dimension
            #array where each row is the xi,xdi, and delta of a given particle
        i=0
        for item in argsArr:
            if item[0]==0 or item[1]==0: #don't include particles with zero position or velocity
                i+=1
                pass
            else:
                self.particles.append(Particle(*item,self.PLS))
                #I exploit symmetry with delta so for every particle with a given delta I consider there to be 2. In addition
                #I consider each particle in the x and y plane.
                if item[2]!=0: #if delta does not equal zero
                    self.numParticlesx+=2
                else: #if it does equal zero, then only one particle
                    self.numParticlesx+=1
                self.numParticlesy += 1
        self.numParticles=self.numParticlesx+self.numParticlesy
    def fill_Parameters_And_Functions(self):
        #fill various functions and parameters of the injector.

        #its convenient to attach these values to the injector rather than the element
        self.M=self.el.M
        self.MVelCor=self.el.MVelCor
        self.Li=self.el.Li
        self.rp=self.el.rp

        self.MFunc=symWrap.autowrap(self.M,args=self.sympyVarList) #function that return numeric valued transfer matrix
        self.LiFunc=sym.lambdify(self.sympyVarList,self.el.Li) #function that return numeric value for shaper image distance
        self.rpFunc = symWrap.autowrap(self.rp, args=self.sympyVarList) #function that return numeric value for bore
            #radius of shaper. This account for wanting to use only self.sigma of the apeture


        args=self.sympyVarList.copy()
        A = self.M[0, 0] #0,0 entry of transfer matrix, sympy version
        C = self.M[1, 0] #1,0 entry of transfer matrix, sympy version
        D = self.M[1, 1] #1,1 entry of transfer matrix, sympy version
        ACor = self.MVelCor[0, 0]  # component of the correction matrix for particle speed, sympy version
        CCor = self.MVelCor[1, 0]  # component of the correction matrix for particle speed, sympy version
        DCor = self.MVelCor[1, 1]  # component of the correction matrix for particle speed, sympy version

        #----convert above sympy expressions into fast fortran/c code
        funcA=symWrap.autowrap(A,args=args)
        funcC = symWrap.autowrap(C, args=args)
        funcD = symWrap.autowrap(D, args=args)
        funcACor=symWrap.autowrap(ACor,args=args)
        funcCCor = symWrap.autowrap(CCor, args=args)
        funcDCor = symWrap.autowrap(DCor, args=args)


        #now construct a function that returns a list of emittance values for each particle.
        def temp(x,beta,alpha):
            sOffset=x[2] #offset from focal plane
            A0=funcA(*x)   #0,0 entry of transfer matrix
            C0 = funcC(*x) #1,0 entry of transfer matrix
            D0 = funcD(*x) #1,1 entry of transfer matrix
            ACor0 = funcACor(*x) #correction factors to transfer matrix for particle speed
            CCor0 = funcCCor(*x) #correction factors to transfer matrix for particle speed
            DCor0 = funcDCor(*x) #correction factors to transfer matrix for particle speed
            epsList=[]
            for particle in self.particles:
                xi=particle.xi
                xdi=particle.xdi/self.PLS.v0 #convert to angles
                xi=xdi*sOffset+xi
                deltaV=particle.deltaV
                xf=(A0+ACor0*deltaV)*xi
                xdf=(C0+CCor0*deltaV)*xi+(D0+DCor0*deltaV)*xdi
                eps=(xf**2+(beta*xdf)**2+(alpha*xf)**2+2*alpha*xdf*xf*beta)/beta
                epsList.append(eps) #this can't go negative. I've tested for many values and by inspection it seems
                    #like it can't. It could be proven no doubt
            return epsList
        self.epsFunc=temp

class PeriodicLatticeSolver:
    def __init__(self,v0,T,axis='both',catchErrors=True):
        #v0: nominal speed of particles
        #T: temperature of particles
        #axis: which axis to model. Typically both
        #catchErrors: Wether to stop the program when certain errors occur that it would otherwise catch. Disabling this
            #has been helpful in the past
        if axis!='x' and axis!='y' and axis!='both': #check wether a valid axis was provided
            raise Exception('INVALID AXIS PROVIDED!!')
        else:
            self.axis=axis #wether we're looking at the x or y axis or both. x is horizontal
        self.v0=v0 #nominal speed of atoms, usually 200 m/s
        self.T=T #temperature of atoms
        self.catchErrors=catchErrors
        self.trackLength=None #total track length of the lattice
        #TL1 and TL2 are used to find the correct angles of the benders to accomdate the bending angle of the combiner
        self.TL1=None #tracklength of section 1, the section from last bender to end of combiner.
        self.TL2=None #tracklength of section 2, the section from end of combiner to beginning of next bender
        self.m = 1.1648E-26 #mass of lithium 7, SI
        self.u0 = 9.274009994E-24 #bohr magneton, SI
        self.kb=1.38064852E-23 #boltzman constant, SI
        self.began=False #check if the lattice has begun
        self.lattice = [] #list to hold lattice magnet objects
        self.benderIndices=[] #list of locations of bends in the lattice
        self.lensIndices=[]
        self.combinerIndex=None #index of combiner's location
        self.delta=np.round(np.sqrt(self.kb*(self.T)/self.m)/self.v0,4) #sigma of velocity spread
        self.numElements = 0 #to keep track of total number of elements. Gets incremented with each additoin
        self.totalLengthListFunc = None  # each entry is the length of that element plus preceding elements.
                                    # arguments are variables declared by user in order user declared them
        self.lengthListFunc = None  # each entry is the length of that element. arguments are variables declared by user
                                    #in order user declared them
        self.sympyVarList = [] #list of sympy object variables. This list is filled in the order that the user
            #declared them
        self.VOList=[] #list of VariableObject objects. These are used to help with interactive plotting
        self.MTot = None #total transfer matrix, can contain sympy symbols
        self.MTotFunc = None#numeric function that returns total transfer matrix based on input arguments
        self.MListFunc=None #returns a list of numeric matrix values
        self.injector = None  # to hold the injector object
        self.emittancex=None #to hold emittance value in x direction.
        self.emittancey=None #to hold emittance value in y direction

    def mathVariable(self,variable,fact):
        #this creates variable that is a previous variable times a factor
        temp=variable.sympyObject*fact
        VO = VariableObject()
        VO.sympyObject=temp
        return VO


    def Variable(self, symbol,varMin=0.0,varMax=1.0,varInit=None):
        #function that is called to create a variable to use. Really it jsut adds things to list, but to the user it looks
        #like a variable
        #symbol: string used for sympy symbol
        sympyVar=sym.symbols(symbol,real=True)
        VO=VariableObject()
        VO.sympyObject=sympyVar
        VO.symbol=symbol
        VO.varMin=varMin
        VO.varMax=varMax
        if varInit==None:
            VO.varInit=(varMin+varMax)/2
        else:
            VO.varInit=varInit
        self.VOList.append(VO)
        self.sympyVarList.append(sympyVar)
        return VO
    def unpack_VariableObjects(self,args):
    #to unpack sympy from VariableObject before sending to Element class. also does error checking and some other
    #things for VariableObject
        for i in range(len(args)): #extract the sympy object from variable to send to Element class. Also note element index
                                    #for later use
            if isinstance(args[i], VariableObject):
                #if args[i].type!=latticetype and args[i].type!=None:
                #    raise Exception('IMPROPER VARIABLE OBJECT USE')
                #args[i].type=latticetype
                args[i]=args[i].sympyObject #replace the argument with the sympy symbol
        return args

    def add_Injector(self):
        #add an injector. This doesn't go into the list of elements though. It's not even a single element, it's more like
            #a mindset maaaan
        #L: length of magnet in injection system
        #Lo: distance from image to front of magnet in injection system
        #Bp: magnetic field strength at pole in magnet
        #rp: radius of lens
        #sigma: fraction of inner bore to use
        #r0: maximum incoming particle offset. This forces the bore to be large
        self.injector=Injector(self)
        if self.numElements>0:
            raise Exception('INJECTOR MUST BE ADDED FIRST')
        args=[self.injector.Lm,self.injector.Lo,self.injector.Bp,self.injector.thetaMax,self.injector.sigma,self.injector.riMax]
        self.injector.el=Element(self,'INJECTOR',args)
        self.injector.fill_Parameters_And_Functions()

    def add_Lens(self,L, Bp, rp,S=None):
        #add a magnetic lens, ie hexapole magnet
        #L: Length of lens
        #Bp: field strength at pole face
        #rp: radius of bore inside magnet
        self.numElements += 1
        args = self.unpack_VariableObjects([L, Bp, rp,S])
        el = Element(self, 'LENS',args)

        self.lattice.append(el)
        el.index = self.numElements - 1
        self.lensIndices.append(el.index)

    def add_Bend(self, angle, rb,Bp,rp=None,S=None):
        #add a bending magnet. A hexapole magnet
        #angle: bending angle, radians
        #rb: bending radius, m
        #Bp: field strength at poles, T
        #rp: Bore radius of hexapole. If None, the bore radius that maximizes capture is calculated
        self.numElements += 1
        self.benderIndices.append(self.numElements-1)
        args = self.unpack_VariableObjects([angle,rb,Bp,rp,S])
        el = Element(self,'BEND',args)
        self.lattice.append(el)
        el.index = self.numElements - 1

    def add_Drift(self, L=None):
        #add a drift section.
        #L: length of drift region
        self.numElements += 1
        if L is None:
            el = Element(self, 'DRIFT',[L])
        else:
            args = self.unpack_VariableObjects([L])
            el = Element(self,'DRIFT', args)
        self.lattice.append(el)
        el.index = self.numElements - 1
    def add_Combiner(self,L=.2,alpha=1.01,beta=22,S=None):
        #add combiner magnet. This is the 'collin' magnet
        #L: length of combiner, current length for collin mag is .187
        #alpha: dipole term
        #beta: quadrupole term
        #NOTE: the form of the potential here quadrupole does not have the 2. ie Vquad=beta*x*y
        self.numElements += 1
        args = self.unpack_VariableObjects([L,alpha,beta,S])
        rb = self.m * self.v0 ** 2 / (self.u0 * beta) #bending radius of the combiner. The combiner is basically a bender
        args.append(rb)
        el = Element(self,'COMBINER',args)
        self.lattice.append(el)
        self.combinerIndex=self.numElements-1
        el.index = self.numElements - 1

    def _compute_M_Total(self): #computes total transfer matric. numeric or symbolic
        M=sym.eye(5) #starting matrix, identity
        for i in range(self.numElements):
            M=self.lattice[i].M @ M #remember, first matrix is on the right!
        return M
    def set_Track_Length(self,trackLength=None,TL1=None,TL2=None):
        #TODO: THIS DOESN'T MAKE SENSE
        #this is to set the length of the straight awayas between bends. As of now this is the same for each
        # straight away

        if TL1==None and TL2==None:
            self.trackLength=self.unpack_VariableObjects([trackLength])[0]
        elif TL1!=None and TL2!=None:
            self.TL1,self.TL2=self.unpack_VariableObjects([TL1,TL2])
            self.trackLength=self.TL1+self.TL2
        else:
            raise Exception('YOU MUST SPECIFY BOTH TRACK LENGTHS OR THE TOTALLENGTH, BUT NOT BOTH')


    def _catch_Errors(self):
        if self.began==False:
            raise Exception("YOU NEED TO BEGIN THE LATTICE BEFORE ENDING IT!")
        if self.trackLength==None:
            raise Exception('TRACK LENGTH WAS NOT SET!!')
        if len(self.benderIndices) == 2:  # can only have two bend as of now
            Exception('the ability to deal with a system with more or less than 2 bends is not implemented')
        if self.lattice[0].elType!='BEND':
            raise Exception('First element must be a bender!')
        #ERROR: Check that there are two benders
        if len(self.benderIndices)!=2:
            raise Exception('There must be 2 benders!!!')
        #ERROR: check that total bending is 360 deg within .1%
        if self.lattice[self.benderIndices[0]].angle is None:
            #if the angle is none, then the user wants the angle to be set by constraints
            pass
        else:
            angle=0
            for i in self.benderIndices:
                angle+=self.lattice[i].angle
            if angle>1.01*2*np.pi or angle <.99*2*np.pi:
                raise Exception('Total bending must be 360 degrees within .1%!')
        #check that every bending angle is either a sympy variable/float or None
        for i in self.benderIndices:
            temp=self.lattice[i].angle
            if i>0:
                if temp is None and self.lattice[0].angle is None:
                    pass
                elif temp is not None and self.lattice[0].angle is not None:
                    pass
                else:
                    raise Exception('IF ONE BENDING ANGLE IS \'None\' THEN ALL MUST BE NONE')

    def _update_Element_Parameters_And_Functions(self):
        #after adding all the elements to the lattice set the lengths and positions such that all the constraints
        #are satisfied. Also set parameters and functions that couldn't be set before
        if self.lattice[self.benderIndices[0]].angle is None: #set the angle with constraints
            if self.combinerIndex==None: #if there is no combiner in the lattice. Settings angles is then easy
                angle=2*np.pi/len(self.benderIndices)
                for i in self.benderIndices:
                    self.lattice[i].angle=angle
            elif len(self.benderIndices)==2: # if there are two benders and a combiner
                #algorithm courtesy of Jamie Hall
                #print(self.lattice[self.combinerIndex].S,self.lattice[self.combinerIndex].Length/2)
                self.TL2=self.lattice[self.combinerIndex].S + self.lattice[self.combinerIndex].Length/2
                self.TL1=self.trackLength-self.TL2
                print(self.TL1,self.TL2)
                L1=self.TL2
                L2=self.TL1
                r1=self.lattice[self.benderIndices[0]].rb
                r2 = self.lattice[self.benderIndices[1]].rb
                theta=self.lattice[self.combinerIndex].Length/self.lattice[self.combinerIndex].rb #approximation, but very accurate
                L3=sym.sqrt((L1-sym.sin(theta)*r2+L2*sym.cos(theta))**2+(L2*sym.sin(theta)-r2*(1-sym.cos(theta))+(r2-r1))**2)
                print(L3)
                raise Exception('SOMETHING DOESN\'T MAKE SENSE HERE, L3 ISN\'T USED')
                angle1=sym.pi*1.5-sym.atan(L1/r1)-sym.acos((L3**2+L1**2-L2**2)/(2*L3*sym.sqrt(L1**2+r1**2)))
                angle2=sym.pi*1.5-sym.atan(L2/r2)-sym.acos((L3**2+L2**2-L1**2)/(2*L3*sym.sqrt(L2**2+r2**2)))
                self.latticeElementList[self.benderIndices[0]].angle=angle1
                self.latticeElementList[self.benderIndices[1]].angle=angle2
                self.latticeElementList[self.benderIndices[0]].Length = r1 * angle1
                self.latticeElementList[self.benderIndices[1]].Length = r2 * angle2
            else:
                raise Exception('INCORRECT NUMBER OF BENDERS. IF COMBINER IS PRESENT THERE SHOULD BE ONLY 2 BENDERS!')

        #---now update the elements. This mostly computes the various functions of each lattice. Somewhat intensive
        for el in self.lattice: #manipulate drift elements only
            if el.elType=='LENS': #-------adjust lens lengths
                if el.index==self.benderIndices[0]+1 or el.index==self.benderIndices[1]+1:#if the lens is right after the bend
                    if el.S==None: #if position is unspecified
                        el.S=el.Length/2
                elif el.index == self.benderIndices[1] - 1 or el.index == self.numElements - 1: #if the lens is right
                        #before the bend or at the end
                    if el.S==None: #if position is unspecified
                        el.S=self.trackLength-el.Length/2

        for el in self.lattice:
            if el.elType=='LENS': #-------adjust lens  lengths
                if el.index==self.benderIndices[0]+1 or el.index==self.benderIndices[1]+1:#if the lens is right after the bend
                    if el.S==None: #if position is unspecified
                        el.S=el.Length/2
                elif el.index == self.benderIndices[1] - 1 or el.index == self.numElements - 1: #if the lens is right
                        #before the bend or at the end
                    if el.S==None: #if position is unspecified
                        el.S=self.trackLength-el.Length/2

            if el.elType=='DRIFT': #-----------adjust the drift lengths
                if el.index==self.numElements-1: #if the drift is the last element
                    if self.lattice[-2].elType=='BEND': #if the drift element is the only element in that track
                        if el.Length is not None:
                            print('User set drift length is being changed!!!!')
                        el.Length=self.trackLength
                        el.S=self.trackLength/2
                    else:
                        edgeL = self.lattice[-2].S + self.lattice[-2].Length / 2
                        el.Length=self.trackLength-edgeL
                        el.S = self.trackLength - el.Length / 2
                elif el.index==self.benderIndices[0]+1 or el.index==self.benderIndices[1]+1: #edge case for drift right
                            # after bend. The lattice starts with first element as bend so there are two cases here for now
                    if self.lattice[el.index+1].elType=='BEND': #the drift is sandwiched between two bends
                        el.Length=self.trackLength
                    elif el.Length!=None:
                        self.lattice[el.index+1].S=el.Length+self.lattice[el.index+1].Length/2 #setting the center of the
                                #next element if the drift length is already set. only for drift after bend
                        el.S=el.Length/2 #also set the drift position
                    else:
                        edgeR=self.lattice[el.index+1].S-self.lattice[el.index+1].Length/2 #the position of the next element edge reletaive
                            #to first bend end
                        el.Length=edgeR #distance from beginning of drift to edge of element
                elif el.index==self.benderIndices[1]-1: #if the drift is right before the bend
                    edgeL = self.lattice[self.benderIndices[1]-2].S+self.lattice[self.benderIndices[1]-2].Length/2
                                # the distance of the element edge from the beggining of the bend
                    el.Length=self.trackLength-edgeL #distance from previous element end to beginning of bend
                    el.S=self.trackLength-el.Length/2
                elif el.index==0 and self.lattice[1].elType=='BEND': #if drift is first and only element in the first track
                    el.Length=self.trackLength
                    el.S=self.trackLength/2
                else: #if drift is somewhere else
                    if self.lattice[el.index+1].S != None and self.lattice[el.index+1].Length != None: #if the next element has
                                #definite position and length
                        edgeL=self.lattice[el.index-1].S+self.lattice[el.index-1].Length/2 #position of prev el edge
                        edgeR = self.lattice[el.index + 1].S-self.lattice[el.index+1].Length/2 #position of next el edge
                        el.Length=edgeR-edgeL
                        el.S=self.lattice[el.index-1].S+self.lattice[el.index-1].Length/2+el.Length/2
                    else: #this really only works for drifts after the bender that is followed by a lens, another drift
                            #and then a bend
                        self.lattice[el.index+1].S=self.trackLength-self.lattice[el.index+2].Length-self.lattice[el.index+1].Length/2
                        el.Length=(self.lattice[el.index+1].S-self.lattice[el.index+1].Length/2)-(self.lattice[el.index-1].S-self.lattice[el.index-1].Length/2)
                        el.S=self.lattice[el.index-1].S+self.lattice[el.index-1].Length/2+self.lattice[el.index].Length/2
        for el in self.lattice:
            if type(el.Length)==float:
                if el.Length<0:
                    raise Exception("ELEMENT LENGTH IS BEING SET TO NEGATIVE VALUE")
        for el in self.lattice:
            el.fill_Params_And_Functions()

    def begin_Lattice(self): #must be called before making lattice
        self.began=True
    def set_apetures(self):
        #update the apeture values for each element. The apeture value is return from a function, even for
        #apetures that don't change with the parameters so that it is general
        for el in self.lattice:
            if el.elType=='BEND':
                #apetures are simply half the bending magnet radius because the vacuum tube would span the center to the
                #edge.
                def temp(func1,func2,*x):
                    return func1(*x)+func2(*x)
                el.apxFuncL = el.rtFunc
                el.apxFuncR=functools.partial(temp,el.rpFunc,el.rtFunc) #there are some subtleties here with functions!!
                el.apyFunc=el.rpFunc
            elif el.elType=='COMBINER':
                #I make a dummy function so that I can still pass the variables and get a constant number
                def temp(*args):
                    return .015
                def temp2(*args):
                    return .005
                el.apxFuncL=temp #15 mm apterure in x plane
                el.apxFuncR = temp  # 15 mm apterure in x plane
                el.apyFunc=temp2 #5 mm apeture in y plane
            elif el.elType=='LENS':
                el.apxFuncL=el.rpFunc
                el.apxFuncR=el.rpFunc
                el.apyFunc=el.rpFunc
            else: #other elements have no apeture
                def temp(*args):
                    return np.inf
                el.apxFuncL=temp
                el.apxFuncR = temp
                el.apyFunc=temp
    def end_Lattice(self):

        #must be called after ending lattice. Prepares functions that will be used later
        if self.catchErrors==True:
            self._catch_Errors()
        self._update_Element_Parameters_And_Functions()

        self.set_apetures()



        self.MTot = self._compute_M_Total() #sympy version of full transfer function

        self.MTotFunc = symWrap.autowrap(self.MTot,args=self.sympyVarList)

        #this loop does 2 things
        # 1:make  an array function where each entry is the length of the corresping optic.
        #2: make an array function each entry is the sum of the lengths of the preceding optics and that optic. ie, the total length at that point
        temp1 = []#temporary array for case 1
        temp2 = [] #temporary aray for case 2
        for i in range(len(self.lattice)):
            temp1.append(self.lattice[i].Length)
            temp2.append(self.lattice[i].Length)
            for j in range(i): #do the sum up to that point
                temp2[i] += self.lattice[j].Length
        self.lengthListFunc = sym.lambdify(self.sympyVarList,temp1) #A function that returns a list where
                                                # each entry is length of that element
        self.totalLengthListFunc = sym.lambdify(self.sympyVarList,temp2) #each entry is an inclusive cumulative sum
                                                                            #of element lengths
        print("Lattice model completed")
        #def tempFunc(args):
        ##takes in arguments and returns a list of transfer matrices
        ##args: the variables defined by the user ie the sympyVarList
        #    tempList=[]
        #    for el in self.lattice:
        #        M_N=sym.lambdify(self.sympyVarList, el.M,'numpy')
        #        tempList.append(M_N(*args))
        #    return tempList
        #self.MListFunc=tempFunc

    def compute_Resonance_Factor(self, tune,res):  # a factor, between 0 and 1, describing how close a tune is a to a given resonance
        # 0 is as far away as possible, 1 is exactly on resonance
        # the procedure is to see how close the tune is to an integer multiples of 1/res. ie a 2nd order resonance(res =2) can occure
        # when the tune is 1,1.5,2,2.5 etc and is maximally far off resonance when the tuen is 1.25,1.75,2.25 etc
        # tune: the given tune to analyze
        # res: the order of resonance interested in. 1,2,3,4 etc. 1 is pure bend, 2 is pure focus, 3 is pure corrector etc.
        resFact = 1 / res  # What the remainder is compare to
        tuneRem = tune%1  # tune remainder, the integer value doens't matter
        tuneResFact =1-np.abs(2*(tuneRem-np.round(tuneRem/resFact)*resFact) / resFact)  # relative nearness to resonances
        return tuneResFact





    def _compute_Lattice_Function_From_M(self,M,funcName,axis,args=None):
        #since many latice functions, such as beta,alpha,gamma and eta are calculated in a very similiar was
        #this function saces space by resusing code
        #M: 5x5 transfer matrix, or 3x3 if using x axis or 2x2 if using x axis and no eta
        if axis==None: #can't put self as keyword arg
            axis=self.axis
        def lattice_Func_From_M_Reduced(Mat): #to save space. This simply computes beta over a given 2x2 matrix
            M11 = Mat[0, 0]
            M12 = Mat[0, 1]
            M21 = Mat[1, 0]
            M22 = Mat[1, 1]
            if funcName=='BETA':
                return np.abs(2 * M12 / np.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2))
            if funcName=='ETA':
                M13 = Mat[0, 2]
                M23 = Mat[1, 2]
                extraFact = 2  # THIS IS A KEY DIFFERENCE BETWEEN NEUTRAL AND CHARGED PARTICLES!!!
                etaPrime=(M21*M13+M23*(1-M11))/(2-M11-M22)
                return extraFact * (M12*etaPrime+M13)/(1-M11)
            if funcName=='ALPHA': #DONT BE STUPID BILLY. Alpha=-deriv(beta)/2.
                #TODO: Figure out how to deal with sign ambiguity...
                return (M11-M22)/np.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2)
        if axis=='x':
            return lattice_Func_From_M_Reduced(M[:3,:3])
        elif axis=='y':
            return lattice_Func_From_M_Reduced(M[3:5, 3:5])
        elif axis=='both':
            funcx= lattice_Func_From_M_Reduced(M[:3, :3])
            funcy=lattice_Func_From_M_Reduced(M[3:5, 3:5])
            return [funcx,funcy]

    def _compute_Lattice_Function_From_MArr(self,MArr,funcName,axis):
        #Yes, this is messy. But this is a bottle neck and this is the fastest way
        #TODO: CLEANUP WITHOUT SACRIFICING PERFORMACE
        #M: 5x5 transfer matrix, or 3x3 if using x axis or 2x2 if using x axis and no eta
        if axis==None: #can't put self as keyword arg
            axis=self.axis
        if axis=='x':
            M11 = MArr[:, 0, 0]
            M12 = MArr[:, 0, 1]
            M21 = MArr[:, 1, 0]
            M22 = MArr[:, 1, 1]
            if funcName=='BETA':
                return np.abs(2 * M12 / np.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2))
            if funcName=='ETA':
                M13 = MArr[:,0, 2]
                M23 = MArr[:,1, 2]
                extraFact = 2  # THIS IS A KEY DIFFERENCE BETWEEN NEUTRAL AND CHARGED PARTICLES!!!
                etaPrime=(M21*M13+M23*(1-M11))/(2-M11-M22)
                return extraFact * (M12*etaPrime+M13)/(1-M11)
            if funcName=='ALPHA': #DONT BE STUPID BILLY. Alpha=-deriv(beta)/2.
                #TODO: Figure out how to deal with sign ambiguity...
                return (M11-M22)/np.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2)
        if axis=='y':
            M11 = MArr[:, 3, 3]
            M12 = MArr[:, 3, 4]
            M21 = MArr[:, 4, 3]
            M22 = MArr[:, 4, 4]
            if funcName=='BETA':
                return np.abs(2 * M12 / np.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2))
            if funcName=='ALPHA': #DONT BE STUPID BILLY. Alpha=-deriv(beta)/2.
                #TODO: Figure out how to deal with sign ambiguity...
                return (M11-M22)/np.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2)
        else:
            M11 = MArr[:, 0, 0]
            M12 = MArr[:, 0, 1]
            M21 = MArr[:, 1, 0]
            M22 = MArr[:, 1, 1]
            M44 = MArr[:, 3, 3]
            M45 = MArr[:, 3, 4]
            M54 = MArr[:, 4, 3]
            M55 = MArr[:, 4, 4]
            if funcName=='BETA':
                betax= np.abs(2 * M12 / np.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2))
                betay = np.abs(2 * M45 / np.sqrt(2 - M44 ** 2 - 2 * M45 * M54 - M55 ** 2))
                return betax,betay
            if funcName=='ALPHA': #DONT BE STUPID BILLY. Alpha=-deriv(beta,z)/2.
                #TODO: Figure out how to deal with sign ambiguity...
                alphax= (M11-M22)/np.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2)
                alphay = (M44 - M55) / np.sqrt(2 - M44 ** 2 - 2 * M45 * M54 - M55 ** 2)
                return alphax,alphay

    def compute_Alpha_From_M(self, M,axis=None):
        if axis==None:
            return self._compute_Lattice_Function_From_M(M,'ALPHA',self.axis)
        else:
            return self._compute_Lattice_Function_From_M(M, 'ALPHA', axis)

    def compute_Eta_From_M(self, M,axis=None):
        if axis == None:
            return self._compute_Lattice_Function_From_M(M, 'Eta', self.axis)
        else:
            return self._compute_Lattice_Function_From_M(M, 'Eta', axis)
    def compute_Beta_From_M(self, M,axis=None):
        if axis == None:
            return self._compute_Lattice_Function_From_M(M, 'Beta', self.axis)
        else:
            return self._compute_Lattice_Function_From_M(M, 'Beta', axis)
    def compute_Beta_Of_Z_Array(self,args,numpoints=None,axis=None,elIndex=None,returZarr=True,zArr=None):
        return self._compute_Lattice_Function_Of_z_Array('BETA',numpoints,elIndex,returZarr,zArr,axis,args)
    def compute_Eta_Of_Z_Array(self,args,numpoints=1000,elIndex=None,returZarr=True,zArr=None):
        return self._compute_Lattice_Function_Of_z_Array('ETA',numpoints,elIndex,returZarr,zArr,'x',args)
    def compute_Alpha_Of_Z_Array(self,args,numpoints=1000,axis=None,elIndex=None,returZarr=True,zArr=None):
        return self._compute_Lattice_Function_Of_z_Array('ALPHA',numpoints,elIndex,returZarr,zArr,axis,args)
    def compute_Beta_At_Z(self,z,args,axis='both'):
        M=self.compute_M_Trans_At_z(z,args)
        beta=self._compute_Lattice_Function_From_M(M,'BETA',axis,args=args)
        return beta
    def compute_Alpha_At_Z(self,z,args,axis='both'):
        M=self.compute_M_Trans_At_z(z,args)
        alpha=self._compute_Lattice_Function_From_M(M,'ALPHA',axis)
        return alpha



    def _compute_Lattice_Function_Of_z_Array(self,funcName,numPoints,elIndex,returnZarr,zArr,axis,args):
       # computes lattice functions over entire lattice, or single element.
       # args: supplied arguments. this depends on the variables created by user if there are any. Order is critical
       # numPoints: number of points compute. Initially none because it has different behaviour wether the user chooses to
       # compute beta over a single element or the while thing
       # elIndex: which element to compute points at. if none compute over whole lattice
       if axis==None:
           axis=self.axis
       totalLengthList = self.totalLengthListFunc(*args)
       if elIndex == None:  # use entire lattice
           if np.any(zArr==None): #if user wants to use default zArr
                if numPoints == None:  # if user is wanting default value
                    numPoints = 500
                zArr = np.linspace(0, totalLengthList[-1], num=numPoints)
       else:  # compute beta array over specific element
           if zArr!=None: #catch an error
               raise Exception("You can't set z array and compute lattice function over single element")
           if numPoints == None:  # if user is wanting default value
               numPoints = 50
           if elIndex == 0:
               zArr = np.linspace(0, totalLengthList[0], num=numPoints)  # different rule for first element
           else:
               zArr = np.linspace(totalLengthList[elIndex - 1], totalLengthList[elIndex], num=numPoints)
       MTranList = self.make_MTrans_List(zArr.shape[0], args)
       MArr = np.asarray(MTranList)
       if axis == 'both':
           latFuncxArr, latFuncyArr = self._compute_Lattice_Function_From_MArr(MArr,funcName,axis=axis)
           latFuncArrReturn = [latFuncxArr, latFuncyArr]
       else:
           latFuncArr=self._compute_Lattice_Function_From_MArr(MArr,funcName,axis=axis)
           latFuncArrReturn = latFuncArr
       if returnZarr == True:
           return zArr, latFuncArrReturn
       else:
           return latFuncArrReturn




    def compute_MTot(self,args):
        M = np.eye(5)
        for j in range(self.numElements):
            M = self.lattice[j].M_Func(*args) @ M
        return M

    def compute_M_Trans_At_z(self, z, args):
        #TODO: speedup!!!
        totalLengthArray = np.asarray(self.totalLengthListFunc(*args))
        lengthArray = np.asarray(self.lengthListFunc(*args))
        temp = totalLengthArray - z
        index = int(np.argmax(temp >= 0)) #to prevent a typecast warning
        M = self.lattice[index].M_Funcz(totalLengthArray[index] - z,*args)  # starting matrix
        # calculate from point z to end of lattice
        for i in range(self.numElements - index - 1):
            j = i + index + 1  # current magnet +1 to get index of next magnet
            M = self.lattice[j].M_Funcz(lengthArray[j],*args) @ M
        # from beginning to current element
        for i in range(index):
            M = self.lattice[i].M_Funcz(lengthArray[i],*args) @ M
        # final step is rest of distance
        if index == 0:  # if the first magnet
            M = self.lattice[index].M_Funcz(z,*args) @ M
        else:  # any other magnet
            M = self.lattice[index].M_Funcz(z - totalLengthArray[index - 1],*args) @ M
        return M
    def make_MTrans_List(self,numPoints,args):
        temp=[]
        z0=0
        index=0
        totalLengthArr = np.asarray(self.totalLengthListFunc(*args))
        lengthArr = np.asarray(self.lengthListFunc(*args))
        zArr = np.linspace(0, totalLengthArr[-1], num=numPoints)

        #----get the algorithm started----

        M1=np.eye(5) #matrix from the left to that element
        M2=np.eye(5) #from that element to the end
        for i in range(1,self.numElements):
            M2=self.lattice[i].M_Func(*args)@M2
        j=1
        for z in zArr:

            if z<=totalLengthArr[index]+1E-10 and z0<=totalLengthArr[index]+1E-10: #small number added to account for
                    #potential numerical issues
                if index==0: #if index==0 then using totalLengthArr[index-1] is equivalent to totalLengthArr[-1]!, but
                        #i want zero!
                    Ma = self.lattice[index].M_Funcz(z, *args)
                else:
                    Ma=self.lattice[index].M_Funcz(z-totalLengthArr[index-1],*args)
                Mb=self.lattice[index].M_Funcz(totalLengthArr[index]-z,*args)

                M=Ma@M1@M2@Mb
                z0=z
                temp.append(M)
            elif j==numPoints:
                M2 = np.eye(5)
                for i in range(0,self.numElements): #go from the element after the current element to the end
                    M2=self.lattice[i].M_Func(*args)@M2 #TODO: REPLACE WITH MFunc
                temp.append(M2)
            else:
                index=int(np.argmax(z<totalLengthArr))
                M1 = np.eye(5)  # matrix from the left to that element
                M2 = np.eye(5)  # from that element to the end
                for i in range(index+1,self.numElements): #go from the element after the current element to the end
                    M2=self.lattice[i].M_Func(*args)@M2
                for i in range(0,index): #go from beginning to current element
                    M1=self.lattice[i].M_Func(*args)@M1
                Ma = self.lattice[index].M_Funcz(z-totalLengthArr[index-1], *args)
                Mb = self.lattice[index].M_Funcz(totalLengthArr[index] - z, *args)
                M=Ma@M1@M2@Mb
                z0=z
                temp.append(M)
            j+=1
        return temp
# PLS = PeriodicLatticeSolver(200, .02)
# PLS.add_Injector()
#
# L1= .5#PLS.Variable('L1', varMin=.01, varMax=.5)
# L2= .5#PLS.Variable('L2', varMin=.01, varMax=.5)
# L3= .5#PLS.Variable('L3', varMin=.01, varMax=.5)
# L4= .5#PLS.Variable('L4', varMin=.01, varMax=.5)
#
# Bp1 = PLS.Variable('Bp1', varMin=.1, varMax=.45)
# Bp2 = PLS.Variable('Bp2', varMin=.1, varMax=.45)
# Bp3 = PLS.Variable('Bp3', varMin=.1, varMax=.45)
# Bp4 = PLS.Variable('Bp4', varMin=.1, varMax=.45)
#
# rp1 =.03# PLS.Variable('rp1', varMin=.005, varMax=.03)
# rp2 =.03# PLS.Variable('rp2', varMin=.005, varMax=.03)
# rp3 =.03# PLS.Variable('rp3', varMin=.005, varMax=.03)
# rp4 =.03# PLS.Variable('rp4', varMin=.005, varMax=.03)
#
# #s = PLS.Variable('s', varMin=.005, varMax=.03)
#
# rb=1
# #TL1=PLS.Variable('TL1',varMin=.5,varMax=1.5)
# #TL2=PLS.Variable('TL2',varMin=.5,varMax=1.5)
#
#
# PLS.set_Track_Length(trackLength=1)
# PLS.begin_Lattice()
#
# PLS.add_Bend(None, rb, .45)
# #PLS.add_Drift(L=test)
# PLS.add_Lens(L4, Bp4, rp4)
# PLS.add_Drift()
# PLS.add_Combiner(S=.5)
# PLS.add_Drift()
# PLS.add_Lens(L1, Bp1,rp1)
# #PLS.add_Drift(L=.05)
# PLS.add_Bend(None, rb, .45)
# #PLS.add_Drift(L=.05)
# PLS.add_Lens(L2, Bp2, rp2)
# PLS.add_Drift()
# PLS.add_Lens(L3, Bp3, rp3)
# #PLS.add_Drift(L=.05)
# PLS.end_Lattice()