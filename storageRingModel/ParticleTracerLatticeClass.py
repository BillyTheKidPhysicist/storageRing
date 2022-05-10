import elementPT
# from geneticLensElement_Wrapper import GeneticLens
from typing import Union,Generator,Iterable
from ParticleClass import Particle
from ParticleTracerClass import ParticleTracer
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import scipy.interpolate as spi
import numpy.linalg as npl
from joblib import Parallel,delayed
from elementPT import *
from storageRingConstraintSolver import build_Particle_Tracer_Lattice,is_Particle_Tracer_Lattice_Closed
from constants import DEFAULT_ATOM_SPEED
#todo: There is a ridiculous naming convention here with r0 r1 and r2. If I ever hope for this to be helpful to other
#people, I need to change that. This was before my cleaner code approach

#todo: refactor!

benderTypes=Union[elementPT.BenderIdeal,elementPT.HalbachBenderSimSegmented]


class ParticleTracerLattice:

    def __init__(self,v0Nominal: float= DEFAULT_ATOM_SPEED,latticeType: str='storageRing',
                 jitterAmp: float=0.0, fieldDensityMultiplier:float =1.0, standardMagnetErrors: bool =False):
        assert fieldDensityMultiplier>0.0
        if latticeType!='storageRing' and latticeType!='injector':
            raise Exception('invalid lattice type provided')
        if jitterAmp>5e-3:
            raise Exception("Jitter values greater than 5 mm may begin to have unexpected results. Several parameters"
                            "depend on this value, and relatively large values were not planned for")
        self.latticeType=latticeType#options are 'storageRing' or 'injector'. If storageRing, the geometry is the the first element's
        #input at the origin and succeeding elements in a counterclockwise fashion. If injector, then first element's input
        #is also at the origin, but seceeding elements follow along the positive x axis
        self.v0Nominal = v0Nominal  # Design particle speed
        self.benderIndices: list[int]=[] #list that holds index values of benders. First bender is the first one that the particle sees
        #if it started from beginning of the lattice. Remember that lattice cannot begin with a bender
        self.combinerIndex: Optional[int]=None #the index in the lattice where the combiner is
        self.totalLength: Optional[float]=None #total length of lattice, m
        self.jitterAmp=jitterAmp
        self.fieldDensityMultiplier=fieldDensityMultiplier
        self.standardMagnetErrors=standardMagnetErrors

        self.bender1: Optional[benderTypes]=None #bender element object
        self.bender2: Optional[benderTypes]=None #bender element object
        self.combiner: Optional[elementPT.Element]=None #combiner element object
        self.linearElementsToConstraint: list[elementPT.HalbachLensSim]=[] #elements whos length will be changed when the
        # lattice is constrained to satisfy geometry. Must be inside bending region

        self.isClosed=None #is the lattice closed, ie end and beginning are smoothly connected?

        self.elList: list=[] #to hold all the lattice elements

    def __iter__(self)-> Iterable[Element]:
        return (element for element in self.elList)

    def find_Optimal_Offset_Factor(self,rp: float,rb: float,Lm: float,parallel: bool=False)-> float:
        #How far exactly to offset the bending segment from linear segments is exact for an ideal bender, but for an
        #imperfect segmented bender it needs to be optimized.
        raise NotImplementedError #this doesn't do much with my updated approach, and needs to be reframed in terms
        #of shifting the particle over to improve performance. It's also bloated. ALso, it's not accurate
        assert rp<rb/2.0 #geometry argument, and common mistake
        numMagnetsHalfBend=int(np.pi*rb/Lm)
        #todo: this should be self I think
        PTL_Ring=ParticleTracerLattice(latticeType='injector')
        PTL_Ring.add_Drift(.05)
        PTL_Ring.add_Halbach_Bender_Sim_Segmented(Lm,rp,numMagnetsHalfBend,rb)
        PTL_Ring.end_Lattice(enforceClosedLattice=False,constrain=False)
        def errorFunc(offset):
            h=5e-6
            particle=Particle(qi=np.array([-1e-10,offset,0.0]),pi=np.array([-self.v0Nominal,0.0,0.0]))
            particleTracer=ParticleTracer(PTL_Ring)
            particle=particleTracer.trace(particle,h,1.0,fastMode=False)
            qoArr=particle.qoArr
            particleAngEl=np.arctan2(qoArr[-1][1],qoArr[-1][0]) #very close to zero, or negative, if particle made it to
            #end
            if particleAngEl<.01:
                error=np.std(1e6*particle.qoArr[:,1])
                return error
            else: return np.nan
        outputOffsetFactArr=np.linspace(-3e-3,3e-3,100)

        if parallel==True: njobs=-1
        else: njobs=1
        errorArr=np.asarray(Parallel(n_jobs=njobs)(delayed(errorFunc)(outputOffset) for outputOffset in outputOffsetFactArr))
        rOffsetOptimal=self._find_rOptimal(outputOffsetFactArr,errorArr)
        return rOffsetOptimal

    def _find_rOptimal(self,outputOffsetFactArr: np.ndarray,errorArr: np.ndarray)-> Optional[float]:
        test=errorArr.copy()[1:]
        test=np.append(test,errorArr[0])
        numValidSolutions=np.sum(~np.isnan(errorArr))
        numNanInitial=np.sum(np.isnan(errorArr))
        numNanAfter=np.sum(np.isnan(test+errorArr))
        valid=True
        if numNanAfter-numNanInitial>1:
            valid=False
        elif numValidSolutions<4:
            valid=False
        elif numNanInitial>0:
            if (np.isnan(errorArr[0])==False and np.isnan(errorArr[-1])==False):
                valid=False
        if valid==False:
            return None
        #trim out invalid points
        outputOffsetFactArr=outputOffsetFactArr[~np.isnan(errorArr)]
        errorArr=errorArr[~np.isnan(errorArr)]
        fit=spi.RBFInterpolator(outputOffsetFactArr[:,np.newaxis],errorArr)
        outputOffsetFactArrDense=np.linspace(outputOffsetFactArr[0],outputOffsetFactArr[-1],10_000)
        errorArrDense=fit(outputOffsetFactArrDense[:,np.newaxis])
        rOptimal=outputOffsetFactArrDense[np.argmin(errorArrDense)]
        rMinDistFromEdge=np.min(outputOffsetFactArr[1:]-outputOffsetFactArr[:-1])/4
        if rOptimal>outputOffsetFactArr[-1]-rMinDistFromEdge or rOptimal<outputOffsetFactArr[0]+rMinDistFromEdge:
            # print('Invalid solution, rMin very near edge. ')
            return None
        return rOptimal

    def set_Constrained_Linear_Element(self,el: Element)-> None:
        if len(self.linearElementsToConstraint)>1: raise Exception("there can only be 2 constrained linear elements")
        self.linearElementsToConstraint.append(el)

    def add_Combiner_Sim(self,sizeScale: float=1.0)-> None:
        """
        Add model of our combiner from COMSOL. rarely used

        :param sizeScale: How much to scale up or down dimensions of combiner
        :return: None
        """

        file='combinerV3.txt'
        el = CombinerSim(self,file,self.latticeType,sizeScale=sizeScale)
        el.index = len(self.elList) #where the element is in the lattice
        assert self.combiner is None #there can be only one!
        self.combiner=el
        self.combinerIndex=el.index
        self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Combiner_Sim_Lens(self,Lm: float,rp: float,loadBeamDiam: float=10e-3,layers: int=2,ap: float =None,
                              seed: int =None)-> None:
        """
        Add halbach hexapole lens combiner element.

        The edge of a hexapole lens is used to deflect high and weak field seeking states. Transvers dimension of
        magnets are the maximum that can be used to for a halbach sextupole of given radius.

        :param Lm: Hard edge length of magnet, m. Total length of element depends on degree of deflection of nominal
        trajectory
        :param rp: Bore radius of hexapole lens, m
        :param loadBeamDiam: Maximum desired acceptance diameter of load beam, m. Circulating beam is not specified
        :param layers: Number of concentric layers of magnets
        :return: None
        """

        if seed is not None:
            np.random.seed(seed)
        el = CombinerHalbachLensSim(self,Lm,rp,loadBeamDiam,layers,ap,self.latticeType,self.standardMagnetErrors)
        el.index = len(self.elList) #where the element is in the lattice
        assert self.combiner is None  # there can be only one!
        self.combiner=el
        self.combinerIndex=el.index
        self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Halbach_Lens_Sim(self,rp: Union[float,tuple],L: Optional[float],ap:Optional[float]=None,
                             constrain: bool=False,magnetWidth: float=None)-> None:
        """
        Add simulated halbach sextupole element to lattice.

        :param rp: Bore radius, m
        :param L: Length of element, m. This includes fringe fields, actual magnet length will be smaller
        :param ap: Size of aperture
        :param constrain: Wether element is being used as part of a constraint. If so, fields construction will be
        deferred
        :param magnetWidth: Width of both side cuboid magnets in polar plane of lens, m. Magnets length is L minus
        fringe fields
        :return: None
        """
        rpLayers=rp if isinstance(rp,tuple) else (rp,)
        magnetWidth=(magnetWidth,) if isinstance(magnetWidth,float) else magnetWidth
        el=HalbachLensSim(self, rpLayers,L,ap,magnetWidth, self.standardMagnetErrors)
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
        if constrain==True: self.set_Constrained_Linear_Element(el)

    # def add_Genetic_lens(self,lens: GeneticLens,ap: float)-> None:
    #     """
    #     Add genetic lens used for minimizing focus size. This is part of an idea to make a low aberration lens
    #
    #     :param lens: GeneticLens object that returns field values. This sextupole lens can be shimmed, and have bizarre
    #     :param ap: Aperture of genetic lens, m
    #     :return: None
    #     """
    #     el=geneticLens(self,lens,ap)
    #     el.index = len(self.elList) #where the element is in the lattice
    #     self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Lens_Ideal(self,L: float,Bp: float,rp: float,constrain: bool=False,ap: float=None)-> None:
        """
        Simple model of an ideal lens. Field norm goes as B0=Bp*r^2/rp^2

        :param L: Length of element, m. Lens hard edge length is this as well
        :param Bp: Field at bore/pole radius of lens
        :param rp: Bore/pole radius of lens
        :param ap: aperture of vacuum tube in magnet
        :param constrain:
        :param bumpOffset:
        :return:
        """

        el=LensIdeal(self, L, Bp, rp, ap) #create a lens element object
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order
        if constrain==True:
            self.set_Constrained_Linear_Element(el)
            print('not fully supported feature')

    def add_Drift(self,L: float,ap: float=.03, outerHalfWidth: float= None)-> None:
        """
        Add drift region. This is simply a vacuum tube.

        :param L: Length of drift region, m
        :param ap: Aperture of drift region, m
        :return:
        """

        el=Drift(self,L,ap,outerHalfWidth)#create a drift element object
        el.index = len(self.elList) #where the element is in the lattice
        self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Halbach_Bender_Sim_Segmented(self,Lm: float,rp: float,numMagnets: Optional[int],rb: float,
                                         extraSpace: float=0.0,rOffsetFact: float=1.0,ap: float=None)->None:
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #Lcap: Length of element on the end/input of bender
        #outputOffsetFact: factor to multply the theoretical offset by to minimize oscillations in the bending segment.
        #modeling shows that ~.675 is ideal
        el = HalbachBenderSimSegmented(self, Lm,rp,numMagnets,rb,ap,extraSpace,rOffsetFact,self.standardMagnetErrors)
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)

    def add_Bender_Ideal(self,ang: float,Bp: float,rb: float,rp: float,ap: float=None)-> None:
        #Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        #ang: Bending angle of bender, radians
        #rb: nominal bending radius of element's centerline. Actual radius is larger because particle 'rides' a little
        # outside this, m
        #Bp: field strength at pole face of lens, T
        #rp: bore radius of element, m
        #ap: size of apeture. If none then a fraction of the bore radius. Can't be bigger than bore radius, unitless

        el=BenderIdeal(self, ang, Bp, rp, rb, ap) #create a bender element object
        el.index = len(self.elList) #where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Combiner_Ideal(self,Lm: float=.2,c1: float=1,c2: float=20,ap: float=.015,sizeScale: float=1.0)-> None:
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

        el=CombinerIdeal(self, Lm, c1, c2, ap,ap,ap/2,self.latticeType,sizeScale) #create a combiner element object
        el.index = len(self.elList) #where the element is in the lattice
        assert self.combiner is None  # there can be only one!
        self.combiner = el
        self.combinerIndex=el.index
        self.elList.append(el) #add element to the list holding lattice elements in order

    def build_Lattice(self,constrain: bool):

        build_Particle_Tracer_Lattice(self, constrain)
        self.isClosed= is_Particle_Tracer_Lattice_Closed(self) #lattice may not have been constrained, but could
                #still be closed
        self.make_Geometry()
        self.totalLength = 0
        for el in self.elList:  # total length of particle's orbit in an element
            self.totalLength += el.Lo

    def end_Lattice(self,constrain: bool=False,enforceClosedLattice: bool=True,buildLattice: bool=True,
                    surpressWarning: bool=False)-> None:
        #todo: reimplement keyword args
        #REALLY NEED TO CLEAN UP ERROR CATCHING
        #document which parameters mean what when using constrain!
        #THIS ALL NEEDS TO MAKE MORE SENSE
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
        self.catch_Errors(constrain,buildLattice)

        if buildLattice==True:
            self.build_Lattice(constrain)

    def make_Geometry(self)-> None:
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
            if el.shape=='STRAIGHT':
                q1Inner=np.asarray([xb-np.sin(theta)*halfWidth,yb+halfWidth*np.cos(theta)]) #top left when theta=0
                q2Inner=np.asarray([xe-np.sin(theta)*halfWidth,ye+halfWidth*np.cos(theta)]) #top right when theta=0
                q3Inner=np.asarray([xe+np.sin(theta)*halfWidth,ye-halfWidth*np.cos(theta)]) #bottom right when theta=0
                q4Inner=np.asarray([xb+np.sin(theta)*halfWidth,yb-halfWidth*np.cos(theta)]) #bottom left when theta=0
                pointsInner=[q1Inner,q2Inner,q3Inner,q4Inner]
                if el.outerHalfWidth is None:
                    pointsOuter=pointsInner.copy()
                else:
                    halfWidth=el.outerHalfWidth
                    if False:#el.fringeFrac is not None:
                        pass
                    else:
                        q1Outer=np.asarray([xb-np.sin(theta)*halfWidth,yb+halfWidth*np.cos(theta)])  #top left when theta=0
                        q2Outer=np.asarray([xe-np.sin(theta)*halfWidth,ye+halfWidth*np.cos(theta)])  #top right when theta=0
                        q3Outer=np.asarray([xe+np.sin(theta)*halfWidth,ye-halfWidth*np.cos(theta)])  #bottom right when theta=0
                        q4Outer=np.asarray([xb+np.sin(theta)*halfWidth,yb-halfWidth*np.cos(theta)])  #bottom left when theta=0
                        pointsOuter=[q1Outer,q2Outer,q3Outer,q4Outer]
            elif el.shape=='BEND':
                phiArr=np.linspace(0,-el.ang,num=benderPoints)+theta+np.pi/2 #angles swept out
                r0=el.r0.copy()
                xInner=(el.rb-halfWidth)*np.cos(phiArr)+r0[0] #x values for inner bend
                yInner=(el.rb-halfWidth)*np.sin(phiArr)+r0[1] #y values for inner bend
                xOuter=np.flip((el.rb+halfWidth)*np.cos(phiArr)+r0[0]) #x values for outer bend
                yOuter=np.flip((el.rb+halfWidth)*np.sin(phiArr)+r0[1]) #y values for outer bend
                if isinstance(el,elementPT.HalbachBenderSimSegmented):
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
            elif el.shape in ('COMBINER_SQUARE','COMBINER_CIRCULAR'):
                if el.shape=='COMBINER_SQUARE':
                    apR=el.apR #the 'right' apeture. here this confusingly means when looking in the yz plane, ie the place
                    #that the particle would look into as it revolves in the lattice
                    apL=el.apL
                else:
                    apR,apL=el.ap,el.ap
                q1Inner=np.asarray([0,apR]) #top left ( in standard xy plane) when theta=0
                q2Inner=np.asarray([el.Lb,apR]) #top middle when theta=0
                q3Inner=np.asarray([el.Lb+(el.La-apR*np.sin(el.ang))*np.cos(el.ang),apR+(el.La-apR*np.sin(el.ang))*np.sin(el.ang)]) #top right when theta=0
                q4Inner=np.asarray([el.Lb+(el.La+apL*np.sin(el.ang))*np.cos(el.ang),-apL+(el.La+apL*np.sin(el.ang))*np.sin(el.ang)]) #bottom right when theta=0
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

    def catch_Errors(self,constrain: bool,builLattice: bool)-> None:
        #catch any preliminary errors. Alot of error handling happens in other methods. This is a catch all for other
        #kinds. This class is not meant to have tons of error handling, so user must be cautious
        if self.elList[0].shape=='BEND': #first element can't be a bending element
            raise Exception('FIRST ELEMENT CANT BE A BENDER')
        if self.elList[0].shape in ('COMBINER_SQUARE','COMBINER_CIRCULAR'): #first element can't be a combiner element
            raise Exception('FIRST ELEMENT CANT BE A COMBINER')
        if len(self.benderIndices)==2: #if there are two benders they must be the same. There could be more benders, but
            #that is not dealth with here
            if type(self.bender1)!=type(self.bender2):
                raise Exception('BOTH BENDERS MUST BE THE SAME KIND')
        if constrain==True:
            if self.latticeType!='storageRing':
                raise Exception('Constrained lattice must be storage ring type')
            if type(self.bender1)!=type(self.bender2): #for continuous benders
                raise Exception('BENDERS BOTH MUST BE SAME TYPE')
            if len(self.benderIndices) != 2:
                raise Exception('THERE MUST BE TWO BENDERS')
            if self.combiner is not None:
                if self.combinerIndex>self.benderIndices[0]:
                    raise Exception('COMBINER MUST BE BEFORE FIRST BENDER')

            if self.bender1.segmented==True:
                if (self.bender1.Lseg != self.bender2.Lseg) or (self.bender1.yokeWidth != self.bender2.yokeWidth):
                    raise Exception('SEGMENT LENGTHS AND YOKEWIDTHS MUST BE EQUAL BETWEEN BENDERS')
            if self.bender1.segmented != self.bender2.segmented:
                raise Exception('BENDER MUST BOTH BE SEGMENTED OR BOTH NOT')

    def get_Element_Before_And_After(self,elCenter: elementPT.Element)-> tuple[Element,Element]:
        if (elCenter.index==len(self.elList)-1 or elCenter.index==0) and self.latticeType=='injector':
            raise Exception('Element cannot be first or last if lattice is injector type')
        elBeforeIndex=elCenter.index-1 if elCenter.index!=0 else len(self.elList)-1
        elAfterIndex=elCenter.index+1 if elCenter.index<len(self.elList)-1 else 0
        elBefore=self.elList[elBeforeIndex]
        elAfter=self.elList[elAfterIndex]
        return elBefore,elAfter

    def get_Lab_Coords_From_Orbit_Distance(self,xPos: np.ndarray)-> tuple[float,float]:
        #xPos: distance along ideal orbit
        assert xPos>=0.0
        xPos=xPos%self.totalLength #xpos without multiple revolutions
        xInOrbitFrame=None
        element=None
        cumulativeLen=0.0
        for latticeElement in self.elList:
            if cumulativeLen+latticeElement.Lo>xPos:
                element=latticeElement
                xInOrbitFrame=xPos-cumulativeLen
                break
            cumulativeLen+=latticeElement.Lo
        xLab,yLab,zLab=element.transform_Orbit_Frame_Into_Lab_Frame(np.asarray([xInOrbitFrame,0,0]))
        return xLab,yLab

    def show_Lattice(self,particleCoords=None,particle=None,swarm=None, showRelativeSurvival=True,showTraceLines=False,
                     showMarkers=True,traceLineAlpha=1.0,trueAspectRatio=True,extraObjects=None,finalCoords=True,
                     saveTitle=None,dpi=150,defaultMarkerSize=1000, plotElementExterior: bool = False):
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
        def plot_Particle(particle,xMarkerSize=defaultMarkerSize):
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
                try:
                    if finalCoords==False:
                        xy=particle.qi[:2]
                    else:
                        xy = particle.qf[:2]
                except: #the coords don't exist. Sometimes this is expected. try and fall back to another
                    if particle.qi is not None: xy=particle.qi[:2]
                    elif particle.qf is not None: xy=particle.qf[:2]
                    else: raise ValueError()
                    color='yellow'
                plt.scatter(*xy, marker='x', s=xMarkerSize, c=color)
                plt.scatter(*xy, marker='o', s=10, c=color)
            if showTraceLines==True:
                if particle.qArr is not None and len(particle.qArr)>0: #if there are lines to show
                    plt.plot(particle.qArr[:,0],particle.qArr[:,1],c=color,alpha=traceLineAlpha)

        for el in self.elList:
            shapelyObject=el.SO if not plotElementExterior else el.SO_Outer
            plt.plot(*shapelyObject.exterior.xy,c='black')
        if particleCoords is not None: #plot from the provided particle coordinate
            if len(particleCoords)==3: #if the 3d value is provided trim it to 2D
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
        if saveTitle is not None:
            plt.savefig(saveTitle,dpi=dpi)
        plt.show()