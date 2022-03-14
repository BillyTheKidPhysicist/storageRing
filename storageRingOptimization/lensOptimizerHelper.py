import time
from SwarmTracerClass import SwarmTracer
from ParticleClass import Swarm
from typing import Union,Optional
import numpy as np
from ParticleTracerLatticeClass import ParticleTracerLattice
from geneticLensElement_Wrapper import GeneticLens
import matplotlib.pyplot as plt
import scipy.optimize as spo



t=time.time()




class Interpolater:

    def __init__(self,swarm: Swarm,PTL: ParticleTracerLattice):
        self.swarm=swarm
        self.PTL=PTL
        self.endDriftLength=abs(self.PTL.elList[-1].r2[0]-self.PTL.elList[-1].r1[0])

    def __call__(self,xOrbit: float,maxRadius: float=np.inf,vTMax: float=np.inf,returnP: bool=False,
                 useAssert: bool=True,useInitial: bool=False,cutoff: bool=False)-> list:
        #xOrbit: Distance in orbit frame, POSITIVE to ease with analyze. Know that the tracing is done with x being negative
        #returns in units of mm
        #vTMax: maximum transverse velocity for interpolation
        #useAssert: I can use this interplater elsewhere if I turn this off
        #cutoff: Remove particles that didn't reach the end of the simulation region
        if useAssert==True:
            assert -self.PTL.elList[-1].r2[0]>xOrbit>-self.PTL.elList[-1].r1[0]
        assert isinstance(xOrbit,float)
        yList=[]
        zList=[]
        pList=[]
        for particle in self.swarm.particles:
            if useInitial==True:
                p,q=particle.pi,particle.qi
            else:
                p,q=particle.pf,particle.qf
            vT=np.sqrt(p[1]**2+p[2]**2)
            if cutoff==True and  abs(q[0]-self.PTL.elList[-1].r2[0])>5e-3:
                pass
            elif (q[0]<-xOrbit and vT<vTMax) or useInitial==True:
                stepFrac=(abs(q[0])-xOrbit)/self.endDriftLength
                ySlope=p[1]/p[0]
                y=q[1]+stepFrac*self.endDriftLength*ySlope
                zSlope=p[2]/p[0]
                z=q[2]+stepFrac*self.endDriftLength*zSlope
                yList.append(y)
                zList.append(z)
                pList.append(p)
        yArr=np.asarray(yList)*1e3
        zArr=np.asarray(zList)*1e3
        rArr=np.sqrt(yArr**2+zArr**2)
        yArr=yArr[rArr<maxRadius]
        zArr=zArr[rArr<maxRadius]
        pArr=np.asarray(pList)[rArr<maxRadius]
        returnArgs=[yArr,zArr]
        if returnP==True:
            returnArgs.append(pArr)
        return returnArgs


class helper:
    def __init__(self,lens: GeneticLens,apMin: float):
        self.lens=lens
        if apMin is None:
            self.apMin=self.lens.minimum_Radius()*.95
        else:
            assert apMin<=self.lens.minimum_Radius()
            self.apMin=apMin
        self.L_Object=.72
        self.L_Image=.8
        self.fringeFrac=4.0
        self.PTL=None

    def make_Lattice(self)-> None:
        self.PTL=ParticleTracerLattice(v0Nominal=210.0,latticeType='injector',parallel=True)
        self.PTL.add_Drift(self.L_Object-self.fringeFrac*self.lens.maximum_Radius(),ap=.07)
        self.PTL.add_Genetic_lens(self.lens,self.apMin)
        self.PTL.add_Drift(1.5*(self.L_Image-self.fringeFrac*self.lens.maximum_Radius()),ap=.07)
        self.PTL.end_Lattice()
        assert self.PTL.elList[1].fringeFracOuter==self.fringeFrac and self.PTL.elList[1].rp==self.lens.maximum_Radius()
        assert abs(abs(self.PTL.elList[1].r2[0])-(self.L_Object+self.lens.length+
                   self.fringeFrac*self.lens.maximum_Radius()))<1e-12

    def make_Interp_Function_Full_Swarm(self,apFrac:float =1.0,particles: int=500)-> Interpolater:
        assert 0.0<=apFrac<=1.0
        swarmTracer=SwarmTracer(self.PTL)
        angle=apFrac*self.apMin/(self.L_Object+self.lens.length/2)
        v0=self.PTL.v0Nominal
        swarm=swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(1e-6,angle*v0,1e-6,particles,sameSeed=True)
        h=1e-5
        fastMode=True
        swarmTraced=swarmTracer.trace_Swarm_Through_Lattice(swarm,h,1.0,fastMode=fastMode)
        # for particle in swarmTraced:
        #     particle.plot_Energies()
        if fastMode==False:
            self.PTL.show_Lattice(swarm=swarmTraced,trueAspectRatio=False,showTraceLines=True,traceLineAlpha=.1)
        interpFunc=Interpolater(swarmTraced,self.PTL)
        return interpFunc

    def make_Interp_Function_Concentric_Swarms(self,apArr: np.ndarray,particles: int=125):
        assert np.all(apArr<=1.0) and np.all(0.0<apArr)
        swarmTracer=SwarmTracer(self.PTL)
        h=1e-5
        fastMode=True
        interpFuncList=[]
        for ap in apArr:
            angle=ap*self.apMin/(self.L_Object+self.lens.length/2)
            swarm=swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(1e-6,angle*self.PTL.v0Nominal,1e-6,particles
                                                                          ,sameSeed=True)
            swarmTraced=swarmTracer.trace_Swarm_Through_Lattice(swarm,h,1.0,fastMode=fastMode)
            if fastMode==False:
                self.PTL.show_Lattice(swarm=swarmTraced,trueAspectRatio=False,showTraceLines=True,traceLineAlpha=.1)
            interpFuncList.append(Interpolater(swarmTraced,self.PTL))
        return interpFuncList

    def atom_Beam_Intensity(self,x: float,interpFunc: Interpolater):
        I= self.beam_Characterization(x,interpFunc)[0]
        return I

    def beam_Characterization(self,x: float,interpFunc: Interpolater)-> tuple:
        yArr, zArr, pArr = interpFunc(x, returnP=True)
        rArr=np.sqrt(yArr**2+zArr**2)
        beamAreaRMS = np.std(yArr) * np.std(zArr)
        beamAreaD90=np.sort(np.sqrt(yArr**2+zArr**2))[int(len(yArr)*.9)]
        radiusRMS=np.sqrt(np.mean(rArr**2))
        I=len(yArr)/(np.pi*radiusRMS**2)
        numD90=int(len(yArr)*.9)
        return I,beamAreaRMS,beamAreaD90,numD90,radiusRMS

    def get_Magnification(self,xFocus: float)-> float:
        L_Focus=xFocus-(abs(self.PTL.elList[1].r2[0])-self.fringeFrac*self.PTL.elList[1].rp)
        return L_Focus/self.L_Object

    def characterize_Focus(self,interpFunc: Interpolater,rejectOutOfRange: bool)-> Union[dict,None]:
        xArr = np.linspace(-self.PTL.elList[-1].r1[0] + 5e-3, -self.PTL.elList[-1].r2[0] - 5e-3, 10)
        IArr = np.asarray([self.atom_Beam_Intensity(x, interpFunc) for x in xArr])
        xi=xArr[np.argmax(IArr)]
        wrapper= lambda x: 1/self.atom_Beam_Intensity(x[0], interpFunc)
        sol=spo.minimize(wrapper,xi,bounds=[(xArr.min(),xArr.max())],method='Nelder-Mead',options=
        {'fatol':1e-9,'xatol':1e-9})
        xFocus=sol.x[0]
        # xFocus=xArr[np.argmax(IArr)]
        if abs(xFocus-xArr.max()) < .1 or abs(xFocus-xArr.min()) < .1 and rejectOutOfRange==True: #if peak is too close, answer is invalid
            return None
        IPeak,beamAreaRMS,beamAreaD90,numD90,radiusRMS = self.beam_Characterization(xFocus, interpFunc)
        m = self.get_Magnification(xFocus)
        results={'I':IPeak,'m':m,'beamAreaRMS':beamAreaRMS,'beamAreaD90':beamAreaD90,'particles in D90':numD90,
                 'radius RMS':radiusRMS,'L_Image':m*self.L_Object}
        return results

    def characterize_Lens_Full_Swarm(self,rejectOutOfRange: bool)-> dict:
        self.make_Lattice()
        interpFuncFullSwarm = self.make_Interp_Function_Full_Swarm()
        results=self.characterize_Focus(interpFuncFullSwarm,rejectOutOfRange)
        return results

    def characterize_Lens_Concentric_Swarms(self,rejectOutOfRange: bool)-> Union[list,None]:
        self.make_Lattice()
        apArr=np.asarray([.25,.5,.75,1.0])
        interpFuncList=self.make_Interp_Function_Concentric_Swarms(apArr)
        resultsList=[]
        for ap,interpFunc in zip(apArr,interpFuncList):
            resultsSingle=self.characterize_Focus(interpFunc,rejectOutOfRange)
            if resultsSingle is None:
                return None
            resultsList.append(resultsSingle)
        return resultsList

def characterize_Lens_Full_Swarm(lens: GeneticLens,rejectOutOfRange: bool,apMin: Optional[float]=None):
    return helper(lens, apMin).characterize_Lens_Full_Swarm(rejectOutOfRange)

def characterize_Lens_Concentric_Swarms(lens: GeneticLens,rejectOutOfRange: bool,apMin: Optional[float]=None):
    return helper(lens, apMin).characterize_Lens_Concentric_Swarms(rejectOutOfRange)