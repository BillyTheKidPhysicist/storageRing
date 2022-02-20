import dill
from ParticleTracerClass import ParticleTracer
from phaseSpaceAnalyzer import SwarmSnapShot
import dill
from tqdm import tqdm
import time
from SwarmTracerClass import SwarmTracer
import warnings
import numpy as np
from ParticleClass import Swarm,Particle
from ParticleClass import Particle as ParticleBase
from ParticleTracerLatticeClass import ParticleTracerLattice
import matplotlib.pyplot as plt
from lensGeneticElement import geneticLensElement
from HalbachLensClass import GeneticLens
import scipy.interpolate as spi


t=time.time()

LObject=72.0E-2
LImage=85e-2
LLensHardEdge=15.24e-2
rpLens=(5e-2,5e-2 +2.54e-2)
magnetWidth=.0254#(.0254,.0254*1.5)
Lm=.3

fringeFrac=1.5
LFringe=fringeFrac*max(rpLens)
LLens=LLensHardEdge+2*LFringe
LObject-=LFringe
LImage-=LFringe


class Interpolater:
    def __init__(self,swarm,PTL):
        self.swarm=swarm
        self.PTL=PTL
        self.endDriftLength=abs(self.PTL.elList[-1].r2[0]-self.PTL.elList[-1].r1[0])
    def __call__(self,xOrbit,maxRadius=np.inf,vTMax=np.inf,returnP=False,useAssert=True,useInitial=False):
        #xOrbit: Distance in orbit frame, POSITIVE to ease with analyze. Know that the tracing is done with x being negative
        #returns in units of mm
        #vTMax: maximum transverse velocity for interpolation
        #useAssert: I can use this interplater elsewhere if I turn this off
        if useAssert==True:
            assert -self.PTL.elList[-1].r2[0]>xOrbit>-self.PTL.elList[-1].r1[0]
        yList=[]
        zList=[]
        pList=[]
        for particle in self.swarm.particles:
            if useInitial==True:
                p,q=particle.pi,particle.qi
            else:
                p,q=particle.pf,particle.qf
            vT=np.sqrt(p[1]**2+p[2]**2)
            if (q[0]<-xOrbit and vT<vTMax) or useInitial==True:
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




def make_Lattice(Lens,apMin):
    PTL=ParticleTracerLattice(v0Nominal=210.0,latticeType='injector',parallel=True)
    PTL.add_Drift(LObject,ap=.07)
    PTL.add_Genetic_lens(Lens,apMin)
    PTL.add_Drift(LImage,ap=.07)
    # assert PTL.elList[1].fringeFracOuter==fringeFrac and abs(PTL.elList[1].Lm-LLensHardEdge)<1e-9
    PTL.end_Lattice()
    return PTL
def make_Interp_Function(PTL):
    swarmTracer=SwarmTracer(PTL)
    angle=.1
    v0=210.0
    swarm=swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(1e-6,angle*v0,1.0,300,sameSeed=True)
    h=1e-5
    fastMode=True
    swarmTraced=swarmTracer.trace_Swarm_Through_Lattice(swarm,h,1.0,fastMode=fastMode,
                                                        jetCollision=False)
    # for particle in swarmTraced:
    #     particle.plot_Energies()
    if fastMode==False:
        PTL.show_Lattice(swarm=swarmTraced,trueAspectRatio=False,showTraceLines=True)
    interpFunc=Interpolater(swarmTraced,PTL)
    return interpFunc
def atom_focus_Intensity(x,interpFunc,Print=False):
    yArr, zArr,pArr = interpFunc(x,returnP=True)
    beamArea=np.std(yArr)*np.std(zArr)
    if Print==True:
        print(':',beamArea,len(yArr))
    return len(yArr)/beamArea

    # rArr = np.sqrt(yArr ** 2 + zArr ** 2)
    # numParticles = len(yArr)
    # D_90 = np.sort(rArr)[int(numParticles * .9)]
    # return numParticles*.9/D_90
def get_Magnification(xFocus):
    LImage=xFocus-(LObject+LLensHardEdge)
    return LImage/LObject
def IPeak_And_Magnification_From_Lens(lens,apMin):
    PTL=make_Lattice(lens,apMin)
    interpFunc=make_Interp_Function(PTL)
    xArr=np.linspace(-PTL.elList[-1].r1[0]+1e-3,-PTL.elList[-1].r2[0]-1e-3,300)
    IArr=np.asarray([atom_focus_Intensity(x,interpFunc) for x in xArr])
    RBF_Func=spi.Rbf(xArr,IArr)
    xDense=np.linspace(xArr.min(),xArr.max(),10_000)
    IDense=RBF_Func(xDense)
    # plt.plot(xArr,IArr)
    # plt.show()
    xFocus=xDense[np.argmax(IDense)]

    IPeak=np.max(IDense)
    m=get_Magnification(xFocus)
    return IPeak,m

