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
from geneticLensClass import GeneticLens
import scipy.interpolate as spi


t=time.time()




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


class helper:
    def __init__(self,lens,apMin):
        self.lens=lens
        if apMin is None:
            self.apMin=self.lens.minimum_Radius()*.95
        else:
            self.apMin=apMin
        self.L_Object=.72
        self.L_Image=.8
        self.fringeFrac=4.0
        self.PTL=None

    def make_Lattice(self):

        self.PTL=ParticleTracerLattice(v0Nominal=210.0,latticeType='injector',parallel=True)
        self.PTL.add_Drift(self.L_Object-self.fringeFrac*self.lens.maximum_Radius(),ap=.07)
        self.PTL.add_Genetic_lens(self.lens,self.apMin)
        self.PTL.add_Drift(1.5*(self.L_Image-self.fringeFrac*self.lens.maximum_Radius()),ap=.07)
        self.PTL.end_Lattice()
        assert self.PTL.elList[1].fringeFracOuter==self.fringeFrac and self.PTL.elList[1].rp==self.lens.maximum_Radius()
    def make_Interp_Function(self):
        swarmTracer=SwarmTracer(self.PTL)
        angle=.09
        v0=210.0
        swarm=swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(1e-6,angle*v0,1.0,1000,sameSeed=True)
        h=1e-5
        fastMode=True
        swarmTraced=swarmTracer.trace_Swarm_Through_Lattice(swarm,h,1.0,fastMode=fastMode,
                                                            jetCollision=False)
        # for particle in swarmTraced:
        #     particle.plot_Energies()
        if fastMode==False:
            self.PTL.show_Lattice(swarm=swarmTraced,trueAspectRatio=False,showTraceLines=True,traceLineAlpha=.1)
        interpFunc=Interpolater(swarmTraced,self.PTL)
        return interpFunc
    def atom_focus_Intensity(self,x,interpFunc,Print=False):
        yArr, zArr,pArr = interpFunc(x,returnP=True)
        beamArea=np.std(yArr)*np.std(zArr)
        if Print==True:
            print(':',beamArea,len(yArr))
        return len(yArr)/beamArea

        # rArr = np.sqrt(yArr ** 2 + zArr ** 2)
        # numParticles = len(yArr)
        # D_90 = np.sort(rArr)[int(numParticles * .9)]
        # return numParticles*.9/D_90
    def get_Magnification(self,xFocus):
        L_Focus=xFocus-(abs(self.PTL.elList[1].r2[0])-self.fringeFrac*self.PTL.elList[1].rp)
        return L_Focus/self.L_Object
    def IPeak_And_Magnification_From_Lens(self):
        self.make_Lattice()
        interpFunc=self.make_Interp_Function()
        xArr=np.linspace(-self.PTL.elList[-1].r1[0]+2e-3,-self.PTL.elList[-1].r2[0]-2e-3,300)
        IArr=np.asarray([self.atom_focus_Intensity(x,interpFunc) for x in xArr])
        RBF_Func=spi.Rbf(xArr,IArr)
        xDense=np.linspace(xArr.min(),xArr.max(),10_000)
        IDense=RBF_Func(xDense)
        # plt.plot(xArr,IArr)
        # plt.show()
        xFocus=xDense[np.argmax(IDense)]

        IPeak=np.max(IDense)
        m=self.get_Magnification(xFocus)
        return IPeak,m
def IPeak_And_Magnification_From_Lens(lens,apMin=None):
    return helper(lens,apMin).IPeak_And_Magnification_From_Lens()

