from injectionOptimizer import ApetureOptimizer
import sys
import scipy.interpolate as spi
import warnings
import time
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
# from storageRingOptimization import elementPT
# from storageRingOptimization.ParticleTracer import ParticleTracer
from storageRingOptimization.ParticleClass import  Swarm,Particle
# import storageRingOptimization.elementPT
from storageRingOptimization.ParticleTracerLatticeClass import ParticleTracerLattice
from storageRingOptimization.SwarmTracer import SwarmTracer
from shapely.geometry import LineString
# from profilehooks import profile

class collimationOptimization(ApetureOptimizer):
    def __init__(self,h=1e-5):
        self.h=1e-5
        self.lattice=None #to hold the lattice object
        self.X0={'LDrift':.5}
        self.X={'Lo':None,'Lm':None,'Bp':None,'rp':None}
        self.v0Nominal=200.0
        self.swarmInitial=None

        self.minAspectRatio=6.1
    def trace_Through_Collimater(self,parallel=False):
        return super().trace_Through_Bumper(parallel=parallel)
    def updateX_With_List(self,args):
        Bp,Lm,rp,Lo=args
        self.X['Bp']=Bp
        self.X['Lm']=Lm
        self.X['rp']=rp
        self.X['Lo']=Lo
    def build_Lattice(self):
        #sigma here moves the element upwards
        self.lattice = ParticleTracerLattice(self.v0Nominal)
        self.lattice.add_Drift(self.X['Lo'])
        self.lattice.add_Lens_Sim_With_Caps('lens2D_Injection_Short.txt','lens3D_Injection_Meshfactor1.txt',self.X['Lm'],rp=self.X['rp'])
        self.lattice.add_Drift(self.X0['LDrift'])
        self.lattice.end_Lattice(enforceClosedLattice=False, latticeType='injector', surpressWarning=True,trackPotential=True)
        self.lattice.elList[1].BpFact=self.X['Bp'] #tune down the models
    def plot_Swarm(self,args,numParticles=1000,Lo=.1,parallel=True):
        super().initialize_Observed_Swarm(numParticles=numParticles)
        Lm=args[0]
        Bp=1.0
        rp=None
        argsLattice=[Bp,Lm,rp,Lo]
        # aspectRatioLens=Lm/rp
        # if aspectRatioLens<self.minAspectRatio:  #not a valid solution.
        #     print('aspect ratio violate')
        self.updateX_With_List(argsLattice)
        self.build_Lattice()
        swarm=self.trace_Through_Collimater(parallel=parallel)
        print('fill',self.fill_Ratio(swarm))
        print('coll',super().collimation_Factor_First_Lens(swarm))

        self.lattice.show_Lattice(swarm=swarm,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,
                                  trueAspectRatio=True)
    def optimize_Collimation(self,Lo=.1,numParticles=500):
        self.initialize_Observed_Swarm(numParticles=numParticles)
        bounds=[(0.0,1.0)]
        rp=.0236
        def cost_Function(args,returnParams=False):
            Lm=args[0]
            aspectRatio=Lm/rp
            if aspectRatio<self.minAspectRatio:
                return np.inf

            Bp=1.0
            argsLattice=[Bp,Lm,rp,Lo]
            self.updateX_With_List(argsLattice)
            self.build_Lattice()
            if returnParams==True:
                swarm=self.trace_Through_Collimater(parallel=True)
            else:
                swarm=self.trace_Through_Collimater(parallel=False)
            collimatino=self.collimation_Factor_First_Lens(swarm)
            fillRatio=self.fill_Ratio(swarm)
            cost=collimatino+fillRatio
            if returnParams==True:
                print(self.collimation_Factor_First_Lens(swarm),self.fill_Ratio(swarm))
                return argsLattice,swarm
            else:
                return cost
        sol1=spo.differential_evolution(cost_Function,bounds,workers=-1,polish=False,disp=True,maxiter=100
                                        ,mutation=(.5,1.0))
        print(sol1)
        args,swarm=cost_Function(sol1.x,returnParams=True)

        self.lattice.show_Lattice(swarm=swarm,showTraceLines=True,showMarkers=False,traceLineAlpha=.1,trueAspectRatio=True,
                                  )

    def fill_Ratio(self,swarm):
        #fraction of the width of the swarm to the aperture of the lens. Measurement can either be RMS or FWHM or any other
        L=self.lattice.elList[1].r2[0]
        width=super().get_Spot_Size(swarm,L,fraction=.95)
        Larr=np.linspace(self.lattice.elList[2].r1[0],self.lattice.elList[2].r2[0]-1e-6,num=100)
        spotSizeList=[]
        for L in Larr:
            spotSizeList.append(1e3*super().get_Spot_Size(swarm,L))
        plt.plot(1e2*Larr,spotSizeList)
        plt.ylabel('95% spot size, mm')
        plt.xlabel('Distance past end of lens, cm')
        plt.grid()
        plt.show()
        print('spot size',width)
        return width/(2*self.lattice.elList[1].ap)



def main():
    collimationOptimizer=collimationOptimization()
    args=np.array([None])
    rp=.02413
    Lo0=.1
    Lo=Lo0-1.5*rp
    collimationOptimizer.plot_Swarm(args,parallel=True,Lo=Lo,numParticles=None)
    # collimationOptimizer.optimize_Collimation()
if __name__=='__main__':
    main()