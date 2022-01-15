import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import joblib
from ParticleClass import Particle, Swarm
from SwarmTracerClass import SwarmTracer
from HalbachLensClassOLD import Sphere
import scipy.optimize as spo
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleTracerClass import ParticleTracer
import numpy as np
import time
import skopt

import multiprocess as mp

import matplotlib.pyplot as plt



class ShimOptimizer:
    def __init__(self):
        self.lattice=None
        self.shimList=[]
        self.shimBVecEvalCoords=None
        self.shimmedLens=None
        self.endDriftElement=None
        self.test=[]
        self.brillianceWithNoShim=None
    def beam_Brilliance_Last_Drift_Region(self,x, qArr, pArr,fracForAnalysis=.9):
        #fracForAnalysis: To choose wether to base beam size on 95% or 90% etc beam size. Ie, the radius up to the 90th
        #percent particle
        xEnd = self.endDriftElement.r2[0]
        xStart = self.endDriftElement.r1[0]
        assert xEnd<x<xStart #remember x is negative
        survivedIndices = np.argwhere(qArr[:, 0] < x)
        survivedIndices = survivedIndices.T[0]
        qArrSurvived = qArr[survivedIndices]
        pArrSurvived = pArr[survivedIndices]
        numSurvived=qArrSurvived.shape[0]
        assert numSurvived>5
        myArr = pArrSurvived[:, 1] / pArrSurvived[:, 0]
        mzArr = pArrSurvived[:, 2] / pArrSurvived[:, 0]
        yEvalArr = myArr * (x - xEnd) + qArrSurvived[:, 1]
        zEvalArr = mzArr * (x - xEnd) + qArrSurvived[:, 2]

        rArr=np.sqrt(yEvalArr**2+zEvalArr**2)
        sortIndices=np.argsort(rArr)
        sortIndices=sortIndices[:int(fracForAnalysis*rArr.shape[0])]
        flux=sortIndices.shape[0]
        meanSpeed=abs(np.mean(pArrSurvived[sortIndices][:,0]))
        spotSize=rArr[sortIndices].max()
        meanDivergence=np.mean((np.abs(pArrSurvived[sortIndices][:,1])+np.abs(pArrSurvived[sortIndices][:,2]))/2)
        brilliance=flux*meanSpeed/(spotSize*meanDivergence)
        return brilliance
    def find_Maximum_Brilliance_And_Location(self,swarm):
        xStart = self.endDriftElement.r1[0]
        xEnd = self.endDriftElement.r2[0]
        pfArr = np.asarray([particle.pf for particle in swarm])
        qfArr = np.asarray([particle.qf for particle in swarm])
        xEvalArr = np.linspace(xStart - 1e-3, xEnd + 1e-3, num=30)
        bounds = [(xEvalArr[-1], xEvalArr[0])]
        brillianceArr = np.asarray([self.beam_Brilliance_Last_Drift_Region(xEval, qfArr, pfArr) for xEval in xEvalArr])
        xInitial = xEvalArr[np.argmax(brillianceArr)]
        scipyWrapper=lambda x:1/(self.beam_Brilliance_Last_Drift_Region(x, qfArr, pfArr)+1e-9)
        sol = spo.minimize(scipyWrapper, xInitial, bounds=bounds,method='Nelder-Mead')
        brilliance=1/sol.fun
        xOptimal=sol.x[0]
        if self.is_Spot_Location_Valid(xOptimal)==False:
            raise Exception("The spot position is invalid. It is likely there is no focus in the region of interest")
        return brilliance,xOptimal
    def is_Spot_Location_Valid(self,xSpot):
        minSepFromEdge=self.lattice.elList[-1].L/10.0
        driftEndx=self.lattice.elList[-1].r2[0]
        driftStartX=self.lattice.elList[-1].r1[0]
        if abs(xSpot-driftEndx)<minSepFromEdge or abs(xSpot-driftStartX)<minSepFromEdge:
            return False
        else: return True
    def trace_Through_Lens(self):
        swarmTracer = SwarmTracer(self.lattice) #82% of time here
        swarm = swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(1e-3, 10.0, 1.0, 1000,sameSeed=True)
        # swarm.particles=swarm.particles[3:4]
        swarm = swarmTracer.trace_Swarm_Through_Lattice(swarm, 5e-6, 1.0, fastMode=True,accelerated=False,parallel=False)
        # for particle in swarm:
        #     print(particle.T)
        # for particle in swarm:
        #     print(particle.q) #[-7.50000000e-01  6.71027029e-04  1.18056459e-03]
        #     particle.plot_Energies()
        #     EArr=particle.EArr
        #     EArr-=EArr[-1]
        #     xArr=particle.qoArr[:,0]
        #     yArr=particle.qoArr[:,1]
        #     zArr=particle.qoArr[:,2]
        #     rArr=np.sqrt(yArr**2+zArr**2)
        #     # plt.plot(xArr,rArr)
        #     # plt.show()
        #     # plt.plot(particle.qoArr[:,0],EArr)
        # # plt.show()
        # self.lattice.show_Lattice(swarm=swarm,showTraceLines=True,trueAspectRatio=False)
        # particleTracer=ParticleTracer(self.PTL)
        # particle=Particle(qi=np.asarray([-1e-10,5e-3,0.0]))
        # particle=particleTracer.trace(particle,1e-6,1.0)
        # particle.plot_Energies()
        return swarm

    def cost_Stitch_Function(self,cost):
        #funcgion that stiches the costs together to make a more continues spectrum. Uses the sigmoid function
        if cost<=1.0:
            return cost
        else:
            return 2 / (1 + np.exp(-(cost - 1) / 1.0))
    def cost(self,args):
        self.update_Shims(args)
        if self.test_If_Shim_Is_Valid(args)==False:
            return self.cost_Stitch_Function(np.inf)
        self.update_Shimmed_Lens() #29% here
        swarm=self.trace_Through_Lens() #71% here
        brilliance=self.find_Maximum_Brilliance_And_Location(swarm)[0]
        cost=1/(brilliance/self.brillianceWithNoShim)
        return self.cost_Stitch_Function(cost)
    def get_Magnification(self,x):
        assert self.is_Spot_Location_Valid(x)
        lensFringeEdgeGap=(self.lattice.elList[1].L-self.lattice.elList[1].Lm)/2 #extra space is added to element edge
        # for fringe fields
        objectDistance=self.lattice.elList[0].L+lensFringeEdgeGap
        imageDistance=(self.lattice.elList[1].r2[0]-lensFringeEdgeGap)-x
        magnification=objectDistance/imageDistance
        return magnification
    def fill_Shim_List(self,numShims):
        self.shimList=[Sphere() for _ in range(numShims)]
    def test_If_Shim_Is_Valid(self,args):
        #enforce geometric constraints of spheres and magnet. Geometry here
        #is cylinder with a cylinderical cutout inside. return True if okay, False is geometry is violated.
        #because of symmetry the sphere is allowed to be as close to the symmetry plane as it like, even right on the
        #middle
        #args: configuraton arguments
        for shim,i in zip(self.shimList,range(len(self.shimList))):
            r,phi,z,theta,psi,radius=self.get_Shim_Parameters_From_Args(args, i)
            shim.update_Size(radius)
            #check if shim is above or below top of yoke
            if z-shim.radius>self.shimmedLens.Lm/2: #bottom shim is above top of magnet and can be anywhere:
                result= True
            elif z>self.shimmedLens.Lm/2 and z-shim.radius<self.shimmedLens.Lm/2: #shim is in the intermediate
                #region where it can kind of roll off the edge
                if self.shimmedLens.rp<r<self.shimmedLens.rp+self.shimmedLens.yokeWidth: #shim is within the yoke radially
                    #so bottom edge is clipping top of magnet
                    result= False
                elif r+shim.radius<self.shimmedLens.rp or r-shim.radius>self.shimmedLens.rp+self.shimmedLens.yokeWidth:
                    #shim is definitely clear of magnet radially
                    result=True
                else: #shim is in intermediate region where it can kind of roll off the edge. Test by finding distance
                    #from edge to center of shim
                    deltaR=r-self.shimmedLens.rp
                    deltaZ=z-self.shimmedLens.Lm/2
                    seperation=np.sqrt(deltaZ**2+deltaR**2)
                    if seperation< shim.radius:
                        result= False #no logic here for now
                    else:
                        result=True
            elif (z<self.shimmedLens.Lm/2 and z-shim.radius>0): #center below top of magnet
                # and above zero
                if self.shimmedLens.rp<r+shim.radius and r-shim.radius<self.shimmedLens.rp+self.shimmedLens.yokeWidth: #check if inside
                    #yoke
                    result= False
                elif r-shim.radius<self.shimmedLens.ap:#check if inside vacuum tube
                    result= False
                else:
                    result= True
            else:  #if anywhere else
                result= False
            if result==False: #if invalid geometry for any shim return False
                return False
        return True #if all tests passed, return True
    def get_Shim_Parameters_From_Args(self, args, i):
        #unpack the parameteres from the differential evolution algorith'm test arguments for the ith sphere
        numVariablesPerShim=6
        assert len(args)//numVariablesPerShim==len(self.shimList) and len(args)%numVariablesPerShim==0
        r,phi,z,theta,psi,radius=args[numVariablesPerShim*i:numVariablesPerShim*(i+1)]
        return r,phi,z,theta,psi,radius
    def update_Shims(self,args):
        for shim,i in zip(self.shimList,range(len(self.shimList))):
            r,phi,z,theta,psi,radius=self.get_Shim_Parameters_From_Args(args, i)
            x=r*np.cos(phi)
            y=r*np.sin(phi)
            shim.r0=np.asarray([x,y,z])
            shim.orient(theta,psi)
            shim.update_Size(radius)
    def build_Lattice(self,parallel):
        self.lattice = ParticleTracerLattice(200.0,latticeType='injector',parallel=parallel)
        self.lattice.add_Drift(.5)
        self.lattice.add_Halbach_Lens_Sim_Shim(.05, .5)
        self.lattice.add_Drift(1.0)
        self.lattice.end_Lattice(enforceClosedLattice=False)
        self.shimmedLens=self.lattice.elList[1]
        self.endDriftElement=self.lattice.elList[2]
    def update_Shimmed_Lens(self):
        evalCoords = self.shimmedLens.BVecCoordsArr
        BVecArr = np.zeros(evalCoords.shape)
        for shim in self.shimList:
            BVecArr += shim.B_Shim(evalCoords)
        self.shimmedLens.update_Force_And_Potential_Function(BVecArr)
    def get_Shim_System_Bounds(self,numShims):
        #bounds are min max for the following
        #[radial position,theta in cylinderical coordinates,z in cylinderical coordinates,azimuthal tilt, polar rotation
        #of shim, and radius of spherical shim]
        minPhiAngle=-30*np.pi/180
        maxPhiAngle=30*np.pi/180
        boundSingleShim = [(self.shimmedLens.ap,
                            self.shimmedLens.rp + self.shimmedLens.yokeWidth*2),
                           (minPhiAngle, maxPhiAngle),
                           (0.0, self.shimmedLens.Lm / 2 + 1.5*self.shimmedLens.rp), (0, np.pi),
                           (0, 2 * np.pi),(.0254/5,.0254)]
        bounds=[]
        for _ in range(numShims): bounds.extend(boundSingleShim)
        return bounds
    def get_Brilliance_With_No_Shims(self):
        swarm = self.trace_Through_Lens()  # 71% here
        brilliance = self.find_Maximum_Brilliance_And_Location(swarm)[0]
        return brilliance

    def optimize(self,numShims):
        self.build_Lattice(True)
        self.brillianceWithNoShim=self.get_Brilliance_With_No_Shims()
        self.fill_Shim_List(numShims)
        bounds=self.get_Shim_System_Bounds(numShims)
        def custom_Map(func,workList):
            with mp.Pool(10,maxtasksperchild=9999) as pool:
                results=pool.map(func,workList,chunksize=1)
            return results
        def function_To_Minimize(args):
            cost=self.cost(args)
            return cost
        np.random.seed(42)
        sol=spo.differential_evolution(function_To_Minimize,bounds,workers=custom_Map,disp=True,polish=False)
        print(sol)