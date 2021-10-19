import dill
from tqdm import tqdm
import time
from SwarmTracerClass import SwarmTracer
import celluloid
import warnings
import numpy as np
from ParticleClass import Swarm
from ParticleClass import Particle as ParticleBase
from ParticleTracerLatticeClass import ParticleTracerLattice
import matplotlib.pyplot as plt


def make_Test_Swarm_And_Lattice(numParticles=100,totalTime=.1)->(Swarm,ParticleTracerLattice):
    # PTL=ParticleTracerLattice(200.0)
    # PTL.add_Lens_Ideal(.4,1.0,.025)
    # PTL.add_Drift(.1)
    # PTL.add_Lens_Ideal(.4,1.0,.025)
    # # PTL.add_Bender_Ideal_Segmented_With_Cap(200,.01,.1,1.0,.01,1.0,.001,1e-6)
    # PTL.add_Bender_Ideal(np.pi,1.0,1.0,.025)
    # PTL.add_Drift(.2)
    # PTL.add_Lens_Ideal(.2,1.0,.025)
    # PTL.add_Drift(.1)
    # PTL.add_Lens_Ideal(.2,1.0,.025)
    # PTL.add_Drift(.2)
    # PTL.add_Bender_Ideal(np.pi,1.0,1.0,.025)
    # PTL.end_Lattice()

    file=open('ringFile','rb')
    PTL=dill.load(file)
    # swarmTracer=SwarmTracer(PTL)
    # swarm=swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(5e-3,5.0,1e-5,numParticles)
    # swarm=swarmTracer.trace_Swarm_Through_Lattice(swarm,5e-6,totalTime,fastMode=False,parallel=True)
    # file=open('swarmFile','wb')
    # dill.dump(swarm,file)
    file=open('swarmFile','rb')
    swarm=dill.load(file)
    print('swarm and lattice done')
    return swarm,PTL
class Particle(ParticleBase):
    def __init__(self,qi=None,pi=None):
        super().__init__(qi=qi,pi=pi)
        self.qo=None
        self.po=None
        self.E=None
        self.deltaE=None
class SwarmSnapShot:
    def __init__(self,swarm:Swarm,xSnapShot):
        assert xSnapShot>0.0 #orbit coordinates, not real coordinates
        self.swarmPhaseSpace=Swarm()
        self._take_SnapShot(swarm,xSnapShot)
    def __iter__(self):
        return (particle for particle in self.swarmPhaseSpace)
    def _take_SnapShot(self,swarm,xSnapShot):
        for particle in swarm:
            assert particle.dataLogging==True
            particleSnapShot=Particle(qi=particle.qi.copy(),pi=particle.pi.copy())
            particleSnapShot.probability=particle.probability
            if self._check_If_Particle_Made_It_To_x(particle,xSnapShot)==False:
                particleSnapShot.qo=particle.qoArr[-1].copy()
                particleSnapShot.po=particle.pArr[-1].copy()
                particleSnapShot.p=particle.p.copy()
                particleSnapShot.q=particle.q.copy()
                particleSnapShot.E=particle.EArr[-1].copy()
                particleSnapShot.clipped=True
            else:
                E,qo,po=self._get_Phase_Space_Coords_And_Energy_SnapShot(particle,xSnapShot)
                particleSnapShot.E=E
                particleSnapShot.deltaE=E-particle.EArr[0].copy()
                particleSnapShot.qo=qo
                particleSnapShot.po=po
                particleSnapShot.clipped=False
            self.swarmPhaseSpace.particles.append(particleSnapShot)
        if self.swarmPhaseSpace.survival_Bool()==0:
            warnings.warn("There are no particles that survived to the snapshot position")
    def _check_If_Particle_Made_It_To_x(self,particle,x):
        #this assumes orbit coordinates
        if len(particle.qoArr)==0: return False #clipped immediately probably
        elif particle.qoArr[-1,0]>x:return True
        else:return False
    def _get_Phase_Space_Coords_And_Energy_SnapShot(self,particle,xSnapShot):
        qoArr=particle.qoArr #position in orbit coordinates
        poArr=particle.pArr
        EArr=particle.EArr
        assert xSnapShot<qoArr[-1,0]
        indexBefore=np.argmax(qoArr[:,0]>xSnapShot)-1
        qo1=qoArr[indexBefore]
        qo2=qoArr[indexBefore+1]
        stepFraction=(xSnapShot-qo1[0])/(qo2[0]-qo1[0])
        qSnapShot=self._interpolate_Array(qoArr,indexBefore,stepFraction)
        pSnapShot=self._interpolate_Array(poArr,indexBefore,stepFraction)
        ESnapShot=self._interpolate_Array(EArr,indexBefore,stepFraction)
        return ESnapShot,qSnapShot,pSnapShot
    def _interpolate_Array(self,arr,indexBegin,stepFraction):
        #v1: bounding vector before
        #v2: bounding vector after
        #stepFraction: fractional position between v1 and v2 to interpolate
        assert 0.0<=stepFraction<=1.0
        v=arr[indexBegin]+(arr[indexBegin+1]-arr[indexBegin])*stepFraction
        return v
    def get_Surviving_Particle_PhaseSpace_Coords(self):
        #get coordinates of surviving particles
        qoList=[]
        poList=[]
        for particle in self.swarmPhaseSpace:
            if particle.clipped==False:
                qoList.append(particle.qo)
                poList.append(particle.po)
        phaseSpaceCoords=np.column_stack((qoList,poList))
        return phaseSpaceCoords
    def get_surviving_Particles_Energy(self,returnChangeInE=False):
        EList=[]
        for particle in self.swarmPhaseSpace:
            if particle.clipped==False:
                if returnChangeInE==True:
                    EList.append(particle.deltaE)
                else:
                    EList.append(particle.E)
        return np.asarray(EList)
class PhaseSpaceAnalyzer:
    def __init__(self,swarm,lattice: ParticleTracerLattice):
        assert lattice.latticeType=='storageRing'
        for particle in swarm: assert particle.dataLogging==True
        self.swarm=swarm
        self.lattice=lattice
    def _get_Plot_Data_From_SnapShot(self,snapShotPhaseSpace,xAxis,yAxis):
        strinNameArr=np.asarray(['y','z','px','py','pz'])
        xAxisIndex=np.argwhere(strinNameArr==xAxis)[0][0]
        yAxisIndex=np.argwhere(strinNameArr==yAxis)[0][0]
        xAxisArr=snapShotPhaseSpace[:,xAxisIndex+1]
        yAxisArr=snapShotPhaseSpace[:,yAxisIndex+1]
        return xAxisArr,yAxisArr
    def _find_Max_Xorbit_For_Swarm(self,timeStep=-1):
        #find the maximum longitudinal distance a particle has traveled
        xMax=0.0
        for particle in self.swarm:
            if len(particle.qoArr)>0:
                xMax=max([xMax,particle.qoArr[timeStep,0]])
        return xMax
    def _find_Inclusive_Min_XOrbit_For_Swarm(self):
        #find the smallest x that as ahead of all particles, ie inclusive
        xMin=0.0
        for particle in self.swarm:
            if len(particle.qoArr)>0:
                xMin=max([xMin,particle.qoArr[0,0]])
        return xMin
    def _make_SnapShot_Position_Arr_At_Same_X(self,xVideoPoint):
        xMax=self._find_Max_Xorbit_For_Swarm()
        revolutionsMax=int((xMax-xVideoPoint)/self.lattice.totalLength)
        assert revolutionsMax>0
        xArr=np.arange(revolutionsMax+1)*self.lattice.totalLength+xVideoPoint
        return xArr
    def plot_Lattice_On_Axis(self,ax,plotPointCoords=None):
        for el in self.lattice.elList:
            ax.plot(*el.SO.exterior.xy,c='black')
        if plotPointCoords is not None:
            ax.scatter(plotPointCoords[0],plotPointCoords[1],c='red',marker='x',s=100)
    def _make_Phase_Space_Video_For_X_Array(self,videoTitle,xOrbitSnapShotArr,xaxis,yaxis,alpha,fps):
        fig,axes=plt.subplots(2,1)
        camera=celluloid.Camera(fig)
        labels,unitModifier=self._get_Axis_Labels_And_Unit_Modifiers(xaxis,yaxis)
        swarmAxisIndex=0
        latticeAxisIndex=1
        axes[swarmAxisIndex].set_xlabel(labels[0])
        axes[swarmAxisIndex].set_ylabel(labels[1])
        axes[swarmAxisIndex].text(0.1,1.1,'Phase space portraint'
                                  ,transform=axes[swarmAxisIndex].transAxes)
        axes[latticeAxisIndex].set_xlabel('meters')
        axes[latticeAxisIndex].set_ylabel('meters')
        for xOrbit,i in zip(xOrbitSnapShotArr,range(len(xOrbitSnapShotArr))):
            snapShotPhaseSpaceCoords=SwarmSnapShot(self.swarm,xOrbit).get_Surviving_Particle_PhaseSpace_Coords()
            if len(snapShotPhaseSpaceCoords)==0:
                break
            else:
                xCoordsArr,yCoordsArr=self._get_Plot_Data_From_SnapShot(snapShotPhaseSpaceCoords,xaxis,yaxis)
                revs=int(xOrbit/self.lattice.totalLength)
                deltaX=xOrbit-revs*self.lattice.totalLength
                axes[swarmAxisIndex].text(0.1,1.01,'Revolutions: '+str(revs)+ ', Distance along revolution: '+str(np.round(deltaX,2))+'m'
                                          ,transform=axes[swarmAxisIndex].transAxes)
                axes[swarmAxisIndex].scatter(xCoordsArr*unitModifier[0],yCoordsArr*unitModifier[1],
                                             c='blue',alpha=alpha,edgecolors=None,linewidths=0.0)
                axes[swarmAxisIndex].grid()
                xSwarmLab,ySwarmLab=self.lattice.get_Lab_Coords_From_Orbit_Distance(xOrbit)
                self.plot_Lattice_On_Axis(axes[latticeAxisIndex],[xSwarmLab,ySwarmLab])
                camera.snap()
        plt.tight_layout()
        animation=camera.animate()
        animation.save(str(videoTitle)+'.gif',fps=fps)
    def _check_Axis_Choice(self,xaxis,yaxis):
        validPhaseCoords=['y','z','px','py','pz']
        assert (xaxis in validPhaseCoords) and (yaxis in validPhaseCoords)
    def _get_Axis_Labels_And_Unit_Modifiers(self,xaxis,yaxis):
        positionLabel='Position, mm'
        momentumLabel='Momentum, 1*m/s'
        positionUnitModifier=1e3
        if xaxis in ['y','z']: #there has to be a better way to do this
            labelsList=[positionLabel]
            unitModifier=[positionUnitModifier]
        else:
            labelsList=[momentumLabel]
            unitModifier=[1]
        if yaxis in ['y','z']:
            labelsList.append(positionLabel)
            unitModifier.append(positionUnitModifier)
        else:
            labelsList.append(momentumLabel)
            unitModifier.append(1.0)
        return labelsList,unitModifier
    def _make_SnapShot_XArr(self,numPoints):
        #revolutions: set to -1 for using the largest number possible based on swarm
        xMax=self._find_Max_Xorbit_For_Swarm()
        xStart=self._find_Inclusive_Min_XOrbit_For_Swarm()
        return np.linspace(xStart,xMax,numPoints)

    def make_Phase_Space_Movie_At_Repeating_Lattice_Point(self,videoTitle,xVideoPoint,xaxis='y',yaxis='z',
                                                          videoLengthSeconds=10.0,alpha=.25):
        # xPoint: x point along lattice that the video is made at
        # xaxis: which cooordinate is plotted on the x axis of phase space plot
        # yaxis: which coordine is plotted on the y axis of phase plot
        # valid selections for xaxis and yaxis are ['y','z','px','py','pz']. Do not confuse plot axis with storage ring
        #axis. Storage ring axis has x being the distance along orbi, y perpindicular and horizontal, z perpindicular to
        #floor
        assert xVideoPoint<self.lattice.totalLength
        self._check_Axis_Choice(xaxis,yaxis)
        numFrames=int(self._find_Max_Xorbit_For_Swarm()/self.lattice.totalLength)
        fpsApprox=min(int(numFrames/videoLengthSeconds),1)
        print(fpsApprox,numFrames)
        xSnapShotArr=self._make_SnapShot_Position_Arr_At_Same_X(xVideoPoint)
        self._make_Phase_Space_Video_For_X_Array(videoTitle,xSnapShotArr,xaxis,yaxis,alpha,fpsApprox)
    def make_Phase_Space_Movie_Through_Lattice(self,title,xaxis,yaxis,videoLengthSeconds=10.0,fps=30,alpha=.25):
        #maxVideoLengthSeconds: The video can be no longer than this, but will otherwise shoot for a few fps
        self._check_Axis_Choice(xaxis,yaxis)
        numFrames=int(videoLengthSeconds*fps)
        xArr=self._make_SnapShot_XArr(numFrames)
        print('making video')
        self._make_Phase_Space_Video_For_X_Array(title,xArr,xaxis,yaxis,alpha,fps)
    def plot_Survival_Versus_Time(self,TMax=None,axis=None):
        TSurvivedList=[]
        for particle in self.swarm:
            TSurvivedList.append(particle.T)
        if TMax is None: TMax=max(TSurvivedList)

        TSurvivedArr=np.asarray(TSurvivedList)
        numTPoints=1000
        TArr=np.linspace(0.0,TMax,numTPoints)
        TArr=TArr[:-1] #exlcude last point because all particles are clipped there
        survivalList=[]
        for T in TArr:
            numParticleSurvived=np.sum(TSurvivedArr>T)
            survival=100*numParticleSurvived/self.swarm.num_Particles()
            survivalList.append(survival)
        TRev=self.lattice.totalLength/self.lattice.v0Nominal
        if axis is None:
            plt.title('Percent particle survival versus revolution time')
            plt.plot(TArr,survivalList)
            plt.xlabel('Time,s')
            plt.ylabel('Survival, %')
            plt.axvline(x=TRev,c='black',linestyle=':')
            plt.text(TRev+TMax*.05,20,'Revolution \n time')
            plt.show()
        else:
            axis.plot(TArr,survivalList)
    def plot_Energy_Growth(self,numPoints=100):
        fig,axes=plt.subplots(2,1)
        xSnapShotArr=self._make_SnapShot_XArr(numPoints)[:-10]
        EList_RMS=[]
        EList_Mean=[]
        EList_Max=[]
        survivalList=[]
        for xOrbit in tqdm(xSnapShotArr):
            snapShot=SwarmSnapShot(self.swarm,xOrbit)
            deltaESnapShot=snapShot.get_surviving_Particles_Energy(returnChangeInE=True)
            minParticlesForStatistics=5
            if len(deltaESnapShot)<minParticlesForStatistics:
                break
            survivalList.append(100*len(deltaESnapShot)/self.swarm.num_Particles())
            E_RMS=np.std(deltaESnapShot)
            EList_RMS.append(E_RMS)
            EList_Mean.append(np.mean(np.abs(deltaESnapShot)))
            EList_Max.append(np.max(deltaESnapShot))
        axes[0].plot(xSnapShotArr[:len(EList_RMS)],EList_RMS,label='RMS')
        axes[0].plot(xSnapShotArr[:len(EList_RMS)],EList_Mean,label='mean')
        axes[1].plot(xSnapShotArr[:len(EList_RMS)],survivalList)
        axes[0].set_ylabel('Energy change, simulation units \n (Mass Li=1.0) ')
        axes[1].set_ylabel('Percent Survival ')
        axes[1].set_xlabel('Distance along orbit, m')
        axes[0].legend()
        plt.show()


    def plot_Standing_Envelope(self):
        raise Exception('This is broken because I am trying to use ragged arrays basically')
        maxCompletedRevs=int(self._find_Max_Xorbit_For_Swarm()/self.lattice.totalLength)
        assert maxCompletedRevs>1
        xStart=self._find_Inclusive_Min_XOrbit_For_Swarm()
        xMax=self.lattice.totalLength
        numEnvPoints=50
        xSnapShotArr=np.linspace(xStart,xMax,numEnvPoints)
        yCoordIndex=1
        envelopeData=np.zeros((numEnvPoints,1)) #this array will have each row filled with the relevant particle
        #particle parameters for all revolutions.
        for revNumber in range(maxCompletedRevs):
            revCoordsList=[]
            for xOrbit in xSnapShotArr:
                xOrbit+=self.lattice.totalLength*revNumber
                snapShotPhaseSpaceCoords=SwarmSnapShot(self.swarm,xOrbit).get_Surviving_Particle_PhaseSpace_Coords()
                revCoordsList.append(snapShotPhaseSpaceCoords[:,yCoordIndex])
            envelopeData=np.column_stack((envelopeData,revCoordsList))
            # rmsArr=np.std(np.asarray(revCoordsList),axis=1)
        meterToMM=1e3
        rmsArr=np.std(envelopeData,axis=1)
        plt.title('RMS envelope of particle beam in storage ring.')
        plt.plot(xSnapShotArr,rmsArr*meterToMM)
        plt.ylabel('Displacement, mm')
        plt.xlabel('Distance along lattice, m')
        plt.ylim([0.0,rmsArr.max()*meterToMM])
        plt.show()