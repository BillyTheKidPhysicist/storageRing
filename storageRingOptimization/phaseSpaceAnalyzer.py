from tqdm import tqdm
import time
from SwarmTracerClass import SwarmTracer
import celluloid
import warnings
import numpy as np
from ParticleClass import Swarm,Particle
from ParticleTracerLatticeClass import ParticleTracerLattice
import matplotlib.pyplot as plt


def make_Test_Swarm_And_Lattice(numParticles=100,totalTime=.1):
    PTL=ParticleTracerLattice(200.0)
    PTL.add_Lens_Ideal(.4,1.0,.025)
    PTL.add_Drift(.1)
    PTL.add_Lens_Ideal(.4,1.0,.025)
    # PTL.add_Bender_Ideal_Segmented_With_Cap(200,.01,.1,1.0,.01,1.0,.001,1e-6)
    PTL.add_Bender_Ideal(np.pi,1.0,1.0,.025)
    PTL.add_Drift(.2)
    PTL.add_Lens_Ideal(.2,1.0,.025)
    PTL.add_Drift(.1)
    PTL.add_Lens_Ideal(.2,1.0,.025)
    PTL.add_Drift(.2)
    PTL.add_Bender_Ideal(np.pi,1.0,1.0,.025)
    PTL.end_Lattice()
    swarmTracer=SwarmTracer(PTL)
    swarm=swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(5e-3,5.0,1e-5,numParticles)
    swarm=swarmTracer.trace_Swarm_Through_Lattice(swarm,5e-6,totalTime,fastMode=False,parallel=True)
    return swarm,PTL
class PhaseSpaceSnapShot:
    def __init__(self,swarm:Swarm,xSnapShot):
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
                particleSnapShot.q=None
                particleSnapShot.p=None
                particleSnapShot.clipped=True
            else:
                q,p=self._get_Phase_Space_Coords_SnapShot(particle,xSnapShot)
                particleSnapShot.q=q
                particleSnapShot.p=p
                particleSnapShot.clipped=False
            self.swarmPhaseSpace.particles.append(particleSnapShot)
        if self.swarmPhaseSpace.survival_Bool()==0:
            warnings.warn("There are no particles that survived to the snapshot position")
    def _check_If_Particle_Made_It_To_x(self,particle,x):
        #this assumes orbit coordinates
        if len(particle.qoArr)==0: return False #clipped immediately probably
        elif particle.qoArr[-1,0]>x:return True
        else:return False
    def _get_Phase_Space_Coords_SnapShot(self,particle,xSnapShot):
        qoArr=particle.qoArr #position in orbit coordinates
        poArr=particle.pArr
        assert xSnapShot<qoArr[-1,0]
        indexBefore=np.argmax(qoArr[:,0]>xSnapShot)-1
        qo1=qoArr[indexBefore]
        qo2=qoArr[indexBefore+1]
        stepFraction=(xSnapShot-qo1[0])/(qo2[0]-qo1[0])
        qSnapshot=self._interpolate_Vector(qo1,qo2,stepFraction)
        po1=poArr[indexBefore]
        po2=poArr[indexBefore]
        pSnapshot=self._interpolate_Vector(po1,po2,stepFraction)
        # pSnapshot[:2]=np.nan
        return qSnapshot,pSnapshot
    def _interpolate_Vector(self,v1,v2,stepFraction):
        #v1: bounding vector before
        #v2: bounding vector after
        #stepFraction: fractional position between v1 and v2 to interpolate
        assert 0.0<=stepFraction<=1.0
        vNew=v1+(v2-v1)*stepFraction
        return vNew
    def get_Surviving_Particle_PhaseSpace_Coords(self):
        #get coordinates of surviving particles
        qList=[]
        pList=[]
        for particle in self.swarmPhaseSpace:
            if particle.clipped==False:
                qList.append(particle.q)
                pList.append(particle.p)
        phaseSpaceCoords=np.column_stack((qList,pList))
        return phaseSpaceCoords
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
            snapShotPhaseSpaceCoords=PhaseSpaceSnapShot(self.swarm,xOrbit).get_Surviving_Particle_PhaseSpace_Coords()
            if len(snapShotPhaseSpaceCoords)==0:
                break
            else:
                xCoordsArr,yCoordsArr=self._get_Plot_Data_From_SnapShot(snapShotPhaseSpaceCoords,xaxis,yaxis)
                revs=int(xOrbit/self.lattice.totalLength)
                deltaX=xOrbit-revs*self.lattice.totalLength
                axes[swarmAxisIndex].text(0.1,1.01,'Revolutions: '+str(revs)+ ', Distance along revolution: '+str(np.round(deltaX,2))+'m'
                        ,transform=axes[swarmAxisIndex].transAxes)
                # bins=np.linspace(-10,10,num=25)
                # image=np.histogram2d(xCoordsArr*unitModifier[0],yCoordsArr*unitModifier[1],bins=bins)[0]
                # image[image==0]=np.nan #to get white background where no particles ares
                # axes[0].imshow(image,extent=[bins.min(),bins.max(),bins.min(),bins.max()],interpolation='bicubic')


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
        self._make_Phase_Space_Video_For_X_Array(videoTitle,xSnapShotArr,xaxis,yaxis,.25,fpsApprox)
    def make_Phase_Space_Movie_Through_Lattice(self,title,xaxis,yaxis,videoLengthSeconds=10.0,fps=30,alpha=.25):
        #maxVideoLengthSeconds: The video can be no longer than this, but will otherwise shoot for a few fps
        self._check_Axis_Choice(xaxis,yaxis)
        numFrames=int(videoLengthSeconds*fps)
        xEnd=self._find_Max_Xorbit_For_Swarm()
        offsetToAvoidElementEdge=1e-9
        xStart=self._find_Inclusive_Min_XOrbit_For_Swarm() + offsetToAvoidElementEdge
        xArr=np.linspace(xStart,xEnd,numFrames)
        print('making video')
        self._make_Phase_Space_Video_For_X_Array(title,xArr,xaxis,yaxis,alpha,fps)
    def plot_Survival_Versus_Time(self,TMax=None):
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
        plt.title('Percent particle survival versus revolution time')
        plt.plot(TArr,survivalList)
        plt.xlabel('Time,s')
        plt.ylabel('Survival, %')
        plt.axvline(x=TRev,c='black',linestyle=':')
        plt.text(TRev+TMax*.05,20,'Revolution \n time')
        plt.show()
    def plot_Standing_Envelope(self):
        maxCompletedRevs=int(self._find_Max_Xorbit_For_Swarm()/self.lattice.totalLength)
        assert maxCompletedRevs>1
        xStart=self._find_Inclusive_Min_XOrbit_For_Swarm()
        xMax=self.lattice.totalLength
        numEnvPoints=250
        xSnapShotArr=np.linspace(xStart,xMax,numEnvPoints)
        envelopeData=np.zeros((numEnvPoints,1)) #this array will have each row filled with the relevant particle
        #particle parameters for all revolutions.
        for revNumber in tqdm(range(maxCompletedRevs)):
            revCoordsList=[]
            for xOrbit in xSnapShotArr:
                xOrbit+=self.lattice.totalLength*revNumber
                snapShotPhaseSpaceCoords=PhaseSpaceSnapShot(self.swarm,xOrbit).get_Surviving_Particle_PhaseSpace_Coords()
                revCoordsList.append(snapShotPhaseSpaceCoords[:,1])
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
# PTL.show_Lattice(swarm=swarm)
# for particle in swarm:
#     particle.plot_Orbit_Reference_Frame_Position()

# phaseSpaceAnalyzer.plot_Standing_Envelope()
# phaseSpaceAnalyzer.make_Phase_Space_Movie_At_Repeating_Lattice_Point('animation',.5)
# phaseSpaceAnalyzer.make_Phase_Space_Movie_Through_Lattice('animation','y','z',videoLengthSeconds=5.0)
