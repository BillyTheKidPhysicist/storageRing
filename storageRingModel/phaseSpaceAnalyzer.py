from tqdm import tqdm
from SwarmTracerClass import SwarmTracer
import celluloid
import warnings
import numpy as np
from ParticleClass import Swarm
from ParticleClass import Particle as ParticleBase
from ParticleTracerLatticeClass import ParticleTracerLattice
import matplotlib.pyplot as plt

cmap = plt.get_cmap('viridis')


def make_Test_Swarm_And_Lattice(numParticles=128, totalTime=.1) -> (Swarm, ParticleTracerLattice):
    PTL = ParticleTracerLattice(v0Nominal=210.0)
    PTL.add_Lens_Ideal(.4, 1.0, .025)
    PTL.add_Drift(.1)
    PTL.add_Lens_Ideal(.4, 1.0, .025)
    PTL.add_Bender_Ideal(np.pi, 1.0, 1.0, .025)
    PTL.add_Drift(.2)
    PTL.add_Lens_Ideal(.2, 1.0, .025)
    PTL.add_Drift(.1)
    PTL.add_Lens_Ideal(.2, 1.0, .025)
    PTL.add_Drift(.2)
    PTL.add_Bender_Ideal(np.pi, 1.0, 1.0, .025)
    PTL.end_Lattice()

    swarmTracer = SwarmTracer(PTL)
    swarm = swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(5e-3, 5.0, 1e-5, numParticles)
    swarm = swarmTracer.trace_Swarm_Through_Lattice(swarm, 1e-5, totalTime, fastMode=False, parallel=True,
                                                    stepsBetweenLogging=4)
    print('swarm and lattice done')
    return swarm, PTL


class Particle(ParticleBase):
    def __init__(self, qi=None, pi=None):
        super().__init__(qi=qi, pi=pi)
        self.qo = None
        self.po = None
        self.E = None
        self.deltaE = None


class SwarmSnapShot:
    def __init__(self, swarm: Swarm, xSnapShot):
        assert xSnapShot >= 0.0  # orbit coordinates, not real coordinates
        for particle in swarm: assert particle.dataLogging == True
        self.particles: list[Particle] = []
        self._take_SnapShot(swarm, xSnapShot)

    def _take_SnapShot(self, swarm, xSnapShot):
        for particle in swarm:
            particleSnapShot = Particle(qi=particle.qi.copy(), pi=particle.pi.copy())
            particleSnapShot.probability = particle.probability
            if self._check_If_Particle_Can_Be_Interpolated(particle, xSnapShot) == False:
                particleSnapShot.qo = particle.qoArr[-1].copy()
                particleSnapShot.po = particle.pArr[-1].copy()
                particleSnapShot.pf = particle.pf.copy()
                particleSnapShot.qf = particle.qf.copy()
                particleSnapShot.E = particle.EArr[-1].copy()
                particleSnapShot.deltaE = particleSnapShot.E - particle.EArr[0]
                particleSnapShot.clipped = True
            else:
                E, qo, po = self._get_Phase_Space_Coords_And_Energy_SnapShot(particle, xSnapShot)
                particleSnapShot.E = E
                particleSnapShot.deltaE = E - particle.EArr[0].copy()
                particleSnapShot.qo = qo
                particleSnapShot.po = po
                particleSnapShot.clipped = False
            self.particles.append(particleSnapShot)
        if self.num_Surviving() == 0:
            warnings.warn("There are no particles that survived to the snapshot position")

    def _check_If_Particle_Can_Be_Interpolated(self, particle, x):
        # this assumes orbit coordinates
        if len(particle.qoArr) == 0:
            return False  # clipped immediately probably
        elif particle.qoArr[-1, 0] > x > particle.qoArr[0, 0]:
            return True
        else:
            return False

    def num_Surviving(self):
        num = sum([not particle.clipped for particle in self.particles])
        return num

    def _get_Phase_Space_Coords_And_Energy_SnapShot(self, particle, xSnapShot):
        qoArr = particle.qoArr  # position in orbit coordinates
        poArr = particle.pArr
        EArr = particle.EArr
        assert xSnapShot < qoArr[-1, 0]
        indexBefore = np.argmax(qoArr[:, 0] > xSnapShot) - 1
        qo1 = qoArr[indexBefore]
        qo2 = qoArr[indexBefore + 1]
        stepFraction = (xSnapShot - qo1[0]) / (qo2[0] - qo1[0])
        qSnapShot = self._interpolate_Array(qoArr, indexBefore, stepFraction)
        pSnapShot = self._interpolate_Array(poArr, indexBefore, stepFraction)
        ESnapShot = self._interpolate_Array(EArr, indexBefore, stepFraction)
        return ESnapShot, qSnapShot, pSnapShot

    def _interpolate_Array(self, arr, indexBegin, stepFraction):
        # v1: bounding vector before
        # v2: bounding vector after
        # stepFraction: fractional position between v1 and v2 to interpolate
        assert 0.0 <= stepFraction <= 1.0
        v = arr[indexBegin] + (arr[indexBegin + 1] - arr[indexBegin]) * stepFraction
        return v

    def get_Surviving_Particle_PhaseSpace_Coords(self):
        # get coordinates of surviving particles
        phaseSpaceCoords = [(*particle.qo, *particle.po) for particle in self.particles if not particle.clipped]
        phaseSpaceCoords = np.array(phaseSpaceCoords)
        return phaseSpaceCoords

    def get_Particles_Energy(self, returnChangeInE=False, survivingOnly=True):
        EList = []
        for particle in self.particles:
            if particle.clipped == True and survivingOnly == True:
                pass
            else:
                if returnChangeInE == True:
                    EList.append(particle.deltaE)
                else:
                    EList.append(particle.E)
        EList = [np.nan] if len(EList) == 0 else np.asarray(EList)
        return EList


class PhaseSpaceAnalyzer:
    def __init__(self, swarm, lattice: ParticleTracerLattice):
        assert lattice.latticeType == 'storageRing'
        assert all(type(particle.clipped) is bool for particle in swarm)
        assert all(particle.traced is True for particle in swarm)
        self.swarm = swarm
        self.lattice = lattice
        self.maxRevs = np.inf

    def _get_Axis_Index(self, xaxis, yaxis):
        strinNameArr = np.asarray(['y', 'z', 'px', 'py', 'pz', 'NONE'])
        assert xaxis in strinNameArr and yaxis in strinNameArr
        xAxisIndex = np.argwhere(strinNameArr == xaxis)[0][0] + 1
        yAxisIndex = np.argwhere(strinNameArr == yaxis)[0][0] + 1
        return xAxisIndex, yAxisIndex

    def _get_Plot_Data_From_SnapShot(self, snapShotPhaseSpace, xAxis, yAxis):
        xAxisIndex, yAxisIndex = self._get_Axis_Index(xAxis, yAxis)
        xAxisArr = snapShotPhaseSpace[:, xAxisIndex]
        yAxisArr = snapShotPhaseSpace[:, yAxisIndex]
        return xAxisArr, yAxisArr

    def _find_Max_Xorbit_For_Swarm(self, timeStep=-1):
        # find the maximum longitudinal distance a particle has traveled
        xMax = 0.0
        for particle in self.swarm:
            if len(particle.qoArr) > 0:
                xMax = max([xMax, particle.qoArr[timeStep, 0]])
        return xMax

    def _find_Inclusive_Min_XOrbit_For_Swarm(self):
        # find the smallest x that as ahead of all particles, ie inclusive
        xMin = 0.0
        for particle in self.swarm:
            if len(particle.qoArr) > 0:
                xMin = max([xMin, particle.qoArr[0, 0]])
        return xMin

    def _make_SnapShot_Position_Arr_At_Same_X(self, xVideoPoint):
        xMax = self._find_Max_Xorbit_For_Swarm()
        revolutionsMax = int((xMax - xVideoPoint) / self.lattice.totalLength)
        assert revolutionsMax > 0
        xArr = np.arange(revolutionsMax + 1) * self.lattice.totalLength + xVideoPoint
        return xArr

    def _plot_Lattice_On_Axis(self, ax, plotPointCoords=None):
        for el in self.lattice:
            ax.plot(*el.SO.exterior.xy, c='black')
        if plotPointCoords is not None:
            ax.scatter(plotPointCoords[0], plotPointCoords[1], c='red', marker='x', s=100, edgecolors=None)

    def _make_Phase_Space_Video_For_X_Array(self, videoTitle, xOrbitSnapShotArr, xaxis, yaxis, alpha, fps, dpi):
        fig, axes = plt.subplots(2, 1)
        camera = celluloid.Camera(fig)
        labels, unitModifier = self._get_Axis_Labels_And_Unit_Modifiers(xaxis, yaxis)
        swarmAxisIndex = 0
        latticeAxisIndex = 1
        axes[swarmAxisIndex].set_xlabel(labels[0])
        axes[swarmAxisIndex].set_ylabel(labels[1])
        axes[swarmAxisIndex].text(0.1, 1.1, 'Phase space portraint'
                                  , transform=axes[swarmAxisIndex].transAxes)
        axes[latticeAxisIndex].set_xlabel('meters')
        axes[latticeAxisIndex].set_ylabel('meters')
        for xOrbit, i in zip(xOrbitSnapShotArr, range(len(xOrbitSnapShotArr))):
            snapShotPhaseSpaceCoords = SwarmSnapShot(self.swarm, xOrbit).get_Surviving_Particle_PhaseSpace_Coords()
            if len(snapShotPhaseSpaceCoords) == 0:
                break
            else:
                xCoordsArr, yCoordsArr = self._get_Plot_Data_From_SnapShot(snapShotPhaseSpaceCoords, xaxis, yaxis)
                revs = int(xOrbit / self.lattice.totalLength)
                deltaX = xOrbit - revs * self.lattice.totalLength
                axes[swarmAxisIndex].text(0.1, 1.01,
                                          'Revolutions: ' + str(revs) + ', Distance along revolution: ' + str(
                                              np.round(deltaX, 2)) + 'm'
                                          , transform=axes[swarmAxisIndex].transAxes)
                axes[swarmAxisIndex].scatter(xCoordsArr * unitModifier[0], yCoordsArr * unitModifier[1],
                                             c='blue', alpha=alpha, edgecolors=None, linewidths=0.0)
                axes[swarmAxisIndex].grid()
                xSwarmLab, ySwarmLab = self.lattice.get_Lab_Coords_From_Orbit_Distance(xOrbit)
                self._plot_Lattice_On_Axis(axes[latticeAxisIndex], [xSwarmLab, ySwarmLab])
                camera.snap()
        plt.tight_layout()
        animation = camera.animate()
        animation.save(str(videoTitle) + '.gif', fps=fps, dpi=dpi)

    def _check_Axis_Choice(self, xaxis, yaxis):
        validPhaseCoords = ['y', 'z', 'px', 'py', 'pz', 'NONE']
        assert (xaxis in validPhaseCoords) and (yaxis in validPhaseCoords)

    def _get_Axis_Labels_And_Unit_Modifiers(self, xaxis, yaxis):
        positionLabel = 'Position, mm'
        momentumLabel = 'Momentum, m/s'
        positionUnitModifier = 1e3
        if xaxis in ['y', 'z']:  # there has to be a better way to do this
            label = xaxis + ' ' + positionLabel
            labelsList = [label]
            unitModifier = [positionUnitModifier]
        else:
            label = xaxis + ' ' + momentumLabel
            labelsList = [label]
            unitModifier = [1]
        if yaxis in ['y', 'z']:
            label = yaxis + ' ' + positionLabel
            labelsList.append(label)
            unitModifier.append(positionUnitModifier)
        else:
            label = yaxis + ' ' + momentumLabel
            labelsList.append(label)
            unitModifier.append(1.0)
        return labelsList, unitModifier

    def _make_SnapShot_XArr(self, numPoints):
        # revolutions: set to -1 for using the largest number possible based on swarm
        xMaxSwarm = self._find_Max_Xorbit_For_Swarm()
        xMax = min(xMaxSwarm, self.maxRevs * self.lattice.totalLength)
        xStart = self._find_Inclusive_Min_XOrbit_For_Swarm()
        return np.linspace(xStart, xMax, numPoints)

    def make_Phase_Space_Movie_At_Repeating_Lattice_Point(self, videoTitle, xVideoPoint, xaxis='y', yaxis='z',
                                                          videoLengthSeconds=10.0, alpha=.25, dpi=None):
        # xPoint: x point along lattice that the video is made at
        # xaxis: which cooordinate is plotted on the x axis of phase space plot
        # yaxis: which coordine is plotted on the y axis of phase plot
        # valid selections for xaxis and yaxis are ['y','z','px','py','pz']. Do not confuse plot axis with storage ring
        # axis. Storage ring axis has x being the distance along orbi, y perpindicular and horizontal, z perpindicular to
        # floor
        assert xVideoPoint < self.lattice.totalLength
        self._check_Axis_Choice(xaxis, yaxis)
        numFrames = int(self._find_Max_Xorbit_For_Swarm() / self.lattice.totalLength)
        fpsApprox = min(int(numFrames / videoLengthSeconds), 1)
        print(fpsApprox, numFrames)
        xSnapShotArr = self._make_SnapShot_Position_Arr_At_Same_X(xVideoPoint)
        self._make_Phase_Space_Video_For_X_Array(videoTitle, xSnapShotArr, xaxis, yaxis, alpha, fpsApprox, dpi)

    def make_Phase_Space_Movie_Through_Lattice(self, title, xaxis, yaxis, videoLengthSeconds=10.0, fps=30, alpha=.25,
                                               maxRevs=np.inf, dpi=None):
        # maxVideoLengthSeconds: The video can be no longer than this, but will otherwise shoot for a few fps
        self.maxRevs = maxRevs
        self._check_Axis_Choice(xaxis, yaxis)
        numFrames = int(videoLengthSeconds * fps)
        xArr = self._make_SnapShot_XArr(numFrames)
        print('making video')
        self._make_Phase_Space_Video_For_X_Array(title, xArr, xaxis, yaxis, alpha, fps, dpi)

    def plot_Survival_Versus_Time(self, TMax=None, axis=None):
        TSurvivedList = []
        for particle in self.swarm:
            TSurvivedList.append(particle.T)
        if TMax is None: TMax = max(TSurvivedList)

        TSurvivedArr = np.asarray(TSurvivedList)
        numTPoints = 1000
        TArr = np.linspace(0.0, TMax, numTPoints)
        TArr = TArr[:-1]  # exlcude last point because all particles are clipped there
        survivalList = []
        for T in TArr:
            numParticleSurvived = np.sum(TSurvivedArr > T)
            survival = 100 * numParticleSurvived / self.swarm.num_Particles()
            survivalList.append(survival)
        TRev = self.lattice.totalLength / self.lattice.v0Nominal
        if axis is None:
            plt.title('Percent particle survival versus revolution time')
            plt.plot(TArr, survivalList)
            plt.xlabel('Time,s')
            plt.ylabel('Survival, %')
            plt.axvline(x=TRev, c='black', linestyle=':', label='One Rev')
            plt.legend()
            plt.show()
        else:
            axis.plot(TArr, survivalList)

    def plot_Energy_Growth(self, numPoints=100, dpi=150, saveTitle=None, survivingOnly=True):
        if survivingOnly == True:
            fig, axes = plt.subplots(2, 1)
        else:
            fig, axes = plt.subplots(1, 1)
            axes = [axes]
        xSnapShotArr = self._make_SnapShot_XArr(numPoints)[:-10]
        EList_RMS = []
        EList_Mean = []
        EList_Max = []
        survivalList = []
        for xOrbit in tqdm(xSnapShotArr):
            snapShot = SwarmSnapShot(self.swarm, xOrbit)
            deltaESnapShot = snapShot.get_Particles_Energy(returnChangeInE=True, survivingOnly=survivingOnly)
            minParticlesForStatistics = 5
            if len(deltaESnapShot) < minParticlesForStatistics:
                break
            survivalList.append(100 * snapShot.num_Surviving() / self.swarm.num_Particles())
            E_RMS = np.std(deltaESnapShot)
            EList_RMS.append(E_RMS)
            EList_Mean.append(np.mean(deltaESnapShot))
            EList_Max.append(np.max(deltaESnapShot))
        revArr = xSnapShotArr[:len(EList_RMS)] / self.lattice.totalLength
        axes[0].plot(revArr, EList_RMS, label='RMS')
        axes[0].plot(revArr, EList_Mean, label='mean')
        axes[0].set_ylabel('Energy change, sim units \n (Mass Li=1.0) ')
        if survivingOnly == True:
            axes[1].plot(revArr, survivalList)
            axes[1].set_ylabel('Survival,%')
            axes[1].set_xlabel('Revolutions')
        else:
            axes[0].set_xlabel('Revolutions')
        axes[0].legend()
        if saveTitle is not None:
            plt.savefig(saveTitle, dpi=dpi)
        plt.show()

    def plot_Acceptance_1D_Histogram(self, dimension: str, numBins: int = 10, saveTitle: str = None,
                                     showInputDist: bool = True,
                                     weightingMethod: str = 'clipped', TMax: float = None, dpi: float = 150,
                                     cAccepted=cmap(cmap.N), cInitial=cmap(0)) -> None:
        """
        Histogram of acceptance of storage ring starting from injector inlet versus initial values ofy,z,px,py or pz in
        the element frame

        :param dimension: The particle dimension to plot the acceptance over. y,z,px,py or pz
        :param numBins: Number of bins spanning the range of dimension values
        :param saveTitle: If not none, the plot is saved with this as the file name.
        :param showInputDist: Plot the initial distribution behind the acceptance plot
        :param weightingMethod: Which weighting method to use to represent acceptence.
        :param TMax: When using 'time' as the weightingMethod this is the maximum time for acceptance
        :param dpi: dot per inch for saved plot
        :param cAccepted: Color of accepted distribution plot
        :param cInitial: Color of initial distribution plot
        :return: None
        """

        assert weightingMethod in ('clipped', 'time')

        self._check_Axis_Choice(dimension, 'NONE')
        labelList, unitModifier = self._get_Axis_Labels_And_Unit_Modifiers(dimension, 'NONE')
        plotIndex, _ = self._get_Axis_Index(dimension, 'NONE')
        vals = np.array([np.append(particle.qi, particle.pi)[plotIndex] for particle in self.swarm])
        fracSurvived = []
        numParticlesInBin = []
        binEdges = np.linspace(vals.min(), vals.max(), numBins)
        for i in range(len(binEdges) - 1):
            isValidList = (vals > binEdges[i]) & (vals < binEdges[i + 1])
            binParticles = [particle for particle, isValid in zip(self.swarm.particles, isValidList) if isValid]
            numSurvived = sum([not particle.clipped for particle in binParticles])
            numParticlesInBin.append(len(binParticles))
            if weightingMethod == 'clipped':
                fracSurvived.append(numSurvived / len(binParticles) if len(binParticles) != 0 else np.nan)
            else:
                survivalTimes = [particle.T for particle in binParticles]
                assert len(survivalTimes) == 0 or max(survivalTimes) <= TMax
                fracSurvived.append(np.mean(survivalTimes) / TMax if len(survivalTimes) != 0 else np.nan)
        plt.title("Particle acceptance")
        if showInputDist:
            numParticlesInBin = [num / max(numParticlesInBin) for num in numParticlesInBin]
            plt.bar(binEdges[:-1], numParticlesInBin, width=binEdges[1] - binEdges[0], align='edge', color=cInitial,
                    label='Initial distribution')
        plt.bar(binEdges[:-1], fracSurvived, width=binEdges[1] - binEdges[0], align='edge', label='Acceptance',
                color=cAccepted)
        plt.xlabel(labelList[0])
        plt.ylabel("Percent survival to end")
        plt.legend()
        plt.tight_layout()

        if saveTitle is not None:
            plt.savefig(saveTitle, dpi=dpi)
        plt.show()

    def plot_Acceptance_2D_ScatterPlot(self, xaxis, yaxis, saveTitle=None, alpha=.5, dpi=150):
        self._check_Axis_Choice(xaxis, yaxis)
        labelList, unitModifier = self._get_Axis_Labels_And_Unit_Modifiers(xaxis, yaxis)
        from matplotlib.patches import Patch
        xPlotIndex, yPlotIndex = self._get_Axis_Index(xaxis, yaxis)
        for particle in self.swarm:
            X = particle.qi.copy()
            assert np.abs(X[0]) < 1e-6
            X = np.append(X, particle.pi.copy())
            xPlot = X[xPlotIndex] * unitModifier[0]
            yPlot = X[yPlotIndex] * unitModifier[1]
            color = 'red' if particle.clipped == True else 'green'
            plt.scatter(xPlot, yPlot, c=color, alpha=alpha, edgecolors='none')
        legendList = [Patch(facecolor='green', edgecolor='green',
                            label='survived'), Patch(facecolor='red', edgecolor='red',
                                                     label='clipped')]
        plt.title('Phase space acceptance')
        plt.legend(handles=legendList)
        plt.xlabel(labelList[0])
        plt.ylabel(labelList[1])
        plt.tight_layout()
        if saveTitle is not None:
            plt.savefig(saveTitle, dpi=dpi)
        plt.show()

    def plot_Acceptance_2D_Histrogram(self, xaxis, yaxis, TMax, saveTitle=None, dpi=150, bins=50, emptyVals=np.nan):

        self._check_Axis_Choice(xaxis, yaxis)
        labelList, unitModifier = self._get_Axis_Labels_And_Unit_Modifiers(xaxis, yaxis)
        xPlotIndex, yPlotIndex = self._get_Axis_Index(xaxis, yaxis)

        xPlotVals, yPlotVals, weights = [], [], []
        for particle in self.swarm:
            X = np.append(particle.qi, particle.pi)
            assert np.abs(X[0]) < 1e-6
            xPlotVals.append(X[xPlotIndex] * unitModifier[0])
            yPlotVals.append(X[yPlotIndex] * unitModifier[1])
            assert particle.T <= TMax
            weights.append(particle.T)
        histogramNumParticles, _, _ = np.histogram2d(xPlotVals, yPlotVals, bins=bins)
        histogramSurvivalTimes, binsx, binsy = np.histogram2d(xPlotVals, yPlotVals, bins=bins, weights=weights)
        histogramNumParticles[histogramNumParticles == 0] = np.nan
        histogramAcceptance = histogramSurvivalTimes / (histogramNumParticles * TMax)
        histogramAcceptance = np.rot90(histogramAcceptance)
        histogramAcceptance[np.isnan(histogramAcceptance)] = emptyVals
        plt.title('Phase space acceptance')
        plt.imshow(histogramAcceptance, extent=[binsx.min(), binsx.max(), binsy.min(), binsy.max()], aspect='auto')
        plt.colorbar()
        plt.xlabel(labelList[0])
        plt.ylabel(labelList[1])
        plt.tight_layout()
        if saveTitle is not None:
            plt.savefig(saveTitle, dpi=dpi)
        plt.show()

    def plot_Standing_Envelope(self):
        raise Exception('This is broken because I am trying to use ragged arrays basically')
        maxCompletedRevs = int(self._find_Max_Xorbit_For_Swarm() / self.lattice.totalLength)
        assert maxCompletedRevs > 1
        xStart = self._find_Inclusive_Min_XOrbit_For_Swarm()
        xMax = self.lattice.totalLength
        numEnvPoints = 50
        xSnapShotArr = np.linspace(xStart, xMax, numEnvPoints)
        yCoordIndex = 1
        envelopeData = np.zeros((numEnvPoints, 1))  # this array will have each row filled with the relevant particle
        # particle parameters for all revolutions.
        for revNumber in range(maxCompletedRevs):
            revCoordsList = []
            for xOrbit in xSnapShotArr:
                xOrbit += self.lattice.totalLength * revNumber
                snapShotPhaseSpaceCoords = SwarmSnapShot(self.swarm, xOrbit).get_Surviving_Particle_PhaseSpace_Coords()
                revCoordsList.append(snapShotPhaseSpaceCoords[:, yCoordIndex])
            envelopeData = np.column_stack((envelopeData, revCoordsList))
            # rmsArr=np.std(np.asarray(revCoordsList),axis=1)
        meterToMM = 1e3
        rmsArr = np.std(envelopeData, axis=1)
        plt.title('RMS envelope of particle beam in storage ring.')
        plt.plot(xSnapShotArr, rmsArr * meterToMM)
        plt.ylabel('Displacement, mm')
        plt.xlabel('Distance along lattice, m')
        plt.ylim([0.0, rmsArr.max() * meterToMM])
        plt.show()
