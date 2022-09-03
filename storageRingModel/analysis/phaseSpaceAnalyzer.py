import warnings

import celluloid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from particle import Particle as ParticleBase
from particle import Swarm
from particle_tracer_lattice import ParticleTracerLattice
from swarm_tracer import SwarmTracer

from helper_tools import parallel_evaluate

cmap = plt.get_cmap('viridis')


def make_Test_Swarm_And_Lattice(num_particles=128, totalTime=.1) -> (Swarm, ParticleTracerLattice):
    PTL = ParticleTracerLattice(speed_nominal=210.0)
    PTL.add_lens_ideal(.4, 1.0, .025)
    PTL.add_drift(.1)
    PTL.add_lens_ideal(.4, 1.0, .025)
    PTL.add_bender_ideal(np.pi, 1.0, 1.0, .025)
    PTL.add_drift(.2)
    PTL.add_lens_ideal(.2, 1.0, .025)
    PTL.add_drift(.1)
    PTL.add_lens_ideal(.2, 1.0, .025)
    PTL.add_drift(.2)
    PTL.add_bender_ideal(np.pi, 1.0, 1.0, .025)
    PTL.end_lattice()

    swarmTracer = SwarmTracer(PTL)
    swarm = swarmTracer.initalize_pseudorandom_swarm_in_phase_space(5e-3, 5.0, 1e-5, num_particles)
    swarm = swarmTracer.trace_swarm_through_lattice(swarm, 1e-5, totalTime, use_fast_mode=False, parallel=True,
                                                    steps_per_logging=4)
    print('swarm and lattice done')
    return swarm, PTL


class Particle(ParticleBase):
    def __init__(self, qi=None, pi=None):
        super().__init__(qi=qi, pi=pi)
        self.qo = None
        self.po = None
        self.q=None
        self.p=None
        self.E = None
        self.deltaE = None
        self.T=None


class SwarmSnapShot:
    def __init__(self, swarm: Swarm, xSnapShot,min_Survival_T=0.0):
        assert xSnapShot > 0.0  # orbit coordinates, not real coordinates
        self.particles: list[Particle] = None
        self.xSnapShot=xSnapShot
        self.swarm=swarm
        self.min_Survival_T=min_Survival_T
        self._take_SnapShot()

    def _take_Particle_Snapshot(self,particle):
        particleSnapShot = Particle(qi=particle.qi.copy(), pi=particle.pi.copy())
        particleSnapShot.probability = particle.probability
        particleSnapShot.T=particle.T
        particleSnapShot.revolutions=particle.revolutions
        if self._check_If_Particle_Can_Be_Interpolated(particle, self.xSnapShot) :
            E, qo, po,q,p = self._get_Phase_Space_Coords_And_Energy_SnapShot(particle, self.xSnapShot)
            particleSnapShot.E = E
            particleSnapShot.deltaE = E - particle.E_vals[0].copy()
            particleSnapShot.qo = qo
            particleSnapShot.po = po
            particleSnapShot.q=q
            particleSnapShot.p=p
            particleSnapShot.clipped = False
        elif particle.qo_vals is not None:
            particleSnapShot.qo = particle.qo_vals[-1].copy()
            particleSnapShot.po = particle.po_vals[-1].copy()
            particleSnapShot.pf = particle.pf.copy()
            particleSnapShot.qf = particle.qf.copy()
            particleSnapShot.E = particle.E_vals[-1].copy()
            particleSnapShot.deltaE = particleSnapShot.E - particle.E_vals[0]
            particleSnapShot.clipped = True
        else: #particle is clipped and there was no logging
            particleSnapShot.clipped = True
        return particleSnapShot

    def _take_SnapShot(self):
        self.particles=[self._take_Particle_Snapshot(particle) for particle in self.swarm.particles]
        if self.num_Surviving() == 0:
            warnings.warn("There are no particles that survived to the snapshot position")

    def _check_If_Particle_Can_Be_Interpolated(self, particle, x):
        # this assumes orbit coordinates
        if particle.qo_vals is None or len(particle.qo_vals) == 0 or particle.T<=self.min_Survival_T:
            return False  # clipped immediately probably
        elif particle.qo_vals[-1, 0] > x > particle.qo_vals[0, 0]:
            return True
        else:
            return False

    def num_Surviving(self):
        num = sum([not particle.clipped for particle in self.particles])
        return num

    def _get_Phase_Space_Coords_And_Energy_SnapShot(self, particle, xSnapShot):
        qo_arr = particle.qo_vals  # position in orbit coordinates
        po_arr = particle.po_vals
        E_arr = particle.E_vals
        assert xSnapShot < qo_arr[-1, 0]
        indexBefore = np.argmax(qo_arr[:, 0] > xSnapShot) - 1
        qo1 = qo_arr[indexBefore]
        qo2 = qo_arr[indexBefore + 1]
        stepFraction = (xSnapShot - qo1[0]) / (qo2[0] - qo1[0])
        qoSnapShot = self._interpolate_Array(qo_arr, indexBefore, stepFraction)
        poSnapShot = self._interpolate_Array(po_arr, indexBefore, stepFraction)
        if np.any(np.isnan(poSnapShot)):
            print(qoSnapShot,poSnapShot)
            print(particle.T,indexBefore,stepFraction)
            print(po_arr[indexBefore-1],po_arr[indexBefore],po_arr[indexBefore+1])
            print(qo_arr[indexBefore-1],qo_arr[indexBefore],qo_arr[indexBefore+1])
        ESnapShot = self._interpolate_Array(E_arr, indexBefore, stepFraction)
        qLabSnapShot=self._interpolate_Array(particle.q_vals, indexBefore, stepFraction)
        pLabSnapShot=self._interpolate_Array(particle.p_vals, indexBefore, stepFraction)
        assert not np.any(np.isnan(poSnapShot)) and not np.any(np.isnan(pLabSnapShot))
        return ESnapShot, qoSnapShot, poSnapShot,qLabSnapShot,pLabSnapShot

    def _interpolate_Array(self, arr, indexBegin, stepFraction):
        # v1: bounding vector before
        # v2: bounding vector after
        # stepFraction: fractional position between v1 and v2 to interpolate
        assert 0.0 <= stepFraction <= 1.0
        v = arr[indexBegin] + (arr[indexBegin + 1] - arr[indexBegin]) * stepFraction
        return v

    def get_Surviving_Particles(self):
        return [particle for particle in self.particles if not particle.clipped]

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
        assert lattice.lattice_type == 'storage_ring'
        assert all(type(particle.clipped) is bool for particle in swarm)
        assert all(particle.traced is True for particle in swarm)
        self.swarm = swarm
        self.lattice = lattice
        self.max_revs = np.inf

    def _get_Axis_Index(self, xaxis, yaxis):
        strinNameArr = np.asarray(['y', 'z', 'px', 'py', 'pz', 'NONE'])
        assert xaxis in strinNameArr and yaxis in strinNameArr
        xAxisIndex = np.argwhere(strinNameArr == xaxis)[0][0] + 1
        yAxisIndex = np.argwhere(strinNameArr == yaxis)[0][0] + 1
        return xAxisIndex, yAxisIndex

    def _get_Plot_Data_From_SnapShot(self, snapShotPhaseSpace, xAxis, y_axis):
        xAxisIndex, yAxisIndex = self._get_Axis_Index(xAxis, y_axis)
        xAxisArr = snapShotPhaseSpace[:, xAxisIndex]
        yAxisArr = snapShotPhaseSpace[:, yAxisIndex]
        return xAxisArr, yAxisArr

    def _find_Max_Xorbit_For_Swarm(self, timeStep=-1):
        # find the maximum longitudinal distance a particle has traveled
        x_max = 0.0
        for particle in self.swarm:
            if len(particle.qo_vals) > 0:
                x_max = max([x_max, particle.qo_vals[timeStep, 0]])
        return x_max

    def _find_Inclusive_Min_XOrbit_For_Swarm(self):
        # find the smallest x that as ahead of all particles, ie inclusive
        x_min = 0.0
        for particle in self.swarm:
            if len(particle.qo_vals) > 0:
                x_min = max([x_min, particle.qo_vals[0, 0]])
        return x_min

    def _make_SnapShot_Position_Arr_At_Same_X(self, xVideoPoint):
        x_max = self._find_Max_Xorbit_For_Swarm()
        revolutionsMax = int((x_max - xVideoPoint) / self.lattice.total_length)
        assert revolutionsMax > 0
        x_arr = np.arange(revolutionsMax + 1) * self.lattice.total_length + xVideoPoint
        return x_arr

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
                revs = int(xOrbit / self.lattice.total_length)
                delta_x = xOrbit - revs * self.lattice.total_length
                axes[swarmAxisIndex].text(0.1, 1.01,
                                          'Revolutions: ' + str(revs) + ', Distance along revolution: ' + str(
                                              np.round(delta_x, 2)) + 'm'
                                          , transform=axes[swarmAxisIndex].transAxes)
                axes[swarmAxisIndex].scatter(xCoordsArr * unitModifier[0], yCoordsArr * unitModifier[1],
                                             c='blue', alpha=alpha, edgecolors=None, linewidths=0.0)
                axes[swarmAxisIndex].grid()
                xSwarmLab, ySwarmLab = self.lattice.get_lab_coords_from_orbit_distance(xOrbit)
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

    def _make_SnapShot_XArr(self, num_points):
        # revolutions: set to -1 for using the largest number possible based on swarm
        xMaxSwarm = self._find_Max_Xorbit_For_Swarm()
        x_max = min(xMaxSwarm, self.max_revs * self.lattice.total_length)
        xStart = self._find_Inclusive_Min_XOrbit_For_Swarm()
        return np.linspace(xStart, x_max, num_points)

    def make_Phase_Space_Movie_At_Repeating_Lattice_Point(self, videoTitle, xVideoPoint, xaxis='y', yaxis='z',
                                                          videoLengthSeconds=10.0, alpha=.25, dpi=None):
        # xPoint: x point along lattice that the video is made at
        # xaxis: which cooordinate is plotted on the x axis of phase space plot
        # yaxis: which coordine is plotted on the y axis of phase plot
        # valid selections for xaxis and yaxis are ['y','z','px','py','pz']. Do not confuse plot axis with storage ring
        # axis. Storage ring axis has x being the distance along orbi, y perpindicular and horizontal, z perpindicular to
        # floor
        assert xVideoPoint < self.lattice.total_length
        self._check_Axis_Choice(xaxis, yaxis)
        numFrames = int(self._find_Max_Xorbit_For_Swarm() / self.lattice.total_length)
        fpsApprox = min(int(numFrames / videoLengthSeconds), 1)
        print(fpsApprox, numFrames)
        xSnapShotArr = self._make_SnapShot_Position_Arr_At_Same_X(xVideoPoint)
        self._make_Phase_Space_Video_For_X_Array(videoTitle, xSnapShotArr, xaxis, yaxis, alpha, fpsApprox, dpi)

    def make_Phase_Space_Movie_Through_Lattice(self, title, xaxis, yaxis, videoLengthSeconds=10.0, fps=30, alpha=.25,
                                               max_revs=np.inf, dpi=None):
        # maxVideoLengthSeconds: The video can be no longer than this, but will otherwise shoot for a few fps
        self.max_revs = max_revs
        self._check_Axis_Choice(xaxis, yaxis)
        numFrames = int(videoLengthSeconds * fps)
        x_arr = self._make_SnapShot_XArr(numFrames)
        print('making video')
        self._make_Phase_Space_Video_For_X_Array(title, x_arr, xaxis, yaxis, alpha, fps, dpi)

    def plot_Survival_Versus_Time(self, TMax=None, axis=None):
        TSurvivedList = []
        for particle in self.swarm:
            TSurvivedList.append(particle.T)
        if TMax is None: TMax = max(TSurvivedList)

        TSurvivedArr = np.asarray(TSurvivedList)
        numTPoints = 1000
        T_arr = np.linspace(0.0, TMax, numTPoints)
        T_arr = T_arr[:-1]  # exlcude last point because all particles are clipped there
        survivalList = []
        for T in T_arr:
            num_particlesurvived = np.sum(TSurvivedArr > T)
            survival = 100 * num_particlesurvived / self.swarm.num_particles()
            survivalList.append(survival)
        TRev = self.lattice.total_length / self.lattice.speed_nominal
        if axis is None:
            plt.title('Percent particle survival versus revolution time')
            plt.plot(T_arr, survivalList)
            plt.xlabel('Time,s')
            plt.ylabel('Survival, %')
            plt.axvline(x=TRev, c='black', linestyle=':', label='One Rev')
            plt.legend()
            plt.show()
        else:
            axis.plot(T_arr, survivalList)

    def plot_Energy_Growth(self, num_points=100, dpi=150, save_title=None, survivingOnly=True):
        if survivingOnly == True:
            fig, axes = plt.subplots(2, 1)
        else:
            fig, axes = plt.subplots(1, 1)
            axes = [axes]
        xSnapShotArr = self._make_SnapShot_XArr(num_points)[:-10]
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
            survivalList.append(100 * snapShot.num_Surviving() / self.swarm.num_particles())
            E_RMS = np.std(deltaESnapShot)
            EList_RMS.append(E_RMS)
            EList_Mean.append(np.mean(deltaESnapShot))
            EList_Max.append(np.max(deltaESnapShot))
        revArr = xSnapShotArr[:len(EList_RMS)] / self.lattice.total_length
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
        if save_title is not None:
            plt.savefig(save_title, dpi=dpi)
        plt.show()

    def plot_Acceptance_1D_Histogram(self, dimension: str, numBins: int = 10, save_title: str = None,
                                     showInputDist: bool = True,
                                     weightingMethod: str = 'clipped', TMax: float = None, dpi: float = 150,
                                     cAccepted=cmap(cmap.N), cInitial=cmap(0)) -> None:
        """
        Histogram of acceptance of storage ring starting from injector inlet versus initial values ofy,z,px,py or pz in
        the element frame

        :param dimension: The particle dimension to plot the acceptance over. y,z,px,py or pz
        :param numBins: Number of bins spanning the range of dimension values
        :param save_title: If not none, the plot is saved with this as the file name.
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
        label_list, unitModifier = self._get_Axis_Labels_And_Unit_Modifiers(dimension, 'NONE')
        plotIndex, _ = self._get_Axis_Index(dimension, 'NONE')
        vals = np.array([np.append(particle.qi, particle.pi)[plotIndex] for particle in self.swarm])
        fracSurvived = []
        num_particlesInBin = []
        binEdges = np.linspace(vals.min(), vals.max(), numBins)
        for i in range(len(binEdges) - 1):
            isValidList = (vals > binEdges[i]) & (vals < binEdges[i + 1])
            binParticles = [particle for particle, isValid in zip(self.swarm.particles, isValidList) if isValid]
            num_survived = sum([not particle.clipped for particle in binParticles])
            num_particlesInBin.append(len(binParticles))
            if weightingMethod == 'clipped':
                fracSurvived.append(num_survived / len(binParticles) if len(binParticles) != 0 else np.nan)
            else:
                survivalTimes = [particle.T for particle in binParticles]
                assert len(survivalTimes) == 0 or max(survivalTimes) <= TMax
                fracSurvived.append(np.mean(survivalTimes) / TMax if len(survivalTimes) != 0 else np.nan)
        plt.title("Particle acceptance")
        if showInputDist:
            num_particlesInBin = [num / max(num_particlesInBin) for num in num_particlesInBin]
            plt.bar(binEdges[:-1], num_particlesInBin, width=binEdges[1] - binEdges[0], align='edge', color=cInitial,
                    label='Initial distribution')
        plt.bar(binEdges[:-1], fracSurvived, width=binEdges[1] - binEdges[0], align='edge', label='Acceptance',
                color=cAccepted)
        plt.xlabel(label_list[0])
        plt.ylabel("Percent survival to end")
        plt.legend()
        plt.tight_layout()

        if save_title is not None:
            plt.savefig(save_title, dpi=dpi)
        plt.show()

    def plot_Acceptance_2D_ScatterPlot(self, xaxis, yaxis, save_title=None, alpha=.5, dpi=150):
        self._check_Axis_Choice(xaxis, yaxis)
        label_list, unitModifier = self._get_Axis_Labels_And_Unit_Modifiers(xaxis, yaxis)
        from matplotlib.patches import Patch
        xPlotIndex, yPlotIndex = self._get_Axis_Index(xaxis, yaxis)
        for particle in self.swarm:
            X = particle.qi.copy()
            assert np.abs(X[0]) < 1e-6
            X = np.append(X, particle.pi.copy())
            xPlot = X[xPlotIndex] * unitModifier[0]
            y_plot = X[yPlotIndex] * unitModifier[1]
            color = 'red' if particle.clipped == True else 'green'
            plt.scatter(xPlot, y_plot, c=color, alpha=alpha, edgecolors='none')
        legendList = [Patch(facecolor='green', edgecolor='green',
                            label='survived'), Patch(facecolor='red', edgecolor='red',
                                                     label='clipped')]
        plt.title('Phase space acceptance')
        plt.legend(handles=legendList)
        plt.xlabel(label_list[0])
        plt.ylabel(label_list[1])
        plt.tight_layout()
        if save_title is not None:
            plt.savefig(save_title, dpi=dpi)
        plt.show()

    def plot_Acceptance_2D_Histrogram(self, xaxis, yaxis, TMax, save_title=None, dpi=150, bins=50, emptyVals=np.nan):

        self._check_Axis_Choice(xaxis, yaxis)
        label_list, unitModifier = self._get_Axis_Labels_And_Unit_Modifiers(xaxis, yaxis)
        xPlotIndex, yPlotIndex = self._get_Axis_Index(xaxis, yaxis)

        xPlotVals, yPlotVals, weights = [], [], []
        for particle in self.swarm:
            X = np.append(particle.qi, particle.pi)
            assert np.abs(X[0]) < 1e-6
            xPlotVals.append(X[xPlotIndex] * unitModifier[0])
            yPlotVals.append(X[yPlotIndex] * unitModifier[1])
            assert particle.T <= TMax
            weights.append(particle.T)
        histogramnum_particles, _, _ = np.histogram2d(xPlotVals, yPlotVals, bins=bins)
        histogramSurvivalTimes, binsx, binsy = np.histogram2d(xPlotVals, yPlotVals, bins=bins, weights=weights)
        histogramnum_particles[histogramnum_particles == 0] = np.nan
        histogramAcceptance = histogramSurvivalTimes / (histogramnum_particles * TMax)
        histogramAcceptance = np.rot90(histogramAcceptance)
        histogramAcceptance[np.isnan(histogramAcceptance)] = emptyVals
        plt.title('Phase space acceptance')
        plt.imshow(histogramAcceptance, extent=[binsx.min(), binsx.max(), binsy.min(), binsy.max()], aspect='auto')
        plt.colorbar()
        plt.xlabel(label_list[0])
        plt.ylabel(label_list[1])
        plt.tight_layout()
        if save_title is not None:
            plt.savefig(save_title, dpi=dpi)
        plt.show()

    def plot_Standing_Envelope(self):
        raise Exception('This is broken because I am trying to use ragged arrays basically')
        maxCompletedRevs = int(self._find_Max_Xorbit_For_Swarm() / self.lattice.total_length)
        assert maxCompletedRevs > 1
        xStart = self._find_Inclusive_Min_XOrbit_For_Swarm()
        x_max = self.lattice.total_length
        numEnvPoints = 50
        xSnapShotArr = np.linspace(xStart, x_max, numEnvPoints)
        yCoordIndex = 1
        envelopeData = np.zeros((numEnvPoints, 1))  # this array will have each row filled with the relevant particle
        # particle parameters for all revolutions.
        for revNumber in range(maxCompletedRevs):
            revCoordsList = []
            for xOrbit in xSnapShotArr:
                xOrbit += self.lattice.total_length * revNumber
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

