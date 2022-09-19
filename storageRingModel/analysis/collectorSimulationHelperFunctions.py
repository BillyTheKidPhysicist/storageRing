import warnings
from typing import Callable


from helper_tools import make_dense_curve_1D_linear
import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

from particle import Swarm
from particle_tracer_lattice import ParticleTracerLattice
from constants import GRAVITATIONAL_ACCELERATION
from lattice_elements.elements import HalbachLensSim, Drift
from type_hints import sequence

meter_to_mm = 1e3


@np.vectorize
def voigt(r, a, b, sigma, gamma):
    # be very cautious about FWHM vs HWHM here for gamma. gamma for scipy is HWHM per the docs
    assert r >= 0
    gamma = gamma / 2.0  # convert to HWHM per scipy docs
    v0 = voigt_profile(0, sigma, gamma)
    v = voigt_profile(r, sigma, gamma) / v0
    v = a * v + b
    return v


def make_Density_And_Pos_Arr(y_arr: np.ndarray, z_arr: np.ndarray, vxArr: np.ndarray, rMax: float, numBins: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(y_arr) == len(z_arr)
    binEdges = np.linspace(-rMax, rMax, numBins + 1)
    image, binx, biny = np.histogram2d(y_arr, z_arr, bins=binEdges, weights=1 / np.abs(vxArr))
    binx = binx[:-1] + (binx[1] - binx[0]) / 2.0
    biny = biny[:-1] + (biny[1] - biny[0]) / 2.0
    binArea = (binx[1] - binx[0]) * (biny[1] - biny[0])
    yHistArr, zHistArr = np.meshgrid(binx, biny)
    yHistArr = np.ravel(yHistArr)
    zHistArr = np.ravel(zHistArr)
    countsArr = np.ravel(image)
    r_arr = np.sqrt(yHistArr ** 2 + zHistArr ** 2)
    r_arr = r_arr[countsArr != 0]
    countsArr = countsArr[countsArr != 0]
    sortIndices = np.argsort(r_arr)
    r_arr = r_arr[sortIndices]
    countsArr = countsArr[sortIndices]
    densityArr = countsArr / binArea
    densityErr = np.sqrt(countsArr) / binArea
    assert len(r_arr) == len(densityArr)
    return r_arr, densityArr, densityErr


def make_Radial_Signal_Arr(y_arr: np.ndarray, z_arr: np.ndarray, vxArr: np.ndarray, numBins: int, rMax: float,
                           numSamples: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert np.all(np.abs(vxArr) >= 1.0)
    r_arr, signalArr, signalErrArr = make_Density_And_Pos_Arr(y_arr, z_arr, vxArr, rMax, numBins)
    for i in range(1, numSamples):
        rArrTemp, signalArrTemp, signalErrArrTemp = make_Density_And_Pos_Arr(y_arr, z_arr, vxArr, rMax, numBins + i)
        r_arr = np.append(r_arr, rArrTemp)
        signalArr = np.append(signalArr, signalArrTemp)
        signalErrArr = np.append(signalErrArr, signalErrArrTemp)
    return r_arr, signalArr, signalErrArr


def check_collector_lattice_is_expected(lattice: ParticleTracerLattice):
    assert len(lattice) == 3
    first_el, middle_el, last_el = lattice.el_list
    assert type(first_el) is Drift and type(last_el) is Drift
    assert type(middle_el) is HalbachLensSim


def build_collector_lattice(interp_density_mult=2.0, ap=None,direction=0.0) -> ParticleTracerLattice:
    distance_nozzle = 72e-2
    magnet_widths = (.0254, 1.5 * .0254)
    rp_layers = (.05, .05 + magnet_widths[0])
    magnet_length = 6 * .0254
    fringe_field_length = rp_layers[1] * HalbachLensSim.fringe_frac_outer
    post_lens_drift_length = 1.25
    pre_lens_drift_length = distance_nozzle - fringe_field_length
    lens_element_length = magnet_length + 2 * fringe_field_length

    lattice = ParticleTracerLattice(lattice_type='injector', initial_ang=direction, field_dens_mult=interp_density_mult,
                                    magnet_grade='N40',use_long_range_fields=False)
    lattice.add_drift(pre_lens_drift_length, ap=rp_layers[1])
    lattice.add_halbach_lens_sim(rp_layers, lens_element_length, ap=ap, magnet_width=magnet_widths)
    lattice.add_drift(post_lens_drift_length, rp_layers[1])
    lattice.end_lattice()
    return lattice


def val_at_cumulative_fraction(values: np.ndarray, fraction: float) -> float:
    assert 0.0 <= fraction <= 1.0
    if len(values)==0:
        warnings.warn("The provided array is empty, returning nan")
        return np.nan
    else:
        arg_at_frac = round((len(values)-1) * fraction)
        return np.sort(values)[arg_at_frac]


# @numba.njit()
# def yz_vals_projected_to_x(q_start: sequence, p: sequence, x_project: float) -> tuple[float, float]:
#     delta_x = x_project - q_start[0]
#     delta_t = delta_x / p[0]
#     y = q_start[1] + p[1] * delta_t
#     z = q_start[2] + p[2] * delta_t - .5 * GRAVITATIONAL_ACCELERATION * delta_t ** 2
#     return y, z
#
#
# @numba.njit()
# def can_be_interpolated(x_interp: float, qf: sequence, pf: sequence, vTMax: float) -> bool:
#     if vTMax != np.inf:
#         vT = np.sqrt(pf[1] ** 2 + pf[2] ** 2)
#         if vT > vTMax:
#             return False
#     elif qf[0] < x_interp:  # particle interpolation region does not overlap requested x value
#         return False
#     else:
#         return True
#
#
# @numba.njit()
# def interpolate_vals(x_interp: float,qf_vals: np.ndarray,pf_vals: np.ndarray,vTMax: float)-> tuple[list,list,list]:
#     y_vals, z_vals, p_vals = [], [], []
#     assert vTMax>0.0
#     for qf,pf in zip(qf_vals,pf_vals):
#         if can_be_interpolated(x_interp, qf, pf, vTMax):
#             y, z = yz_vals_projected_to_x(qf, pf, x_interp)
#             y_vals.append(y)
#             z_vals.append(z)
#             p_vals.append(pf)
#     return y_vals,z_vals,p_vals

def yz_vals_projected_to_x(q_vals,p_vals, x_project: float) -> tuple[np.ndarray,np.ndarray]:
    delta_x_vals=x_project-q_vals[:,0]
    delta_t_vals=delta_x_vals/p_vals[:,0]
    y_vals=q_vals[:,1]+p_vals[:,1]*delta_t_vals
    z_vals=q_vals[:,2]+p_vals[:,2]*delta_t_vals
    return np.array(y_vals), np.array(z_vals)

def interpolate_values(x_interp,qf_vals,pf_vals,vT_max,max_radius)-> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    assert vT_max>0.0 and max_radius>0.0
    y_vals_all,z_vals_all=yz_vals_projected_to_x(qf_vals,pf_vals,x_interp)
    vt=np.linalg.norm(pf_vals[:,1])
    is_valid_vt=vt<vT_max
    is_valid_x=qf_vals[:,0]>x_interp
    r_vals=np.sqrt(y_vals_all**2 + z_vals_all**2)
    is_valid_r=r_vals<max_radius
    is_valid= (is_valid_vt & is_valid_x) & is_valid_r
    y_vals_valid,z_vals_valid=y_vals_all[is_valid],z_vals_all[is_valid]
    pf_vals_valid=pf_vals[is_valid]
    valid_indices=np.arange(len(is_valid))[is_valid]
    return y_vals_valid,z_vals_valid,pf_vals_valid,valid_indices

def radial_density_histogram(values,bin_center_sep):
    r_bin_centers=np.arange(0,np.max(values)+1e-9,bin_center_sep)
    r_bin_edges=r_bin_centers[1:]-bin_center_sep/2
    r_bin_edges=np.append(0,r_bin_edges)
    r_bin_edges=np.append(r_bin_edges,r_bin_edges[-1]+bin_center_sep)
    hist_r,_=np.histogram(values,bins=r_bin_edges)
    d_area=np.pi*(r_bin_edges[1:]**2-r_bin_edges[:-1]**2)
    density=hist_r/d_area
    return r_bin_centers,density

class CollectorSwarmAnalyzer:
    def __init__(self, swarm: Swarm, lattice: ParticleTracerLattice):
        self.check_swarm(swarm)
        assert lattice.initial_ang == 0.0
        self.swarm = swarm
        self.pf_vals=np.array([particle.pf for particle in swarm])
        self.qf_vals=np.array([particle.qf for particle in swarm])
        self.lattice = lattice
        self.end_drift_length = abs(self.lattice.el_list[-1].r2[0] - self.lattice.el_list[-1].r1[0])
        self.x_min, self.x_max = self.end_drift_x_min_max()

    def end_drift_x_min_max(self):
        return self.lattice.el_list[-1].r1[0], self.lattice.el_list[-1].r2[0]

    def check_swarm(self, swarm: Swarm):
        assert all([particle.pi[0] > 0 for particle in swarm])

    def _get_Sweep_Range_Trans_Vel_Max(self, voltageRange: float) -> float:
        freqRange = voltageRange * 565e6
        vTMax = freqRange * 671e-9
        vTMax /= 2  # need to half because validity is assesed with radial velocity
        return vTMax

    def interpolate(self, x_interp: float, max_radius_mm: float = np.inf, laser_scan_range_volts: float = None,vtrans_max: float=None,
                    return_p: bool = False, return_valid_indices: bool=False, enforce_location: bool=True) -> list[np.ndarray, ...]:
        if enforce_location:
            assert self.x_min < x_interp < self.x_max
        assert not ( laser_scan_range_volts is not None and vtrans_max is not None)
        if laser_scan_range_volts is not None:
            vTMax = self._get_Sweep_Range_Trans_Vel_Max(laser_scan_range_volts)
        elif vtrans_max is not None:
            vTMax=vtrans_max
        else:
            vTMax=np.inf
        max_radius_meter=max_radius_mm/meter_to_mm
        y_vals, z_vals, p_vals,valid_indices=interpolate_values(x_interp,self.qf_vals,self.pf_vals,vTMax,max_radius_meter)

        y_vals= meter_to_mm*np.array(y_vals)
        z_vals= meter_to_mm*np.array(z_vals)
        r_vals = np.sqrt(y_vals ** 2 + z_vals ** 2)
        return_args = [y_vals, z_vals]
        if return_p:
            p_arr = np.array(p_vals)[r_vals < max_radius_mm]
            return_args.append(p_arr)
        if return_valid_indices:
            return_args.append(valid_indices)
        return return_args

    def D_90(self, x_position: float) -> float:
        y_arr, z_arr = self.interpolate(x_position)
        r_arr = np.sqrt(y_arr ** 2 + z_arr ** 2)
        R90= val_at_cumulative_fraction(r_arr, .9)
        D90=2*R90
        return D90

    def S_90(self, x_position: float,speed_trans_max: float=np.inf,r_max: float=np.inf) -> float:
        speed_trans=self.transverse_speeds(x_position,speed_trans_max,r_max)
        S90=val_at_cumulative_fraction(speed_trans,.9)
        return S90

    def transvers_position_density(self, x_interp: float, r_max: float, bins_center_sep_mm=.1) -> tuple[np.ndarray, np.ndarray]:
        """Return coordinates (radial values) and density of transverse particles distribution"""
        assert r_max > 0.0
        y_vals, z_vals, _ = self.interpolate(x_interp, return_p=True)
        r_vals = np.sqrt(y_vals ** 2 + z_vals ** 2)
        r_vals = r_vals[r_vals < r_max]
        density_r_vals, density=radial_density_histogram(r_vals,bins_center_sep_mm)
        return density_r_vals, density

    def transverse_speeds(self, x_interp: float, speed_trans_max, r_max) -> np.ndarray:
        assert speed_trans_max > 0.0
        y_vals,z_vals, p_vals = self.interpolate(x_interp, return_p=True)
        speed_trans=np.linalg.norm(p_vals[:,1:],axis=1)
        r_vals = np.sqrt(y_vals ** 2 + z_vals ** 2)
        speed_trans=speed_trans[r_vals<r_max]
        speed_trans=speed_trans[speed_trans<speed_trans_max]
        return speed_trans

    def transverse_speed_density(self, x_position: float, speed_trans_max: float=np.inf, r_max: float=np.inf, bins_center_sep_mm=.1) -> tuple[np.ndarray, np.ndarray]:
        """Return coordinates (radial values) and density of transverse particles distribution"""
        speed_trans=self.transverse_speeds(x_position, speed_trans_max, r_max)
        bins_speed_vals, density=radial_density_histogram(speed_trans,bins_center_sep_mm)
        return bins_speed_vals, density

    def fwhm(self, x_interp, r_max: float,  bins_center_sep_mm=.1) -> float:
        assert r_max > 0.0
        r_vals, density = self.transvers_position_density(x_interp, r_max,  bins_center_sep_mm)
        r_vals, density=make_dense_curve_1D_linear(r_vals, density) #fill in more points between values so that FWHM
        # values as a function of x_interp isn't so coarse
        half_max = density[0] / 2.0
        if np.min(density) > half_max:  # there is no half max!
            return np.nan
        r_vals_flipped=np.flip(r_vals)
        density_flipped=np.flip(density)
        half_width_half_max=r_vals_flipped[np.argmax(density_flipped-half_max>0)]
        full_width_half_max=2*half_width_half_max
        return full_width_half_max



def get_FWHM(x: float, interpFunction: Callable, Plot: bool = False, rMax: float = 10.0, w: float = .3,
             laser_scan_range_volts: float = np.inf) -> float:
    """

    :param x: location to get FWHM, cm
    :param interpFunction: Function for interpolating particle positions
    :param Plot: wether to plot the focus
    :param rMax: Maximum radius to consider, mm
    :param w: Interrogation size, mm. This will be edge size of bins used to construct the histogram for signal
    :param laser_scan_range_volts: Full voltage scan range. Used to limit transverse velocities contributing as per our
        actual method
    :return: the FWHM of the voigt
    """
    numBins = int(2 * rMax / w)
    y_arr, z_arr, p_arr = interpFunction(x, max_radius_mm=rMax, laser_scan_range_volts=laser_scan_range_volts,
                                         return_p=True)
    vxArr = p_arr[:, 0]
    r_arr, signalArr, signalErrArr = make_Radial_Signal_Arr(y_arr, z_arr, vxArr, numBins, rMax, numSamples=3)
    sigmaArr = signalErrArr / signalArr

    guess = [signalArr.max(), signalArr.min(), 1.0, 1.0]
    bounds = [0.0, np.inf]
    params = curve_fit(voigt, r_arr, signalArr, p0=guess, bounds=bounds, sigma=sigmaArr)[0]
    if Plot == True:
        plt.scatter(r_arr, signalArr)
        rPlot = np.linspace(0.0, r_arr.max(), 1_000)
        plt.plot(rPlot, voigt(rPlot, *params), c='r')
        plt.xlabel('radial position, mm')
        plt.ylabel('signal, au')
        plt.show()
    sigma = params[2]
    gamma = params[3]
    fL = gamma  # already FWHM
    fG = 2 * sigma * np.sqrt(2 * np.log(2))  # np.log is natural logarithm
    FWHM = .5346 * fL + np.sqrt(.2166 * fL ** 2 + fG ** 2)
    return FWHM


def make_Fake_Flat_Data():
    numGridEdge = 500
    rMax0 = 1.0
    xGridArr = np.linspace(-rMax0, rMax0, numGridEdge)
    yGridArr = xGridArr.copy()
    coords = np.asarray(np.meshgrid(xGridArr, yGridArr)).T.reshape(-1, 2)
    coords = coords[np.linalg.norm(coords, axis=1) < rMax0]
    x_arr, y_arr = coords.T
    density0 = 1.0 / (xGridArr[1] - xGridArr[0]) ** 2
    pxArr = np.ones((len(x_arr), 3))
    return x_arr, y_arr, pxArr, density0

# def make_Fake_Image(rMax, numSamplesGoal, sigma, gamma):
#     np.random.seed(42)
#     numSamples = 0
#     data = []
#     while numSamples < numSamplesGoal:
#         x, y = rMax * (2 * (np.random.random_sample(2) - .5))
#         r = np.sqrt(x ** 2 + y ** 2)
#         prob = voigt(r, 1.0, 0.0, sigma, gamma)
#         accept = np.random.random_sample() < prob
#         if accept == True:
#             data.append([x, y])
#             numSamples += 1
#     y_arr, z_arr = np.asarray(data).T
#     return y_arr, z_arr

# 
# def test__make_Density_And_Pos_Arr():
#     numBins = 100
#     x_arr, y_arr, p_arr, density0 = make_Fake_Flat_Data()
#     vxArr = p_arr[:, 0]
#     r_arr, densityArr, densityErrArr = make_Density_And_Pos_Arr(x_arr, y_arr, vxArr, numBins)
#     assert np.all(densityErrArr <= densityArr)
#     assert abs(len(densityArr) / (numBins ** 2 * (np.pi / 4.0)) - 1.0) < .1  # geometry works as expected
#     assert abs(np.mean(densityArr) / density0 - 1.0) < .1  # assert density mostly works
# 
# 
# def test__Radial_Signal_Arr():
#     numBins = 100
#     x_arr, y_arr, p_arr, density0 = make_Fake_Flat_Data()
#     vxArr = p_arr[:, 0]
#     rArr1, signalArr1, signalErrArr1 = make_Radial_Signal_Arr(x_arr, y_arr, vxArr, numBins, numSamples=1)
#     rArr2, signalArr2, signalErrArr2 = make_Radial_Signal_Arr(x_arr, y_arr, vxArr, numBins, numSamples=4)
#     assert abs(signalArr1.mean() / signalArr2.mean() - 1.0) < .05
#     assert abs(signalArr1.mean() / density0 - 1.0) < .05
#     assert abs(signalArr2.mean() / density0 - 1.0) < .05
#     assert np.all(signalErrArr1 <= signalArr1) and np.all(signalErrArr2 <= signalArr2)

#
# def test__get_FWHM():
#     sigma0, gamma0 = 1.0, 2.0
#     rMax0 = 10.0
#     w0 = .3
#     numBins0 = int(2 * rMax0 / w0)
#     numSamplesGoal0 = 100_000
#     y_arr, z_arr = make_Fake_Image(rMax0, numSamplesGoal0, sigma0, gamma0)
#     vxArr = np.ones(len(y_arr))
#     r_arr, signalArr, signalErrArr = make_Density_And_Pos_Arr(y_arr, z_arr, vxArr, numBins0)
#     sigmaArr = signalErrArr / signalArr
#
#     guess = [signalArr.max(), signalArr.min(), 1.0, 1.0]
#     bounds = [0.0, np.inf]
#     params = curve_fit(voigt, r_arr, signalArr, p0=guess, bounds=bounds, sigma=sigmaArr)[0]
#     # plt.scatter(r_arr,signalArr)
#     # rPlot=np.linspace(0.0,r_arr.max(),1_000)
#     # plt.plot(rPlot,voigt(rPlot,*params),c='r')
#     # plt.plot(rPlot,voigt(rPlot,*[signalArr.max(),0.0,sigma0,gamma0]),c='g')
#     # plt.show()
#     sigma = params[2]
#     gamma = params[3]
#     fL = gamma
#     fG = 2 * sigma * np.sqrt(2 * np.log(2))  # np.log is natural logarithm
#     FWHM = .5346 * fL + np.sqrt(.2166 * fL ** 2 + fG ** 2)
#     pFakeArr = np.ones((len(y_arr), 3))
#     fakeInterpFunc = lambda x, **kwargs: (y_arr, z_arr, pFakeArr)
#
#     FWHM_FromMyFunc = get_FWHM(np.nan, fakeInterpFunc, Plot=False, w=w0, rMax=rMax0)
#     assert abs(gamma / gamma0 - 1) < .1 and abs(sigma / sigma0 - 1) < .1
#     assert abs(FWHM_FromMyFunc / FWHM - 1.0) < 1e-3
#
#
# def test__Interpolator():
#     """dirty tester"""
#     LObject = 72.0E-2
#     LImage = 85E-2
#     LLensHardEdge = 15.24e-2
#     rpLens = (5e-2, 5e-2 + 2.54e-2)
#     magnet_width = (.0254, .0254 * 1.5)
#
#     voltScanRange = .25
#     freqRange = voltScanRange * 565e6
#     vTMax = freqRange * 671e-9
#     vTMax /= 2  # need to half because validity is assesed with radial velocity
#
#     fringe_frac = 1.5
#     LFringe = fringe_frac * max(rpLens)
#     LLens = LLensHardEdge + 2 * LFringe
#     LObject -= LFringe
#     LImage -= LFringe
#
#     lattice = ParticleTracerLattice(lattice_type='injector', field_dens_mult=1.0,
#                                 include_mag_errors=False)
#     lattice.add_drift(LObject, ap=.07)
#     lattice.add_halbach_lens_sim(rpLens, LLens, apFrac=None, magnet_width=magnet_width)
#     lattice.add_drift(LImage * 2, ap=.07)
#     assert lattice.el_list[1].fringe_frac_outer == fringe_frac and abs(lattice.el_list[1].Lm - LLensHardEdge) < 1e-9
#     lattice.end_lattice()
#     lensIndex = 1
#     assert type(lattice.el_list[lensIndex]) == HalbachLensSim
#     assert abs((lattice.el_list[lensIndex].L - lattice.el_list[lensIndex].Lm) / 2 - LFringe) < 1e-9
#
#     xEnd = -lattice.el_list[-1].r2[0] + 1e-3
#     swarmTest = Swarm()
#     theta_arr = np.linspace(.01, 1.5 * vTMax / v0, 30)
#     thetaPhiList = []
#     for theta in theta_arr:
#         numSamples = int(np.random.random_sample() * 15 + 3)
#         phiArr = np.linspace(0.0, 2 * np.pi, numSamples)[:-1]
#         for phi in phiArr:
#             thetaPhiList.append([theta, phi])
#             particle = Particle()
#             # tan is used because of the slope
#             p = np.array([-1.0, np.tan(theta) * np.sin(phi), np.tan(theta) * np.cos(phi)]) * v0
#             q = np.array([-xEnd, 0.0, 0.0])
#             particle.qf = q
#             particle.pf = p
#             swarmTest.particles.append(particle)
#     thetaPhiArr = np.array(thetaPhiList)
#     interpFunc = Interpolater(swarmTest, lattice)
#
#     xTest = 1.75
#     delta_x = abs(xEnd) - xTest
#     y_arr, z_arr, p_arr = interpFunc(xTest, return_p=True)
#     r_arr = np.sqrt(y_arr ** 2 + z_arr ** 2)
#     assert abs(r_arr.max() - delta_x * np.tan(theta_arr.max()) * 1e3) < 1e-12
#     assert len(thetaPhiArr) == len(y_arr) and len(thetaPhiArr) == len(z_arr) and len(thetaPhiArr) == len(p_arr)
#     # test that interpolation results agree with geoemtric model
#     yGeomArr = np.sin(thetaPhiArr[:, 1]) * delta_x * np.tan(-thetaPhiArr[:, 0]) * 1e3
#     zGeomArr = np.cos(thetaPhiArr[:, 1]) * delta_x * np.tan(-thetaPhiArr[:, 0]) * 1e3
#     assert np.all(np.abs(np.sort(z_arr) - np.sort(zGeomArr)) < 1e-6)
#     assert np.all(np.abs(np.sort(y_arr) - np.sort(yGeomArr)) < 1e-6)
#     # test that momentum is reduced by using laser scan range as expcted
#     y_arr, z_arr, p_arr = interpFunc(xTest, return_p=True, laser_scan_range_volts=voltScanRange)
#     vTArr = np.sqrt(p_arr[:, 1] ** 2 + p_arr[:, 2] ** 2)
#     assert vTArr.max() <= interpFunc._get_Sweep_Range_Trans_Vel_Max(voltScanRange)
#     assert len(y_arr) < len(yGeomArr)
#
#     import pytest
#     testVals = [0.0, -lattice.el_list[-1].r2[0], -1.0, np.inf]
#     for val in testVals:
#         with pytest.raises(ValueError):
#             interpFunc(val)
