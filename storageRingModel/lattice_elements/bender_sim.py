import warnings
from math import isclose, tan, cos, sin, sqrt, atan, pi
from typing import Optional

import numpy as np
import scipy.optimize as spo
from scipy.spatial.transform import Rotation as Rot

from constants import MIN_MAGNET_MOUNT_THICKNESS, SIMULATION_MAGNETON, TUBE_WALL_THICKNESS
from field_generators import BenderSim as HalbachBender_FieldGenerator
from helper_tools import arr_product, round_and_make_odd
from lattice_elements.bender_ideal import BenderIdeal
from lattice_elements.utilities import TINY_OFFSET, is_even, mirror_across_angle, full_arctan2, \
    max_tube_IR_in_segmented_bend, halbach_magnet_width, calc_unit_cell_angle, B_GRAD_STEP_SIZE, \
    INTERP_MAGNET_MATERIAL_OFFSET, TINY_INTERP_STEP
from numba_functions_and_objects import bender_sim_numba_functions
from type_hints import sequence, ndarray, RealNum

dummy_field_data_empty = (np.ones(1) * np.nan,) * 7


def speed_with_energy_correction(U_longitudinal: RealNum, atom_speed: RealNum) -> float:
    """Energy conservation for atom speed accounting for reduction from potential"""
    E0 = .5 * atom_speed ** 2
    KE = E0 - U_longitudinal
    speed_corrected = np.sqrt(2 * KE)
    return speed_corrected


# todo: fix bug with coord outside when right on edge on input

class BenderSim(BenderIdeal):
    fringe_frac_outer: float = 1.5  # multiple of bore radius to accomodate fringe field

    num_model_lenses: int = 7  # number of lenses in halbach model to represent repeating system. Testing has shown
    # this to be optimal

    num_points_bore_ap_default = 25

    def __init__(self, PTL, Lm: float, rp: float, num_magnets: Optional[int], rb: float, ap: Optional[float],
                 r_offset_fact: float):
        assert all(val > 0 for val in (Lm, rp, rb, r_offset_fact))
        assert rb > rp * 10  # this would be very dubious
        super().__init__(PTL, None, None, rp, rb, None)
        self.rb = rb
        self.Lm = Lm
        self.rp = rp
        self.ap = ap
        self.magnet_width = halbach_magnet_width(rp, use_standard_sizes=PTL.use_standard_mag_size)
        self.ucAng: Optional[float] = None
        self.r_offset_fact = r_offset_fact  # factor to times the theoretic optimal bending radius by
        self.L_cap = self.fringe_frac_outer * self.rp
        self.num_magnets = num_magnets
        self.numPointsBoreAp: int = round_and_make_odd(self.num_points_bore_ap_default * self.PTL.field_dens_mult)
        # This many points should span the bore ap for good field sampling

    def compute_maximum_aperture(self) -> float:
        # beacuse the bender is segmented, the maximum vacuum tube allowed is not the bore of a single magnet
        # use simple geoemtry of the bending radius that touches the top inside corner of a segment
        ap_max_geom = max_tube_IR_in_segmented_bend(self.rb, self.rp, self.Lm, TUBE_WALL_THICKNESS,
                                                    use_standard_sizes=self.PTL.use_standard_tube_OD)
        delta = sqrt(2) * self.rp / self.numPointsBoreAp  # maximum interp grid spacing
        ap_max_interp = (self.rp - INTERP_MAGNET_MATERIAL_OFFSET) - delta
        # without particles seeing field interpolation reaching into magnetic materal. Will not be exactly true for
        # several reasons (using int, and non equal grid in xy), so I include a small safety factor
        if ap_max_interp < ap_max_geom:  # for now, I want this to be the case
            warnings.warn("bender aperture being limited by the good field region")
        ap_max = min([ap_max_geom, ap_max_interp])
        assert ap_max < self.rp
        return ap_max

    def fill_pre_constrained_parameters(self) -> None:
        self.output_offset = self.find_optimal_radial_offset() * self.r_offset_fact
        self.ro = self.output_offset + self.rb

    def find_optimal_radial_offset(self) -> float:
        """Find the radial offset that accounts for the centrifugal force moving the particles deeper into the
        potential well"""

        m = 1  # in simulation units mass is 1kg
        uc_ang_approx = self.unit_cell_angle()  # this will be different if the bore radius changes
        lens = HalbachBender_FieldGenerator(self.rp, self.rb, uc_ang_approx, self.Lm, self.PTL.magnet_grade, 10,
                                            (False, False), use_pos_mag_angs_only=False,
                                            magnet_width=self.magnet_width,
                                            use_method_of_moments=True, use_solenoid_field=self.PTL.use_solenoid_field)
        lens.rotate(Rot.from_rotvec([np.pi / 2, 0, 0]))
        theta_arr = np.linspace(0.0, 2 * uc_ang_approx, 100)
        z_arr = np.zeros(len(theta_arr))

        def offset_error(r_offset):
            assert abs(r_offset) < self.rp
            r = self.rb + r_offset
            x_arr = r * np.cos(theta_arr)
            y_arr = r * np.sin(theta_arr)
            coords = np.column_stack((x_arr, y_arr, z_arr))
            norms = [coord / np.linalg.norm(coord) for coord in coords]
            forces = lens.B_norm_grad(coords) * SIMULATION_MAGNETON
            V_vals = lens.B_norm(coords) * SIMULATION_MAGNETON
            Fr = np.array([np.dot(force, norm) for force, norm in zip(forces, norms)])
            atom_speeds = np.array([speed_with_energy_correction(V, self.PTL.speed_nominal) for V in V_vals])
            FCen = m * atom_speeds ** 2 / r
            error = np.sqrt(np.sum((Fr - FCen) ** 2)) / np.mean(FCen)
            return error

        r_offset_max = .9 * self.rp
        bounds = [(0.0, r_offset_max)]
        sol = spo.minimize(offset_error, np.array([self.rp / 3.0]), bounds=bounds, method='Nelder-Mead',
                           options={'xatol': 500e-9, 'ftol': 1e-6})
        r_offset_optimal = sol.x[0]
        if isclose(r_offset_optimal, r_offset_max, abs_tol=1e-6):
            raise Exception("The bending bore radius is too large to accomodate a reasonable solution")
        return r_offset_optimal

    def unit_cell_angle(self) -> float:
        """Get the angle that a single unit cell spans. Each magnet is composed of two unit cells because of symmetry.
        The unit cell includes half of the magnet and half the gap between the two"""

        # todo: why is this a function actually? and why is it called in the roffset thing?

        return calc_unit_cell_angle(self.Lm, self.rb, self.rp + self.magnet_width)

    def fill_post_constrained_parameters(self) -> None:
        self.ap = self.ap if self.ap is not None else self.compute_maximum_aperture()
        assert self.ap <= self.compute_maximum_aperture()
        self.ucAng = self.unit_cell_angle()
        self.ang = 2 * self.num_magnets * self.ucAng
        self.fill_in_and_out_rotation_matrices()
        assert self.ang < 2 * pi * 3 / 4  # not sure why i put this here
        self.ro = self.rb + self.output_offset
        self.L = self.ang * self.rb + 2 * self.L_cap

        self.Lo = self.ang * self.ro + 2 * self.L_cap
        self.outer_half_width = self.rp + self.magnet_width + MIN_MAGNET_MOUNT_THICKNESS
        self.make_orbit()

    def build_fast_field_helper(self, extra_magnets=None) -> None:
        """compute field values and build fast numba helper"""
        field_data_seg = self.generate_segment_field_data()
        field_data_internal = self.generate_internal_fringe_field_data()
        field_data_cap = self.generate_cap_field_data()
        field_data_perturbation = self.generate_perturbation_data() if self.PTL.use_mag_errors else dummy_field_data_empty
        assert np.all(field_data_cap[0] == field_data_internal[0]) and np.all(
            field_data_cap[2] == field_data_internal[2])
        field_data = (field_data_seg, field_data_internal, field_data_cap, field_data_perturbation)
        use_field_perturbations = self.PTL.use_mag_errors

        m = np.tan(self.ucAng)
        M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        m = np.tan(self.ang / 2)
        M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])

        numba_func_constants = (self.rb, self.ap, self.L_cap, self.ang, self.num_magnets,
                                self.ucAng, M_ang, RIn_Ang, M_uc, self.field_fact, use_field_perturbations)

        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(bender_sim_numba_functions, force_args, potential_args, is_coord_in_vacuum_args)

    def make_Grid_Coords(self, x_min: float, x_max: float, y_min: float, y_max: float) -> ndarray:
        """Make Array of points that the field will be evaluted at for fast interpolation. only x and s values change.
        """
        assert not is_even(self.numPointsBoreAp)  # points should be odd to there is a point at zero field, if possible
        longitudinal_coord_spacing: float = (.8 * self.rp / 10.0) / self.PTL.field_dens_mult  # Spacing
        # through unit cell. .8 was carefully chosen
        num_x = round_and_make_odd(self.numPointsBoreAp * (x_max - x_min) / self.ap)
        z_min, z_max = -TINY_INTERP_STEP, (self.ap + TINY_INTERP_STEP)  # same for every part of bender
        num_z = self.numPointsBoreAp
        num_y = round_and_make_odd((y_max - y_min) / longitudinal_coord_spacing)
        assert (num_x + 1) / num_z >= (x_max - x_min) / (z_max - z_min)  # should be at least this ratio
        coord_arrs = []
        for (start, stop, num) in ((x_min, x_max, num_x), (y_min, y_max, num_y), (z_min, z_max, num_z)):
            coord_arrs.append(np.linspace(start, stop, num))
        grid_coords = np.asarray(np.meshgrid(*coord_arrs)).T.reshape(-1, 3)
        return grid_coords

    def convert_center_to_cartesian_coords(self, s: float, xc: float,
                                           yc: float, r_offset=0.0) -> tuple[float, float, float]:
        """Convert center coordinates [s,xc,yc] to cartesian coordinates[x,y,z]"""
        r_center = self.rb + r_offset
        if -TINY_OFFSET <= s < self.L_cap:
            x, y, z = r_center + xc, s - self.L_cap, yc
        elif self.L_cap <= s < self.L_cap + self.ang * r_center:
            theta = (s - self.L_cap) / r_center
            r = r_center + xc
            x, y, z = cos(theta) * r, sin(theta) * r, yc
        elif self.L_cap + self.ang * r_center <= s <= self.ang * r_center + 2 * self.L_cap + TINY_OFFSET:
            theta = self.ang
            r = r_center + xc
            x0, y0 = cos(theta) * r, sin(theta) * r
            delta_s = s - (self.ang * r_center + self.L_cap)
            theta_perp = pi + atan(-1 / tan(theta))
            x, y, z = x0 + cos(theta_perp) * delta_s, y0 + sin(theta_perp) * delta_s, yc
        else:
            raise ValueError
        return x, y, z

    def convert_orbit_to_cartesian_coords(self, s: float, xo: float, yo: float) -> tuple[float, float, float]:
        """Convert orbit coordinates [s,xo,yo] to cartesian coordinates[x,y,z]"""

        return self.convert_center_to_cartesian_coords(s, xo, yo, r_offset=self.ro - self.rb)

    def make_perturbation_data_coords(self) -> tuple[ndarray, ndarray]:
        """Make coordinates for computing and interpolation perturbation data. The perturbation field exists in an
        evenly spaced grid in "center" coordinates [s,xc,yc] where s is distance along bender through center, xc is
        radial distance from center with positive meaning along larger radius and 0 meaning right  at the center,pu
        and yc is distance along z axis. HalbachLensClass.BenderSim is in (x,z) plane with z=0 at start
        and going clockwise in +y. This needs to be converted to cartesian coordinates to actually evaluate the field
        value"""

        Ls = 2 * self.L_cap + self.ang * self.rb
        num_s = round_and_make_odd(5 * (self.num_magnets + 2))  # carefully measured
        num_yc = round_and_make_odd(35 * self.PTL.field_dens_mult)
        num_xc = num_yc

        s_arr = np.linspace(-TINY_OFFSET, Ls + TINY_OFFSET, num_s)  # distance through bender along center
        xc_arr = np.linspace(-self.ap - TINY_OFFSET, self.ap + TINY_OFFSET,
                             num_xc)  # radial deviation along major radius
        yc_arr = np.linspace(-self.ap - TINY_OFFSET, self.ap + TINY_OFFSET,
                             num_yc)  # deviation in vertical from center of
        # bender, along y in cartesian
        assert not is_even(len(s_arr)) and not is_even(len(xc_arr)) and not is_even(len(yc_arr))
        coords_center = arr_product(s_arr, xc_arr, yc_arr)
        coords = np.asarray([self.convert_center_to_cartesian_coords(*coordCenter) for coordCenter in coords_center])
        return coords_center, coords

    def generate_perturbation_data(self) -> tuple[ndarray, ...]:
        coords_center, coords_cartesian = self.make_perturbation_data_coords()
        lens_imperfect = self.build_bender(True, (True, True), use_method_of_moments=False,
                                           num_lenses=self.num_magnets + 1,
                                           use_mag_errors=True)
        lens_perfect = self.build_bender(True, (True, True), use_method_of_moments=False,
                                         num_lenses=self.num_magnets + 1)
        r_center_arr = np.linalg.norm(coords_center[:, 1:], axis=1)
        valid_indices = r_center_arr < self.rp
        vals_imperfect = np.column_stack(self.compute_valid_field_vals(lens_imperfect, coords_cartesian, valid_indices))
        vals_perfect = np.column_stack(self.compute_valid_field_vals(lens_perfect, coords_cartesian, valid_indices))
        vals_perturbation = vals_imperfect - vals_perfect
        vals_perturbation[np.isnan(vals_perturbation)] = 0.0
        interp_data = np.column_stack((coords_center, vals_perturbation))
        interp_data = self.shape_field_data_3D(interp_data)
        return interp_data

    def generate_cap_field_data(self) -> tuple[ndarray, ...]:
        # x and y bounds should match with internal fringe bounds
        x_min = (self.rb - self.ap) * cos(2 * self.ucAng) - TINY_INTERP_STEP
        x_max = self.rb + self.ap + TINY_INTERP_STEP
        y_min = -(self.L_cap + TINY_INTERP_STEP)
        y_max = TINY_INTERP_STEP
        field_coords = self.make_Grid_Coords(x_min, x_max, y_min, y_max)
        valid_indices = np.sqrt(
            (field_coords[:, 0] - self.rb) ** 2 + field_coords[:, 2] ** 2) < self.rp - B_GRAD_STEP_SIZE
        lens = self.build_bender_fringe_cap_model()
        return self.compute_valid_field_data(lens, field_coords, valid_indices)

    def generate_internal_fringe_field_data(self) -> tuple[ndarray, ...]:
        """An magnet slices are required to model the region going from the cap to the repeating unit cell,otherwise
        there is too large of an energy discontinuity"""
        # x and y bounds should match with cap bounds
        x_min = (self.rb - self.ap) * cos(2 * self.ucAng) - TINY_INTERP_STEP  # inward enough to account for the tilt
        x_max = self.rb + self.ap + TINY_INTERP_STEP
        y_min = -TINY_INTERP_STEP
        y_max = tan(2 * self.ucAng) * (self.rb + self.ap) + TINY_INTERP_STEP
        field_coords = self.make_Grid_Coords(x_min, x_max, y_min, y_max)
        lens = self.build_bender_fringe_cap_model()
        valid_indices = self.get_valid_indices_internal(field_coords, 3)
        return self.compute_valid_field_data(lens, field_coords, valid_indices)

    def is_valid_in_lens_of_bender(self, x: bool, y: bool, z: bool) -> bool:
        """Check that the coordinates x,y,z are valid for a lens in the bender. The lens is centered on (self.rb,0,0)
        aligned with the z axis. If the coordinates are outside the double unit cell containing the lens, or inside
        the toirodal cylinder enveloping the magnet material, the coordinate is invalid"""
        y_uc_line = tan(self.ucAng) * x
        minor_radius = sqrt((x - self.rb) ** 2 + z ** 2)
        lens_radius_valid_inner = self.rp - INTERP_MAGNET_MATERIAL_OFFSET
        lens_radius_valid_outer = self.rp + self.magnet_width + INTERP_MAGNET_MATERIAL_OFFSET
        if abs(y) < (self.Lm + B_GRAD_STEP_SIZE) / 2.0 and \
                lens_radius_valid_inner <= minor_radius < lens_radius_valid_outer:
            return False
        elif abs(y) <= y_uc_line:
            return True
        else:
            return False

    def get_valid_indices_internal(self, coords: ndarray, max_rotations: int) -> list[bool]:
        """Check if coords are not in the magnetic material region of the bender. Check up to max_rotations of the
        coords going counterclockwise about y axis by rotating coords"""
        R = Rot.from_rotvec([0, 0, -self.ucAng]).as_matrix()
        valid_indices = []
        for [x, y, z] in coords:
            if self.is_valid_in_lens_of_bender(x, y, z):
                valid_indices.append(True)
            else:
                loop_start, loop_stop = 1, max_rotations + 1
                for i in range(loop_start, loop_stop):
                    num_rotations = (i + 1) // 2
                    x, y, z = (R ** num_rotations) @ [x, y, z]
                    if self.is_valid_in_lens_of_bender(x, y, z):
                        valid_indices.append(True)
                        break
                    elif i == loop_stop - 1:
                        valid_indices.append(False)
        return valid_indices

    def generate_segment_field_data(self) -> tuple[ndarray, ...]:
        """Internal repeating unit cell segment. This is modeled as a tilted portion with angle self.ucAng to the
        z axis, with its bottom face at z=0 alinged with the xy plane. In magnet frame coordinates"""
        x_min = (self.rb - self.ap) * cos(self.ucAng) - TINY_INTERP_STEP
        x_max = self.rb + self.ap + TINY_INTERP_STEP
        y_min = -TINY_INTERP_STEP
        y_max = tan(self.ucAng) * (self.rb + self.ap) + TINY_INTERP_STEP
        field_coords = self.make_Grid_Coords(x_min, x_max, y_min, y_max)

        valid_indices = self.get_valid_indices_internal(field_coords, 1)
        lens = self.build_bender_internal_model()
        return self.compute_valid_field_data(lens, field_coords, valid_indices)

    def compute_valid_field_vals(self, lens: HalbachBender_FieldGenerator, field_coords: ndarray,
                                 valid_indices: sequence) -> tuple[ndarray, ndarray]:
        B_norm_grad_arr, B_norm_arr = np.zeros((len(field_coords), 3)) * np.nan, np.zeros(len(field_coords)) * np.nan
        B_norm_grad_arr[valid_indices], B_norm_arr[valid_indices] = lens.B_norm_grad(field_coords[valid_indices],
                                                                                     return_norm=True, use_approx=True)
        return B_norm_grad_arr, B_norm_arr

    def compute_valid_field_data(self, lens: HalbachBender_FieldGenerator, field_coords: ndarray,
                                 valid_indices: sequence) -> tuple[ndarray, ...]:
        B_norm_grad_arr, B_norm_arr = self.compute_valid_field_vals(lens, field_coords, valid_indices)
        field_data_unshaped = np.column_stack((field_coords, B_norm_grad_arr, B_norm_arr))
        return self.shape_field_data_3D(field_data_unshaped)

    def build_bender(self, use_pos_mag_angs_only: bool, use_half_cap_end: tuple[bool, bool],
                     use_method_of_moments: bool = True, num_lenses: int = None, use_mag_errors: bool = False):
        num_lenses = self.num_model_lenses if num_lenses is None else num_lenses
        bender_field_generator = HalbachBender_FieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                                              self.PTL.magnet_grade,
                                                              num_lenses, use_half_cap_end,
                                                              use_method_of_moments=use_method_of_moments,
                                                              use_pos_mag_angs_only=use_pos_mag_angs_only,
                                                              use_solenoid_field=self.PTL.use_solenoid_field,
                                                              use_mag_errors=use_mag_errors,
                                                              magnet_width=self.magnet_width)
        bender_field_generator.rotate(Rot.from_rotvec([-np.pi / 2, 0, 0]))
        return bender_field_generator

    def build_full_bender_model(self):
        return self.build_bender(True, (True, True), num_lenses=self.num_magnets + 1)

    def build_bender_internal_model(self):
        num_lenses = self.num_model_lenses
        assert not is_even(num_lenses)
        return self.build_bender(False, (False, False))

    def build_bender_fringe_cap_model(self):
        return self.build_bender(True, (True, False))

    def in_which_section_of_bender(self, q_el: ndarray) -> str:
        """Find which section of the bender q_el is in. options are:
            - 'IN' refers to the westward cap. at some angle
            - 'OUT' refers to the eastern. input is aligned with y=0
            - 'ARC' in the bending arc between input and output caps
        Return 'NONE' if not inside the bender"""

        angle = full_arctan2(q_el)
        if 0 <= angle <= self.ang:
            return 'ARC'
        cap_names = ['IN', 'OUT']
        for name in cap_names:
            x_cap, y_cap = mirror_across_angle(q_el[0], q_el[1], self.ang / 2.0) if name == 'IN' else q_el[:2]
            if (self.rb - (self.ap + TINY_OFFSET) <= x_cap <= self.rb + (self.ap + TINY_OFFSET)) and (
                    TINY_OFFSET >= y_cap >= -(self.L_cap + TINY_OFFSET)):
                return name
        return 'NONE'

    def transform_element_coords_into_local_orbit_frame(self, q_el: ndarray) -> ndarray:

        which_section = self.in_which_section_of_bender(q_el)
        if which_section == 'ARC':
            phi = self.ang - full_arctan2(q_el)
            xo = sqrt(q_el[0] ** 2 + q_el[1] ** 2) - self.ro
            so = self.ro * phi + self.L_cap  # include the distance traveled throught the end cap
        elif which_section == 'OUT':
            so = self.L_cap + self.ang * self.ro + (-q_el[1])
            xo = q_el[0] - self.ro
        elif which_section == 'IN':
            x_mirror, y_mirror = mirror_across_angle(q_el[0], q_el[1], self.ang / 2.0)
            so = self.L_cap + y_mirror
            xo = x_mirror - self.ro
        else:
            raise ValueError
        qo = np.array([so, xo, q_el[2]])
        return qo

    def transform_element_momentum_into_local_orbit_frame(self, q_el: ndarray, p_el: ndarray) -> ndarray:
        """Overrides abstract method from Element class. Mildly tricky. Need to determine if the position is in
        one of the caps or the bending segment, then handle accordingly"""

        which_section = self.in_which_section_of_bender(q_el)
        if which_section == 'ARC':
            return super().transform_element_momentum_into_local_orbit_frame(q_el, p_el)
        elif which_section == 'OUT':
            pso, pxo = -p_el[1], p_el[0]
        elif which_section == 'IN':
            pxo, pso = mirror_across_angle(p_el[0], p_el[1], self.ang / 2.0)
        else:
            raise ValueError
        p_orbit = np.array([pso, pxo, p_el[2]])
        return p_orbit