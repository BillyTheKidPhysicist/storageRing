import numba
import numpy as np

from constants import FLAT_WALL_VACUUM_THICKNESS
from numba_functions_and_objects.interpFunctions import vec_interp3D, scalar_interp3D

spec_Combiner_Halbach = [
    ('field_data_internal', numba.types.UniTuple(numba.float64[::1], 7)),
    ('field_data_external', numba.types.UniTuple(numba.float64[::1], 7)),
    ('La', numba.float64),
    ('Lb', numba.float64),
    ('Lm', numba.float64),
    ('space', numba.float64),
    ('ap', numba.float64),
    ('ang', numba.float64),
    ('field_fact', numba.float64),
    ('extra_field_length', numba.float64),
    ('use_symmetry', numba.boolean),
    ('acceptance_width', numba.float64)
]


class CombinerHalbachLensSimFieldHelper_Numba:

    def __init__(self, field_data_internal, field_data_external, La, Lb, Lm, space, ap, ang, field_fact,
                 extra_field_length,
                 use_symmetry, acceptance_width):
        self.field_data_internal = field_data_internal
        self.field_data_external = field_data_external
        self.La = La
        self.Lb = Lb
        self.Lm = Lm
        self.space = space
        self.ap = ap
        self.ang = ang
        self.field_fact = field_fact
        self.extra_field_length = extra_field_length
        self.use_symmetry = use_symmetry
        self.acceptance_width = acceptance_width

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return (self.field_data_internal, self.La, self.Lb, self.Lm, self.space, self.ap, self.ang, self.field_fact,
                self.extra_field_length, self.use_symmetry, self.acceptance_width), ()

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.field_fact, ()

    def _force_Func_Internal(self, x, y, z):
        x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = self.field_data_internal
        Fx, Fy, Fz = vec_interp3D(x, y, z, x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr)
        return Fx, Fy, Fz

    def _force_Func_External(self, x, y, z):
        x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = self.field_data_external
        Fx, Fy, Fz = vec_interp3D(x, y, z, x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr)
        return Fx, Fy, Fz

    def _magnetic_potential_Func_Internal(self, x, y, z):
        x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = self.field_data_internal
        return scalar_interp3D(x, y, z, x_arr, y_arr, z_arr, V_arr)

    def _magnetic_potential_Func_External(self, x, y, z):
        x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = self.field_data_external
        return scalar_interp3D(x, y, z, x_arr, y_arr, z_arr, V_arr)

    def force(self, x, y, z):
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan, np.nan, np.nan
        else:
            return self.force_Without_isInside_Check(x, y, z)

    def force_Without_isInside_Check(self, x0, y0, z0):
        # this function uses the symmetry of the combiner to extract the force everywhere.
        # I believe there are some redundancies here that could be trimmed to save time.
        # x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        x, y, z = x0, y0, z0
        symmetryPlaneX = self.Lm / 2 + self.space  # field symmetry plane location
        if self.use_symmetry:
            FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
            FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
            y = abs(y)  # confine to upper right quadrant
            z = abs(z)

            if -self.extra_field_length <= x <= self.space:
                # print(x,y,z,self.Lm,self.space)
                Fx, Fy, Fz = self._force_Func_External(x, y, z)
            elif self.space < x <= symmetryPlaneX:
                Fx, Fy, Fz = self._force_Func_Internal(x, y, z)
            elif symmetryPlaneX < x <= self.Lm + self.space:
                x = 2 * symmetryPlaneX - x
                Fx, Fy, Fz = self._force_Func_Internal(x, y, z)
                Fx = -Fx
            elif self.space + self.Lm < x:
                x = 2 * symmetryPlaneX - x
                Fx, Fy, Fz = self._force_Func_External(x, y, z)
                Fx = -Fx
            else:
                print(x, y, z, self.Lm, self.space)
                raise ValueError
            Fy = Fy * FySymmetryFact
            Fz = Fz * FzSymmetryFact
        else:
            Fx, Fy, Fz = self._force_Func_Internal(x, y, z)
        # Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        Fx *= self.field_fact
        Fy *= self.field_fact
        Fz *= self.field_fact
        return Fx, Fy, Fz

    def magnetic_potential(self, x, y, z):
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan
        # x, y, z = self.baseClass.misalign_Coords(x, y, z)
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        symmetryPlaneX = self.Lm / 2 + self.space  # field symmetry plane location
        if self.use_symmetry:
            if -self.extra_field_length <= x <= self.space:
                V = self._magnetic_potential_Func_External(x, y, z)
            elif self.space < x <= symmetryPlaneX:
                V = self._magnetic_potential_Func_Internal(x, y, z)
            elif symmetryPlaneX < x <= self.Lm + self.space:
                x = 2 * symmetryPlaneX - x
                V = self._magnetic_potential_Func_Internal(x, y, z)
            elif self.Lm + self.space < x:  # particle can extend past 2*symmetryPlaneX
                x = 2 * symmetryPlaneX - x
                V = self._magnetic_potential_Func_External(x, y, z)
            else:
                print(x, y, z, self.Lm, self.space)
                raise ValueError
        else:
            V = self._magnetic_potential_Func_Internal(x, y, z)
        V = V * self.field_fact
        return V

    def is_Coord_Inside_Vacuum(self, x, y, z):
        # q: coordinate to test in element's frame
        standOff = 10e-6  # first valid (non np.nan) interpolation point on face of lens is 1e-6 off the surface of the lens
        assert FLAT_WALL_VACUUM_THICKNESS > standOff
        if not -self.ap <= z <= self.ap:  # if outside the z apeture (vertical)
            return False
        elif 0 <= x <= self.Lb + FLAT_WALL_VACUUM_THICKNESS:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner.
            if np.sqrt(y ** 2 + z ** 2) < self.ap:
                return True
            else:
                return False
        elif x < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            # todo: these should be in functions if they are used elsewhere
            m = np.tan(self.ang)
            Y1 = m * x + (self.acceptance_width - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * x + (-self.acceptance_width - m * self.Lb)
            if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
                return True
            elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
                return True
            else:
                return False