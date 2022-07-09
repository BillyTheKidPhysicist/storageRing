from HalbachLensClass import Collection,HalbachLens
from latticeElements.utilities import MAGNET_ASPECT_RATIO

from scipy.spatial.transform import Rotation as Rot
from HalbachLensClass import Collection

from helperTools import *

from contextlib import contextmanager
import random
@contextmanager
def temporary_seed(seed: int)-> None:
    """context manager that temporarily override numpy and python random
    generators by seeding with the value 'seed', then restore the previous state"""

    state_np=np.random.get_state()
    state_py=random.getstate()
    np.random.seed(seed)
    yield
    np.random.set_state(state_np)
    random.setstate(state_py)



B_Vec_Arr,B_Norm_Arr=np.ndarray,np.ndarray
Dim1_Arr=np.ndarray

class MagneticOptic:
    def __init__(self,seed: int=None):
        self.r_in=None #input of magnet system in lab coordinates.
        self.r_out=None #output of magnet system in lab coordinates
        self.seed=int(time.time()) if seed is None else seed
        self.neighbors: list[MagneticOptic]=[]

    def get_magpylib_magnets(self,*args,**kwargs)-> Collection:
        with temporary_seed(self.seed):
            return self._make_magpylib_magnets(*args,**kwargs)

    def _make_magpylib_magnets(self,*args,**kwargs)-> Collection:
        raise NotImplemented

class MagneticLens(MagneticOptic):
    def __init__(self,Lm,rp_layers,magnet_widths,magnet_grade,use_solenoid,x_in_offset):
        assert all(rp1==rp2 for rp1,rp2 in zip(rp_layers,sorted(rp_layers)))
        assert len(rp_layers)==len(magnet_widths)
        super().__init__()
        self.Lm=Lm
        self.rp_layers=rp_layers
        self.magnet_widths=magnet_widths
        self.magnet_grade=magnet_grade
        self.use_solenoid=use_solenoid

        self.x_in_offset=x_in_offset

    def num_disks(self,magnet_errors)-> int:
        if magnet_errors:
            L_magnets = min([(MAGNET_ASPECT_RATIO * min(self.magnet_widths)), self.Lm])
            return round(self.Lm/L_magnets)
        else:
            return 1

    def _make_magpylib_magnets(self,magnet_errors)-> Collection:
        position=(self.Lm/2.0+self.x_in_offset, 0, 0)
        orientation=Rot.from_rotvec([0, np.pi / 2.0, 0.0])
        magnets = HalbachLens(self.rp_layers, self.magnet_widths, self.Lm, self.magnet_grade,
                            applyMethodOfMoments=True, useStandardMagErrors=magnet_errors,
                            numDisks=self.num_disks(magnet_errors), use_solenoid_field=self.use_solenoid,
                            orientation=orientation,position=position)
        return magnets

    def get_valid_coord_indices(self,coords: np.ndarray,interp_step_size: float)-> Dim1_Arr:

        valid_x_a = coords[:, 0] < self.x_in_offset - interp_step_size
        valid_x_b = coords[:, 0] > self.x_in_offset+self.Lm + interp_step_size
        rarr=np.linalg.norm(coords[:, 1:], axis=1)

        r_inner=self.rp_layers[0]
        r_outer=self.rp_layers[-1]+self.magnet_widths[-1]
        valid_r_a =  rarr< r_inner - interp_step_size
        valid_r_b =  rarr> r_outer + interp_step_size
        valid_indices=valid_x_a+valid_x_b+valid_r_a+valid_r_b
        return valid_indices
    
    def get_valid_field_values(self,coords: np.ndarray,interp_step_size: float,magnet_errors: bool,
                               interp_rounding_guard: float=1e-12)-> tuple[B_Vec_Arr,B_Norm_Arr]:
        assert interp_step_size>0.0 and interp_rounding_guard>0.0
        interp_step_size_valid=interp_step_size+interp_rounding_guard
        valid_indices=self.get_valid_coord_indices(coords,interp_step_size_valid)
        magnets=self.get_magpylib_magnets(magnet_errors)
        B_norm_grad, B_norm = np.zeros((len(valid_indices), 3)) * np.nan, np.ones(len(valid_indices)) * np.nan
        B_norm_grad[valid_indices], B_norm[valid_indices] = magnets.B_norm_grad(coords[valid_indices],
                                                                                return_norm=True,dx=interp_step_size)
        return B_norm_grad,B_norm
