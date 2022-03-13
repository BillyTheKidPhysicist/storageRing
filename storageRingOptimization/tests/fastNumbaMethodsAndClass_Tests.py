import numpy as np

from fastNumbaMethodsAndClass import BaseClassFieldHelper_Numba
from hypothesis import given,settings,strategies as st

def full_Arctan(x2,x1):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    phi = np.arctan2(x2, x1)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi
class BaseFieldHelper_TestHelper:
    def __init__(self):
        self.baseFieldHelper=BaseClassFieldHelper_Numba()
        self.tol=1e-10
    def run_Tests(self):
        self.test1()
        self.test2()
    def test1(self):
        """Test that coords and forces are not changed by misalignment procedure when there is no misalignment"""""
        testRules=[st.floats(min_value=-1.0,max_value=1.0)]*6
        @given(*testRules)
        @settings(max_examples=100,deadline=None)
        def coord_And_Force_Alignment_Check(x0,y0,z0,Fx0,Fy0,Fz0):
            self.baseFieldHelper.update_Element_Perturb_Params(*(0.0,)*4)
            x,y,z=self.baseFieldHelper.misalign_Coords(x0,y0,z0)
            assert x==x0 and y==y0 and z==z0
            Fx,Fy,Fz=self.baseFieldHelper.rotate_Force_For_Misalignment(Fx0,Fy0,Fz0)
            assert Fx0==Fx and Fy0==Fy and Fz0==Fz
        coord_And_Force_Alignment_Check()
    def test2(self):
        """Test that mislalingment changes coords as expected"""
        testRules=[st.floats(min_value=.2,max_value=1.0)]*8
        testRules.extend([st.floats(min_value=.01,max_value=.1)]*2)
        @given(*testRules)
        @settings(max_examples=100,deadline=None)
        def coord_And_Force_Aligntment_Check(x0,y0,z0,Fx0,Fy0,Fz0,shiftY,shiftZ,rotY,rotZ):
            arr=np.asarray([x0,y0,z0,Fx0,Fy0,Fz0,shiftY,shiftZ,rotY,rotZ])
            if np.any(np.abs(arr)<1e-2):
                return
            self.baseFieldHelper.update_Element_Perturb_Params(shiftY,shiftZ,0.0,0.0)
            x, y, z = self.baseFieldHelper.misalign_Coords(x0, y0, z0)
            assert abs(y-(y0-shiftY))<self.tol and abs(z-(z0-shiftZ))<self.tol
            self.baseFieldHelper.update_Element_Perturb_Params(0.0,0.0,rotY*0.0, rotZ)
            x,y,z=self.baseFieldHelper.misalign_Coords(x0,y0,z0)
            Fx,Fy,Fz=self.baseFieldHelper.rotate_Force_For_Misalignment(Fx0,Fy0,Fz0)
            assert abs(full_Arctan(y,x)-full_Arctan(y0,x0))-abs(rotZ)<self.tol
            assert abs(full_Arctan(Fy, Fx) - full_Arctan(Fy0, Fx0)) - abs(rotZ) < self.tol
            self.baseFieldHelper.update_Element_Perturb_Params(0.0,0.0,rotY, rotZ*0.0)
            x,y,z=self.baseFieldHelper.misalign_Coords(x0,y0,z0)
            Fx,Fy,Fz=self.baseFieldHelper.rotate_Force_For_Misalignment(Fx0,Fy0,Fz0)
            assert abs(full_Arctan(z,x)-full_Arctan(z0,x0))-abs(rotY)<self.tol
            assert abs(full_Arctan(Fz, Fx) - full_Arctan(Fz0, Fx0)) - abs(rotY) < self.tol
        coord_And_Force_Aligntment_Check()
def run_Tests():
    BaseFieldHelper_TestHelper().run_Tests()
run_Tests()