#simple unit tests. Should have been doing this earlier



from HalbachLensClass import B_NUMBA,HalbachLens
import numpy as np
sufficientPrecision=1e-12
def test_Magnetic_Dipole_Field():
    r=np.asarray([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    r0=np.asarray([.1,.1,-.5])
    m=np.asarray([.5,.5,.5])
    expectedResult=np.array([[1.4983384417993377e-11, 8.4788328294762813e-10, 2.1805231205950436e-09],
        [8.1406384161066075e-11 ,1.1971428894877305e-10, 1.8100693660910416e-10]])
    assert np.all(np.abs(B_NUMBA(r,r0,m)-expectedResult)<sufficientPrecision)
def test_Halbach_Lens():
    expectedResult1=np.array([4.2614844645801936e-01, 4.2618299183022451e-01,
       1.3803794641619599e-04])
    expectedResult2=np.array([[-1.7482760031328759e-11,  4.2613693978892087e-01,
         6.9031586003021062e-05],
       [ 9.2740888961855106e-09, -8.2496573306534037e-09,
        -8.8440601139681529e-08]])
    rTest1 = np.asarray([1e-3, 1e-3, -1e-3])
    rTest2 = np.asarray([[0.0, 1e-3, -1e-3], [.1, .5, 1.0]])
    magnetWidth = (2.54e-2, 2.54e-2 * 1.5)
    rp = (.06, .09)
    length = .1
    lens = HalbachLens(length, magnetWidth, rp)
    assert np.all(np.abs(lens.BNorm_Gradient(rTest1)-expectedResult1) <sufficientPrecision)
    assert np.all(np.abs(lens.BNorm_Gradient(rTest2)-expectedResult2) <sufficientPrecision)