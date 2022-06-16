# for genetic lens
# @numba.njit()
# def genetic_Lens_Force_NUMBA(x,y,z, L,ap,  force_Func):
#     FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
#     FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
#     y = abs(y)  # confine to upper right quadrant
#     z = abs(z)
#     if np.sqrt(y**2+z**2)>ap:
#         return np.nan,np.nan,np.nan
#     if 0<=x <=L/2:
#         x = L/2 - x
#         Fx,Fy,Fz= force_Func(x, y, z)
#         Fx=-Fx
#     elif L/2<x<L:
#         x=x-L/2
#         Fx,Fy,Fz = force_Func(x, y, z)
#     else:
#         return np.nan,np.nan,np.nan
#     Fy = Fy * FySymmetryFact
#     Fz = Fz * FzSymmetryFact
#     return Fx,Fy,Fz
