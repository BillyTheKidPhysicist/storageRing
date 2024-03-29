"""
Old class used for trying to implement lens shimming to reduce impact of unwanted fringing field effects.
"""
#
# class geneticLens(LensIdeal):
#     def __init__(self, PTL, geneticLens, ap):
#         # if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
#         # to accomdate the new rp such as force values and positions
#         # super().__init__(PTL, geneticLens.length, geneticLens.maximum_Radius(), np.nan,np.nan,'injector',fillParams=False)
#         raise NotImplementedError #under construction still
#         super().__init__(PTL, geneticLens.length, None, geneticLens.maximum_Radius(), ap, 0.0, fillParams=False)
#         self.fringe_frac_outer = 4.0
#         self.L = geneticLens.length + 2 * self.fringe_frac_outer * self.rp
#         self.Lo = None
#         self.shape = 'STRAIGHT'
#         self.lens = geneticLens
#         assert self.lens.minimum_Radius() >= ap
#         self.fringe_frac_inner_min = np.inf  # if the total hard edge magnet length is longer than this value * rp, then it can
#         # can safely be modeled as a magnet "cap" with a 2D model of the interior
#         self.self.effectiveLength = None  # if the magnet is very long, to save simulation
#         # time use a smaller length that still captures the physics, and then model the inner portion as 2D
#
#         self.magnetic_potential_Func_Fringe = None
#         self.magnetic_potential_Func_Inner = None
#         self.field_fact = 1.0  # factor to multiply field values by for tunability
#         if self.L is not None:
#             self.build()
#
#     def set_length(self, L):
#         assert L > 0.0
#         self.L = L
#         self.build()
#
#     def build(self):
#         more robust way to pick number of points in element. It should be done by using the typical lengthscale
#         # of the bore radius
#
#         numPointsLongitudinal = 31
#         numPointsTransverse = 31
#
#         self.Lm = self.L - 2 * self.fringe_frac_outer * self.rp  # hard edge length of magnet
#         assert np.abs(self.Lm - self.lens.length) < 1e-6
#         assert self.Lm > 0.0
#         self.Lo = self.L
#
#         numXY = numPointsTransverse
#         # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
#         # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
#         # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
#         yArr_Quadrant = np.linspace(-TINY_STEP, self.ap + TINY_STEP, numXY)
#         xArr_Quadrant = np.linspace(-(self.ap + TINY_STEP), TINY_STEP, numXY)
#
#         z_min = -TINY_STEP
#         z_max = self.L / 2 + TINY_STEP
#         z_arr = np.linspace(z_min, z_max, num=numPointsLongitudinal)  # add a little extra so interp works as expected
#
#         # assert (z_arr[-1]-z_arr[-2])/self.rp<.2, "spatial step size must be small compared to radius"
#         assert len(xArr_Quadrant) % 2 == 1 and len(yArr_Quadrant) % 2 == 1
#         assert all((arr[-1] - arr[-2]) / self.rp < .1 for arr in [xArr_Quadrant, yArr_Quadrant]), "" \
#                                                                "spatial step size must be small compared to radius"
#
#         volume_coords = np.asarray(np.meshgrid(xArr_Quadrant, yArr_Quadrant, z_arr)).T.reshape(-1,
#                                                                             3)  # note that these coordinates can have
#         # the wrong value for z if the magnet length is longer than the fringe field effects. This is intentional and
#         # input coordinates will be shifted in a wrapper function
#         B_norm_grad, B_norm = self.lens.B_norm_grad(volume_coords, return_norm=True)
#         data_3D = np.column_stack((volume_coords, B_norm_grad, B_norm))
#         self.fill_Field_Func(data_3D)
#         # self.compile_fast_Force_Function()
#
#         # F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
#         # F_center = np.linalg.norm(self.force(np.asarray([self.L / 2, self.ap / 2, .0])))
#         # assert F_edge / F_center < .01
#
#     def force(self, q, searchIsCoordInside=True):
#         raise Exception("under construction")
#
#         # if np.isnan(F[0])==False:
#         #     if q[0]<2*self.rp*self.fringe_frac_outer or q[0]>self.L-2*self.rp*self.fringe_frac_outer:
#         #         return np.zeros(3)
#         # F = self.field_fact * np.asarray(F)
#         # return F
#
#     def fill_Field_Func(self, data):
#         interpF, interpV = self.make_Interp_Functions(data)
#
#         # wrap the function in a more convenietly accesed function
#         #-@numba.njit(numba.types.UniTuple(numba.float64, 3)(numba.float64, numba.float64, numba.float64))
#         def force_Func(x, y, z):
#             Fx0, Fy0, Fz0 = interpF(-z, y, x)
#             Fx = Fz0
#             Fy = Fy0
#             Fz = -Fx0
#             return Fx, Fy, Fz
#
#         self.force_Func = force_Func
#         self.magnetic_potential_Func = lambda x, y, z: interpV(-z, y, x)
#
#     def magnetic_potential(self, q):
#         # this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
#         x, y, z = q
#         y = abs(y)  # confine to upper right quadrant
#         z = abs(z)
#         if self.is_coord_inside(q) == False:
#             raise Exception(ValueError)
#
#         if 0 <= x <= self.L / 2:
#             x = self.L / 2 - x
#             V = self.magnetic_potential_Func(x, y, z)
#         elif self.L / 2 < x:
#             x = x - self.L / 2
#             V = self.magnetic_potential_Func(x, y, z)
#         else:
#             raise Exception(ValueError)
#         return V
