from periodicLatticeSolver import PeriodicLatticeSolver
from minimizer import Minimizer
import numpy as np
from FloorPlanClass import FloorPlan
from plotter import Plotter
PLS = PeriodicLatticeSolver(200, .02, axis='both')
PLS.add_Injector()

L1 =PLS.Variable('L1', varMin=.01, varMax=.5)
L2= PLS.Variable('L2', varMin=.01, varMax=.5)
L3 =PLS.Variable('L3', varMin=.01, varMax=.5)
L4= PLS.Variable('L4', varMin=.01, varMax=.5)

Bp1 = .45#PLS.Variable('Bp1', varMin=.1, varMax=.45)
Bp2 = .45#PLS.Variable('Bp2', varMin=.1, varMax=.45)
Bp3 = .45#PLS.Variable('Bp3', varMin=.1, varMax=.45)
Bp4 = .45#PLS.Variable('Bp4', varMin=.1, varMax=.45)

rp1 = .025#PLS.Variable('rp1', varMin=.01, varMax=.03)
rp2 = .025#PLS.Variable('rp2', varMin=.01, varMax=.03)
rp3 = .025#PLS.Variable('rp3', varMin=.01, varMax=.03)
rp4 = .025#PLS.Variable('rp4', varMin=.01, varMax=.03)

r0=1#PLS.Variable('r0',varMin=1,varMax=1.3)
TL1=1#PLS.Variable('TL1',varMin=.5,varMax=1.5)
TL2=1#PLS.Variable('TL2',varMin=.5,varMax=1.5)


PLS.set_Track_Length(TL1=TL1,TL2=TL2)
PLS.begin_Lattice()

PLS.add_Bend(np.pi, r0, .45)
PLS.add_Lens(L4, Bp4, rp4)
PLS.add_Drift()
PLS.add_Combiner()
PLS.add_Drift()
PLS.add_Lens(L1, Bp1,rp1)
PLS.add_Bend(np.pi, r0, .45)
PLS.add_Lens(L2, Bp2, rp2)
PLS.add_Drift()
PLS.add_Lens(L3, Bp3, rp3)
PLS.end_Lattice()
for el in PLS.lattice:
    args = [1, 1, 1, 1]
    if el.elType=='BEND':
        print(el.apxFuncL(*args))

        #print(el.elType, el.penisFunc, el.rpFunc, el.apxFuncR(*args), el.apxFuncL(*args))