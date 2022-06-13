from optimizer import Solver

xRing = (0.01232265, 0.00998983, 0.03899118, 0.00796353, 0.10642821, 0.4949227)
sol=Solver('ring', xRing).solve(xRing)
print(sol)
assert sol.cost==0.6938718845512213
assert sol.fluxMultiplication==96.19201217596292

