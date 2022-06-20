from optimizer import Solver
from helperTools import tool_Parallel_Process

def _test_Ring_Solver():
    xRing = (0.01232265, 0.00998983, 0.03899118, 0.00796353, 0.10642821, 0.4949227)
    sol=Solver('ring', xRing).solve(xRing)
    print(sol)
    assert sol.cost==0.9369001327646969
    assert sol.fluxMultiplication==19.827330098386547
def _test_Injector_Surrogate_Solver():
    xInjector=(0.1513998, 0.02959859, 0.12757497, 0.022621, 0.19413599, 0.01606958, 0.20289069, 0.23274805, 0.17537412)
    sol=Solver('injector_Surrogate_Ring',None).solve(xInjector)
    print(sol)
    assert sol.cost == 1.3160207927763141
    assert sol.survival == 66.40625
def _test_Injector_Actual_Ring():
    xInjector=(0.1513998, 0.02959859, 0.12757497, 0.022621, 0.19413599, 0.01606958, 0.20289069, 0.23274805, 0.17537412)
    xRing = (0.01232265, 0.00998983, 0.03899118, 0.00796353, 0.10642821, 0.4949227)
    sol=Solver('injector_Actual_Ring',xRing).solve(xInjector)
    print(sol)
    assert sol.cost == 0.9369001327646969
    assert sol.fluxMultiplication == 19.827330098386547
def _test_Both():
    xInjector=(0.1513998, 0.02959859, 0.12757497, 0.022621, 0.19413599, 0.01606958, 0.20289069, 0.23274805, 0.17537412)
    xRing = (0.01232265, 0.00998983, 0.03899118, 0.00796353, 0.10642821, 0.4949227)
    params=(*xRing,*xInjector)
    sol=Solver('both',None).solve(params)
    print(sol)
    assert sol.cost == 0.9369001327646969
    assert sol.fluxMultiplication == 19.827330098386547
def test():
    funcList=[_test_Both,_test_Ring_Solver,_test_Injector_Surrogate_Solver,_test_Injector_Surrogate_Solver]
    run=lambda func: func()
    tool_Parallel_Process(run,funcList)
