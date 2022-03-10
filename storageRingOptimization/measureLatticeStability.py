import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import skopt
import numpy as np
import multiprocess as mp
from optimizerHelperFunctions import solve_For_Lattice_Params
import matplotlib.pyplot as plt
import skopt
import time

bumpAmp=1e-3
varJitterAmp=1e-3

def wrapper(args,seed):
    np.random.seed(seed)
    try:
        tuning=None
        sol=solve_For_Lattice_Params(args,tuning,bumpOffsetAmp=bumpAmp)
        cost=sol.swarmCost
        print(cost)
    except:
        np.set_printoptions(precision=100)
        print('assert during evaluation on args: ',args)
        assert False
    return cost
'''
------ITERATIONS:  3480
POPULATION VARIABILITY: [0.01475089 0.01717158 0.01157133 0.01893284]
BEST MEMBER BELOW
---population member---- 
DNA: array([0.02417499, 0.02112171, 0.02081137, 0.22577471])
cost: 0.7099381604306393
'''
numSamples=50
#rplens rplensfirst rplenslast rpBend LLens
Xi=np.array([0.02398725, 0.02110859, 0.02104631, 0.22405252])


wrapper(Xi,1)
exit()
deltaXTest=np.ones(len(Xi))*varJitterAmp/2
boundsUpper=Xi+deltaXTest
boundsLower=Xi-deltaXTest
bounds=np.row_stack((boundsLower,boundsUpper)).T

samples=np.asarray(skopt.sampler.Sobol().generate(bounds, numSamples-1))
samples=np.row_stack((samples,Xi))
seedArr=np.arange(numSamples)+int(time.time())


with mp.Pool() as pool:
    results=np.asarray(pool.starmap(wrapper,zip(samples,seedArr)))
print(results)
data=np.column_stack((samples-Xi,results))
np.savetxt('stabilityData',data)
# plt.hist(data[:,-1])
# plt.show()
