from ParaWell import ParaWell
import time
import numpy as np
from joblib import Parallel, delayed
import dill as pickle
class Data:
    def __init__(self):
        self.data=np.ones((3000,3000))*np.pi


def main():
    dataClass = Data()
    def test(x):
        return np.mean(dataClass.data)
    t=time.time()
    for i in range(32):
        test(1)

    print((time.time()-t)/32)

    pool=ParaWell()
    for i in range(10):
        t=time.time()
        args = np.arange(32)
        pool.parallel_Problem(test,args)
        print((time.time()-t)/32.0)
if __name__=='__main__':
    main()