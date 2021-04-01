from ParaWell import ParaWell
import time
import numpy as np
from joblib import Parallel, delayed
import dill as pickle
import pathos as pa
class Data:
    def __init__(self):
        self.num=1.0
        self.data=np.ones((10,10))


# def main():
#     dataClass = Data()
#     def test(x):
#         dataClass.num=x
#         time.sleep(.05)
#         return np.mean(dataClass.data)*dataClass.num
#
#     argArr=np.arange(100)
#     helper=ParaWell()
#     t=time.time()
#     #res=helper.parallel_Problem(test,argArr)
#     res = helper.parallel_Problem(test, argArr)
#     print((time.time()-t)/10.0)
#
#     #
#     # argArr=np.arange(10)
#     # helper=ParaWell()
#     # t=time.time()
#     # #res=helper.parallel_Problem(test,argArr)
#     # res = helper.parallel_Problem(test, argArr)
#     # print(res)
#     # print((time.time()-t)/10.0)
#
# if __name__=='__main__':
#     main()
def meep():
    dataClass = Data()
    def test(x):
        dataClass.num=x
        time.sleep(.05)
        return np.mean(dataClass.data)*dataClass.num
    argArr=np.arange(100)
    helper=ParaWell()
    t=time.time()
    #res=helper.parallel_Problem(test,argArr)
    res = helper.parallel_Chunk_Problem(test, argArr)
    print((time.time()-t)/10.0)
    time.sleep(5)
meep()
meep()