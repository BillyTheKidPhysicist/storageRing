import pathos as pa
import numpy as np
import pathos.multiprocessing
import time
from multiprocessing import Process
from profilehooks import profile
import sys
import multiprocess as mp


def test_Function(runs):
    # function to show the performance benefits of using ParaWell. Simple function that tracks a particle
    # in a harmonic potential in 2D with a mass of 1

    for i in range(int(runs)):
        loops = 10000
        K = 1e6
        h = 1e-5
        q = np.asarray([1.0, 1.5])
        p = np.asarray([0.0, 0.0])
        force = lambda x: -K * x
        qList = []
        pList = []
        for i in range(loops):
            F = force(q)
            q_n = q + p * h + .5 * F * h ** 2
            F_n = force(q_n)
            p_n = p + .5 * (F + F_n) * h
            p = p_n
            q = q_n
            pList.append(p)
            qList.append(q)
    return runs


class ParaWell:
    def __init__(self):
        self.pool=None
        #if len(pathos.multiprocessing._ProcessPool__STATE)!=0:
        #    print(pa.multiprocessing._ProcessPool__STATE['pool'])
    def parallel_Problem(self,func,argsList,numWorkers=pa.helpers.cpu_count()):
        #func: the function that is being fed the arguments
        #argsList: a list of arguments
        #returns a list of tuples of the results paired with the arguments as (args,results)

        def wrapper(args):
            result=func(args)
            return args,result

        if self.pool is None:
            self.pool = pa.pools.ProcessPool(nodes=numWorkers)
        else:
            if self.pool.ncpus != numWorkers:
                self.pool = pa.pools.ProcessPool(nodes=numWorkers)
        jobs = []  # list of jobs. jobs are distributed across processors
        results = []  # list to hold results of particle tracing.
        for arg in argsList:
            jobs.append(self.pool.apipe(wrapper, arg))  # create job for each argument in arglist, ie for each particle
        for job in jobs:
            results.append(job.get())  # get the results. wait for the result in order given


        return results
    def parallel_Chunk_Problem(self,func,argsList,numWorkers=pa.helpers.cpu_count()):
        #solve a small problem in parallel. the logic is
        #1: split the arguments into evenly sized chunk for each work (some will have more or less)
        #2: create a wrapper function that parses these chunks out
        #3: use pathos to work on each chunk
        #4: stitch the results together
        if numWorkers>len(argsList):
            raise Exception('MORE WORKERS THAN ARGUMENTS IN ARGLIST')
        argChunkList=[]
        for i in range(numWorkers):
            argChunkList.append([])
        j=0 #to keep track of which worker the next argument should be sent to
        for i in range(len(argsList)):
            argChunkList[j].append(argsList[i])
            if j==numWorkers-1: #reset j when we've distributed to the last worker to start over at the beginning
                j=0
            else:
                j+=1
        def wrapper(chunk):
            funcResults=[]
            for arg in chunk:
                funcResults.append((arg,func(arg)))
            return funcResults


        if self.pool is None:
            self.pool=pa.pools.ProcessPool(nodes=numWorkers)
        else:
            if self.pool.ncpus != numWorkers:
                self.pool = pa.pools.ProcessPool(nodes=numWorkers)

        jobs = []  # list of jobs. jobs are distributed across processors
        chunkResults = []  # list to hold results of particle tracing.
        for chunk in argChunkList:
            jobs.append(self.pool.apipe(wrapper,chunk))  # create job for each argument in arglist, ie for each particle
        for job in jobs:
            chunkResults.append(job.get())  # get the results. wait for the result in order given
        results=[]
        for result in chunkResults:
            results.extend(result)
        return results


