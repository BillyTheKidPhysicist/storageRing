import warnings
import pathos as pa
import numpy as np
'''
This is a wrapper that I made for the pathos multiprocessing library. I found myself writing alot of the same code
over and over when using pathos/multiprocessing, so I decided to make a class that offloads that work. Along the way,
I learned alot about multiprocessing that I include in this class. 

Features:

Two classes, parallel_Problem and parallel_Chunk_Problem. Both classes take the sampe input format, and return the same
output. Input is the function and an iterable of arguments(list/array etc) for the function to work on in parallel. 
In parallel_Problem a number of processes are started and the arguments are passed to the function in each process after
the previous argument has been solved. This is the standard way and is well suited for problems that take a while to solve
>10sec? parallel_Chunk_Problem on the other hand takes the list of arguments, breaks it into equal size (as possible)
lists of arguments, and passes a chunk to each process. This method avoids the overhead of passing the argument to each 
process over and over. This method is much faster with quicker functions, <~1 sec, because the main thread isn't a bottleneck
as it attempts to recieve answers from each process, and then give it the next argument. On a system with many processors
this becomes a very very serious bottleneck. On a 16 CPU computer this brough a 3x speedup.

pathos attempts to save time by reusing the pool once it has been created by caching it in the backend. This is a great feature
except that if it is reused hundreds and thousands of time system memory will be gobbled up. Thus the user can set a limit
to how many times to reuse the same pool before destroying and starting again. User can also choose wether to enforce this
at all. Calling a new ParaWell will not erase the previous pool either, so it is possible to create a new ParaWell over
and over, and eat up memory this way, so when a new instance of Parawell is called any pools are cleared. 


IT IS RECOMENDED TO NOT CREATE MULTIPLE INSTANCES OF ParaWell TO PRESERVE PERFORMANCE. 
It is also not recomendded to change number of workers.
'''


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
    def __init__(self,saveMemory=True,maxPoolUses=100):
        #saveMemory: calling pathos.ProcessPool over and over wihtout clearing after each calls will slowly eat up
        # memory, though it is far faster. I think this is a bug, but maybe it is a necessity with how it works.
        #regardless, clearing the pool after so many runs will maintain performance benefits and save memory. a hundred
        #calls to pool is a reasonable compromise
        #maxPoolUses: maximum number of times to reuse the cached pool before clearing it
        self.pool=None
        self.poolUse=0 #pool use starts at 0
        if saveMemory==False:
            self.maxPoolUses=99999999999 #set to a huge number
        else:
            self.maxPoolUses= maxPoolUses# number of times to use pool before cleraing it

        #if there are any existing pools caches, clear them to prevent memory leak
        if len(pa.multiprocessing._ProcessPool__STATE)!=0:
            pa.helpers.shutdown()
    def parallel_Problem(self,func,argsList,numWorkers=pa.helpers.cpu_count()):
        #func: the function that is being fed the arguments
        #argsList: a list of arguments
        #returns a list of tuples of the results paired with the arguments as (args,results)
        #workers: processes to work on the problem
        # returns a list of tuples of the results paired with the arguments as (args,results)
        def wrapper(args):
            result=func(args)
            return args,result

        self.manage_Pool(numWorkers)
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

        #func: the function that is being fed the arguments
        #argsList: a list of arguments to work on
        # numWorkers: processes to work on the problem
        #returns a list of tuples of the results paired with the arguments as (args,results)

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

        self.manage_Pool(numWorkers)
        jobs = []  # list of jobs. jobs are distributed across processors
        chunkResults = []  # list to hold results of particle tracing.
        for chunk in argChunkList:
            jobs.append(self.pool.apipe(wrapper,chunk))  # create job for each argument in arglist, ie for each particle
        for job in jobs:
            chunkResults.append(job.get())  # get the results. wait for the result in order given
        resultsList=[]
        for result in chunkResults:
            resultsList.extend(result)
        return resultsList

    def manage_Pool(self,numWorkers):
        #keep track of the number of time pool has been called, and make a new pool if the previous one gets changed
        #with a new selection of workers
        if self.pool is None:
            self.pool=pa.pools.ProcessPool(nodes=numWorkers)
            self.poolUse=0
        else:
            if self.pool.ncpus!=numWorkers: #if the requested worker is different the pool changes
                pa.helpers.shutdown()
                self.pool = pa.pools.ProcessPool(nodes=numWorkers)
                self.poolUse=0
            elif self.poolUse>=self.maxPoolUses:
                pa.helpers.shutdown()
                self.pool = pa.pools.ProcessPool(nodes=numWorkers)
                self.poolUse=0
        self.poolUse+=1
