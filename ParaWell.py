import os
import random
import time
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

When using a unix like system forking can be taken advanteged of. This is done by using a global function, globalFunc
to hold onto the supplised function. Then no pickling is required. There is a method to inelligently choose the right
solution depending on windows or unix like system detected. Pools need to be closed when using a global funcgion because
changes in the function may not be reflected because the first global function will always be used.


IT IS RECOMENDED TO NOT CREATE MULTIPLE INSTANCES OF ParaWell TO PRESERVE PERFORMANCE. 
It is also not recomendded to change number of workers.
'''

globalFuncDict={} #Multiple instance of parawell may be created, in which the previous global func could be overwritten.
#therefore maintain a dictionary of global functions tied to each parawell


class ParaWell:
    def __init__(self,saveMemory=True,maxPoolUses=100):
        #saveMemory: calling pathos.ProcessPool over and over wihtout clearing after each calls will slowly eat up
        # memory, though it is far faster. I think this is a bug, but maybe it is a necessity with how it works.
        #regardless, clearing the pool after so many runs will maintain performance benefits and save memory. a hundred
        #calls to pool is a reasonable compromise
        #maxPoolUses: maximum number of times to reuse the cached pool before clearing it
        global globalFuncDict
        self.id=len(globalFuncDict)
        globalFuncDict[self.id]=None #function is initially none valued
        self.pool=None
        self.poolUse=0 #pool use starts at 0
        self.test=0
        self.useGlobal=None
        self.useDelete=None #this is to decided wether to allow the garbage collector to call delete. There is a subtelty
        #because the garbage collector is called in each thread and would call __del__ for each subprcess which is not what
        #I want
        if os.name=='nt':
            self.useGlobal=False
        elif os.name=='posix':
            self.useGlobal=True
        else:
            warnings.warn('OS TYPE UNKNOWN. ONLY \'posix\' AND \'nt\' KNOWN. DEFAULTING TO SLOWER SAFE BEHAVIOUR')
        if saveMemory==False:
            self.maxPoolUses=99999999999 #set to a huge number
        else:
            self.maxPoolUses= maxPoolUses# number of times to use pool before cleraing it
    def __del__(self):
        #properly close the processpool so there aren't a bunch of processes hanging around.
        if self.pool is not None:
            self.pool.close()
        if pa.helpers is not None: #sometimes this appears to be garbage collected before this is reach
            pa.helpers.shutdown()
    def prepare_Wrapper(self, func, parallelType):
        #This method prepares the provided function by wrapping it in another function to accept and return data in the
        #correct format. Returned data is formatted as (provided arguments,results). THis method also takes advantage of
        #forking when possible by using global functions to dramatically reduce startup time of processes by avoiding
        #pickling time


        #TODO: THOUROUGHLY TEST AND TIME IN MY SITUATION
        #test when having many particles that take short/medium time, compare chunk to standard
        if self.useGlobal==True: #more investigation needed
            global globalFuncDict
            # if a new function is used then the old pool needs to be cleared else the old function
            # will still superseded the new one because of how global functions and multiprocessing work.
            if globalFuncDict[self.id] is not None:  #if the same parawell instance is being used again
                if func != globalFuncDict[self.id]: #it's a new function
                    self.reset()
                    globalFuncDict[self.id] = func
            else: #if its the first time the parawell instance is being accessed
                globalFuncDict[self.id] = func
            if parallelType== 'STANDARD':
                def wrapper(args):
                    result = globalFuncDict[self.id](args)
                    return args, result
            elif parallelType== 'CHUNK':
                def wrapper(chunk):
                    funcResults = []
                    for arg in chunk:
                        funcResults.append((arg, globalFuncDict[self.id](arg)))
                    return funcResults
            else:
                raise Exception('NO VALID NAME PROVIDED')
            return wrapper
        else:
            if parallelType== 'STANDARD':
                def wrapper(args):
                    result = func(args)
                    return args, result
            elif parallelType== 'CHUNK':
                def wrapper(chunk):
                    funcResults = []
                    for arg in chunk:
                        funcResults.append((arg, func(arg)))
                    return funcResults
            else:
                raise Exception('NO PROPER NAME PROVIDED')
            return wrapper
    def parallel_Problem(self,func,argsList,numWorkers=pa.helpers.cpu_count(),mutableFunction=True):
        """
        solve func in parallel over the provided arguments argsList.

        :param func: Function that is to be solved in parallel.
        :param argsList: List of arguments to be work on in parallel with func.
        :param numWorkers: Number of processes to work on the problem in parallel with. Defaults to the available number
        of processors (which may be double from hyper threading)
        :param mutableFunction: Some functions may be considered mutable which here I mean that the function may depend
        on other parameters which will change from call to call. If using a posix system global variables are used to
        improve performance but they will prevent functions from being able to reflect these changes.
        :return: list of tuples like (args,results) with length len(argslist).
        """
        wrapper=self.prepare_Wrapper(func,'STANDARD')
        self.manage_Pool(numWorkers)
        jobs = []  # list of jobs. jobs are distributed across processors
        results = []  # list to hold results of particle tracing.
        for arg in argsList:
            jobs.append(self.pool.apipe(wrapper, arg))  # create job for each argument in arglist, ie for each particle

        for job in jobs:
            results.append(job.get())  # get the results. wait for the result in order given

        if mutableFunction==True:
            self.pool.clear()
        return results
    def parallel_Chunk_Problem(self,func,argsList,numWorkers=pa.helpers.cpu_count(),mutableFunction=True):
        #solve a small problem in parallel. the logic is
        #1: split the arguments into evenly sized chunk for each work (some will have more or less)
        #2: create a wrapper function that parses these chunks out
        #3: use pathos to work on each chunk
        #4: stitch the results together

        #func: the function that is being fed the arguments
        #argsList: a list of arguments to work on
        # numWorkers: processes to work on the problem
        #returns a list of tuples of the arguments paired with the results as (args,results)
        random.shuffle(argsList)
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

        wrapper=self.prepare_Wrapper(func,'CHUNK')
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
        if mutableFunction==True:
            self.pool.clear()

        return resultsList

    def manage_Pool(self,numWorkers):
        #keep track of the number of time pool has been called, and make a new pool if the previous one gets changed
        #with a new selection of workers

        if self.pool is None:
            self.pool=pa.pools.ProcessPool(nodes=numWorkers)
            self.poolUse=0
        else:
            if self.pool.ncpus!=numWorkers: #if the requested worker is different the pool changes
                self.pool.close()
                pa.helpers.shutdown()
                self.pool = pa.pools.ProcessPool(nodes=numWorkers)
                self.poolUse=0
            elif self.poolUse>=self.maxPoolUses:
                print('Pool restarted to save memory, maximum pool uses limited to '+str(self.maxPoolUses))
                self.pool.close()
                pa.helpers.shutdown()
                self.pool = pa.pools.ProcessPool(nodes=numWorkers)
                self.poolUse=1
            self.poolUse+=1
    def reset(self):
        self.pool.close()
        pa.helpers.shutdown()
        self.pool=None