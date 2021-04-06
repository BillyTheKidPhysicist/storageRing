#when using many libraries the basic multiprocessing module is used which is very slow often because of long picling
#time. This is a way to avoid pickling time. My custom ParaWell library uses the same trick basically, but is not
#easy to hook into other libraries necesarily

global func
func=None

global lattice
lattice = None



def solve(numParticles=3000):
    #this method solves a mode matching problem using scipy differential evolution in parallel
    from SwarmTracer import SwarmTracer
    import numpy as np
    import scipy.optimize as spo
    swarmTracer = SwarmTracer(lattice)

    def mode_Match(swarm, mode_Func):
        #iterate over a swarm and find the number of expected revolutions for each particle.
        #swarm: particle swarm being mode matched into lattice
        #mode_Func: The function that reprsents the coupling. Must return number of revolutions at a point in phase space
        #returns a few different operations on the results.
        temp = []
        for particle in swarm:
            if particle.clipped == False:
                q = particle.q[1:]
                p = particle.p
                Xi = np.append(q, p)
                temp.append(mode_Func(*Xi))
        temp = np.asarray(temp)
        meanVal=np.nansum(temp)/swarm.num_Particles()
        return meanVal
    def inject(args):
        #inject a swarm into the lattice
        #args: injection system paramters
        #return: some paramter of merit such as average revolutions are sum of revolutions for each particle
        Lo, Li, LOffset = args
        swarmNew = swarmTracer.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, labFrame=False,
                                                                   numParticles=numParticles)
        return mode_Match(swarmNew, func)

    temp = []
    def minimize(args):
        temp.append(0)
        val = inject(args) #mean number of revolutions
        return -val
    bounds = [(.15, .25), (.5, 1.5), (-.1, .1)]
    sol = spo.differential_evolution(minimize, bounds, maxiter=10, workers=1, polish=False, disp=True, popsize=10,mutation=0.1)
    print('optimal injector args:', sol.x)
    return sol.fun
