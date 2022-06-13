import numpy as np
import skopt
import random
import time
import numbers
import multiprocess as multiprocessing
import scipy.optimize as spo
import scipy.interpolate as spi


def low_Discrepancy_Sample(bounds, num: int) -> np.ndarray:
    """
    Make a low discrepancy sample (ie well spread)

    :param bounds: sequence of lower and upper bounds of the sampling
    :param num: number of samples. powers of two are best for nice sobol properties
    :return:
    """

    from scipy.stats.qmc import Sobol
    bounds = np.array(bounds).astype(float)
    samples = np.array(Sobol(len(bounds), scramble=True).random(num))
    scaleFactors = bounds[:, 1] - bounds[:, 0]
    offsets = bounds[:, 0]
    for i, sample in enumerate(samples):
        samples[i] = sample * scaleFactors + offsets
    return samples


HUGE_INT = int(1e12)


class asyncSolver:
    def __init__(self, workers):
        self.jobs = []
        if workers == None or isinstance(workers, int):
            if workers == None:
                numProcesses = multiprocessing.cpu_count()
            elif isinstance(workers, int):
                numProcesses = workers
            else:
                raise ValueError
            self.pool = multiprocessing.Pool(processes=numProcesses, maxtasksperchild=10)
        else:  # for using other apply_async type methods
            self.pool = workers

    def add_Jobs(self, job):
        self.jobs.append(self.pool.apply_async(job))

    def get_Job(self, wait=.05):
        # work thorugh the list of jobs
        assert len(self.jobs) > 0
        while True:
            time.sleep(wait)  # without this I would be ripping through the list
            job = self.jobs.pop(0)
            if job.ready() == True:  # try first entry
                return job.get()
            else:
                self.jobs.append(job)

    def close(self):
        self.pool.terminate()
        self.pool.close()


class RBF_Predictor:
    def __init__(self, coords, vals, bounds):
        assert len(coords.shape) == 2 and len(bounds) == len(coords[0]) and len(vals.shape) == 1
        self.coords = coords
        self.vals = vals
        self.bounds = np.asarray(bounds)
        self.smoothing = 1e-3
        self.surrogate = None

    def train(self):
        scaledCoords = self.scale(self.coords)
        self.surrogate = spi.RBFInterpolator(scaledCoords, self.vals, smoothing=1e-3)

    def scale(self, X):
        assert len(X.shape) == 2
        xScaled = (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        return xScaled

    def descale(self, X):
        assert len(X.shape) == 2
        xOriginal = (self.bounds[:, 1] - self.bounds[:, 0]) * X + self.bounds[:, 0]
        return xOriginal

    def predict(self, nSample=10_000, w=None):
        self.train()
        if w is None:
            w = np.sqrt(np.random.rand())  # weight towards explotation
        assert 0 <= w <= 1
        scaledCoords = self.scale(self.coords)

        def surrogate(x):
            val = self.surrogate([x])[0]
            if w != 1:
                nearestDist = np.min(np.linalg.norm(scaledCoords - x, axis=1))
                cost = w * val + (1 - w) * (1 - nearestDist)
            else:  # no nearest distance being used
                cost = val
            return cost

        # surrogate =lambda x: self.surrogate([x]) #obnoxious format demands for rbf
        bounds = [(0, 1)] * len(self.bounds)
        sol = spo.differential_evolution(surrogate, bounds, tol=1e-6, atol=1e-6, maxiter=nSample, mutation=1.0)
        xOptimal = self.descale(np.asarray([sol.x]))[0]
        return xOptimal


class Member:
    def __init__(self, func, DNA, tag=None):
        if tag is None:
            ID = None
        else:
            ID = tag
        assert callable(DNA) or len(DNA) > 0
        if not isinstance(DNA, np.ndarray) and not callable(DNA):
            DNA = np.asarray(DNA)
        self.func = func
        self.DNA = DNA
        self.grown = False
        self.hasChild = False
        self.firstGen = False
        self.dead = False
        self.parentIsAlive = None
        self.fitness = None
        self.cost = None
        self.parent = None
        self.ID = ID  # each member has an ID to keep track of its clones when it's sent off to be solved in
        # parallel so it can be replaced by its new clone

    def __str__(self):
        string = "---population member---- \n"
        string += 'DNA: ' + repr(self.DNA) + '\n'
        string += 'cost: ' + repr(self.cost)
        return string

    def grow(self, knownCost=None):
        assert self.grown == False
        if knownCost is not None:
            cost = knownCost
        else:
            if callable(self.DNA) == True:  # the DNA is a function to get the DNA. This is usually a surrogate method
                self.DNA = self.DNA()
            cost = self.func(self.DNA)
        assert isinstance(cost, numbers.Number), str(repr(self.DNA))
        self.cost = cost
        self.fitness = -cost
        self.grown = True
        return self

    def absorb_Clone(self, cloneMember):
        self.grown = cloneMember.grown
        self.fitness = cloneMember.fitness
        self.cost = cloneMember.cost
        self.DNA = cloneMember.DNA


class Population:
    def __init__(self):
        self.adultMembers = []
        self.childMembers = []
        self.memberHistory = []  # list of all members

    def add_Adult(self, member: Member):
        assert member.grown == True
        self.adultMembers.append(member)

    def add_child(self, member: Member):
        assert member.grown == False
        self.childMembers.append(member)

    def remove_Child(self, member: Member):
        assert member.grown == True
        assert (member.firstGen == True or member.parent is not None)
        self.childMembers.remove(member)  # child grew up into adult. it will try and challenge adult

    def remove_adult(self, member: Member):
        assert member.grown == True and (member.hasChild == True or member.parent is not None)  # lost to child
        self.adultMembers.remove(member)

    def num_Adults(self):
        return len(self.adultMembers)

    def num_Childs(self):
        return len(self.childMembers)

    def get_Viable_Breeders(self):
        members = []
        for adultMember in self.adultMembers:
            if adultMember.hasChild == True or adultMember.firstGen == True:
                members.append(adultMember)
        return members

    def num_Breedable_Adults(self):
        return len(self.get_Viable_Breeders())

    def get_Most_Fit_Member(self, viableBreeder=False):
        if viableBreeder == True:
            memberList = self.get_Viable_Breeders()
        else:
            memberList = self.adultMembers
        fitness = [mem.fitness for mem in memberList]
        return memberList[np.argmax(fitness)]

    def get_And_Update_Original_Member(self, possibleCloneMemb):
        # if possibleCloneMemb is a clone, replace all instances of the original
        for memberList in (self.childMembers, self.adultMembers):
            for i in range(len(memberList)):
                if memberList[i].ID == possibleCloneMemb.ID:  # could be true if not a clone, but go ahead
                    memberList[i].absorb_Clone(possibleCloneMemb)  # member is now replaced
                    return memberList[i]
        raise Exception()  # loop should find at least a clone or an original


class asyncDE:
    def __init__(self, func, numMembers, bounds, maxEvals=None, timeOut_Seconds=None, initialVals=None,
                 surrogateMethodProb=0.0, disp=True, tol=None, workers=None, saveData=None):
        assert numMembers >= 5
        for bound in bounds:
            assert len(bound) == 2 and bound[1] > bound[0]
        bounds = np.asarray(bounds).astype(float)
        if not isinstance(bounds, np.ndarray): bounds = np.asarray(bounds)
        assert saveData is None or isinstance(saveData, str)
        self.initialVals = [] if initialVals is None else initialVals
        self.numMembers = numMembers
        self.numEvals = 0
        self.disp = disp
        self.saveData = saveData
        assert (timeOut_Seconds is not None) ^ (maxEvals is not None) ^ (tol is not None)
        self.maxEvals = maxEvals if maxEvals is not None else HUGE_INT
        self.timeOut = timeOut_Seconds if timeOut_Seconds is not None else np.inf
        self.tol = tol if tol is not None else -np.inf
        self.mutationFactor = (.5, 1.0)
        self.crossProbability = .7
        self.func = func
        self.bounds = bounds
        self.asyncManager = asyncSolver(workers)
        self.population = Population()
        self.surrogateMethodProb = surrogateMethodProb  # try surrogate method instead of breeding this fraction of time
        self.currentID = 0  # ID tracker to tag each member. If I don't do this they get all mixed up in parallel code
        # because they get cloned so the original member doens't get updated, only its clone does

    def generate_Initial_Coords(self, num):
        initialCoords = low_Discrepancy_Sample(self.bounds, num)
        return initialCoords

    def initialize_Population(self):

        assert len(self.initialVals) <=self.numMembers
        for initialVal in self.initialVals:
            assert len(initialVal) == 2
            DNA, cost = initialVal
            assert len(DNA) == len(self.bounds)
            if cost is None:
                newChild = Member(self.func, DNA, tag=self.currentID)
                newChild.firstGen = True
                newChild.parentIsAlive = False
                self.population.add_child(newChild)
                self.asyncManager.add_Jobs(newChild.grow)
            else:  # already grown member
                newAdult = Member(self.func, DNA, tag=self.currentID)
                newAdult.grow(knownCost=cost)
                newAdult.firstGen = True
                self.population.add_Adult(newAdult)
        self.evolve() #initial population needs to be breed

        numRandom = self.numMembers - len(self.initialVals)
        initialCoords = self.generate_Initial_Coords(numRandom)
        for coord in initialCoords:
            newChild = Member(self.func, coord, tag=self.currentID)
            self.currentID += 1
            newChild.firstGen = True
            newChild.parentIsAlive = False
            self.population.add_child(newChild)
            self.asyncManager.add_Jobs(newChild.grow)

    def update_Population(self):
        adultMember_Clone = self.asyncManager.get_Job()  # pool returns a "clone"
        adultMember = self.population.get_And_Update_Original_Member(adultMember_Clone)
        self.population.add_Adult(adultMember)
        self.population.remove_Child(adultMember)
        self.population.memberHistory.append(adultMember)
        self.numEvals += 1

    def evolve(self):
        # attempt to defeat a parent and add a new member
        random.shuffle(self.population.adultMembers)
        for adultMember in self.population.adultMembers:
            if adultMember.firstGen == True and adultMember.hasChild == False and self.population.num_Breedable_Adults() >= 5:
                # childless first gen needs its first child to be bread
                adultFirstGenMember = adultMember
                newChildMember = self.breed_New_Member(adultFirstGenMember)
                adultFirstGenMember.hasChild = True  # first gen now has a child
                self.population.add_child(newChildMember)
                self.asyncManager.add_Jobs(newChildMember.grow)
            elif adultMember.parentIsAlive == True and self.population.num_Breedable_Adults() >= 5:
                # new adult offspring can now challenger parent
                adultOffspringMember = adultMember
                if self.offspring_Wins(adultOffspringMember) == True:  # new adult offspring wins
                    self.population.remove_adult(adultOffspringMember.parent)  # parent is discarded
                    adultOffspringMember.parentIsAlive = False  # offspring's parent was defeated
                    adultOffspringMember.parent.dead = True  # parent is now dead
                    adultOffspringMember.parent = None  # it now has no parent
                    newChild = self.breed_New_Member(adultOffspringMember)  # offspring produces a new child
                    adultOffspringMember.hasChild = True
                    self.population.add_child(newChild)
                    self.asyncManager.add_Jobs(newChild.grow)
                else:  # offspring lost. Make new offspring
                    self.population.remove_adult(adultOffspringMember)  # offspring is discarded because it lost
                    adultOffspringMember.dead = True  # offspring is now dead
                    newChild = self.breed_New_Member(
                        adultOffspringMember.parent)  # offspring parent produces a new child
                    self.population.add_child(newChild)
                    self.asyncManager.add_Jobs(newChild.grow)

    def dithered_Mutation_Factor(self):
        return np.random.random_sample() * (self.mutationFactor[1] - self.mutationFactor[0]) + self.mutationFactor[0]

    def offspring_Wins(self, offspringMember):
        assert offspringMember.parentIsAlive == True and offspringMember.grown == True
        assert np.isnan(offspringMember.fitness) == False and np.isnan(offspringMember.parent.fitness) == False
        if offspringMember.fitness > offspringMember.parent.fitness:
            return True
        else:
            return False

    def create_Mutant_DNA(self, targetMember: Member):
        viableBreeders = self.population.get_Viable_Breeders()
        bestMember = self.population.get_Most_Fit_Member(viableBreeder=True)
        viableBreeders.remove(bestMember)
        if targetMember in viableBreeders:
            viableBreeders.remove(targetMember)
        assert len(viableBreeders) >= 2  # Must be at least 2 members for next step
        random.shuffle(viableBreeders)  # mix things up
        memB, memC = viableBreeders[:2]
        x1 = bestMember.DNA
        x2 = memB.DNA
        x3 = memC.DNA
        x4 = x1 + self.dithered_Mutation_Factor() * (x2 - x3)
        xNew = targetMember.DNA.copy()
        # new DNA may be out of bounds, so clip
        x4[x4 < self.bounds[:, 0]] = self.bounds[:, 0][x4 < self.bounds[:, 0]]
        x4[x4 > self.bounds[:, 1]] = self.bounds[:, 1][x4 > self.bounds[:, 1]]
        # mitosis! (sort of)
        crossOverIndices = np.random.rand(len(self.bounds)) < self.crossProbability
        xNew[crossOverIndices] = x4[crossOverIndices]  # replace the crossover genes
        return xNew

    def create_Random_DNA(self):
        DNAList = []
        for bound in self.bounds:
            DNAList.append(np.random.rand() * (bound[1] - bound[0]) + bound[0])
        return np.asarray(DNAList)

    def create_Predictor_Model(self):
        coordsTrain = np.asarray([mem.DNA for mem in self.population.memberHistory])
        valsTrain = np.asarray([mem.cost for mem in self.population.memberHistory])
        if np.any(valsTrain == np.inf):
            raise Exception('You cant use surrogate model with infinite cost functions')
        if len(coordsTrain.shape) != 2:
            coordsTrain = coordsTrain[:, np.newaxis]
        predictor = RBF_Predictor(coordsTrain, valsTrain, self.bounds)
        return predictor.predict

    def breed_New_Member(self, adultMember):
        # newAdultMember: The soon to be parent new adult
        assert adultMember.grown == True
        if np.random.rand() < self.surrogateMethodProb and self.numEvals > 5 * len(self.bounds):
            newChildDNA = self.create_Predictor_Model()
        else:
            newChildDNA = self.create_Mutant_DNA(adultMember)
        newChildMember = Member(self.func, newChildDNA, tag=self.currentID)
        self.currentID += 1
        newChildMember.parentIsAlive = True
        newChildMember.parent = adultMember
        return newChildMember

    def found_Poisson_Pill(self) -> bool:
        try:
            open('poisonPill.txt')
            return True
        except:
            return False

    def get_Population_Variability(self):
        if self.population.num_Adults() < 2:
            return None
        DNAArr = np.asarray([mem.DNA for mem in self.population.adultMembers])
        variability = np.std(DNAArr, axis=0) / (self.bounds[:, 1] - self.bounds[:, 0])
        return variability

    def tolerance_Met(self):
        costArr = np.asarray([mem.cost for mem in self.population.adultMembers])
        if sum([cost == np.inf for cost in costArr]):  # any infinites prevent tolerance being met
            return False
        if len(costArr) >= self.numMembers:
            meanCost = np.mean(costArr)
            window = (meanCost - self.tol, meanCost + self.tol)
            numSatisfied = sum(costArr[costArr > window[0]] < window[1])
            if numSatisfied / self.numMembers >= .9:
                return True
            else:
                return False

    def resave_Progress(self):
        costArr = np.asarray([mem.cost for mem in self.population.memberHistory])
        DNA_Arr = np.asarray([mem.DNA for mem in self.population.memberHistory])
        historyArr = np.column_stack((DNA_Arr, costArr))
        try:
            np.savetxt(self.saveData, historyArr)
        except:
            print('error encountered with file saving!! proceeding')

    def solve(self):
        self.initialize_Population()
        t0 = time.time()
        while True:
            self.update_Population()
            if self.numEvals >= self.maxEvals or self.timeOut <= time.time() - t0 or self.tolerance_Met() \
                    or self.found_Poisson_Pill():
                if self.saveData is not None:
                    self.resave_Progress()
                print('finished with total evals: ', self.numEvals)
                break
            self.evolve()
            if self.numEvals % self.numMembers == 0:
                if self.disp == True:
                    print('------ITERATIONS: ', self.numEvals)
                    print("POPULATION VARIABILITY: " + str(self.get_Population_Variability()))
                    print('BEST MEMBER BELOW')
                    print(self.population.get_Most_Fit_Member())
                if self.saveData is not None:
                    self.resave_Progress()
            assert self.population.num_Adults() <= self.numMembers
            assert self.population.num_Childs() <= self.numMembers
        self.asyncManager.close()
        if self.saveData is not None:
            self.resave_Progress()
        return self.population

def load_Previous_Population(num,file):
    data = np.loadtxt(file)
    X = data[:, :len(data[0]) - 1]
    vals = data[:, -1]

    indexSort = np.argsort(vals)[:num]

    population = [(DNA, cost) for DNA, cost in zip(X[indexSort], vals[indexSort])]
    return population

def solve_Async(func, bounds, popsize, timeOut_Seconds=None, initialVals=None, savePopulation=None,
                surrogateMethodProb=0.0,
                disp=True, maxEvals=None, tol=None, workers=None, saveData=None,reloadPopulation: str=None) -> Member:
    if reloadPopulation is not None:
        assert initialVals is None
        initialVals=load_Previous_Population(popsize,reloadPopulation)
    np.set_printoptions(precision=1000)
    solver = asyncDE(func, popsize, bounds, timeOut_Seconds=timeOut_Seconds, initialVals=initialVals,
                     surrogateMethodProb=surrogateMethodProb, disp=disp, maxEvals=maxEvals, tol=tol, workers=workers,
                     saveData=saveData)
    pop = solver.solve()
    if savePopulation is not None:
        assert type(savePopulation) == str
        import pickle
        with open(savePopulation, 'wb') as file:
            pickle.dump(pop, file)
    return pop.get_Most_Fit_Member()
