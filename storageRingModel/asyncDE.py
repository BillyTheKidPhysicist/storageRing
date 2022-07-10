import numpy as np
import skopt
import random
import time
import multiprocess as multiprocessing
import scipy.optimize as spo
import scipy.interpolate as spi
from helperTools import low_discrepancy_sample



HUGE_INT = int(1e12)
real_number=(int,float)

class AsyncSolver:
    def __init__(self, workers):
        self.jobs = []
        if workers is None or isinstance(workers, int):
            if workers is None:
                numProcesses = multiprocessing.cpu_count()
            elif isinstance(workers, int):
                numProcesses = workers
            else:
                raise ValueError
            self.pool = multiprocessing.Pool(processes=numProcesses, maxtasksperchild=10)
        else:  # for using other apply_async type methods
            self.pool = workers

    def add_jobs(self, job):
        self.jobs.append(self.pool.apply_async(job))

    def get_job(self, wait=.05):
        # work thorugh the list of jobs
        assert len(self.jobs) > 0
        while True:
            time.sleep(wait)  # without this I would be ripping through the list
            job = self.jobs.pop(0)
            if job.ready() is True:  # try first entry
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
        scaled_coords = self.scale(self.coords)
        self.surrogate = spi.RBFInterpolator(scaled_coords, self.vals, smoothing=1e-3)

    def scale(self, X):
        assert len(X.shape) == 2
        x_scaled = (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        return x_scaled

    def descale(self, X):
        assert len(X.shape) == 2
        x_original = (self.bounds[:, 1] - self.bounds[:, 0]) * X + self.bounds[:, 0]
        return x_original

    def predict(self, nSample=10_000, w=None):
        self.train()
        if w is None:
            w = np.sqrt(np.random.rand())  # weight towards explotation
        assert 0 <= w <= 1
        scaled_coords = self.scale(self.coords)

        def surrogate(x):
            val = self.surrogate([x])[0]
            if w != 1:
                nearestDist = np.min(np.linalg.norm(scaled_coords - x, axis=1))
                cost = w * val + (1 - w) * (1 - nearestDist)
            else:  # no nearest distance being used
                cost = val
            return cost

        # surrogate =lambda x: self.surrogate([x]) #obnoxious format demands for rbf
        bounds = [(0, 1)] * len(self.bounds)
        sol = spo.differential_evolution(surrogate, bounds, tol=1e-6, atol=1e-6, maxiter=nSample, mutation=1.0)
        x_optimal = self.descale(np.asarray([sol.x]))[0]
        return x_optimal


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
        self.has_child = False
        self.firstGen = False
        self.dead = False
        self.parent_is_alive = None
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

    def grow(self, known_cost=None):
        assert self.grown is False
        if known_cost is not None:
            cost = known_cost
        else:
            if callable(self.DNA) == True:  # the DNA is a function to get the DNA. This is usually a surrogate method
                self.DNA = self.DNA()
            cost = self.func(self.DNA)
        assert isinstance(cost, real_number), str(repr(self.DNA))
        self.cost = cost
        self.fitness = -cost
        self.grown = True
        return self

    def absorb_clone(self, cloneMember):
        self.grown = cloneMember.grown
        self.fitness = cloneMember.fitness
        self.cost = cloneMember.cost
        self.DNA = cloneMember.DNA


class Population:
    def __init__(self):
        self.adult_members = []
        self.child_members = []
        self.member_history = []  # list of all members

    def add_adult(self, member: Member):
        assert member.grown is True
        self.adult_members.append(member)

    def add_child(self, member: Member):
        assert member.grown is False
        self.child_members.append(member)

    def remove_child(self, member: Member):
        assert member.grown is True
        assert (member.firstGen is True or member.parent is not None)
        self.child_members.remove(member)  # child grew up into adult. it will try and challenge adult

    def remove_adult(self, member: Member):
        assert member.grown is True and (member.has_child is True or member.parent is not None)  # lost to child
        self.adult_members.remove(member)

    def num_adults(self):
        return len(self.adult_members)

    def num_childs(self):
        return len(self.child_members)

    def get_viable_breeders(self):
        members = []
        for adult_member in self.adult_members:
            if adult_member.has_child or adult_member.firstGen:
                members.append(adult_member)
        return members

    def num_breedable_adults(self):
        return len(self.get_viable_breeders())

    def get_most_fit_member(self, viable_breeder=False):
        if viable_breeder:
            members = self.get_viable_breeders()
        else:
            members = self.adult_members
        fitness = [mem.fitness for mem in members]
        return members[np.argmax(fitness)]

    def get_and_update_original_member(self, possibleCloneMemb):
        # if possibleCloneMemb is a clone, replace all instances of the original
        for members in (self.child_members, self.adult_members):
            for i in range(len(members)):
                if members[i].ID == possibleCloneMemb.ID:  # could be true if not a clone, but go ahead
                    members[i].absorb_clone(possibleCloneMemb)  # member is now replaced
                    return members[i]
        raise Exception()  # loop should find at least a clone or an original


class AsyncDE:
    def __init__(self, func, num_members, bounds, max_evals=None, time_out_seconds=None, initial_vals=None,
                 surrogate_method_prob=0.0, disp=True, tol=None, workers=None, save_data=None):
        assert num_members >= 5
        for bound in bounds:
            assert len(bound) == 2 and bound[1] > bound[0]
        bounds = np.asarray(bounds).astype(float)
        if not isinstance(bounds, np.ndarray): bounds = np.asarray(bounds)
        assert save_data is None or isinstance(save_data, str)
        self.initial_vals = [] if initial_vals is None else initial_vals
        self.num_members = num_members
        self.num_evals = 0
        self.disp = disp
        self.save_data = save_data
        assert (time_out_seconds is not None) ^ (max_evals is not None) ^ (tol is not None)
        self.max_evals = max_evals if max_evals is not None else HUGE_INT
        self.time_out = time_out_seconds if time_out_seconds is not None else np.inf
        self.tol = tol if tol is not None else -np.inf
        self.mutation_factor = (.5, 1.0)
        self.cross_probability = .7
        self.func = func
        self.bounds = bounds
        self.async_manager = AsyncSolver(workers)
        self.population = Population()
        self.surrogate_method_prob = surrogate_method_prob  # try surrogate method instead of breeding this fraction of time
        self.current_ID = 0  # ID tracker to tag each member. If I don't do this they get all mixed up in parallel code
        # because they get cloned so the original member doens't get updated, only its clone does

    def generate_initial_coords(self, num):
        initial_coords = low_discrepancy_sample(self.bounds, num)
        return initial_coords

    def initialize_population(self):

        assert len(self.initial_vals) <=self.num_members
        for initial_val in self.initial_vals:
            assert len(initial_val) == 2
            DNA, cost = initial_val
            assert len(DNA) == len(self.bounds)
            if cost is None:
                new_child = Member(self.func, DNA, tag=self.current_ID)
                new_child.firstGen = True
                new_child.parent_is_alive = False
                self.population.add_child(new_child)
                self.async_manager.add_jobs(new_child.grow)
            else:  # already grown member
                new_adult = Member(self.func, DNA, tag=self.current_ID)
                new_adult.grow(known_cost=cost)
                new_adult.firstGen = True
                self.population.add_adult(new_adult)
        self.evolve() #initial population needs to be breed

        num_random = self.num_members - len(self.initial_vals)
        initial_coords = self.generate_initial_coords(num_random)
        for coord in initial_coords:
            new_child = Member(self.func, coord, tag=self.current_ID)
            self.current_ID += 1
            new_child.firstGen = True
            new_child.parent_is_alive = False
            self.population.add_child(new_child)
            self.async_manager.add_jobs(new_child.grow)

    def update_population(self):
        adult_member_clone = self.async_manager.get_job()  # pool returns a "clone"
        adult_member = self.population.get_and_update_original_member(adult_member_clone)
        self.population.add_adult(adult_member)
        self.population.remove_child(adult_member)
        self.population.member_history.append(adult_member)
        self.num_evals += 1

    def evolve(self):
        # attempt to defeat a parent and add a new member
        random.shuffle(self.population.adult_members)
        for adult_member in self.population.adult_members:
            if adult_member.firstGen and not adult_member.has_child and self.population.num_breedable_adults() >= 5:
                # childless first gen needs its first child to be bread
                adult_first_gen_member = adult_member
                new_child_member = self.breed_new_member(adult_first_gen_member)
                adult_first_gen_member.has_child = True  # first gen now has a child
                self.population.add_child(new_child_member)
                self.async_manager.add_jobs(new_child_member.grow)
            elif adult_member.parent_is_alive and self.population.num_breedable_adults() >= 5:
                # new adult offspring can now challenger parent
                adult_offspring_member = adult_member
                if self.offspring_wins(adult_offspring_member) is True:  # new adult offspring wins
                    self.population.remove_adult(adult_offspring_member.parent)  # parent is discarded
                    adult_offspring_member.parent_is_alive = False  # offspring's parent was defeated
                    adult_offspring_member.parent.dead = True  # parent is now dead
                    adult_offspring_member.parent = None  # it now has no parent
                    new_child = self.breed_new_member(adult_offspring_member)  # offspring produces a new child
                    adult_offspring_member.has_child = True
                    self.population.add_child(new_child)
                    self.async_manager.add_jobs(new_child.grow)
                else:  # offspring lost. Make new offspring
                    self.population.remove_adult(adult_offspring_member)  # offspring is discarded because it lost
                    adult_offspring_member.dead = True  # offspring is now dead
                    new_child = self.breed_new_member(
                        adult_offspring_member.parent)  # offspring parent produces a new child
                    self.population.add_child(new_child)
                    self.async_manager.add_jobs(new_child.grow)

    def dithered_mutation_factor(self):
        return np.random.random_sample() * (self.mutation_factor[1] - self.mutation_factor[0]) + self.mutation_factor[0]

    def offspring_wins(self, offspringMember):
        assert offspringMember.parent_is_alive == True and offspringMember.grown == True
        assert np.isnan(offspringMember.fitness) == False and np.isnan(offspringMember.parent.fitness) == False
        if offspringMember.fitness > offspringMember.parent.fitness:
            return True
        else:
            return False

    def create_mutant_DNA(self, targetMember: Member):
        viableBreeders = self.population.get_viable_breeders()
        bestMember = self.population.get_most_fit_member(viable_breeder=True)
        viableBreeders.remove(bestMember)
        if targetMember in viableBreeders:
            viableBreeders.remove(targetMember)
        assert len(viableBreeders) >= 2  # Must be at least 2 members for next step
        random.shuffle(viableBreeders)  # mix things up
        memB, memC = viableBreeders[:2]
        x1 = bestMember.DNA
        x2 = memB.DNA
        x3 = memC.DNA
        x4 = x1 + self.dithered_mutation_factor() * (x2 - x3)
        xNew = targetMember.DNA.copy()
        # new DNA may be out of bounds, so clip
        x4[x4 < self.bounds[:, 0]] = self.bounds[:, 0][x4 < self.bounds[:, 0]]
        x4[x4 > self.bounds[:, 1]] = self.bounds[:, 1][x4 > self.bounds[:, 1]]
        # mitosis! (sort of)
        crossOverIndices = np.random.rand(len(self.bounds)) < self.cross_probability
        xNew[crossOverIndices] = x4[crossOverIndices]  # replace the crossover genes
        return xNew

    def create_random_DNA(self):
        DNAList = []
        for bound in self.bounds:
            DNAList.append(np.random.rand() * (bound[1] - bound[0]) + bound[0])
        return np.asarray(DNAList)

    def create_predictor_model(self):
        coordsTrain = np.asarray([mem.DNA for mem in self.population.member_history])
        valsTrain = np.asarray([mem.cost for mem in self.population.member_history])
        if np.any(valsTrain == np.inf):
            raise Exception('You cant use surrogate model with infinite cost functions')
        if len(coordsTrain.shape) != 2:
            coordsTrain = coordsTrain[:, np.newaxis]
        predictor = RBF_Predictor(coordsTrain, valsTrain, self.bounds)
        return predictor.predict

    def breed_new_member(self, adult_member):
        # newAdultMember: The soon to be parent new adult
        assert adult_member.grown == True
        if np.random.rand() < self.surrogate_method_prob and self.num_evals > 5 * len(self.bounds):
            newChildDNA = self.create_predictor_model()
        else:
            newChildDNA = self.create_mutant_DNA(adult_member)
        new_child_member = Member(self.func, newChildDNA, tag=self.current_ID)
        self.current_ID += 1
        new_child_member.parent_is_alive = True
        new_child_member.parent = adult_member
        return new_child_member

    def found_poisson_pill(self) -> bool:
        try:
            open('poisonPill.txt')
            return True
        except:
            return False

    def get_population_variability(self):
        if self.population.num_adults() < 2:
            return None
        DNAArr = np.asarray([mem.DNA for mem in self.population.adult_members])
        variability = np.std(DNAArr, axis=0) / (self.bounds[:, 1] - self.bounds[:, 0])
        return variability

    def tolerance_met(self):
        cost_arr = np.asarray([mem.cost for mem in self.population.adult_members])
        if sum([cost == np.inf for cost in cost_arr]):  # any infinites prevent tolerance being met
            return False
        if len(cost_arr) >= self.num_members:
            mean_cost = np.mean(cost_arr)
            window = (mean_cost - self.tol, mean_cost + self.tol)
            num_satisfied = sum(cost_arr[cost_arr > window[0]] < window[1])
            if num_satisfied / self.num_members >= .9:
                return True
            else:
                return False

    def resave_progress(self):
        cost_arr = np.asarray([mem.cost for mem in self.population.member_history])
        DNA_Arr = np.asarray([mem.DNA for mem in self.population.member_history])
        history_arr = np.column_stack((DNA_Arr, cost_arr))
        try:
            np.savetxt(self.save_data, history_arr)
        except:
            raise IOError('error encountered with file saving!! proceeding')

    def solve(self):
        self.initialize_population()
        t0 = time.time()
        while True:
            self.update_population()
            if self.num_evals >= self.max_evals or self.time_out <= time.time() - t0 or self.tolerance_met() \
                    or self.found_poisson_pill():
                if self.save_data is not None:
                    self.resave_progress()
                print('finished with total evals: ', self.num_evals)
                break
            self.evolve()
            if self.num_evals % self.num_members == 0:
                if self.disp is True:
                    print('------ITERATIONS: ', self.num_evals)
                    print("POPULATION VARIABILITY: " + str(self.get_population_variability()))
                    print('BEST MEMBER BELOW')
                    print(self.population.get_most_fit_member())
                if self.save_data is not None:
                    self.resave_progress()
            assert self.population.num_adults() <= self.num_members
            assert self.population.num_childs() <= self.num_members
        self.async_manager.close()
        if self.save_data is not None:
            self.resave_progress()
        return self.population

def load_previous_population(num,file):
    data = np.loadtxt(file)
    X = data[:, :len(data[0]) - 1]
    vals = data[:, -1]

    index_sort = np.argsort(vals)[:num]

    population = [(DNA, cost) for DNA, cost in zip(X[index_sort], vals[index_sort])]
    return population

def solve_async(func, bounds, popsize, time_out_seconds=None, initial_vals=None, save_population=None,
                surrogate_method_prob=0.0,
                disp=True, max_evals=None, tol=None, workers=None, save_data=None,reload_population: str=None) -> Member:
    if reload_population is not None:
        assert initial_vals is None
        initial_vals=load_previous_population(popsize,reload_population)
    np.set_printoptions(precision=1000)
    solver = AsyncDE(func, popsize, bounds, time_out_seconds=time_out_seconds, initial_vals=initial_vals,
                     surrogate_method_prob=surrogate_method_prob, disp=disp, max_evals=max_evals, tol=tol, workers=workers,
                     save_data=save_data)
    pop = solver.solve()
    if save_population is not None:
        assert type(save_population) == str
        import pickle
        with open(save_population, 'wb') as file:
            pickle.dump(pop, file)
    return pop.get_most_fit_member()
