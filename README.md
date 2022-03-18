
# storageRing
Code and algorithms for design of storage ring and injector. Anyone who comes looking, I apologize for my previous selfs not strictly adhering to clean coding and decent documentation. This will be removed when these issues are remedied

## storageRingOptimization
This is constantly evolving. The main files are as follows:

- element.py: Contains the classes which represent elements in the storage ring such as drift regions, lenses, combiners, and benders. Linear combinations of these elements form a lattice through which particles trace.

- fastNumbaMethodsAndClass.py: A file of helper Numba classes attached to Elements in element.py, as well as helper functions. Classes use "@numba.jitclass". Jitclasses have no inheritance, and can't be pickled directly so some challenges must be overcome.

- particleTracerLattice.py: Wrangles the classes from elements.py into a functioning storage ring. There is a decent amount of geometry required to get everything lined up correctly. When using the combining magnet an implicit problem appears that must be solved numerically, and it may have multiple solutions.   
- particleTracer.py: Does the actual particle tracing using Velocity Verlet. Can use an entirely "Numbafied" loop, with jitclasses attached to element object in element.py, for rapid compiled language speed

- particleClass.py: Contains two classes, Particle and Swarm. Particle represents a single particle with properties like momentum and position, wether it has clipped an apeture, a history of it's trajectory through phase space if so needed and etc. Swarm represents a cloud of particle in phase space. It is more a covenience class primarily

- swarmTracer.py: A class that creates and manipulates Swarm objects. Has methods to create a random swarm, create a swarm from observed data, move a swarm to various positions, inject a swam into the lattice, trace a swarm and etc.

- HalbachLensClass.py:. Wrapper for magpylib class to generate fields for halbach hexapole lenses, and bender elements composed of halbach lenses

- injectionOptimizer.py: Optimize survival through injection system. This is achieved by optimizing atom survival through combiner and an atom lens which follows

- storageRingOptimizer.py: Optimizer storage ring. Different options for which knobs can be tuned.

- measureLatticeStability.py: Once an optimal solution is found, verify it is stable to realistic perturbations. Work in progress

- parallel_Gradient_Descent.py: Mostly a learning exercise. Jacobian is found in parallel.

- runOptimizer.py: Launch storage ring optimization

- constants.py: File of constants used globally.

- test: Folder containing unit tests. There is decent coverage


## phaseSpaceGenerationAndExtraction
Contains a class and files to primarily generate nice plots of tracing.
