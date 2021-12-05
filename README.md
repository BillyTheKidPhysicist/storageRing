
# storageRing
Created 9/20/2019 by Billy.
Code and algorithms for design of storage ring and injector. There are two folders, storageRingOptimization and injectionSystemOptimization. storageRingOptimization deals with the analysis and optimization of the storage ring, including injection components. injectionSystemOptimization deals with the analysis and optimization of the injection system, but would not be used in ringSystem, though the results from the analysis are.

## storageRingOptimization
This is constantly evolving.
The main files are as follows:
- element.py: Contains the classes which represent elements in the storage ring such as drift regions, lenses, combiners, and benders. There are two main classes, ideal and sim. Ideal represents the ideal hard edge version of an element. Sim is for using the results of numerical modeling, particularily form COMSOL. Each elements has methods for transforming coordinates between the element frame and lab frame, getting the force and magnetic field, determining if a point lies within the element's vacuum chamber, etc. Cython and numba is used to accelerate computation
- particleTracerLattice.py: Wrangles the classes from elements.py into a functioning storage ring. There is a decent amount of geometry required to get everything lined up correctly. When using the combining magnet an implicit problem appears that must be solved numerically, and it may have multiple solutions.   
- particleTracer.py: Does the actual particle tracing using Velocity Verlet. I've tried RK4 and implicit trapezoid, but neither performed as well. Particle tracing is done in the lab frame. A single particle is traced at a time.
- particleClass.py: Contains two classes, Particle and Swarm. Particle represents a single particle with properties like momentum and position, wether it has clipped an apeture, a history of it's trajectory through phase space if so needed and etc. Swarm represents a cloud of particle in phase space. It is more a covenience class that saves me from typing    the same thing over and over and improves readability.
- ParaWell.py: My custom class that takes advantage of forking on posix systems to do fast parallelization, and with minimal typing. Needs to be included with a setup so it can be install system wide, and moved out of this rep.
- generatePhaseSpaceCloud.py: Model out observed atomic focus to generate a text file of particle's initial condition. Uses the very low discrepancy poisson disc method to sample from.
- swarmTracer.py: A class that creates and manipulates Swarm objects. Has methods to create a random swarm, create a swarm from observed data, move a swarm to various positions, inject a swam into the lattice, trace a swarm and etc.
- OptimizerClass.py: A class to do the actual optimization of the lattice. I use scikit-optimize gaussian proccess. However I use the ask/tell interface so I can have more control because there are many configurations of the storage ring that are unstable, which is easy to evaluate and I don't want to count towards my evaluations. The basic logic is at a stable configuration do a monte carlo integartion with a large cloud of particles with nearest neighbor (kdtree) interpolate the results. Then use differential evolution to find the optimal injection system parameters that best mode match into the lattice by projecting the swarm at the end of the injection system onto the lattice's phase space with the nearest neighbor function. After this is optimized, move on. I have found this to reliabely give the same results
- Various notebooks to test and use the above classes, mostly OptimizerClass, on different configurations.

## phaseSpaceGenerationAndExtraction
Contains a class and files to analyze results of tracing.
