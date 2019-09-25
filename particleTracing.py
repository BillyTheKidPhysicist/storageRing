import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
import sys
import time


class ParticleTrace:
    def __init__(self,LLS,v0,T):
        #LLS: LinearLatticeSolverObject. has information on lattice and lengths, forces,etc
        #nozzleDiam: diameter in input nozzle, known to be about 400E-6 um
        #v0: nominal velocity of atoms
        #T: temperature of atoms
        self.m = 1.1650341e-26
        self.u0 = 9.274009994E-24
        self.u0Sim = self.u0 / self.m
        self.lattice=LLS.lattice #list of lattive elements
        self.v0=v0
        self.T=T
        self.lengthArray = np.asarray([LLS.totalLengthArrayFunc()]) #array of element lengths
        self.timeSteps=None
        self.numParticles=None
        self.kArr,self.apetureArr=self.compute_Apeture_And_K_Array() #array of k values and apeture sizes
        self.thetaMax=self.compute_ThetaMax()
        self.ForceHelper=None #an array to hold the force values at a given time step. Cheaper than recreating every timestep
        self.dt=None #time step soze
        self.simulationTime=None #total time in the particle tracing
        self.firstLensIndex=None #index of first lens in the lattice

    def compute_ThetaMax(self):
        #computes maximum input angle for first lens from ideal picture.
        i=0
        Ld=0 #to keep track of drift length before first lens, should be just one element
        for el in self.lattice: #find index of first lens
            if el.type == 'LENS':
                self.firstLensIndex=i
                break
            else:
                Ld+=el.Length
            i+=1
        if self.firstLensIndex==None:
            print('NO LENS WAS FOUND IN THE LATTICE!!!')
            sys.exit()

        lens=self.lattice[self.firstLensIndex]
        return lens.rp / (Ld * np.sqrt(1 + self.m * (lens.rp * self.v0) ** 2 / (2 * self.u0 * lens.Bp * Ld ** 2)))
    def compute_Apeture_And_K_Array(self):
        #return 2 arrays. first array is of the values of k (spring constant form, F=u0*grad(B0)=u0*grad(B0*r^2/rp^2))
        #=2*u0*B0/rp^2 * r=k*r). Second is of aperture values of each element. As of now this is just rp for lenses.
        #in the future I would like to add an actual apeture
        apetureList = []  # list of aperture sizes
        kList = []  # List that hold k values for each total length array entry
        for el in self.lattice:
            if el.type == 'DRIFT':
                k = 0
                apeture = np.inf #drift has infinite apeture
            else:
                k = 2 * self.u0Sim * el.Bp / el.rp ** 2
                apeture = el.rp #lens has finite aperture defined by wall size
            kList.append(k)
            apetureList.append(apeture)
        # now add an element at the end for drift. There cannot be a drift in the final element from LLS because it makes
        #stuff more difficult
        kList.append(0) #no k for drift
        apetureList.append(np.inf) #infinite apeture for drift
        return np.asarray(kList),np.asarray(apetureList)

    def initialize_q_And_v(self):
        #initialize reasonable values for q and v.
        # q: array of size(3*number of particles,timesteps). It contains position data for all particles at single timestep
        #in each row. Number of rows is number of timesteps. Goes as x1,y1,s1,x2,y2,s2...
        # v: same as above except velocity
        #ideally represent a small nozzle that has expanded for about 5 cm, then seeded with particles that come to thermal
        #equlibrium. As of now it's represented by a point, but this isn't so bad cause the thermal distribution makes
        # a massive impact versus nozzle size. I also trim out particles that will be clipped by the first magnet


        #make initial empty array to hold position and speed
        q = np.zeros((self.timeSteps, 3 * self.numParticles))
        v = q.copy()




        #Here i find particles to include in the trace. Any that exceed the max transverese energy
        #of the first magnet are excluded. I do this by generating a bunch of particles then removing the ones that
        #would clip, and adding the remainder to the above arrays


        particles=0 #tracks number of particles that are added

        numSample=self.numParticles*10 #use lots of particles in each sample
        while particles<self.numParticles:
            qTemp=np.zeros(3 *numSample) #temporary array of first time step
            vTemp=qTemp.copy()
            vy = np.linspace(-1, 1, num=numSample) * self.v0*.25 # large spread of sample velocities to filter. an
                                                                #angle of .25 would be crazy, so big overshoot
            vx = vy.copy()
            vz = np.ones(numSample) * self.v0
            np.random.shuffle(vx) #shuffle the transverse velocities so you don't just have a plane 45 degrees between
                                    #x and y
            np.random.shuffle(vy)
            vTemp[0::3] = vz
            vTemp[1::3] = vy
            vTemp[2::3] = vx
            dtArr = .05 / vz  # array of time steps required to bring particles up to seeding distance
            qTemp[0::3]+=vTemp[0::3]*dtArr  #add the required distance
            qTemp[1::3] += vTemp[1::3] * dtArr
            qTemp[2::3] += vTemp[2::3] * dtArr

            sigma = np.sqrt(1.38E-23 * self.T / 1.16E-26) #add temperatuer spread
            vTemp += np.random.normal(scale=sigma, size=(vTemp.shape[0]))


            #rArr = np.sqrt(QxArr ** 2 + QyArr ** 2)

            # now step up to the magnet to find parameters at the input.
            dtArr = (self.lengthArray[0][self.firstLensIndex-1]-qTemp[0::3]) / vTemp[0::3]
            qInput=qTemp.copy() #array to hold values at input of the magnet
            qInput[0::3]+=vTemp[0::3]*dtArr #step up to the magnet
            qInput[ 1::3] += vTemp[1::3] * dtArr
            qInput[ 2::3] += vTemp[2::3] * dtArr



            rArr = np.sqrt(qInput[1::3] ** 2 + qInput[2::3] ** 2)
            vArr = np.sqrt(vTemp[ 1::3] ** 2 + vTemp[ 2::3] ** 2)
            lens=self.lattice[self.firstLensIndex] #get the first lens

            Et=.5*vArr**2+self.u0Sim*lens.Bp*rArr**2/lens.rp**2 #transverse input energy of each particles
            Emax=self.u0Sim*lens.Bp #max transverse energy is field strength at pole
            removeArr=Et>Emax #which particle to remove
            newParticles=np.sum(~removeArr) #the number of particles to remove is sum of falses because true is removed.

            numExclude=0
            if particles+newParticles>self.numParticles: #if it's overshot
                numExclude=particles+newParticles-self.numParticles
            t1,t2=self.remove_Particles(qTemp,vTemp,removeArr) #the particles that won't clip
            q[0,3*particles:3*(newParticles+particles-numExclude)]=t1[:3 * (newParticles-numExclude)] #now insert them
            v[0,3*particles:3*(newParticles+particles-numExclude)] = t2[:3 * (newParticles-numExclude)]
            particles += newParticles
            #now find transverse energy of each particle
        return q,v
    def Force(self,q):
        #cute trick here.
        temp = -self.kArr[np.sum((q[0::3] > self.lengthArray.T), axis=0)] #find which optic the particle is inside. Do this
                #comparing qz to the total length of each optic. Each 'time' it's less than the total, there is a False.
                #sum it up then the value is the optic it's currently in!
        self.ForceHelper[1::3] = temp #cheaper method to track the force than restating an array each time
        self.ForceHelper[2::3] = temp
        return self.ForceHelper * q
    def trace_Particles(self):
        #do the time stepping. Uses velocity verlet
        q,v=self.initialize_q_And_v()
        accelPrev = np.zeros(3 * self.numParticles)
        for i in range(self.timeSteps - 1):
            i += 1  # need to start at second step
            q[i] = q[i - 1] + v[i - 1] * self.dt
            accelCurr = self.Force(q[i])
            accelArr = (accelCurr + accelPrev) / 2
            v[i] = v[i - 1] + accelArr * self.dt
            accelPrev = accelCurr
        return q,v
    def find_Wall_Collision_Indices(self,q):
        #find indices where a collision has occured. If a particles has collided it's marked True. I use a cute trick here
        #this method would NOT work for small aperture, only bores
        rArr = np.sqrt(q[:, 1::3] ** 2 + q[:, 2::3] ** 2)
        collisionArr=np.zeros(self.numParticles) #to hold the True/False for a collision. True means there was a collision
        for i in range(self.timeSteps):
            z = q[i, 0::3]
            indices = np.sum(~(z < self.lengthArray.T), axis=0) #find where each particle is in the lattice. Gives the index
                                        #of which optic it's in.
            collisionArr+= rArr[i] > self.apetureArr[indices]  # if the particle is outside the apeture, record the difference
                                                                #Remember, drifts are infinite
        collisionArr=collisionArr>0 #if any array entry is greater than zero, it has spent sometime outside the apeture
                    #and needs to be removed
        return collisionArr
    def find_Particle_Distributions(self,totalLength):
        #trace many particles and converge on the distribution of particle parameters


        self.thetaMax=self.compute_ThetaMax()
        self.numParticles=1000
        self.timeSteps = 1000
        self.initialize_Parameters(totalLength)
        q,v=self.trace_Particles()

        q,v=self.remove_Collided_Particles(q,v)
        numRemoved=int(self.numParticles-q[0].shape[0]/3)
        transmission=100*(self.numParticles-numRemoved)/self.numParticles

        peaks=self.find_Focuses(q)

        xRMSArr=np.std(q[:,1::3],axis=1)
        yRMSArr = np.std(q[:,2::3], axis=1)
        spotSizes = np.sqrt(xRMSArr[peaks] ** 2 + yRMSArr[peaks] ** 2)  # combine the values at the focus

        VxRMSArr=np.std(v[:,1::3],axis=1)
        VyRMSArr = np.std(v[:,2::3], axis=1)
        spotVelRMS = np.sqrt(VxRMSArr[peaks] ** 2 + VyRMSArr[peaks] ** 2)  # combine the values at the focus


        return spotSizes,spotVelRMS,transmission


    def find_Focuses(self,q):
        #find spot sizes at the particle focuses
        rArr=np.sqrt(q[:,1::3]**2+q[:,2::3]**2)
        rRMSArr=np.std(rArr,axis=1) #used to find the position of the focus
        peaks=sps.find_peaks(-rRMSArr)[0] #make negative because the focus occurs at valleys, not peaks
        return peaks


    def initialize_Parameters(self,totalLength):
        #initialize parameters for tracing
        self.ForceHelper=np.zeros(3*self.numParticles)
        self.xPosArr = np.arange(1, self.numParticles * 3 + 1, 3)
        self.yPosArr = np.arange(2, self.numParticles * 3 + 1, 3)
        simulationTime = 1.1 * totalLength / self.v0
        self.dt = simulationTime / (self.timeSteps - 1)

    def remove_Particles(self,q,v,indices):
        #silly method to remove particles. I'm sure theres a better way
        if len(q.shape)==1:
            qNew = np.zeros(3*np.sum(~indices))
            vNew = qNew.copy()
            qNew[0::3] = q[0::3][~indices]
            qNew[1::3] = q[1::3][~indices]
            qNew[2::3] = q[2::3][~indices]
            vNew[0::3] = v[0::3][~indices]
            vNew[1::3] = v[1::3][~indices]
            vNew[2::3] = v[2::3][~indices]

        else:
            qNew = np.zeros((self.timeSteps,3*np.sum(~indices)))
            vNew = qNew.copy()
            qNew[:, 0::3] = q[:, 0::3][:, ~indices]
            qNew[:, 1::3] = q[:, 1::3][:, ~indices]
            qNew[:, 2::3] = q[:, 2::3][:, ~indices]
            vNew[:, 0::3] = v[:, 0::3][:, ~indices]
            vNew[:, 1::3] = v[:, 1::3][:, ~indices]
            vNew[:, 2::3] = v[:, 2::3][:, ~indices]
        return qNew, vNew
    def remove_Collided_Particles(self,q,v):
        #remove particles that collided with the wall
        indices = self.find_Wall_Collision_Indices(q)
        qNew,vNew=self.remove_Particles(q,v,indices)
        return qNew,vNew




