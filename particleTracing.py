import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps
import time

#integrate into linearlattice

class ParticleTrace:
    def __init__(self,LLS,nozzleDiam=400E-6,v0=None,T=None):
        self.m = 1.1650341e-26
        self.u0 = 9.274009994E-24
        self.u0Sim = self.u0 / self.m
        self.LLS = LLS
        #LinearLatticeSolver object
        if v0==None:
            v0=LLS.v0
        if T==None:
            T=LLS.T
        self.v0=v0
        self.T=T
        self.LLS=LLS
        self.lengthArray = np.asarray([LLS.totalLengthArrayFunc()])
        self.timeSteps=None
        self.numParticles=None
        self.kArr,self.apetureArr=self.compute_Apeture_And_K_Array()
        self.ForceHelper=None
        self.xPosArr=None
        self.yPosArr=None
        self.dt=None
        self.simulationTime=None

    def compute_ThetaMax(self):
        firstLensIndex = None
        i = 0
        for el in self.LLS.lattice:
            if el.type == 'LENS':
                firstLensIndex = i
                break
            i += 1
        return self.LLS.lattice[firstLensIndex].params[2] / self.LLS.totalLengthArrayFunc()[firstLensIndex - 1]
    def compute_Apeture_And_K_Array(self):
        apetureList = []  # list of aperture size
        kList = []  # List that hold k values for each total length array entry
        for el in self.LLS.lattice:
            if el.type == 'DRIFT':
                k = 0
                apeture = np.inf
            else:
                B = el.params[1]
                rp = el.params[2]
                k = 2 * self.LLS.u0Sim * B / rp ** 2
                apeture = rp
            kList.append(k)
            apetureList.append(apeture)
        # now add an element at the end for drift
        kList.append(0)
        apetureList.append(np.inf)
        return np.asarray(kList),np.asarray(apetureList)

    def initialize_q_And_v(self):
        # create q array
        q = np.zeros((self.timeSteps, 3 * self.numParticles))
        # create v array
        v = q.copy()
        vy = np.linspace(-1, 1, num=self.numParticles) * self.thetaMax * self.v0
        vx = vy.copy()
        vz = np.ones(self.numParticles) * self.v0
        np.random.shuffle(vx)
        np.random.shuffle(vy)
        v[0, 0::3] = vz
        v[0, 1::3] = vy
        v[0, 2::3] = vx

        # now seed at 5 cm from the nozzle
        #use cylinderical coordinates
        dtArr = .05/ vz[0]  # array of time steps required to bring particles up to seeding distance
        dtArr = dtArr[np.newaxis].T  # numpy tricks
        q[0]+=v[0] * dtArr
        sigma = np.sqrt(1.38E-23 * (self.T/1000) / 1.16E-26)
        v[0] += np.random.normal(scale=sigma, size=(v[0].shape[0]))

        return q,v
    def Force(self,q):
        temp = -self.kArr[np.sum(~(q[0::3] < self.lengthArray.T), axis=0)]
        self.ForceHelper[self.xPosArr] = temp
        self.ForceHelper[self.yPosArr] = temp
        return self.ForceHelper * q
    def trace_Particles(self):
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
        rArr = np.sqrt(q[:, 1::3] ** 2 + q[:, 2::3] ** 2)
        collisionArr=np.zeros(self.numParticles)
        for i in range(self.timeSteps):
            z = q[i, 0::3]
            indices = np.sum(~(z < self.lengthArray.T), axis=0)
            collisionArr+= rArr[i] > self.apetureArr[indices]  # record the collision
            #if np.any(collisionIndices):
            #    q[i:, collisionIndices * 3] = np.nan
            #    q[i:, 1 + collisionIndices * 3] = np.nan
            #    q[i:, 2 + collisionIndices * 3] = np.nan
        collisionArr=collisionArr>0
        return collisionArr
    def find_Particle_Distributions(self,totalLength):
        #trace many particles and converge on the distribution of particle parameters
        #solves for particle density distribution as a function of radius for each position

        self.thetaMax=self.compute_ThetaMax()
        self.numParticles=1000
        self.timeSteps = 1000
        self.initialize_Parameters(totalLength)
        q,v=self.trace_Particles()
        q,v=self.remove_Collided_Particles(q,v)
        spotsArr=self.find_Spot_Sizes(totalLength)[0]
        qSpots=q[spotsArr]
        vSpots=v[spotsArr]

        qRMS=(np.std(qSpots[:,1::3],axis=1)+np.std(qSpots[:, 2::3],axis=1))/2
        vRMS=(np.std(vSpots[:,1::3],axis=1)+np.std(vSpots[:, 2::3],axis=1))/2
        return qRMS,vRMS




        #zArr=q[:,0]
        #xArr=q[:,1::3]
        #yArr=q[:,2::3]
        #rArr=np.sqrt(xArr**2 + yArr**2)
        #plt.plot(zArr,np.sqrt(2)*1000*np.std(q[:,1::3],axis=1),c='black')#np.std(q[:,1::3],axis=1))
        #plt.grid()
        #plt.show()
    def find_Spot_Sizes(self,totalLength=10):
        self.thetaMax=self.compute_ThetaMax()
        self.numParticles=1000
        self.timeSteps = 1000
        self.initialize_Parameters(totalLength)
        q,v=self.trace_Particles()
        q,v=self.remove_Collided_Particles(q,v)
        zArr=q[:,0]
        xArr=q[:,1::3]
        yArr=q[:,2::3]
        rArr=np.sqrt(xArr**2 + yArr**2)
        stdArr=1000*np.std(q[:,1::3],axis=1)
        peaks=sps.find_peaks(-stdArr)[0]
        spotSizes=np.sqrt(2)*stdArr[peaks]
        spotPos=zArr[peaks]
        return peaks,spotSizes


    def initialize_Parameters(self,totalLength):
        self.ForceHelper=np.zeros(3*self.numParticles)
        self.xPosArr = np.arange(1, self.numParticles * 3 + 1, 3)
        self.yPosArr = np.arange(2, self.numParticles * 3 + 1, 3)
        simulationTime = 1.1 * totalLength / self.LLS.v0
        self.dt = simulationTime / (self.timeSteps - 1)
    def remove_Collided_Particles(self,q,v):

        temp = self.find_Wall_Collision_Indices(q)
        qNew = np.zeros((self.timeSteps, 3 * np.sum(~temp)))
        vNew = qNew.copy()

        qNew[:, 0::3] = q[:, 0::3][:, ~temp]
        qNew[:, 1::3] = q[:, 1::3][:, ~temp]
        qNew[:, 2::3] = q[:, 2::3][:, ~temp]
        vNew[:, 0::3] = v[:, 0::3][:, ~temp]
        vNew[:, 1::3] = v[:, 1::3][:, ~temp]
        vNew[:, 2::3] = v[:, 2::3][:, ~temp]
        return qNew,vNew




