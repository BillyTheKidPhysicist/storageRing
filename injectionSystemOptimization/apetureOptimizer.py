import time
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt
from ParticleClass import Swarm
def lorentz(x,x0,gamma,a):
    return a*(gamma/2)**2/((x-x0)**2+(gamma/2)**2)
class ApetureOptimizer:
    def __init__(self):
        pass
        #self.swarm=Swarm() #particle swarm class
    def load_Swarm(self,numParticles=None,fileName='phaseSpaceCloud.dat'):
        #load a swarm from a file.
        #numParticles: Number of particles to load from the file. if None load all
        #fileName: Name of the file with the particle data in the format (y,z,px,py,pz)
        swarm=Swarm()
        data= np.loadtxt(fileName)
        if numParticles is not None:
            data=data[:numParticles] #restrict the data
        for coords in data:
            y,z,px,py,pz=coords
            q=np.asarray([0,y,z])
            p=np.asarray([px,py,pz])
            swarm.add_Particle(qi=q,pi=p)
        return swarm
    def move_Swarm_Through_Free_Space(self,swarm,L):
        #Move a swarm through free space by simple explicit euler. Distance is along the x axis. Moves TO a position
        #swarm: swarm to move
        #L: the distance to move to, positive or negative, along the x axis
        #Return: A new swarm object
        for particle in swarm:
            q=particle.q
            p=particle.p
            dt=(L-q[0])/p[0]
            q=q+p*dt
            particle.q=q
        return swarm
    def get_RMS_Width(self,swarm):
        #Get the RMS width of the beam. Swarm is assumed to have the same x value for all particles. RMS is of the y axis
        temp=[]
        for particle in swarm:
            temp.append(particle.q[1])
        rms=np.std(np.asarray(temp))
        return rms

apetureOptimizer=ApetureOptimizer()
swarm=apetureOptimizer.load_Swarm()





rList=[]
gamma0=4.2
for particle in swarm:
    q=particle.q
    if np.abs(q[1])<.25*gamma0:
        rList.append(q[2])

rArr=np.asarray(rList)
rHist,binEdges=np.histogram(rArr,bins=50)
binsCenter=binEdges[:-1]+ (binEdges[1]-binEdges[0])/2
xPlot=np.linspace(binsCenter[0],binsCenter[-1],num=10000)
guess=[0,5.0,np.max(rHist)]
bounds=[(-np.inf,0,0),(np.inf)]
params,pcov=spo.curve_fit(lorentz,binsCenter,rHist,bounds=bounds,p0=guess)
print(params[1])
plt.plot(binsCenter,rHist)
plt.plot(xPlot,lorentz(xPlot,*params))
plt.show()




rmsList=[]
posArr=np.linspace(-.1,.1)
for pos in posArr:
    print(pos)
    rmsList.append(apetureOptimizer.get_RMS_Width(apetureOptimizer.move_Swarm_Through_Free_Space(swarm,pos)))
plt.plot(posArr,rmsList)
plt.grid()
plt.show()



#print(apetureOptimizer.get_RMS_Width(swarm))