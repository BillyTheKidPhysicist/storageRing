import matplotlib.pyplot as plt
import skopt
import poisson_disc
import numpy as np

#generate a sample of the particle that represent the focus. Right now this is rather simple. A radial lorentzian to
#represent the spatial dependence and a radial gaussian to represent the momentum dependence. For the lorentzian
#I have to use the accept and reject method to construct the distribution so it is not deterministic for the number of
#generated particles if the random state isn't set


#----constants----
gamma0=4.2 #spot size FWHM, mm
sigmaTrans0=4.0 #Sigma of the entire spot. dubious with multipeaks.m/s
sigmaLong0=1.5 #longitudinal sigma. A guess based on temperature m/s
v0Nominal=-200.0 #longitudinal speed m/s
np.random.seed(42) #set to repeatable seed
apetureDiam=5.0 #mm. Diameter of the apeture at the focus


def lorentz(x,gamma):
    #the distribution to sample from.
    #x: position, mm
    #gamma: FWHM, mm
    return (1/np.pi)*(gamma/2)/(x**2+(gamma/2)**2)



numSamples=20000#final count will not be this value because of the sample reject method being used

yzBound=apetureDiam/2
bounds=[(-yzBound,yzBound),(-yzBound,yzBound)]
#--------poisson disc sampling--------------------
r = np.sqrt(2 * (1.0 / numSamples) / np.pi)  # this is used to get near the required amount of points with
# poission disc method, which does not give an exact number of points
samples = poisson_disc.Bridson_sampling(dims=np.asarray([1.0,1.0]),radius=r,k=100)
np.random.shuffle(samples)
samples[:,0]=samples[:,0]*(bounds[0][1]-bounds[0][0])+bounds[0][0]
samples[:,1]=samples[:,1]*(bounds[1][1]-bounds[1][0])+bounds[1][0]
# plt.scatter(samples[:,0],samples[:,1])
# plt.show()





#----------sobol sampling-----------------
#
# sampler=skopt.sampler.Sobol()
# samples=np.asarray(sampler.generate(bounds,numSamples))
# np.random.shuffle(samples)
# plt.scatter(samples[:,0],samples[:,1])
# plt.show()



#--------position space--------------
acceptedList=[]
prob0=lorentz(0.0,gamma0) #the peak probability value. use this to scale each probability to dramatically
#increase the rate at which values are accepted without distorting the distribution at all
randList=[]
probList=[]
for sample in samples:
    r=np.sqrt(sample[0]**2+sample[1]**2)
    if r<bounds[0][1]:
        prob=lorentz(r,gamma0)/prob0 #scale the probability
        rand=np.random.rand()
        if prob>rand: #simple test and reject/accept method. robust
            acceptedList.append(sample)
positionArr=np.asarray(acceptedList)
xPos=np.zeros(positionArr.shape[0]) #particles start at x=0
positionArr=np.column_stack((xPos,positionArr))



#--------momentum space--------
#now use a multivariate gaussian to construct the momentum space sample before stitching together
varianceArr=np.zeros((3, 3))
varianceArr[0, 0]= sigmaLong0 ** 2
varianceArr[1, 1]= sigmaTrans0 ** 2
varianceArr[2, 2]= sigmaTrans0 ** 2
meanArr=np.zeros(3)
meanArr[0]=v0Nominal
meanArr[1:]=0
momentumSamples=np.random.multivariate_normal(meanArr, varianceArr, size=len(acceptedList))
print(np.std(momentumSamples[:,1]),np.std(positionArr[:,1]))
# plt.hist(momentumSamples[:,0],bins=50)
# plt.show()
#
# plt.hist(momentumSamples[:,1],bins=50)
# plt.show()
#
# plt.hist(momentumSamples[:,2],bins=50)
# plt.show()



#save the results.
#SAVE IN UNITS OF M AND M/S
positionArr=positionArr*1e-3 #convert to meter from mm
phaseSpaceCoords=np.column_stack((positionArr,momentumSamples)) #combine into 6d phase space coords
print('Total number of particle generated:',phaseSpaceCoords.shape[0])
# np.savetxt('phaseSpaceCloud.dat',phaseSpaceCoords)

