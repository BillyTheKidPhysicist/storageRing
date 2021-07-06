import matplotlib.pyplot as plt
import skopt
import poisson_disc
import numpy as np
import scipy.interpolate as spi

#Here mass is taken to have a value of 1, with no units

numParticlesProbe=100000 #number of particles to project onto the dsitrbution. Final number will be much smaller


#----constants----
v0=200
T=.001 #temperature in kelvin
sigma=np.sqrt(1.38e-23*T/1.16e-26)
magnification=3
binning=4
pixelSize=24E-6 #m
apertureDiam=.01 #aperture diameter, m
maxPTrans=10 #maximum transverse momentum accepted
lam=671e-9 #wavelength of lithium transition



fileName='run40FarphaseSpaceData.dat'
data=np.loadtxt(fileName)
xArr=np.arange(data.shape[1]-1)*magnification*binning*pixelSize
pxArr=lam*data[:,0]*1e6  #transverse momentum, with m=1. m/s
pxArr=pxArr*2/3



momentumProfile2DArr=data[:,1:] #pick the momentum profiles


#----build spatial probability function. Center also----
spatialProfile=np.sum(data[:,1:],axis=0) #I consider this a probability function
spatialProbabilityArr=spatialProfile/spatialProfile.max() #rescale from 0 to 1 probability

tempPeakFunc=spi.Rbf(xArr,spatialProbabilityArr,smooth=1) #make interpolating function to find peak. SMooth as well a little
xArrDense=np.linspace(xArr[0],xArr[-1],num=1000)

xPeak=xArrDense[np.argmax(tempPeakFunc(xArrDense))]
xArr=xArr-xPeak #shift the x array to have x=0 at the peak
spatial_Probability_Function=spi.Rbf(xArr,spatialProbabilityArr,smooth=0) #make the final spatial probability function. no smoothing

# plt.plot(xArr,spatialProbabilityArr)
# plt.plot(xArrDense-xPeak,spatial_Probability_Function(xArrDense-xPeak))
# plt.show()

#----build momentum probability function. center also


#rescale the momentum probabilty
momentumProbability2DArr=momentumProfile2DArr.copy() #this should not be rescaled by the same amount for each curve.
#that is already done in the spatial rescaling. each curve should be scaled to a peak of one




momentumPeakProbabilityArr=momentumProbability2DArr[:,np.argmin(np.abs(xArr))] #the central profile curve. Use the momentum
#that corresponds to the peak value to adjust the momentum array
tempPeakFunc=spi.Rbf(pxArr,momentumPeakProbabilityArr,smooth=1) #slight smoothing
pxArrDense=np.linspace(pxArr[0],pxArr[-1],1000)
pxPeak=pxArrDense[np.argmax(tempPeakFunc(pxArrDense))]
pxArr=pxArr-pxPeak
# plt.plot(pxArr,momentumProbabilityArr)
# plt.plot(pxArrDense-pxPeak,tempPeakFunc(pxArrDense))
# plt.axvline(x=0)
# plt.show()


momentumProbabilityFuncList=[]
for i in range(momentumProbability2DArr.shape[1]):
    func=spi.Rbf(pxArr,momentumProbability2DArr[:,i]/momentumProbability2DArr[:,i].max())
    #
    pxArrDense=np.linspace(pxArr[0],pxArr[-1],1000)
    # plt.plot(pxArrDense,func(pxArrDense))
    # plt.show()
    momentumProbabilityFuncList.append(func)

def momentum_Probability_Function(x,px):
    #x: spatial position, md
    #px: momentum with mass=1, m/s
    if (not xArr.max()>x>xArr.min()) or ( not pxArr.max()>px>pxArr.min()):
        print(x,px)
        raise Exception("outside range")
    momentumProfileFunc=momentumProbabilityFuncList[np.argmin(np.abs(xArr-x))]
    # plt.plot(momentumProfileFunc(pxArrDense-xPeak))
    # plt.show()
    return momentumProfileFunc(px)

def probability_Function(x,px,momentumOnly=False,positionOnly=False):
    assert momentumOnly==False or positionOnly==False
    if momentumOnly==True:
        return momentum_Probability_Function(x,px)
    if positionOnly==True:
        return spatial_Probability_Function(x)
    return spatial_Probability_Function(x)*momentum_Probability_Function(x,px)

num=200
xPlot=np.linspace(-apertureDiam/2,apertureDiam/2,num)*.95 #position
pPlot=np.linspace(-maxPTrans,maxPTrans,num)*.95 #momentum
image=np.zeros((num,num))
for i in range(num):
    for j in range(num):
        xPos=xPlot[i]
        pPos=pPlot[j]
        image[j,i]=probability_Function(xPos,pPos,momentumOnly=False,positionOnly=False)
image=np.flip(image,axis=0)

for i in range(num):
    profile=image[:,i]
    # plt.plot(pPlot,profile)
    # plt.show()
    image[np.argmax(profile),i]=0



extent=[xPlot[0],xPlot[-1],pPlot[0],pPlot[-1]]
aspect=(extent[1]-extent[0])/(extent[3]-extent[2])
plt.title('Phase space plot of far field data')
plt.ylabel("velocity, m/s")
plt.xlabel('Position, m')
plt.imshow(image,extent=extent,aspect=aspect)
plt.show()








r=(1/numParticlesProbe**(.25))*.9


samples=poisson_disc.Bridson_sampling(dims=np.asarray([1.0,1.0,1.0,1.0]),radius=r,k=30) #returned samples are in the
#form [[x,y,px,py]..]
samples=(samples-.5)*2
samples[:,:2]=samples[:,:2]*apertureDiam/2
samples[:,2:]=samples[:,2:]*maxPTrans






particleList=[]

for sample in samples:
    y,z,py,pz=sample
    r=np.sqrt(y**2+z**2)
    pr=np.sqrt(py**2+pz**2)
    if r<apertureDiam/2 and pr<maxPTrans:
        probability=probability_Function(y,py)*probability_Function(z,pz)
        if np.random.rand()<probability:
            px=v0+np.random.normal(scale=sigma)
            particleList.append([0,y,z,px,py,pz])
            # plt.scatter(y,z,alpha=.5,c='r')

np.savetxt('phaseSpaceParticleCloudOriginal.dat',np.asarray(particleList))