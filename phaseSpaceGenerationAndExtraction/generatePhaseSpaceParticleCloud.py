import matplotlib.pyplot as plt
import skopt
import poisson_disc
import numpy as np
import scipy.interpolate as spi
import scipy.ndimage as spni

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



runName='run45Far'
fileName=runName+'phaseSpaceData.dat'
data=np.loadtxt(fileName)
xArr=np.arange(data.shape[1]-1)*magnification*binning*pixelSize
pxArr=data[:,0]  #transverse momentum, with m=1. m/s



#-------center spatially-------------
spatialProfile=np.sum(data[:,1:],axis=0) #I consider this a probability function
spatialProfile=spatialProfile/spatialProfile.max()
tempPeakFunc=spi.Rbf(xArr,spatialProfile,smooth=1) #make interpolating function to find peak. SMooth as well a little
xArrDense=np.linspace(xArr[0],xArr[-1],num=1000)

xPeak=xArrDense[np.argmax(tempPeakFunc(xArrDense))] #value to subtract from xarr
xArr=xArr-xPeak #center
#------center in momentu-------

momentumProfile=np.sum(data[:,1:],axis=1) #I consider this a probability function
momentumProfile=np.flip(momentumProfile) #because bottom left is lower frequency, not top
momentumProfile=momentumProfile/momentumProfile.max()


tempPeakFunc=spi.Rbf(pxArr,momentumProfile,smooth=1) #make interpolating function to find peak. SMooth as well a little
pxArrDense=np.linspace(pxArr[0],pxArr[-1],num=1000)

pxPeak=pxArrDense[np.argmax(tempPeakFunc(pxArrDense))] #value to subtract from xarr
# plt.plot(pxArrDense,tempPeakFunc(pxArrDense))
# plt.axvline(x=pxPeak)
# plt.show()
# print(pxPeak)
pxArr=pxArr-pxPeak #center

#---------build phase space function--------------
phaseSpace2DArray=data[:,1:] #phase space data


phaseSpace2DArray=spni.gaussian_filter(phaseSpace2DArray,.5)
phaseSpace2DArray=phaseSpace2DArray/phaseSpace2DArray.max() #normalize to probability function



#----build spatial probability function. ----
spatialProfile=np.sum(phaseSpace2DArray,axis=0) #I consider this a probability function
spatialProbabilityArr=spatialProfile/spatialProfile.max() #rescale from 0 to 1 probability

spatial_Probability_Function=spi.Rbf(xArr,spatialProbabilityArr,smooth=0) #make the final spatial probability function. no smoothing

# plt.plot(xArr,spatialProbabilityArr)
# plt.plot(xArrDense-xPeak,spatial_Probability_Function(xArrDense-xPeak))
# plt.show()

#----build momentum probability function. center also


momentumProfile=np.sum(data[:,1:],axis=1) #I consider this a probability function
momentumProfile=np.flip(momentumProfile) #because bottom left is lower frequency, not top
momentumProfile=momentumProfile/momentumProfile.max()

# plt.plot(pxArrDense,tempPeakFunc(pxArrDense))
# plt.axvline(x=pxPeak)
# plt.show()
# print(pxPeak)


momentumProbabilityFuncList=[]

for i in range(phaseSpace2DArray.shape[1]):
    func=spi.Rbf(pxArr,np.flip(phaseSpace2DArray[:,i]/phaseSpace2DArray[:,i].max()))
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


#simpler version. Slighly more coarse though
# coords=[] #there's a faster way to do this, but this is simpler and more intuitive to me
# points=[]
# i=0
# for x in xArr:
#     j=0
#     for px in pxArr:
#         coords.append((x,px))
#         points.append(phaseSpace2DArray[pxArr.shape[0]-1-j,i])
#         j+=1
#     i+=1
# coords=np.asarray(coords)
# probability_Function=spi.LinearNDInterpolator(coords,points,rescale=True) #this is the fit, but it is not
# #centered like I would like





# num=100
# xPlot=np.linspace(-apertureDiam/2,apertureDiam/2,num)*.99 #position
# pPlot=np.linspace(-maxPTrans,maxPTrans,num)*.99 #momentum
# image=np.zeros((num,num))
# for i in range(num):
#     for j in range(num):
#         xPos=xPlot[i]
#         pPos=pPlot[j]
#         image[num-j-1,i]=probability_Function(xPos,pPos)
# image=spni.gaussian_filter(image,1)
# for i in range(num):
#     profile=image[:,i]
#     image[np.argmax(profile),i]=0
#
#
#
# extent=[xPlot[0],xPlot[-1],pPlot[0],pPlot[-1]]
# aspect=(extent[1]-extent[0])/(extent[3]-extent[2])
# plt.title('Phase space plot of far field data')
# plt.ylabel("velocity, m/s")
# plt.xlabel('Position, m')
# plt.imshow(image,extent=extent,aspect=aspect)
# plt.show()
#
#






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

# np.savetxt('phaseSpaceParticleCloudOriginal.dat',np.asarray(particleList))