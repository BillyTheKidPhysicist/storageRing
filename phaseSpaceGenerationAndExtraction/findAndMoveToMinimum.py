import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps



#analyze the phase space data to find where the minimum is. The phase space data can also used to created a new cloud
#at the minimum



def get_Frac_Width(qArr, fraction=.95):
    # get the width of the beam that captures fraction of the atoms
    # fraction: fraction of the atoms in that width
    if not 0 < fraction < 1.0:
        raise Exception('fraction must be number between 0 and 1')
    numParticles = qArr.shape[0]
    rArr = np.sqrt(qArr[:, 1] ** 2 + qArr[:, 2] ** 2)
    rArrSorted = np.sort(rArr)
    cutoffIndex = int(fraction * numParticles)  # the 100*fraction% particle. There is some minor rounding error
    width = 2 * rArrSorted[cutoffIndex]
    return width
moveCloudToMinimimum=False
particleCoords=np.loadtxt('run40PhaseSpaceParticleCloudOriginal.dat')


xStart=-.05
xEnd=.05

widthList=[]
numSteps=100
xArr=np.linspace(xStart,xEnd,numSteps)
for x in xArr:
    qList=[]
    for coord in particleCoords:
        x0,y0,z0,px0,py0,pz0=coord
        dt=(x-x0)/px0
        y=y0+py0*dt
        z=z0+pz0*dt
        qList.append([x0,y,z])
    width=get_Frac_Width(np.asarray(qList),fraction=.95)
    widthList.append(width)
widthArr=np.asarray(widthList)
widthArr=sps.savgol_filter(widthArr,11,2)
xMin=xArr[np.argmin(widthArr)]
print(xMin)
plt.title('Spot size vs position of phase space cloud')
plt.axvline(x=1e3*xMin,c='black',linestyle=':')
plt.xlabel('Position, mm')
plt.ylabel('Size, mm')
plt.plot(xArr*1e3,widthArr*1e3)
plt.show()


if moveCloudToMinimimum==True:
    phaseList = []
    for coord in particleCoords:
        x0, y0, z0, px0, py0, pz0 = coord
        dt = (xMin - x0) / px0
        x=0 #set the new origin of expansion to zero
        y = y0 + py0 * dt
        z = z0 + pz0 * dt
        phaseList.append([x, y, z, px0, py0, pz0])
    np.savetxt('run40PhaseSpaceParticleCloudMinimum.dat',np.asarray(phaseList))