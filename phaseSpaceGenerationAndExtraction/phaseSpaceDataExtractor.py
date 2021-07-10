from DeconvolveNaturalLineWidth import deconvolve
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.signal as sps
import scipy.optimize as spo
import scipy.ndimage as spni
import scipy.special as spspec

def voigt(x,x0,a,b,sigma,gamma):
    v0=spspec.voigt_profile(0,sigma,gamma)
    v=spspec.voigt_profile(x-x0,sigma,gamma)
    return a*v/v0+b


#extract the deconvoluted data
lam=671e-9 #wavelength of lithium transition
fileName = 'run41Far'
deltaF=230


#42 and 45 are two opposites of the observation cell

# Opening fits file and creating array. Flip the y axis such that image is oriented correctly.
fitsFile = fits.open(fileName + '.fits')
imagesList = fitsFile[0].data
imagesArr = imagesList.astype(float)
imagesArr = np.flip(imagesArr, axis=1)
imageFreqArr=np.linspace(0,1,imagesArr.shape[0])*deltaF

# Find background
imageBackGround = (np.mean(imagesArr[-3:], axis=0) + np.mean(imagesArr[:3],
                                                             axis=0)) / 2  # take images from beginning and end
# to get average of background noise

imagesArr = imagesArr - imageBackGround  # extract background.



trimValue = 2e3  # maximum positive trim value for cosmic rays
imagesArr[imagesArr > trimValue] = trimValue
# maximum negative trim
trimValue = -10
imagesArr[imagesArr < trimValue] = trimValue

# go through each image and apply a median filter to remove hot/dead pixels and cosmic rays
for i in range(imagesArr.shape[0]):
    imagesArr[i] = spni.median_filter(imagesArr[i], size=1)


imagesArr = imagesArr[:, 75:125, 130:160]



#
# plt.imshow(np.mean(imagesArr,axis=0))
# plt.show()

phaseSpaceArr = np.mean(imagesArr, axis=2) #collapsed along x axis
#orient correctly such that the bottom left is 0,0 and the x axis is position and the y axis is frequency
phaseSpaceArr=np.flip(phaseSpaceArr,axis=0) #switch frequency
phaseSpaceArr=np.flip(phaseSpaceArr,axis=1) #switch space



spatialProfile=np.mean(phaseSpaceArr,axis=0)
spatialProfile=spatialProfile/spatialProfile.max()
xArr=np.arange(0,spatialProfile.shape[0])*3*4*24e-6



# guess=[xArr[int(xArr.shape[0]/2)],1.0,0.0,.001,.001]
# bounds=[(-np.inf,-np.inf,-np.inf,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf)]
# params=spo.curve_fit(voigt,xArr,spatialProfile,p0=guess,bounds=bounds)[0]
# sig=params[3]
# gamma=params[4]
# fwhmImage=.5346*(2*params[4])+np.sqrt(.2166*(2*params[4])**2+(params[3]*2.335)**2)
# print(fwhmImage*1e3)
# # print(1e3*sig,1e3*gamma) #0.6802467258181029 2.3344808083133364
# plt.plot(xArr,spatialProfile)
# plt.plot(xArr,voigt(xArr,*params))
# plt.grid()
# plt.show()



# profile=np.sum(phaseSpaceArr,axis=0)
# plt.plot(profile)
# plt.show()


imStart=28
imEnd=-20



# phaseSpaceArr=phaseSpaceArr[imStart:imEnd] #don't do this here here to help with convolution
# plt.imshow(phaseSpaceArr)
# plt.show()

pxArr=lam*imageFreqArr*1e6  #transverse momentum, with m=1. m/s

# plt.plot(pxArr,np.mean(phaseSpaceArr,axis=1))
# plt.show()


data=pxArr[imStart:imEnd] #array to hold data to save as 2d array with first column as frequency. second
#column is the profile for the topmost pixel
j=0
for i in range(0, phaseSpaceArr.shape[1],1):
    print(i)
    profile = phaseSpaceArr[:, i]
    # plt.plot(pxArr,profile)
    # plt.show()
    # profile=sps.savgol_filter(profile,5,2)
    profile = deconvolve(profile,deltaF)
    profile=profile[imStart:imEnd]
    data=np.column_stack((data,profile))

    plt.plot(np.arange(profile.shape[0])+j*profile.shape[0],profile)
    j+=1
    # plt.show()
plt.show()
np.savetxt(fileName+'phaseSpaceData.dat',data)