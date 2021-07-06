from DeconvolveNaturalLineWidth import deconvolve
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.special as sps
import scipy.optimize as spo
import scipy.ndimage as spni
#extract the deconvoluted data
lam=671e-9 #wavelength of lithium transition
fileName = 'run45Far'

#42 and 45 are two opposites of the observation cell

# Opening fits file and creating array. Flip the y axis such that image is oriented correctly.
fitsFile = fits.open(fileName + '.fits')
imagesList = fitsFile[0].data
imagesArr = imagesList.astype(float)
imagesArr = np.flip(imagesArr, axis=1)
imageFreqArr=np.linspace(0,1,imagesArr.shape[0])*230

# Find background
imageBackGround = (np.mean(imagesArr[-5:], axis=0) + np.mean(imagesArr[:5],
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
    imagesArr[i] = spni.median_filter(imagesArr[i], size=3)

imagesArr = imagesArr[:, 80:123, 135:145]

# plt.imshow(np.sum(imagesArr,axis=0))
# plt.show()
phaseSpaceArr = np.sum(imagesArr, axis=2) #collapsed along x axis
#orient correctly such that the bottom left is 0,0 and the x axis is position and the y axis is frequency
phaseSpaceArr=np.flip(phaseSpaceArr,axis=0) #switch frequency
phaseSpaceArr=np.flip(phaseSpaceArr,axis=1) #switch space

plt.imshow(phaseSpaceArr)
plt.show()
indexStart=25
indexEnd=-15
print(phaseSpaceArr.shape)
pxArr=lam*imageFreqArr*1e6  #transverse momentum, with m=1. m/s
data=pxArr[indexStart:indexEnd] #array to hold data to save as 2d array with first column as frequency. second
#column is the profile for the topmost pixel
for i in range(0, phaseSpaceArr.shape[1]):
    profile = phaseSpaceArr[:, i]

    # profile = deconvolve(profile)
    profile=profile[indexStart:indexEnd]
    # plt.plot(profile)
    # plt.show()
    data=np.column_stack((data,profile))
np.savetxt(fileName+'phaseSpaceData.dat',data)
