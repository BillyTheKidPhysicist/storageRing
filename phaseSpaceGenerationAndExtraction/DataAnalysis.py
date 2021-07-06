import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
import scipy.special as spf
import scipy.signal as spsi
import globalVariables as gv
import time
import numba

#FUTURE PHYSICISTS, THIS USES OBJECT ORIENTED PROGRAMMING.
#Written by William Debenham(Huntington), billydebenham@gmail.com, 6197870499. Feel free to contact





class DataAnalyzer:
    def __init__(self):
        self.params=None #params for the current solution of fit_Spectral_Profile
        self.fitFunc=None #fit for the current solution of fit_Spectral_Profile
        self.T=None
        self.F0=None #center frequency of the profile
    def fit_Spectral_Profile(self,freqData, signalData, peakMode='multi', lensDistribution=False,vTMaxLens=10.0
                             ,gamma=gv.LiTau/1E6,gammaFloor=False,laserJitter=2.0):
        #Fit spectral data. Can be (single peak or multipeak) and include the velocity distribution of lens.
        #freqData: frequency, or anything else, scale
        #signalData: Flourescence signal
        #peakMode: Use either multipeak (6 peaks) or single peak.
        #lensDistribution: Include the effects of the convolution with the lens output transverse velocity distribution
        #vTMaxLens: transverse velocity maximum at lens output.
        #gamme: value of gamma. Set to None to enable free fitting of gamma. Set to a specific value to lock to that
        #value, or allow to go above depending on gammaFloor
        #gammaFloor: If true, the provided gamme value is the floor of the fit, and values can go above
        #laserJitter: Jitter of laser, standard deviation, MHz
        if peakMode!= 'multi' and peakMode!='single':
            raise Exception('Invalid peak mode. Choose multipeak or single peak')
        if gammaFloor==True and gamma is None:
            raise Exception('Gamma cannot be both free to vary, and be a floor value')
        if lensDistribution==False:
            freqTMaxLens=None
        else:
            freqTMaxLens=(vTMaxLens/gv.cLight)*gv.Li_D2_Freq/1e6 #transverse frequency maximum lens, MhZ



        def minimize(params):
            v0, a, b, sigma, gamma=params
            fit=self._spectral_Profile(freqData,v0,a,b,sigma,gamma,freqTMaxLens,peakMode,laserJitter)
            return self._cost(signalData,fit)
        bounds=[(-30,30),(0,1000),(-100,100),(0,30),(0,30)]


        np.random.seed(42) #seed the generator to make results repeatable
        sol=spo.differential_evolution(minimize,bounds,polish=False)
        np.random.seed(int(time.time())) #resead to make pseudorand
        self.params=sol.x
        self.T=self._calculate_Temp(self.params[3])
        self.F0=self.params[0]
        self.fitFunc=lambda v: self._spectral_Profile(v,*self.params,freqTMaxLens,peakMode,laserJitter)
        self.fitFunc(freqData)
        # plt.plot(freqData,signalData)
        # plt.plot(freqData,self.fitFunc(freqData))
        # plt.show()
        return self.params

    def _spectral_Profile(self,v,v0,a,b,sigma,gamma,vTMaxLens,peakMode,laserJitter):
        #create spectral profile
        profile=np.zeros(v.shape)
        sigma=np.sqrt(sigma**2+laserJitter**2)
        if peakMode=='multi':
            profile+=self.multi_Voigt(v,v0,a,b,sigma,gamma)
        else:
            profile+=self.voigt(v,v0,a,sigma,gamma)
        if vTMaxLens is not None:
            peak0=profile.max() #to aid in normalizing and rescaling the profile after convolution
            profile=profile/peak0
            vLens=np.linspace(-(v.max()-v.min())/2,(v.max()-v.min())/2,num=v.shape[0]) #need to have the lens profile
            #centered for the convolution to preserve the center of the spectral profile that results
            lensVelProfile=np.vectorize(self.lens_Velocity_Spread)(vLens,vTMaxLens) #not very efficient
            profile=np.convolve(lensVelProfile,profile,mode='same') #convolution has distributive property so don't need
            #to perform on each peak individually
            profile=profile*peak0/profile.max() #rescale the peak
        return profile
    @staticmethod
    @numba.njit()
    def _cost(signalData,fit):
        return np.sqrt(np.sum((signalData-fit)**2))

    def gauss(self,T,v,m=gv.massLi7,v0=0):
        t1=np.sqrt(m/(2*np.pi*gv.kB*T))
        t2=np.exp(-m*np.power(v-v0,2)/(2*gv.kB*T))
        return t1*t2
    @staticmethod
    @numba.njit()
    def lens_Velocity_Spread(x, x0, a=1):
        #1D transvers velocity distribution for the output of the lens. This is because the lens a circular input
        #and so creates a unique semicircular distirbution. Can be derived considering the density and y velocity as a
        # function of y in the circular aperture easily.
        #x: either velocity or frequency value. Must have same units as x0
        #x0: maximum transverse velocity of frequency. Same units as x
        if np.abs(x)>x0:  #to prevent imaginary
            return 0.0
        else:
            return a*np.sqrt(1-(x/x0)**2)

        #---------SINGLE VOIGT FIT---------
        #this voigt is normalized to a height of 1, then multiplied by the variable a
        #there is no yaxis offset in the voigt, that is done later
    @staticmethod
    def voigt(f,f0, a, sigma, gamma):
        #units must be consistent!!
        #f, frequency value
        #f0, center frequency
        #a, height of voigt
        #sigma,standard deviation of the gaussian
        #gamma, FWHM of the lorentzian
        gamma=gamma/2  #convert lorentzian FWHM to HWHM
        x=f-f0
        z=(x+gamma*1.0J)/(sigma*np.sqrt(2.0))  #complex argument
        V=np.real(spf.wofz(z))/(sigma*np.sqrt(2*np.pi))  #voigt profile

        #now normalize to 1 at peak, makes fitting easier
        z0=(gamma*1.0J)/(sigma*np.sqrt(2.0))  #z at peak
        V0=np.real(spf.wofz(z0))/(sigma*np.sqrt(2*np.pi))  #height of voigt at peak
        return a*V/V0  #makes the height equal to a

        #------SEXTUPLE VOIGT FIT----------

    def multi_Voigt(self,freq,freq0, a, b, sigma, gamma=gv.LiTau/1E6):
        #units must be consitant!
        #freq: frequency
        #a: height constant
        #b: center frequency
        #c: vertical offset
        #sigma: standard deviation of gaussian profile
        #gamma: FWHM of lorentzian
        aRatio=4*gv.F2F1Ratio  #ratio of intensity of f=2 to f=1. First number is power ratio in sideband. Second
        # fraction is ratio of hyperfine transitions (see globalVariable.py for more info in the comments for
        #F2F1Ratio).
        a2=(aRatio/(aRatio+1))*a #I do some funny business here to try and get the parameter "a" to more closely match
        #the total height. a2/a1 still equals the parameter "aRatio", but now they also add up the parameter "a"
        a1=a*1/(aRatio+1) #same funny business
        val=b  #start with the offset


        #F=2 transition
        val+=a2*self.voigt(freq, freq0+gv.F1Sep/1E6, gv.S21, sigma, gamma)
        val+=a2*self.voigt(freq, freq0+gv.F2Sep/1E6, gv.S22, sigma, gamma)
        val+=a2*self.voigt(freq, freq0+gv.F3Sep/1E6, gv.S23, sigma, gamma)
        #F=1 transition
        val+=a1*self.voigt(freq,freq0+gv.F0Sep/1E6, gv.S10,  sigma, gamma)
        val+=a1*self.voigt(freq,freq0+gv.F1Sep/1E6, gv.S11,  sigma, gamma)
        val+=a1*self.voigt(freq,freq0+gv.F2Sep/1E6, gv.S12,  sigma, gamma)
        return val

    def _calculate_Temp(self,sigma,f0=gv.Li_D2_Freq, MHz=True):
        dF0=2*np.sqrt(2*np.log(2))*sigma
        if MHz==True:
            dF0=dF0*1E6
        return (dF0/f0)**2*(gv.massLi7*gv.cLight**2)/(8*gv.kB*np.log(2))
    def calculate_Temp(self,sigma,MHz=True):
        return self._calculate_Temp(sigma,MHz=MHz)


