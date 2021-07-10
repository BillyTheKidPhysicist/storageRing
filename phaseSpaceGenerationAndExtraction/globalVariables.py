import numpy as np
import time
import sys

#this file contains the global variables to be used in all other fields
#any edit here is represented across all files that use these variables
#
#
#
#   FUTURE IMPROVEMENT:
#   cite where this data comes from and describe it better
#
#
#
#

#search name convention
#FIX  something is wrong and needs to be repaired
#REMOVE  something is temporarily added and needs to be removed
#IMPROVE something needs to be improved




mTOnm=1e9 #number of nanometers in a meter
cLight = 2.998792458E8  # Speed of light







kB=1.38064852e-23 # boltzman constant, SI
massLi7=1.165034771e-26 # mass of lithium7 atom, kg





debugFolder = "C:\\Desktop\\deBug\\"  #location of debug folder
dataFolder="C:\\"
settingsFolder="C:\\Desktop\\mainProgram\\savedSettings"
settingFileName="dataGuiSettings.txt"


#-------------DAQ BOARD STUFF------------------
#ACTIVE DATA INPUT PINS: Galvo input pin and lithium reference photodiode pin
numActiveDataInputPins=2
galvoOutPin = "ao0"
flowOutPin  = "ao1"
galvoInPin  = "ai0"
pin1Pin     = "ai1"
etalonInPin = "ai2"
lithiumRefInPin = "ai3"
laserWidthInPin = "ai4"
flowInPin  = "ai5"
pin6Pin    ="ai6"
galvoInRawPin="ai7"
servoPin="ctr0"
shutterPin="port0/line1"
pinNameList      =[galvoOutPin,flowOutPin,galvoInPin,pin1Pin       ,etalonInPin    ,lithiumRefInPin,laserWidthInPin
    ,flowInPin,galvoInRawPin,pin6Pin,servoPin,shutterPin]
pinVoltRangeList =[[-5.0,5.0]  ,[0,4.9]   ,[-5.0,5.0] ,[-10.0,10.0],[-10.0,10.0]     ,[-5.0,0.0 ]  ,[-10.0,10.0]
    ,[-10.0,10.0],[-1.0,1.0]   ,[-10.0,10.0],[None,None],[None,None]]
pinTypeList      =["out","out","in","in","in","in", "in", "in","in","in","counterOut","digitalOut"]



#-------INFORMATION ABOUT TRANSITIONS IN THE D2 LINE-------
#some notation that needs to be observed
#  L: lower frequency transition, EX F=2 of 1p1/2 level
#  H: higher frequency transition, EX F=1 of 1p1/2 level
#  F0,F1,F2,F3: the hyperfine states of the 2p3/2 level
#  Source: [1],[4]

Li_2S_F2_F1_Sep=803.504E6   # seperation of 2S1/2 hyperfine levels,hz [1]
Li_D2_Freq=446810184E6 #[1], center value
LiTau=5.86E6 #natural linewidth of transition,  FWHM. [2]

F2F1Ratio=1/.404 #the strength of the F2 transition if greater than the strength of the F1 transition by this factor.
#This number comes from using data from the observation cell, which does not have the sidebands, and fitting the
#profile with a parameter for the ratio of the peaks. This may only be valid with broadening because otherwise there
#are distinct peaks in F=1 and F=2 and the method used here may not apply. If the number A is used to weight each
#hyperfine peak in the F=2 transition, then A/F2F1Ratio must be used for the F=1 transition if doing a multi-peak
#spectral fit


#--Hyperfine structure-----------

#transition probabilities
# Sij signifies relative probability between lower and upper hyperfine state i->j
#Upper 2S_1/2 hyperfine state, F=2->F=1,F=2->F=2,F=2->F=3 [4]
S21=1/20
S22=1/4
S23=7/10
#Lower 2S_1/2 hyperfine state< F=1>F=0,F=1->F=2,F=1->F=3 [4]
S10=1/6
S11=5/12
S12=5/12

#hyperfine seperation of 2P_3/2,hz,[1]. This is the seperation around the 2P_3/2 fine structure line
F0Sep=11.180E6
F1Sep=8.346E6
F2Sep=2.457E6
F3Sep=-6.929E6





#----------Laser scanning constants--------
minScanVal=-3.0 #minimum expected scan value
maxScanVal=3.0 #maximum expected scan value
timePerVolt=.250 #seconds that it should take to scan a volt. This is so the laser doesn't lose lock
               #you can't let it scan too fast
fullScanRange=maxScanVal-minScanVal #the total scan range
stepsPerVoltGalvo=25 #number of steps per volt when taking the laser to or from a value. This is only used in DAQPin
    #class to prevent the galvo from sweeping too fast
samplesPerVoltDAQ=100 #Number of DAQ readings per scanned volt. This is independent of image aquisition, but at every
    #image aquisition an additional point is collected as well.
typicalMHzPerVolt=565.0 #typical MhZ per volt of laser
#-----TYPICAL MHZ/VOLT OF GALVO IS 565--------
#----------



DAQAveragePoints=1000
lambdipPeakSepVolts=2.0 #roughly how Far apart are two lambdip peaks in terms of piezo volts


#--------camera stuff-----------
#imageFormatRound=16 #image size and offset will be integer multiple of this
#cam1SN='b\'44550450\''
#cam1Name='Near'
#cam2SN='b\'CEMAU1827006\''
#cam2Name='Far'
#camSNList=[cam1SN,cam2SN]
#camNameList=[cam1Name,cam2Name]
#
#def get_Camera_SN(name):
#    for i in range(len(camNameList)):
#        if name==camNameList[i]:
#            return camSNList[i]
#    error_Sound()
#    print(name+" does not correlate with a camera serial number!")
#    sys.exit()
#def get_Camera_Name(SN):
#    for i in range(len(camSNList)):
#        if SN==camSNList[i]:
#            return camNameList[i]
#    error_Sound()
#    print(SN+" does not correlate with a camera name!")
#    sys.exit()



#
#
# def piezoWait():
#     time.sleep(.05)
# def etalonWait():
#     time.sleep(.01)
# def error_Sound():
#     ws.Beep(700, 500)
#     ws.Beep(1000, 500)
#     ws.Beep(700, 500)
#     ws.Beep(1000, 500)
#     time.sleep(1) #if there is another noise after this from another error or warning this will prevent them from
#             #sounding like one warning/error
# def begin_Flow_Sound():
#     ws.Beep(750, 500)
# def begin_Sound(noWait=False):
#     ws.Beep(500, 1000)
#     if noWait==False:
#         time.sleep(1) #if there is another noise after this from another error or warning this will prevent them from
#             #sounding like one warning/error
# def finished_Sound(noWait=False):
#     ws.Beep(1100, 1000)
#     if noWait==False:
#         time.sleep(1) #if there is another noise after this from another error or warning this will prevent them from
#             #sounding like one warning/error
#
#
# def warning_Sound(noWait=False):
#     ws.Beep(800, 250)
#     ws.Beep(1200, 250)
#     if noWait==False:
#         time.sleep(.5) #if there is another noise after this from another error or warning this will prevent them from
#             #sounding like one warning/error
#



#-----CITATIONS-------------
#[1]

#[2]  W. I. McAlexander, E. R. I. Abraham, and R. G. Hulet, Phys.
#     Rev. A 54, R5 (1996).

#[3] Borysow thesis

#[4] Rubidium 87 D Line Data, Daniel A. Steck, Los Alamos National Laboratory