#-improved error handling 
from particleTracing import ParticleTrace
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from magnet import Magnet
import time
import sys


class LinearLatticeSolver:

    def __init__(self,v0=None,T=None):
        if v0==None or T==None:
            raise Exception('YOU DID NOT STATE WHAT V0 and/or T IS')
        else:
            self.v0=v0
            self.T=T
        self.type='LINEAR' #this to so the magnet class knows what's calling it
        self.m=1.1650341e-26
        self.u0=9.274009994E-24


        #use good units for program when possible
        self.mSim = 1#  #mass of lithium, kg
        self.u0Sim = self.u0/self.m #bohr magneton, J/T



        self.k = 1.38064852E-23 #bolztman constant, J/K
        self.began = False  # check if the lattice has begun
        self.lattice = []  # list to hold lattice magnet objects
        self.delta=np.round(np.sqrt(self.k * self.T / self.m) / self.v0,3)  # RMS velocity spread
        self.numElements = 0  # to keep track of total number of elements. Gets incremented with each additoin
        self.totalLengthArrayFunc = None  # a function that recieves values and returns an array
        # of lengths
        self.lengthArrayFunc = None  # length of each element
        self.imageDistanceFunc=None #function that calculates image distance
        self.angularMagnificationFunc=None #function that calculates magnification
        self.imageMagnificationFunc=None #function that calcultes imag magnification
        self.sympyVariableList = []  # list of sympy objects
        self.sympyStringList = []  # list of strings used to represent sympy variables
        self.M_Tot = None  # total transfer matrix, can contain sympy symbols
        self.M_Tot_N = None  # numeric function that returns total transfer matrix based on input arguments

    def add_Focus(self, L, Bp, rp):
        # L: Length of LENS
        # Bp: field strength at pole face
        # rp: radius of bore inside magnet
        self.numElements += 1
        args = [L, Bp, rp]
        el = Magnet(self, 'LENS', args)
        self.lattice.append(el)



    def add_Drift(self, L):
        # L: length of drift region
        self.numElements += 1
        args = [L]
        el = Magnet(self, 'DRIFT', args)
        self.lattice.append(el)


    def get_Element_Names(self):
        namesList = []
        for item in self.lattice:
            namesList.append(item.type)
        return namesList

    def compute_M_Total(self,trimFromEnd=0):  # computes total transfer matric. numeric or symbolic
        M = sym.Matrix([[1, 0], [0, 1]])  # starting matrix, identity
        for i in range(self.numElements-trimFromEnd):
            M = self.lattice[i].M @ M  # remember, first matrix is on the right!
        return M

    def begin_Lattice(self):  # must be called before making lattice
        #v0: nominal atomic velocity
        self.began = True

    def end_Lattice(self):
        # must be called after ending lattice. Prepares functions that will be used later
        if self.began == False:
            raise Exception("YOU NEED TO BEGIN THE LATTICE BEFORE ENDING IT!")
        if self.lattice[-1].type == "DRIFT":
            raise Exception('THE LAST ELEMENT CANNOT BE A DRIFT')
        self.M_Tot = self.compute_M_Total()
        self.M_Tot_N = sym.lambdify(self.sympyVariableList, self.M_Tot,
                                    'numpy')  # numeric version of M_Total that takes in arguments
        # arguments are in the order of sympyVariableList, which is order that they were created

        # make two functions that return 1: an array where each entry is the length of the corresping optic.
        # 2: each entry is the sum of the preceding optics. ie, the total length at that point
        temp = []  # temporary holders
        temp1 = []  # temporary holders
        for i in range(len(self.lattice)):
            temp.append(self.lattice[i].Length)
            temp1.append(self.lattice[i].Length)
            for j in range(i):  # do the sum up to that point
                temp[i] += self.lattice[j].Length
        self.totalLengthArrayFunc = sym.lambdify(self.sympyVariableList, temp,'numpy')
        self.lengthArrayFunc = sym.lambdify(self.sympyVariableList, temp1,'numpy')
        # for image distance and image magnification depends on wether there is a drift region at the end or not

        self.imageDistanceFunc=sym.lambdify(self.sympyVariableList,-self.M_Tot[0,1]/self.M_Tot[1,1],'numpy')
        Li = -self.M_Tot[0, 1] / self.M_Tot[1, 1]
        Mtemp = sym.Matrix([[1, Li], [0, 1], [0, 1]])
        Mtemp = Mtemp @ self.M_Tot
        self.imageMagnificationFunc = sym.lambdify(self.sympyVariableList, Mtemp[0, 0], 'numpy')
        self.angularMagnificationFunc = sym.lambdify(self.sympyVariableList, self.M_Tot[1,1], 'numpy')

    def Variable(self, symbol):
        # function that is called to create a variable to use. Really it jsut adds things to list, but to the user it looks
        # like a variable
        # symbol: string used for sympy symbol
        var = sym.symbols(symbol)
        self.sympyVariableList.append(var)
        self.sympyStringList.append(symbol)

        return var
    def compute_M_Trans_At_z(self, z, *args):
        totalLengthArray = np.asarray(self.totalLengthArrayFunc(*args))
        lengthArray = np.asarray(self.lengthArrayFunc(*args))
        temp = totalLengthArray - z
        index = np.argmax(temp >= 0)
        M = self.lattice[index].M_Funcz(totalLengthArray[index] - z, *args)  # starting matrix
        # calculate from point z to end of lattice
        for i in range(self.numElements - index - 1):
            j = i + index + 1  # current magnet +1 to get index of next magnet
            M = self.lattice[j].M_Funcz(lengthArray[j], *args) @ M
        # from end to point z
        for i in range(index):
            M = self.lattice[i].M_Funcz(lengthArray[i], *args) @ M
        # final step is rest of distance
        if index == 0:  # if the first magnet
            M = self.lattice[index].M_Funcz(z, *args) @ M
        else:  # any other magnet
            M = self.lattice[index].M_Funcz(z - totalLengthArray[index - 1], *args) @ M
        return M
    def compute_Output_Grid(self,xMin,xMax,yMin,yMax,output,numPoints=50):
        #trimUpper: image distances larger than this will be pegged
        #trimLower: image distances smaller than this will be pegged
        xArr = np.linspace(xMin, xMax, num=numPoints)
        yArr = np.linspace(yMin, yMax, num=numPoints)

        coords = np.meshgrid(xArr, yArr)
        if output=='image distance':
            imageDistanceGrid = self.imageDistanceFunc(*coords)
            return imageDistanceGrid
        if output=='angular magnification':
            angularMagnificationGrid = self.angularMagnificationFunc(*coords)
            return angularMagnificationGrid
        if output=='image magnification':
            imageMagnificationGrid = self.imageMagnificationFunc(*coords)
            return imageMagnificationGrid
        else:
            raise Exception("not a valid output!")
    def generate_2D_Output_Plot(self,xMin,xMax,yMin,yMax,output,numPoints=1000,trimUpper=None,trimLower=None):
        plotData = self.compute_Output_Grid(xMin, xMax, yMin, yMax,output, numPoints=numPoints)

        #condition the data
        if output=='image distance':
            if trimUpper==None:
                trimUpper=3
            if trimLower==None:
                trimLower=.01
            oldSettings = np.seterr(invalid='ignore', divide='ignore')  # temporarily numpy ignore errors

            plotData[plotData < 0] = np.nan
            plotData[plotData > trimUpper] = trimUpper  # trim unreasonable values
            plotData[plotData < trimLower] = trimLower  # trim unreasonable values
            plotData = np.log10(plotData)  # take log of the data to make easier to read
            plotData[np.isinf(plotData)] = np.nan  # make infinity values nan. Remember, log(0)=-inf
            np.seterr(**oldSettings)  # return to previous
            titleString = 'Image distance from end of last magnet, log scale \n'
            titleString += "data > " + str(trimLower) + 'm and < ' + str(
                trimUpper) + 'm trimmed.\n Grey region is defocusing'
            cmap = matplotlib.cm.inferno  # Can be any colormap that you want after the cm
        if output=='angular magnification':
            focusedData=self.compute_Output_Grid(xMin, xMax, yMin, yMax,'image distance', numPoints=numPoints)#exclude magnification that
                            # is in a defocusing region. Because this is a thick lens, there could
                            #be negative magnification for both focusing and defocusing
            plotData[focusedData<0]=np.nan
            titleString='Angular magnificaton after last magnet, linear scale \n'
            titleString+='Grey region is defocusing'
            cmap = matplotlib.cm.inferno  # Can be any colormap that you want after the cm
        if output=='image magnification':
            if trimUpper==None:
                trimUpper=10
            if trimLower==None:
                trimLower=.01
            focusedData=self.compute_Output_Grid(xMin, xMax, yMin, yMax,'image distance', numPoints=numPoints)#exclude magnification that
                            # is in a defocusing region. Because this is a thick lens, there could
                            #be negative magnification for both focusing and defocusing
            oldSettings = np.seterr(invalid='ignore', divide='ignore')  # temporarily numpy ignore errors

            plotData[focusedData<0]=np.nan
            plotData=np.abs(plotData)
            plotData[plotData > trimUpper] = trimUpper  # trim unreasonable values
            plotData[plotData < trimLower] = trimLower  # trim unreasonable values
            plotData[np.isinf(plotData)] = np.nan  # make infinity values nan. Remember, log(0)=-inf
            plotData=np.log10(plotData)
            np.seterr(**oldSettings)  # return to previous
            cmap = matplotlib.cm.inferno_r  # Can be any colormap that you want after the cm

        plotData = np.flip(plotData, axis=0)
        masked_array = np.ma.masked_where(plotData == np.nan, plotData)
        plt.figure(figsize=(8, 6))
        plt.grid()
        cmap.set_bad(color='grey')
        xMin = xMin
        xMax = xMax
        yMin = yMin
        yMax = yMax
        plt.imshow(masked_array, cmap=cmap, interpolation='bilinear', extent=[xMin, xMax, yMin, yMax],
                   aspect=xMax / yMax)
        plt.colorpar()

        #plt.title(titleString)
        plt.xlabel(self.sympyStringList[0] + ' m')
        plt.ylabel(self.sympyStringList[1] + ' m')
        plt.show()

    def generate_2D_Image_Distance_Plot(self, xMin, xMax, yMin, yMax, numPoints=1000, trimUpper=3, trimLower=.01,zUnits='cm'):
        self.generate_2D_Output_Plot(xMin, xMax, yMin, yMax, 'image distance',numPoints=numPoints, trimUpper=trimUpper, trimLower=trimLower,zUnits=zUnits)

    def generate_2D_Angular_Magnification_Plot(self, xMin, xMax, yMin, yMax, numPoints=1000, trimUpper=10, trimLower=-10,zUnits=''):
        self.generate_2D_Output_Plot(xMin, xMax, yMin, yMax, 'angular magnification',numPoints=numPoints, trimUpper=trimUpper, trimLower=trimLower,zUnits=zUnits)

    def generate_2D_Image_Magnification_Plot(self, xMin, xMax, yMin, yMax, numPoints=1000, trimUpper=10, trimLower=-10,zUnits=''):
        self.generate_2D_Output_Plot(xMin, xMax, yMin, yMax, 'image magnification',numPoints=numPoints, trimUpper=trimUpper, trimLower=trimLower,zUnits=zUnits)
    def _1D_Array_Splitter(self,x,y): #takes in x and y value arrays and return 4 lists of lists.
        # [[pos y values]..],[[x values for neg values]..],  [[neg y values]..],[[x values for neg values]..]

        posList = []  # list of LIST of positive values
        posCoordList = []  # list of LIST of positive values corresponding x axis values
        negList = []  # list of LIST of negative values
        negCoordList = []  # list of LIST of negative values corresponding x axis values
        neg = False
        pos = False
        currentList = []
        currentCoordList = []
        for i in range(x.shape[0]):
            val = y[i]
            if (val < 0):
                if neg == False:  # if it just switched
                    posList.append(currentList)
                    posCoordList.append(currentCoordList)

                    currentList = []
                    currentCoordList = []
                neg = True
                pos = False
                currentList.append(val)
                currentCoordList.append(x[i])
            else:
                if pos == False:  # if it just switched
                    negList.append(currentList)
                    negCoordList.append(currentCoordList)
                    currentList = []
                    currentCoordList = []

                pos = True
                neg = False
                currentList.append(val)
                currentCoordList.append(x[i])
        # append the last bit, it doesn't get appended in the end
        if neg == True:
            negList.append(currentList)
            negCoordList.append(currentCoordList)
        if pos == True:
            posList.append(currentList)
            posCoordList.append(currentCoordList)

        # one of the list will have an empty first entry, try to remove it, if i can't then don't crash and
        # try the other one
        try:
            posList.remove([])
            posCoordList.remove([])
        except:
            None
        try:
            negList.remove([])
            negCoordList.remove([])
        except:
            None


        return posList,posCoordList,negList,negCoordList


    def generate_1D_Image_Distance_Plot(self,xMin,xMax,numPoints=100,trimUpper=3):
        #xMin: minimum x value
        #xMax: maximum x value
        # numPoints: number of points generate between xMin and xMax
        #trimUpper: values above this will be pegged
        xArr = np.linspace(xMin, xMax, num=numPoints) #x axis array
        imageDistanceArr=self.imageDistanceFunc(xArr) #array to hold image distances
        imageDistanceArr[np.abs(imageDistanceArr)>trimUpper]=trimUpper #trim the image distance array

        #I want to remove negative values because these are diverging beams and are not helpful to use
        #Basically i break the graph down into segments that are either positive valued or negative valued.
        # Then I set the negative ones to zero and paint them red.
        posList, posCoordList, negList, negCoordList=self._1D_Array_Splitter(xArr,imageDistanceArr)


        #add negative values to plot and paint, also add markers
        plt.figure(figsize=(10,5))
        for i in range(len(negList)):
            plt.plot(negCoordList[i],np.zeros(len(negList[i])),c='r')
            plt.axvline(x=negCoordList[i][0],c='black',ymin=0,ymax=.1,linestyle=':')
            plt.axvline(x=negCoordList[i][-1], c='black', ymin=0, ymax=.1, linestyle=':')
        # add pos values to plot

        for i in range(len(posList)):
            posArrSeq=np.asarray(posList[i])
            posArrSeq[posArrSeq>trimUpper]=trimUpper
            plt.plot(posCoordList[i], posArrSeq,c='C0')

        titleString='Image distance relative to back of last magnet\n'
        titleString+='Red regions are defocusing. Maximum image distance clipped to '+str(100*trimUpper)
        plt.title(titleString)
        plt.xlabel(self.sympyStringList[0])
        plt.ylabel('Image distance')
        plt.grid()
        plt.show()
    def generate_1D_Lattice_Length_Plot(self,xMin,xMax,numPoints=100,trimUpper=4):
        #trimUpper: values above this will be pegged
        xArr = np.linspace(xMin, xMax, num=numPoints)
        latticeLengthArr=self.imageDistanceFunc(xArr)#+self.totalLengthArrayFunc(xArr)[-1]
        #latticeLengthArr[latticeLengthArr<0]=0 #to prevent some negative values
        #latticeLengthArr+=self.totalLengthArrayFunc(xArr)[-1] #now add the totalLength



        #I want to remove negative values because these are diverging beams and are not helpful to use
        #Basically i break the graph down into segments that are either positive valued or negative valued.
        # Then I set the negative ones to zero and paint them red.
        posList, posCoordList, negList, negCoordList=self._1D_Array_Splitter(xArr,latticeLengthArr)


        #add negative values to plot and paint, also add markers
        plt.figure(figsize=(10,5))
        for i in range(len(negList)):
            plt.plot(negCoordList[i],np.zeros(len(negList[i])),c='r')
            plt.axvline(x=negCoordList[i][0],c='black',ymin=0,ymax=.1,linestyle=':')
            plt.axvline(x=negCoordList[i][-1], c='black', ymin=0, ymax=.1, linestyle=':')
        # add pos values to plot

        for i in range(len(posList)):
            posArrSeq=np.asarray(posList[i])+self.totalLengthArrayFunc(np.asarray(posCoordList[i]))[-1]
            posArrSeq[posArrSeq>trimUpper]=trimUpper
            plt.plot(posCoordList[i], posArrSeq,c='C0')


        titleString='Total lattice length from beginning of first element\n'
        titleString+='Red regions are defocusing. Maximum image distance clipped to '+str(100*trimUpper)
        plt.title(titleString)
        plt.xlabel(self.sympyStringList[0])
        plt.ylabel('lattice Length')
        plt.grid()
        plt.show()
    def compute_Bp(self,thetaMax=.075,Ld=.6,rp=.05,v0=None):
        if v0==None:
            v0=self.v0
        return .5*self.m*(v0*thetaMax)**2/(self.u0*(1-(Ld*thetaMax/rp)**2))

    def compute_ThetaMax(self,rp=.05,Ld=.6,Bp=.5,v0=None):
        if v0 == None:
            v0 = self.v0
        return rp/(Ld*np.sqrt(1+self.m*(rp*v0)**2/(2*self.u0*Bp*Ld**2)))

    def compute_rp(self,thetaMax=.075,Ld=.6,Bp=.5,v0=None):
        if v0 == None:
            v0 = self.v0
        return thetaMax*Ld/np.sqrt(1-self.m*(v0*thetaMax)**2/(2*self.u0*Bp))
if __name__ == '__main__':

    thetaMax1 = .07

    LLS=LinearLatticeSolver(v0=200,T=.025)
    rp1=LLS.compute_rp(thetaMax=thetaMax1,Bp=.5)
    LLS.begin_Lattice()
    LLS.add_Drift(.6)
    LLS.add_Focus(.384,.5,rp1)
    LLS.end_Lattice()


    thetaMax2=np.abs(LLS.angularMagnificationFunc()*thetaMax1)
    rp2=LLS.compute_rp(thetaMax=thetaMax2,Bp=.5,Ld=.5)
    print(rp1,rp2)


    LLS=LinearLatticeSolver(v0=200,T=.025)
    #Lm=LLS.Variable('Lm')
    LLS.begin_Lattice()
    LLS.add_Drift(.6)
    LLS.add_Focus(.384,.5,rp1)
    LLS.add_Drift(1)
    LLS.add_Focus(.4,.5,rp2)
    LLS.end_Lattice()


    PT = ParticleTrace(LLS,200,0.01)
    print(PT.find_Spot_Sizes(3))
    print(PT.find_RMS_Emittance(3,.18))
    PT.plot_RMS_Envelope(3)
















