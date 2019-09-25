#version .1,sep 18th


#Fix max dispersion!!



import sympy as sym
import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib
import sys
from magnet import Magnet
import time




class PeriodicLatticeSolver:
    def __init__(self,v0=None,T=None):

        if v0==None or T==None:
            print('YOU DID NOT STATE WHAT V0 and/or T IS')
            sys.exit()
        else:
            self.v0=v0
            self.T=T
        self.type = 'PERIODIC'  # this to so the magnet class knows what's calling it
        self.m = 1.16503E-26
        self.u0 = 9.274009E-24
        self.k=1.38064852E-23
        self.began=False #check if the lattice has begun
        self.lattice = [] #list to hold lattice magnet objects
        self.delta=np.round(np.sqrt(self.k*(self.T)/self.m)/self.v0,3) #RMS longitudinal velocity spread
        self.numElements = 0 #to keep track of total number of elements. Gets incremented with each additoin
        self.totalLengthArrayFunc = None  # a function that recieves values and returns an array
        # of lengths
        self.lengthArrayFunc = None  # length of each element
        self.sympyVariableList = [] #list of sympy objects
        self.sympyStringList=[] #list of strings used to represent sympy variables
        self.M_Tot = None #total transfer matrix, can contain sympy symbols
        self.M_Tot_N = None#numeric function that returns total transfer matrix based on input arguments
        self.combinerIndex=None

    def add_Focus(self, L, Bp, rp):
        #L: Length of focuser
        #Bp: field strength at pole face
        #rp: radius of bore inside magnet
        self.numElements += 1
        args = [L, Bp, rp]
        el = Magnet(self, 'LENS',args)
        self.lattice.append(el)

    def add_Bend(self, ang, r0, Bdd):
        #Ang: angle of bending
        #r0: nominal radius of curvature.
        #Bdd: second derivative of field. This causes focusing during the bending.
                #this is used instead of Bp and rp because the bending region will probably not be a simple revolved hexapole
        self.numElements += 1
        args = [ang, r0, Bdd]
        el = Magnet(self,'BEND', args)
        self.lattice.append(el)

    def add_Drift(self, L):
        #L: length of drift region
        self.numElements += 1
        args = [L]
        el = Magnet(self,'DRIFT', args)
        self.lattice.append(el)
    def add_Combiner(self,Ld):
        #Ld: drift region size
        self.combinerIndex=self.numElements
        self.numElements += 1
        L=.187+Ld #fixed length plus drift
        args = [L]
        el = Magnet(self,'COMBINER', args)
        self.lattice.append(el)

    def compute_M_Total(self): #computes total transfer matric. numeric or symbolic
        M=sym.Matrix([[1,0,0],[0,1,0],[0,0,1]]) #starting matrix, identity
        for i in range(self.numElements):
            M=self.lattice[i].M @ M #remember, first matrix is on the right!
        return M

    def begin_Lattice(self): #must be called before making lattice
        self.began=True

    def end_Lattice(self):
        #must be called after ending lattice. Prepares functions that will be used later
        if self.began==False:
            print("YOU NEED TO BEGIN THE LATTICE BEFORE ENDING IT!")
            sys.exit()
        self.M_Tot = self.compute_M_Total()
        self.M_Tot_N = sym.lambdify(self.sympyVariableList, self.M_Tot, 'numpy') #numeric version of M_Total that takes in arguments
                    #arguments are in the order of sympyVariableList, which is order that they were created


        #make two functions that return 1: an array where each entry is the length of the corresping optic.
        #2: each entry is the sum of the preceding optics. ie, the total length at that point
        temp = []#temporary holders
        temp1 = [] #temporary holders
        for i in range(len(self.lattice)):
            temp.append(self.lattice[i].Length)
            temp1.append(self.lattice[i].Length)
            for j in range(i): #do the sum up to that point
                temp[i] += self.lattice[j].Length
        self.totalLengthArrayFunc = sym.lambdify(self.sympyVariableList, temp)
        self.lengthArrayFunc = sym.lambdify(self.sympyVariableList, temp1)

    def compute_Tune(self,*args):
        #args are input values
        #this method is slow, but very accruate because it used default sample size of compute_Beta_Of_z_Array
        x,y=self.compute_Beta_Of_z_Array(*args)
        tune=np.trapz(np.power(y,-2),x=x)/(2*np.pi)
        return tune




    def Variable(self, symbol):
        #function that is called to create a variable to use. Really it jsut adds things to list, but to the user it looks
        #like a variable
        #symbol: string used for sympy symbol
        var=sym.symbols(symbol)
        self.sympyVariableList.append(var)
        self.sympyStringList.append(symbol)
        return var

    def compute_Beta_From_M(self, M):
        M11 = M[0, 0]
        M12 = M[0, 1]
        M21 = M[1, 0]
        M22 = M[1, 1]
        return 2 * M12 / sym.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2)

    def compute_Alpha_From_M(self, M):
        M11 = M[0, 0]
        M12 = M[0, 1]
        M21 = M[1, 0]
        M22 = M[1, 1]
        return (M11-M22)*self.compute_Beta_From_M(M)/(2*M12)
    def compute_Eta_From_M(self, M):
        M11 = M[0, 0]
        M12 = M[0, 1]
        M13 = M[0, 2]
        M21 = M[1, 0]
        M22 = M[1, 1]
        M23 = M[1, 2]
        extraFact=2 #  THIS IS A KEY DIFFERENCE BETWEEN NEUTRAL AND CHARGED PARTICLES!!!
        return extraFact*((1-M22)*M13+M12*M23)/(2-M11-M22)
        #return -extraFact*(M13 - M13 * M22 + M12 * M23) / (M11 + M12 * M21 + M22 - M11 * M22 - 1)

    def compute_Tune_Array(self, *args,numPoints=250):
        temp = []
        x,y=self.compute_Beta_Of_z_Array(*args,numPoints=250)
        for i in range(numPoints):
            integral=np.trapz(y[:i],x=x[:i])
            temp.append(integral)

        return np.asarray(temp)

    def compute_Alpha_Of_z_Array(self, *args,numPoints=1000):
        #args: supplied arguments. this depends on the variables created by user if there are any. Order is critical
        #numPoints: number of points compute
        totalLengthArray = np.asarray(self.totalLengthArrayFunc(*args))
        zArr = np.linspace(0, totalLengthArray[-1], num=numPoints)
        betaList = []
        for z in zArr:
            M = self.compute_M_Trans_At_z(z, *args)
            betaList.append(self.compute_Alpha_From_M(M))
        alphaArr = np.asarray(betaList)
        return zArr, alphaArr

    def compute_Beta_Of_z_Array(self, *args,numPoints=None,elementIndex=None,returnZarr=True):
        #computes beta over entire lattice, or single element.
        #args: supplied arguments. this depends on the variables created by user if there are any. Order is critical
        #numPoints: number of points compute. Initially none because it has different behaviour wether the user chooses to
                    #compute beta over a single element or the while thing
        #elementIndex: which element to compute points for
        totalLengthArray =  self.totalLengthArrayFunc(*args)
        if elementIndex==None:
            if numPoints==None: #if user is wanting default value
                numPoints=500
            zArr = np.linspace(0, totalLengthArray[-1], num=numPoints)
        else:
            if numPoints==None: #if user is wanting default value
                numPoints=50
            if elementIndex==0:
                zArr = np.linspace(0, totalLengthArray[elementIndex], num=numPoints)
            else:
                zArr = np.linspace(totalLengthArray[elementIndex-1], totalLengthArray[elementIndex], num=numPoints)


        betaArr=np.empty(zArr.shape)
        i=0
        for z in zArr:
            M = self.compute_M_Trans_At_z(z, *args)
            betaArr[i]=self.compute_Beta_From_M(M)
            i+=1
        betaArr = np.abs(betaArr)
        if returnZarr==True:
            return zArr, betaArr
        else:
            return betaArr
    def compute_Eta_Of_z_Array(self, *args,numPoints=None,elementIndex=None,returnZarr=True):
        #computes beta over entire lattice, or single element.
        #args: supplied arguments. this depends on the variables created by user if there are any. Order is critical
        #numPoints: number of points compute. Initially none because it has different behaviour wether the user chooses to
                    #compute beta over a single element or the while thing
        #elementIndex: which element to compute points for
        totalLengthArray =  self.totalLengthArrayFunc(*args)
        if elementIndex==None:
            if numPoints==None: #if user is wanting default value
                numPoints=500
            zArr = np.linspace(0, totalLengthArray[-1], num=numPoints)
        else:
            if numPoints==None: #if user is wanting default value
                numPoints=50
            if elementIndex==0:
                zArr = np.linspace(0, totalLengthArray[elementIndex], num=numPoints)
            else:
                zArr = np.linspace(totalLengthArray[elementIndex-1], totalLengthArray[elementIndex], num=numPoints)


        EtaArr=np.empty(zArr.shape)
        i=0
        for z in zArr:
            M = self.compute_M_Trans_At_z(z, *args)
            EtaArr[i]=self.compute_Eta_From_M(M)
            i+=1
        if returnZarr==True:
            return zArr, EtaArr
        else:
            return EtaArr
    def generate_2D_Stability_Plot(self, xMin, xMax, yMin, yMax, numPoints=500):
        plotData=self.compute_Stability_Grid(xMin, xMax, yMin, yMax, numPoints=numPoints)
        plotData=np.transpose(plotData)
        plotData=np.flip(plotData,axis=0)
        plt.imshow(plotData, extent=[xMin,xMax, yMin,yMax], aspect=(xMax-xMin) / (yMax-yMin))
        plt.title("x stability, yellow is STABLE")
        plt.show()


    def compute_M_Trans_At_z(self, z, *args):
        totalLengthArray = np.asarray(self.totalLengthArrayFunc(*args))
        lengthArray = np.asarray(self.lengthArrayFunc(*args))
        temp = totalLengthArray - z
        index = np.argmax(temp >= 0)
        M = self.lattice[index].M_Funcz(totalLengthArray[index] - z,*args)  # starting matrix
        #print(M,index)
        # calculate from point z to end of lattice
        for i in range(self.numElements - index - 1):
            j = i + index + 1  # current magnet +1 to get index of next magnet
            M = self.lattice[j].M_Funcz(lengthArray[j],*args) @ M
        # from beginning to point z
        for i in range(index):
            M = self.lattice[i].M_Funcz(lengthArray[i],*args) @ M
        # final step is rest of distance
        if index == 0:  # if the first magnet
            M = self.lattice[index].M_Funcz(z,*args) @ M
        else:  # any other magnet
            M = self.lattice[index].M_Funcz(z - totalLengthArray[index - 1],*args) @ M
        return M

    def plot_Beta_And_Eta(self,*args):
        self._1D_Plot_Helper("BETA AND ETA",*args)
    def _1D_Plot_Helper(self,plotType,*args,emittance=None): #used to plot different functions witout repetativeness of code
        fig, ax1 = plt.subplots(figsize=(10, 5))
        totalLengthArray = self.totalLengthArrayFunc(*args)
        zArr, y1 = self.compute_Beta_Of_z_Array(*args) #compute beta array
        zArr, y2 = self.compute_Eta_Of_z_Array(*args) #compute periodic dispersion array
        #y3 = [] #to hold tune array
        #for i in range(zArr.shape[0]):
        #    y3.append(np.trapz(1 / (y1[:i]) ** 2, x=zArr[:i]) / (2 * np.pi))
        #y3=np.asarray(y3)

        if plotType=="BETA AND ETA":
            y2 = y2 * 1000  # dispersion shift is the periodic dispersion times the velocity shift. convert to mm
            ax1Name='Beta'
            ax1yLable='Beta, m^2'
            ax2Name='Eta'
            ax2yLable='Eta, mm'
            xLable='Nominal trajectory distance,m'
            titleString='Beta and Eta versus trajectory'
            ax2 = ax1.twinx()
            ax1.plot(zArr, y1, c='black', label=ax1Name)
            ax2.plot(zArr, y2, c='red', alpha=1, linestyle=':', label=ax2Name)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc=4)
            ax1.set_xlabel(xLable)
            ax1.set_ylabel(ax1yLable, color='black')
            ax2.set_ylabel(ax2yLable, color='black')

        #if plotType=="ENVELOPE":




        #ax2 = ax1.twinx()
        #ax3 = ax1.twinx()
#
        #ax1.plot(zArr, y1, c='black', label=ax1Name)
        #ax2.plot(zArr, y2, c='red', alpha=1, linestyle=':', label=ax2Name)
        ## ax3.plot(zArr, y, c='blue', alpha=.5, linestyle='-.', label='envelope')
#
        #lines, labels = ax1.get_legend_handles_labels()
        #lines2, labels2 = ax2.get_legend_handles_labels()
        #lines3, labels3 = ax3.get_legend_handles_labels()
        #ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc=4)
        #ax3.get_yaxis().set_visible(False)

        #titleString = "Beam envelope in mm\n"
        #titleString += "Emittance is " + str(np.round(emittance, 6)) + ". Delta is " + str(
        #    self.delta) + '. Total tune is ' + str(np.round(y3[-1], 2)) + '.'
        #titleString += " Time of flight ~" + str(int(1000 * zArr[-1] / self.v0)) + " ms"

        for i in range(self.numElements):
            if i == 0:
                center = totalLengthArray[0] / 2
            else:
                center = (totalLengthArray[i - 1] + totalLengthArray[i]) / 2
            plt.axvline(totalLengthArray[i], c='black', linestyle=':')
            ax1.text(center, np.max(y1), self.lattice[i].type, rotation=45)
        #ax1.set_xlabel('Nominal trajectory distance,m')
        #ax1.set_ylabel('Betraton envelope,mm', color='black')
        #ax2.set_ylabel('Dispersion shift,mm', color='black')
        plt.title(titleString)
        plt.draw()
        plt.show()





    def compute_Stability_Grid(self, xMin, xMax, yMin, yMax, numPoints=500):
        #uses parallelism. Tricky
        #This returns a grid of stable points. The top left point on the grid is 0,0. Going down rows
        #is going up in x and going across columns is going up in y.
        #TO GET 0,0 ON THE BOTTOM LEFT AND NORMAL BEHAVIOUR, THE OUTPUT MUST BE TRANSPOSED AND FLIPPED ABOUT AXIS=0

        x = np.linspace(xMin, xMax, num=numPoints)
        y = np.linspace(yMin, yMax, num=numPoints)
        inputCoords = np.meshgrid(x, y)

        #stupid trick to get around weird behaviour with lambdify.
        one = sym.symbols('one')
        zero=sym.symbols('zero')
        newMatrix=self.M_Tot.copy()
        for i in range(3):
            for j in range(3):
                if newMatrix[i,j]==0:
                    newMatrix[i,j]=zero
                if newMatrix[i,j]==1:
                    newMatrix[i,j]=one
        newArgs=self.sympyVariableList.copy()
        newArgs.extend([zero,one])
        func=sym.lambdify(newArgs,newMatrix,'numpy')
        zeroArg=np.zeros(numPoints**2).reshape(numPoints,numPoints)
        oneArg = np.ones(numPoints**2).reshape(numPoints,numPoints)
        inputCoords.extend([zeroArg,oneArg])
        matrixArr =func(*inputCoords)

        eigValsArr = np.abs(npl.eigvals(matrixArr.flatten(order='F').reshape(numPoints ** 2, 3, 3))) #some tricky shaping
        results = ~np.any(eigValsArr > 1 + 1E-5, axis=1)
        stabilityGrid=results.reshape((numPoints, numPoints))
        return stabilityGrid
    

    def compute_Resonance_Factor(self,tune,res):# a factor, between 0 and 1, describing how close a tune is a to a given resonance
    #0 is as far away as possible, 1 is exactly on resonance
    #the precedure is to see how close the tune is to an integer multiples of 1/res. ie a 2nd order resonance(res =2) can occure
    # when the tune is 1,1.5,2,2.5 etc and is maximally far off resonance when the tuen is 1.25,1.75,2.25 etc
    #tune: the given tune to analyze
    #res: the order of resonance interested in. 1,2,3,4 etc. 1 is pure bend, 2 is pure focus, 3 is pure corrector etc.
        resFact=1/res #What the remainder is compare to
        tuneRem = tune - tune.astype(int)  # tune remainder, the integer value doens't matter
        tuneResFactArr = 1-np.abs(2*(tuneRem-np.round(tuneRem/resFact)*resFact)/resFact)  # relative nearness to resonances
        print('here')
        return tuneResFactArr



    def plot_Beta_Min_2D(self,xMin,xMax,yMin,yMax,elementIndex,useLogScale=False,numPoints=100,trim=2.5):
        self._2D_Plot_Parallel_Helper(xMin,xMax,yMin,yMax,'BETA MIN',elementIndex,useLogScale,numPoints,trim,None)
    def plot_Dispersion_Min_2D(self,xMin,xMax,yMin,yMax,elementIndex=None,useLogScale=False,numPoints=100,trim=100):
        self._2D_Plot_Parallel_Helper(xMin,xMax,yMin,yMax,'DISPERSION MIN',elementIndex,useLogScale,numPoints,trim,None)
    def plot_Tune_2D(self,xMin,xMax,yMin,yMax,useLogScale=False,numPoints=50):
        self._2D_Plot_Parallel_Helper(xMin,xMax,yMin,yMax,'TUNE',None,useLogScale,numPoints,None,None)
    def plot_Resonance_2D(self,xMin,xMax,yMin,yMax,resonance=1,numPoints=50,useLogScale=False):
        self._2D_Plot_Parallel_Helper(xMin,xMax,yMin,yMax,'RESONANCE',None,useLogScale,numPoints,None,resonance)
    def plot_Dispersion_Max_2D(self,xMin,xMax,yMin,yMax,elementIndex=None,useLogScale=False,numPoints=100,trim=None):
        self._2D_Plot_Parallel_Helper(xMin,xMax,yMin,yMax,'DISPERSION MAX',elementIndex,useLogScale,numPoints,trim,None)

    def _2D_Plot_Parallel_Helper(self,xMin,xMax,yMin,yMax,output,elementIndex,useLogScale,numPoints,trim,resonance):

        plotData = self._compute_Grid_Parallel(xMin, xMax, yMin, yMax,output,elementIndex=elementIndex, numPoints=numPoints)
        plotData = np.transpose(plotData) #need to reorient the data so that 0,0 is bottom left
        plotData = np.flip(plotData, axis=0) #need to reorient the data so that 0,0 is bottom left




        if trim!=None: #if plot values are clipped pegged above some value
            titleExtra=' \n Data clipped at a values greater than '+str(trim)
        else:
            titleExtra=''
        if output=='BETA MIN':
            plotData[plotData>=trim]=trim #trim values
            cmap = matplotlib.cm.inferno_r  # Can be any colormap that you want after the cay
            title='Beta,m^2.'
        if output=="DISPERSION MIN":
            plotData = plotData *1000 #convert to mm
            if trim!=None:
                plotData[plotData>=trim]=trim #trim values
            cmap = matplotlib.cm.inferno_r  # Can be any colormap that you want after the cm
            title='Disperison Minimum, mm.'
        if output=="DISPERSION MAX":
            plotData = plotData *1000 #convert to mm
            if trim!=None:
                plotData[plotData>=trim]=trim #trim values
            cmap = matplotlib.cm.inferno_r  # Can be any colormap that you want after the cm
            title='Disperison Maximum, mm.'
        if output=="TUNE":
            cmap = matplotlib.cm.inferno  # Can be any colormap that you want after the cm
            title='tune'
        if output=='RESONANCE':
            plotData=self.compute_Resonance_Factor(plotData,resonance) #compute the tune
            cmap = matplotlib.cm.inferno  # Can be any colormap that you want after the cm
            title='Resonances of order '+str(resonance)+'. \n 1.0 indicates exactly on resonance, 0.0 off'

        if useLogScale == True:
            plotData = np.log10(plotData)
        masked_array = np.ma.masked_where(plotData == np.nan, plotData) #a way of marking ceratin values to have different colors

        plt.subplots(figsize=(8,8)) #required to plot multiple plots
        plt.grid()
        cmap.set_bad(color='grey')#set the colors of values in masked array
        plt.title(title+titleExtra)
        plt.imshow(masked_array, cmap=cmap, extent=[xMin, xMax, yMin, yMax],aspect=(xMax - xMin) / (yMax - yMin))
        plt.colorbar()
        plt.xlabel(self.sympyStringList[0])
        plt.ylabel(self.sympyStringList[1])
        plt.draw()
        plt.pause(.01)
        plt.show()



    def _compute_Grid_Parallel(self,xMin, xMax, yMin, yMax,output,elementIndex=None, numPoints=100):
        stableCoords = []
        gridPosList = []
        x = np.linspace(xMin, xMax, num=numPoints)
        y = np.linspace(yMin, yMax, num=numPoints)
        stableGrid=self.compute_Stability_Grid(xMin,xMax,yMin,yMax,numPoints=numPoints) #grid of stable solutions to compute tune for
        outputGrid=np.empty((numPoints,numPoints))
        for i in range(x.shape[0]):
           for j in range(y.shape[0]):
               if(stableGrid[i,j]==True):
                   stableCoords.append([x[i],y[j]])
                   gridPosList.append([i,j])
               else:
                   outputGrid[i,j]=np.nan #no tune

        processes=mp.cpu_count()-1
        jobSize=int(len(stableCoords)/processes)+1 #in case rounding down. will make a small difference
        manager = mp.Manager()
        resultList = manager.list()
        jobs=[]

        loop=True
        i=0
        while loop==True:
            arg1List=[]
            arg2List=[]
            for j in range(jobSize):
                if i==len(stableCoords):
                    loop=False
                    break
                arg1List.append(stableCoords[i])
                arg2List.append(gridPosList[i])
                i+=1
            p = mp.Process(target=self._parallel_Helper, args=(arg1List,arg2List,elementIndex,resultList,output))
            p.start()
            jobs.append(p)
        for proc in jobs:
            proc.join()

        for item in resultList:
            i,j=item[0]
            outputGrid[i,j]=item[1]
        return outputGrid

    def _parallel_Helper(self, argList, gridPos, elementIndex, results,output):
        if output=='BETA MIN':
            for i in range(len(argList)):
                try:  # in case there is imaginary numbers
                    betaMin = np.min(self.compute_Beta_Of_z_Array(*argList[i], elementIndex=elementIndex, returnZarr=False))
                    results.append([gridPos[i], betaMin])
                except:
                    results.append([gridPos[i], np.nan])
        if output=="DISPERSION MIN":
            for i in range(len(argList)):
                try:  # in case there is imaginary numbers
                    dispersionMin = self.delta*np.min(np.abs(self.compute_Eta_Of_z_Array(*argList[i], elementIndex=elementIndex, returnZarr=False)))
                    #print(dispersionMin)
                    results.append([gridPos[i], dispersionMin])
                except:
                    results.append([gridPos[i], np.nan])
        if output=="DISPERSION MAX":
            for i in range(len(argList)):
                try:  # in case there is imaginary numbers
                    dispersionMax = self.delta*np.max(np.abs(self.compute_Eta_Of_z_Array(*argList[i], elementIndex=elementIndex, returnZarr=False)))
                    #print(dispersionMin)
                    results.append([gridPos[i], dispersionMax])
                except:
                    results.append([gridPos[i], np.nan])
        if output=="TUNE" or output=="RESONANCE":
            for i in range(len(argList)):
                try: #in case there is imaginary numbers
                    x,y=self.compute_Beta_Of_z_Array(*argList[i],numPoints=200)
                    tune = np.trapz(np.power(y, -1), x=x) / (2 * np.pi)
                    results.append([gridPos[i], tune])
                except:
                    results.append([gridPos[i], np.nan])

if __name__ == '__main__':

    PLS=PeriodicLatticeSolver(v0=200,T=.001)
    print(PLS.delta)
    Lm=PLS.Variable('Lm')
    Bp=PLS.Variable('Bp')

    PLS.begin_Lattice()
    PLS.add_Bend(np.pi,1,0)
    PLS.add_Drift(.05)
    PLS.add_Focus(Lm,Bp,.05)
    PLS.add_Combiner(.5)
    PLS.add_Focus(Lm,Bp,.05)
    PLS.add_Drift(.05)
    PLS.add_Bend(np.pi,1,0)
    #PLS.add_Focus(.25,Bp,.05)
    #PLS.add_Bend(np.pi,1,0)
    PLS.add_Drift(.1+2*Lm+.5)



    PLS.end_Lattice()





    #PLS.plot_Dispersion_Min_2D(*args,numPoints=numPoints,elementIndex=0)
    #PLS.plot_Dispersion_Min_2D(*args,numPoints=numPoints,elementIndex=elementIndex)
    #PLS.plot_Dispersion_Max_2D(*args,numPoints=numPoints,trim=1000)
    #PLS.plot_Dispersion_Min_2D(*args,numPoints=numPoints,elementIndex=1)
    PLS.plot_Beta_And_Eta(.4,.4)

    args=(.1,.5,1E-3,.5)
    numPoints=100
    elementIndex=3
    #PLS.plot_Beta_Min_2D(*args,numPoints=numPoints,elementIndex=elementIndex)
    #PLS.plot_Dispersion_Min_2D(*args,numPoints=numPoints,elementIndex=elementIndex)
    #PLS.plot_Dispersion_Max_2D(*args,numPoints=numPoints,trim=1000)
    #PLS.plot_Resonance_2D(*args,numPoints=numPoints,resonance=1)
    #PLS.plot_Resonance_2D(*args,numPoints=numPoints,resonance=2)
    #PLS.plot_Resonance_2D(*args,numPoints=numPoints,resonance=3)


