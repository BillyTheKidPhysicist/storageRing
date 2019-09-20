#version .1,sep 18th


#more commenting
#generalize the 2d plots like in the other program
#properly decorate plots



import sympy as sym
import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib

import sys



class PeriodicLatticeSolver:
    class Magnet:
        def __init__(self, PeriodicLattice, type,args):
            self.type=type #type of magnet, DRIFT,FOCUS,BEND,COMBINER
            self.Length = None  # to keep track of length
            self.z = sym.symbols('Delta_z')  # variable to hold onto z variable used in self.Mz
            self.params = []  # input parameters. The order needs to be conserved and is extremely important
            self.unpack_Arguments(args) #fill up self.params with the supplied arguments. Handling the sympy objects appropiately
            self.M = self.calc_M(PeriodicLattice)  # calculate the sympy matrix
            self.M_Funcz, self.Mz = self.calc_M_Of_z(PeriodicLattice)
        def unpack_Arguments(self, args):
            for i in range(len(args)):
                if isinstance(args[i], tuple(sym.core.all_classes)): #if the variable is a sympy symbol
                    self.params.append(args[i])
                else:
                    self.params.append(args[i])
            self.Length = self.params[0]
        def calc_M_Of_z(self, PL):
            sympyArguments = []  # collect all the sympy variables in the matrix, including the NEW one self.L.
                                #order is extremely important
            for item in self.params:
                # test to see if the parameter is a sympy variable
                if isinstance(item, tuple(sym.core.all_classes)): #check if it's a sympy object
                    sympyArguments.append(item)
            Mnew = self.calc_M(PL, useZ=True)
            args=[self.z]
            args.extend(PL.sympyVariableList) #add the other variables which appear
            temp1 = sym.lambdify(args, Mnew, 'numpy')  # make function version
            temp2 = Mnew  # sympy version
            return temp1, temp2
        def calc_M(self, PL, useZ=False):
            #PL: outer particle lattice class
            #useZ: wether or not to replace the length of the magnet with the user provided value, or a new value.
            #       this is used to create a matrix function that depends on an arbitray magnet length
            if self.type=='DRIFT': #if drift type magnet
                if useZ == True:
                    L = self.z #length of magnet
                else:
                    L = self.params[0]
                return sym.Matrix([[1, L, 0], [0, 1, 0], [0, 0, 1]])
            if self.type=='FOCUSER': #if focusing magnet, probably a hexapole
                if useZ == True:
                    L = self.z
                else:
                    L = self.params[0]
                Bp = self.params[1] #field strength at pole face, B_pole
                rb = self.params[2] #bore radius, r_bore
                kappa = 2 * PL.u0 * Bp / (PL.m * PL.v0 ** 2 * rb ** 2) #'spring constant' of the magnet
                phi = sym.sqrt(kappa) * L
                C = sym.cos(phi)
                S = sym.sin(phi) / sym.sqrt(kappa)
                Cd = -sym.sqrt(kappa) * sym.sin(phi)
                Sd = sym.cos(phi)
                return sym.Matrix([[C, S, 0], [Cd, Sd, 0], [0, 0, 1]])
            if self.type=='COMBINER': #combiner magnet, ie Stern Gerlacht magnet. Made by Collin. Have yet to characterise fully
                                        #as of now it's basically a square bender, which seems to be mostly true based on Comsol
                                        #except it is square. Rather than add the matrices to account for that, which is small
                                        #I will wait till we characterize it in a general way

                                        #VERY temporarily, it's a drift region
                if useZ == True:
                    L = self.z #length of magnet
                else:
                    L = self.params[0]
                return sym.Matrix([[1, L, 0], [0, 1, 0], [0, 0, 1]])

            if self.type=='BENDER': #bending magnet. Will probably be a quadrupole cut in half, this introduces
                                    #a gradient in the force in addition to the bending force. The total norm of the field is written
                                    #as B0=.5*Bdd*r^2, where Bdd is the second derivative of the field.thus k=Bdd*u0/m*v^2
                if useZ == True:
                    L = self.z
                else:
                    L =self.params[0] * self.params[1]#angle*r
                r0 = self.params[1]
                Bdd = self.params[2]
                k = PL.u0 * Bdd / (PL.m * PL.v0 ** 2)
                k0 = 1 / r0
                kappa = k + k0 ** 2
                phi = sym.sqrt(kappa) * L
                C = sym.cos(phi)
                S = sym.sin(phi)/sym.sqrt(kappa)
                Cd = -sym.sin(phi)*sym.sqrt(kappa)
                Sd = sym.cos(phi)
                D = (1 - sym.cos(phi)) / (r0 * kappa)  # dispersion
                Dd = (sym.sin(phi)) / (r0 * sym.sqrt(kappa))
                return sym.Matrix([[C, S, D], [Cd, Sd, Dd], [0, 0, 1]])




    def __init__(self):
        self.m = 1.16503E-26
        self.u0 = 9.274009E-24
        self.k=1.38064852E-23
        self.v0 = 200
        self.began=False #check if the lattice has begun
        self.lattice = [] #list to hold lattice magnet objects
        self.T=25 #temperature of atoms, mk
        self.delta=np.round(np.sqrt(self.k*(self.T/1000)/self.m)/self.v0,3) #RMS longitudinal velocity spread
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
        el = self.Magnet(self, 'FOCUSER',args)
        self.lattice.append(el)

    def add_Bend(self, ang, r0, Bdd):
        #Ang: angle of bending
        #r0: nominal radius of curvature.
        #Bdd: second derivative of field. This causes focusing during the bending.
                #this is used instead of Bp and rp because the bending region will probably not be a simple revolved hexapole
        self.numElements += 1
        args = [ang, r0, Bdd]
        el = self.Magnet(self,'BENDER', args)
        self.lattice.append(el)

    def add_Drift(self, L):
        #L: length of drift region
        self.numElements += 1
        args = [L]
        el = self.Magnet(self,'DRIFT', args)
        self.lattice.append(el)
    def add_Combiner(self,Ld):
        #Ld: drift region size
        self.combinerIndex=self.numElements
        self.numElements += 1
        L=.187+Ld #fixed length plus drift
        args = [L]
        el = self.Magnet(self,'COMBINER', args)
        self.lattice.append(el)


    def get_Element_Names(self):
        namesList = []
        for item in self.lattice:
            namesList.append(item.type)
        return namesList

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
    def compute_EtaD_From_M(self,M):
        M11 = M[0, 0]
        M12 = M[0, 1]
        M13 = M[0, 2]
        M21 = M[1, 0]
        M22 = M[1, 1]
        M23 = M[1, 2]
        return (M21*M13+M23*(1-M11))/(2-M11-M22)
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
                numPoints=1000
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
                numPoints=1000
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
        print(plotData.shape)
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
    def generate_Envelope_Graph(self,*args):
        totalLengthArray=self.totalLengthArrayFunc(*args)
        zArr,y1=self.compute_Beta_Of_z_Array(*args)
        zArr, y2 = self.compute_Eta_Of_z_Array(*args)
        y3=[]
        for i in range(zArr.shape[0]):
            y3.append(np.trapz(1/(y1[:i])**2,x=zArr[:i])/(2*np.pi))

        #end=totalLengthArray[self.combinerIndex]
        #start=totalLengthArray[self.combinerIndex-1]
        #startIndex=np.argmax(zArr-start>0)
        #endIndex=np.argmax(zArr - end > 0)
        #betaCombiner=np.min(y1[startIndex:endIndex])
        emittance=.001#self.emmitance_Function(betaCombiner)


        #y1=np.sqrt(y1*emittance) #the envelope is the square root of  emittance times the square root of the beta function
        y1=y1*1000 #convert to mm
        y2=y2*self.delta*1000 #dispersion shift is the periodic dispersion times the velocity shift. convert to mm
        fig, ax1 = plt.subplots(figsize=(10,5))
        ax2=ax1.twinx()
        ax3=ax1.twinx()

        ax1.plot(zArr, y1, c='black', label='betatron oscillation')
        ax2.plot(zArr, y2,c='red',alpha=1,linestyle=':',label='RMS dispersion shift')
        #ax3.plot(zArr, y, c='blue', alpha=.5, linestyle='-.', label='envelope')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines + lines2+lines3, labels + labels2+labels3,loc=4)
        ax3.get_yaxis().set_visible(False)


        titleString="Beam envelope in mm\n"
        titleString+="Emittance is "+str(np.round(emittance,6))+". Delta is "+str(self.delta)+'. Total tune is '+str(np.round(y3[-1],2))+'.'
        titleString+=" Time of flight ~"+str(int(1000*zArr[-1]/self.v0))+" ms"
        plt.title(titleString)
        for i in range(self.numElements):
            if i==0:
                center=totalLengthArray[0]/2
            else:
                center=(totalLengthArray[i-1]+totalLengthArray[i])/2
            plt.axvline(totalLengthArray[i],c='black',linestyle=':')
            ax1.text(center,np.max(y1),self.lattice[i].type,rotation=45)
        ax1.set_xlabel('Nominal trajectory distance,m')
        ax1.set_ylabel('Betraton envelope,mm', color='black')
        ax2.set_ylabel('Dispersion shift,mm', color='black')
        plt.show()


    def _compute_Tune_Parallel(self,coords,gridPos,results):
        for i in range(len(coords)):
            try: #in case there is imaginary numbers
                x,y=self.compute_Beta_Of_z_Array(*coords[i],numPoints=200)
                tune = np.trapz(np.power(y, -2), x=x) / (2 * np.pi)
                results.append([gridPos[i], tune])
            except:
                results.append([gridPos[i], np.nan])

    def generate_2D_Resonance_Plot(self,xMin,xMax,yMin,yMax,resonance=1,numPoints=50):
        tuneGrid=self.compute_Tune_Grid(xMin, xMax, yMin, yMax, numPoints=numPoints)
        plotData=self.compute_Resonance_Factor(tuneGrid,resonance)

        plotData=np.transpose(plotData)
        plotData=np.flip(plotData,axis=0)
        masked_array = np.ma.masked_where(plotData == np.nan, plotData)
        cmap = matplotlib.cm.inferno  # Can be any colormap that you want after the cm
        cmap.set_bad(color='grey')
        plt.imshow(masked_array, cmap=cmap,interpolation='bilinear', extent=[xMin, xMax, yMin, yMax],aspect=(xMax-xMin) / (yMax-yMin))
        plt.colorbar()
        plt.show()
    def generate_2D_Tune_Plot(self,xMin,xMax,yMin,yMax,numPoints=25):
        plotData=self.compute_Tune_Grid(xMin,xMax,yMin,yMax,numPoints=numPoints)
        plotData=np.transpose(plotData)
        plotData=np.flip(plotData,axis=0)

        masked_array = np.ma.masked_where(plotData==np.nan, plotData)
        cmap = matplotlib.cm.inferno  # Can be any colormap that you want after the cm
        cmap.set_bad(color='grey')
        plt.imshow(masked_array, cmap=cmap,interpolation='bilinear',extent=[xMin, xMax, yMin, yMax], aspect=(xMax-xMin) / (yMax-yMin))
        plt.colorbar()
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
    def compute_Tune_Grid(self,xMin,xMax,yMin,yMax,numPoints=25):
        # uses parallelism. Tricky
        #a function that generates a grid of the tune for 2 parameters. Rather complicated because it utilized parallel processing
        #It uses the grid of stable solutions to filter out tunes that shouldn't be computed because these would through an error
        #cause imaginary numbers appear, and computing tune is expensive.


        stableCoords = []
        gridPosList = []
        x = np.linspace(xMin, xMax, num=numPoints)
        y = np.linspace(yMin, yMax, num=numPoints)
        stableGrid=self.compute_Stability_Grid(xMin,xMax,yMin,yMax,numPoints=numPoints) #grid of stable solutions to compute tune for
        tuneGrid=np.empty((numPoints,numPoints))
        for i in range(x.shape[0]):
           for j in range(y.shape[0]):
               if(stableGrid[i,j]==True):
                   stableCoords.append([x[i],y[j]])
                   gridPosList.append([i,j])
               else:
                   tuneGrid[i,j]=np.nan #no tune

        #parallel part. Split up tasks and distribute
        processes=mp.cpu_count()
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
            p = mp.Process(target=self._compute_Tune_Parallel, args=(arg1List,arg2List,resultList))
            p.start() #start the process
            jobs.append(p)
        for proc in jobs:
            proc.join()#wait for it to be done and get the results

        for item in resultList:
            i,j=item[0]
            tuneGrid[i,j]=item[1]
        #plt.imshow(tuneGrid)
        #plt.show()
        return tuneGrid




    def compute_Resonance_Factor(self,tune,res):# a factor, between 0 and 1, describing how close a tune is a to a given resonance
    #0 is as far away as possible, 1 is exactly on resonance
    #the precedure is to see how close the tune is to an integer multiples of 1/res. ie a 2nd order resonance(res =2) can occure
    # when the tune is 1,1.5,2,2.5 etc and is maximally far off resonance when the tuen is 1.25,1.75,2.25 etc
    #args--
    #tune: the given tune to analyze
    #res: the order of resonance interested in. 1,2,3,4 etc. 1 is pure bend, 2 is pure focus, 3 is pure corrector etc.
        resFact=1/res #What the remainder is compare to
        tuneRem = tune - tune.astype(int)  # tune remainder, the integer value doens't matter
        tuneResFactArr = 1-np.abs(2*(tuneRem-np.round(tuneRem/resFact)*resFact)/resFact)  # relative nearness to resonances
        return tuneResFactArr

    def _compute_Dispersion_Max_Parallel(self, coords, gridPos, elementIndex, results):
        # parallel helper routine. Processes can finish in random order so it's important to keep track of the order
        #

        # coords: coordinates of the calculation
        # gridPos: position in the plot of the calculation. So I don't need to track the order of solution, just plop the results
        # where it belongs
        for i in range(len(coords)):
            try:  # in case there is imaginary numbers
                betaMin = np.max(self.compute_Eta_Of_z_Array(*coords[i], elementIndex=elementIndex, returnZarr=False))
                results.append([gridPos[i], betaMin])
            except:
                results.append([gridPos[i], np.nan])

    def compute_Dispersion_Max_Grid(self, xMin, xMax, yMin, yMax,whichElement=None,  numPoints=25):
        # function that makes a grid of minimum envelope sizes for a given element
        # whichElement: which element to analyze, 1st,2nd,3rd, etc
        # numPoints: number of points along each axis
        if whichElement!=None:
            elementIndex = whichElement - 1  # to adjust to computer style where 1st is actually zeroth in an array

        stableCoords = []
        gridPosList = []
        x = np.linspace(xMin, xMax, num=numPoints)
        y = np.linspace(yMin, yMax, num=numPoints)
        stableGrid = self.compute_Stability_Grid(xMin, xMax, yMin, yMax,
                                                 numPoints=numPoints)  # grid of stable solutions to compute tune for
        envelopeMinGrid = np.empty((numPoints, numPoints))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                if (stableGrid[i, j] == True):
                    stableCoords.append([x[i], y[j]])
                    gridPosList.append([i, j])
                else:
                    envelopeMinGrid[i, j] = np.nan  # no tune

        processes = mp.cpu_count()
        jobSize = int(len(stableCoords) / processes) + 1  # in case rounding down. will make a small difference
        manager = mp.Manager()
        resultList = manager.list()
        jobs = []

        loop = True
        i = 0
        while loop == True:
            arg1List = []
            arg2List = []
            for j in range(jobSize):
                if i == len(stableCoords):
                    loop = False
                    break
                arg1List.append(stableCoords[i])
                arg2List.append(gridPosList[i])
                i += 1
            p = mp.Process(target=self._compute_Dispersion_Max_Parallel,
                           args=(arg1List, arg2List,whichElement, resultList))
            p.start()
            jobs.append(p)
        for proc in jobs:
            proc.join()

        for item in resultList:
            i, j = item[0]
            envelopeMinGrid[i, j] = item[1]
        return envelopeMinGrid

    def _compute_Beta_Min_Parallel(self, coords, gridPos,elementIndex, results):
        #parallel helper routine. Processes can finish in random order so it's important to keep track of the order
        #
        
        #coords: coordinates of the calculation
        #gridPos: position in the plot of the calculation. So I don't need to track the order of solution, just plop the results
        #where it belongs
        for i in range(len(coords)):
            try:  # in case there is imaginary numbers
                betaMin=np.min(self.compute_Beta_Of_z_Array(*coords[i],elementIndex=elementIndex,returnZarr=False))
                envelope=self.emmitance_Function(betaMin)
                results.append([gridPos[i], betaMin])
            except:
                results.append([gridPos[i], np.nan])
    def compute_Beta_Min_Grid(self,xMin,xMax,yMin,yMax,whichElement,numPoints=25):
        #function that makes a grid of minimum envelope sizes for a given element
        #whichElement: which element to analyze, 1st,2nd,3rd, etc
        #numPoints: number of points along each axis
        
        if whichElement!=None:
            elementIndex = whichElement - 1  # to adjust to computer style where 1st is actually zeroth in an array


        stableCoords = []
        gridPosList = []
        x = np.linspace(xMin, xMax, num=numPoints)
        y = np.linspace(yMin, yMax, num=numPoints)
        stableGrid=self.compute_Stability_Grid(xMin,xMax,yMin,yMax,numPoints=numPoints) #grid of stable solutions to compute tune for
        envelopeMinGrid=np.empty((numPoints,numPoints))
        for i in range(x.shape[0]):
           for j in range(y.shape[0]):
               if(stableGrid[i,j]==True):
                   stableCoords.append([x[i],y[j]])
                   gridPosList.append([i,j])
               else:
                   envelopeMinGrid[i,j]=np.nan #no tune

        processes=mp.cpu_count()
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
            p = mp.Process(target=self._compute_Beta_Min_Parallel, args=(arg1List,arg2List,elementIndex,resultList))
            p.start()
            jobs.append(p)
        for proc in jobs:
            proc.join()

        for item in resultList:
            i,j=item[0]
            envelopeMinGrid[i,j]=item[1]
        return envelopeMinGrid
    def generate_2D_Beta_Min_Plot(self,xMin,xMax,yMin,yMax,whichElement,numPoints=25,useLogScale=False):
        #whichElement: the desired element to find the minimum mover
        #numPoints: number of points to along each axis for the graph.
        #useLogScale: wether to take the base 10 log of the results. Data is kind of hard to see otherwise
        envelopeMinGrid=self.compute_Beta_Min_Grid(xMin,xMax,yMin,yMax,whichElement,numPoints=numPoints)
        if useLogScale==True:
            envelopeMinGrid=np.log10(envelopeMinGrid)
        plotData = np.transpose(envelopeMinGrid)
        plotData = np.flip(plotData, axis=0)
        masked_array = np.ma.masked_where(plotData == np.nan, plotData)
        cmap = matplotlib.cm.inferno_r  # Can be any colormap that you want after the cm
        cmap.set_bad(color='grey')
        plt.imshow(masked_array, cmap=cmap, interpolation='bilinear', extent=[xMin, xMax, yMin, yMax],
                   aspect=(xMax-xMin) / (yMax-yMin))
        plt.colorbar()
        plt.show()
    def generate_2D_Dispersion_Max_Plot(self,xMin,xMax,yMin,yMax,numPoints=25,useLogScale=False,whichElement=None):
        #whichElement: the desired element to find the minimum mover
        #numPoints: number of points to use along each axis for the graph.
        #useLogScale: wether to take the base 10 log of the results. Data is kind of hard to see otherwise
        envelopeMinGrid=self.compute_Dispersion_Max_Grid(xMin,xMax,yMin,yMax,numPoints=numPoints,whichElement=whichElement)
        if useLogScale==True:
            envelopeMinGrid=np.log10(envelopeMinGrid)
        plotData = np.transpose(envelopeMinGrid)
        plotData = np.flip(plotData, axis=0)
        plotData=plotData*self.delta
        plotData=plotData*1000#convert to mm
        #data can be excessively large
        plotData[plotData>500]=500 #clip dispersions over 500mm
        plotData[plotData<.1]=.1 #clip dispersion under .1 mm
        plotData=np.log10(plotData)
        masked_array = np.ma.masked_where(plotData == np.nan, plotData)
        cmap = matplotlib.cm.inferno_r  # Can be any colormap that you want after the cm
        cmap.set_bad(color='grey')
        plt.imshow(masked_array, cmap=cmap, interpolation='bilinear', extent=[xMin, xMax, yMin, yMax],
                   aspect=(xMax-xMin) / (yMax-yMin))
        plt.colorbar()
        plt.show()

if __name__ == '__main__':

    PLS=PeriodicLatticeSolver()
    L=PLS.Variable('L')
    Lm=PLS.Variable('Lm')

    PLS.begin_Lattice()
    PLS.add_Bend(np.pi,1,0)
    PLS.add_Drift(L)
    PLS.add_Focus(Lm,.5,.05)
    PLS.add_Drift(L)
    PLS.add_Bend(np.pi,1,0)
    PLS.add_Drift(L)
    PLS.add_Focus(Lm,.5,.05)
    PLS.add_Drift(L)

    PLS.end_Lattice()

    #PLS.generate_2D_Stability_Plot(.01,.5,.01,.6)
    #PLS.generate_2D_Beta_Min_Plot(.1,.3,.001,.6,4,numPoints=50)
    PLS.generate_Envelope_Graph(.3,.3)
    #x,y=PLS.compute_Beta_Of_z_Array(.2,.4)
    #plt.plot(x,y)
    #plt.show()

    #PLS.generate_2D_Dispersion_Max_Plot(.35,.45,.15,.25,numPoints=50)
    #PLS.generate_2D_Tune_Plot(.38,.42,.18,.22)
    #xArr=np.linspace(.01,1,num=100)
    #yArr=np.linspace(.01,1,num=100)
    #img=np.ones((xArr.shape[0],yArr.shape[0]))
    #for i in range(xArr.shape[0]):
    #    for j in range(yArr.shape[0]):
    #        M=PLS.M_Tot_N(xArr[i],yArr[j])
    #        M11 = M[0, 0]
    #        M12 = M[0, 1]
    #        M21 = M[1, 0]
    #        M22 = M[1, 1]
    #        root=2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2
    #        if root<0:
    #            img[i,j]=0
    #        else:
    #            img[i,j]=1
    #plt.imshow(img)
    #plt.show()

    #PLS.generate_2D_Beta_Min_Plot(.01,1,.01,1,whichElement=4,numPoints=100)
