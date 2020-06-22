import sympy as sym
import numpy as np
from element import Element


#version 1.0. June 22, 2020. William Huntington


class VariableObject:
    # class used to hold information about the variables made by the user when they call PLS.Variable()
    def __init__(self,sympyVar,symbol):
        self.vareMin=None
        self.varMax=None
        self.varInit=None #initial value of variable
        self.symbol=symbol
        self.sympyObject=sympyVar
        self.elIndex=None #index of use of variables. Can be overwritten if reused.
class Injector:
    def __init__(self):
        self.M=None #to hold transfer matrix
        self.MFunc=None #to hold numpy function for matrix
        self.el=None #to hold element object
        self.xi=None #To hold xRMS for input to injector
        self.xdi=None #To hold xDotRMS for input to inject
    def compute_Xf_Xdf_RMS(self,*args):
        M=self.MFunc(*args)
        A=M[0,0]
        C=M[1,0]
        D=M[1,1]
        xRMS=A*self.xi
        xdRMS=C*self.xi +D*self.xdi
        return float(xRMS),float(xdRMS)
class PeriodicLatticeSolver:
    def __init__(self,v0,T,axis=None,catchErrors=True):

        if axis==None:
            self.axis='both'
        else:
            self.axis=axis
        if self.axis!='x' and self.axis!='y' and self.axis!='both': #check wether a valid axis was provided
            raise Exception('INVALID AXIS PROVIDED!!')
        else:
            self.axis=axis #wether we're looking at the x or y axis or both. x is horizontal
        self.v0=v0 #nominal speed of atoms, usually 200 m/s
        self.T=T #temperature of atoms
        self.catchErrors=catchErrors
        self.trackLength=None
        self.m = 1.16503E-26 #mass of lithium 7, SI
        self.u0 = 9.274009E-24 #bohr magneton, SI
        self.kb=1.38064852E-23 #boltzman constant, SI
        self.began=False #check if the lattice has begun
        self.lattice = [] #list to hold lattice magnet objects
        self.bendIndices=[] #list of locations of bends in the lattice
        self.combinerIndex=None #index of combiner's location
        self.delta=np.round(np.sqrt(self.kb*(self.T)/self.m)/self.v0,4) #sigma of velocity spread
        self.numElements = 0 #to keep track of total number of elements. Gets incremented with each additoin
        self.totalLengthListFunc = None  # each entry is the length of that element plus preceding elements.
                                    # arguments are variables declared by user in order user declared them
        self.lengthListFunc = None  # each entry is the length of that element. arguments are variables declared by user
                                    #in order user declared them
        self.sympyVarList = [] #list of sympy object variables. This list is filled in the order that the user
            #declared them
        self.VOList=[] #list of VariableObject objects. These are used to help with interactive plotting
        self.MTot = None #total transfer matrix, can contain sympy symbols
        self.MTotFunc = None#numeric function that returns total transfer matrix based on input arguments
        self.MListFunc=None #returns a list of numeric matrix values
        self.injector=Injector() #to hold the injector object
        self.emittancex=None #to hold emittance value in x direction.
        self.emittancey=None #to hold emittance value in y direction

    def Variable(self, symbol,varMin=0.0,varMax=1.0,varInit=None):
        #function that is called to create a variable to use. Really it jsut adds things to list, but to the user it looks
        #like a variable
        #symbol: string used for sympy symbol
        sympyVar=sym.symbols(symbol)
        VO=VariableObject(sympyVar,symbol)
        VO.varMin=varMin
        VO.varMax=varMax
        if varInit==None:
            VO.varInit=(varMin+varMax)/2
        else:
            VO.varInit=varInit
        self.VOList.append(VO)
        self.sympyVarList.append(sympyVar)
        return VO
    def unpack_VariableObjects(self,args):
    #to unpack sympy from VariableObject before sending to Element class. also does error checking and some other
    #things for VariableObject
        for i in range(len(args)): #extract the sympy object from variable to send to Element class. Also note element index
                                    #for later use
            if isinstance(args[i], VariableObject):
                #check if the variable has been used in another element, that is currently not allowed
                args[i]=args[i].sympyObject
        return args

    def add_Injector(self,L,Lo,Bp,rp,xiRMS=None,xdiRMS=None):
        #add an injector. This doesn't go into the list of elements though
        #L: length of magnet in injection system
        #Lo: distance from image to front of magnet in injection system
        #Bp: magnetic field strength at pole in magnet
        #rp: radius of lens
        args=self.unpack_VariableObjects([L,Lo,Bp,rp])
        self.injector.el=Element(self,'INJECTOR',args=args)
        self.injector.M=self.injector.el.M
        self.injector.xiRMS=xiRMS
        self.injector.xdiRMS=xdiRMS


    def add_Lens(self,L, Bp, rp,S=None):
        #add a magnetic lens, ie hexapole magnet
        #L: Length of lens
        #Bp: field strength at pole face
        #rp: radius of bore inside magnet
        self.numElements += 1
        args = self.unpack_VariableObjects([L, Bp, rp,S])

        el = Element(self, 'LENS',args=args)
        self.lattice.append(el)
        el.index = self.numElements - 1

    def add_Bend(self, angle, alpha,beta,S=None):
        #add a bending magnet. At this point it's some kind of dipole+quadrupole magnet. The radius of curvature
        #is decided by bending power
        #Ang: angle of bending
        #alpha: dipole term
        #beta: quadrupole terms
        self.numElements += 1
        self.bendIndices.append(self.numElements-1)
        args = self.unpack_VariableObjects([angle,alpha,beta,S])
        el = Element(self,'BEND',args=args)
        self.lattice.append(el)
        el.index = self.numElements - 1

    def add_Drift(self, L=None,defer=True):
        #add a drift section.
        #L: length of drift region
        self.numElements += 1
        if L==None:
            defer=True
            el = Element(self, 'DRIFT',[L],defer=defer)
        else:
            args = self.unpack_VariableObjects([L])
            el = Element(self,'DRIFT', args)
        self.lattice.append(el)
        el.index = self.numElements - 1
    def add_Combiner(self,L=.187,alpha=1.01,beta=20,S=None):
        #add combiner magnet. This is the 'collin' magnet
        #L: length of combiner, current length for collin mag is .187
        #alpha: dipole term
        #beta: quadrupole term
        #NOTE: the form of the potential here quadrupole does not have the 2. ie Vquad=beta*x*y
        self.numElements += 1
        args = self.unpack_VariableObjects([L,alpha,beta,S])
        el = Element(self,'COMBINER',args=args)
        self.lattice.append(el)
        self.combinerIndex=self.numElements-1
        el.index = self.numElements - 1

    #def add_Injector(self):


    def compute_M_Total(self): #computes total transfer matric. numeric or symbolic
        M=sym.eye(5) #starting matrix, identity
        for i in range(self.numElements):
            M=self.lattice[i].M @ M #remember, first matrix is on the right!
        return M
    def set_Track_Length(self,value):
        #this is to set the length of the straight awayas between bends. As of now this is the same for each
        # straight away
        self.trackLength=value

    def catch_Errors(self):
        if self.began==False:
            raise Exception("YOU NEED TO BEGIN THE LATTICE BEFORE ENDING IT!")
        if self.trackLength==None:
            raise Exception('TRACK LENGTH WAS NOT SET!!')
        if len(self.bendIndices) == 2:  # can only have two bend as of now
            Exception('the ability to deal with a system with more or less than 2 bends is not implemented')
        if self.lattice[0].elType!='BEND':
            raise Exception('First element must be a bender!')
        #ERROR: Check that there are two benders
        if len(self.bendIndices)!=2:
            raise Exception('There must be 2 benders!!!')
        #ERROR: check that total bending is 360 deg within .1%
        angle=0
        for i in self.bendIndices:
            angle+=self.lattice[i].angle
        if angle>1.01*2*np.pi or angle <.99*2*np.pi:
            raise Exception('Total bending must be 360 degrees within .1%!')
















            #ERROR: the total length of elements is greater than the track length or edges overlap
            ##length1=0
            ##edgeList1 = []
            ##for el in self.lattice[1:self.bendIndices[1]]: #first element is a bend always
            ##    if isinstance(el.Length,numbers.Number): #sometimes the length of the element is declared
            ##        length1+=el.Length
            ##    if isinstance(el.Length,numbers.Number): #sometimes the length of the element may be an undetermind variable
            ##    if isinstance(el.S,numbers.Number) and isinstance(el.Length, numbers.Number): #sometimes
            ##        edgeList1.append(el.S-el.Length/2)
            ##        edgeList1.append(el.S + el.Length / 2)
            ##length2=0
            ##edgeList2 = []
            ##for el in self.lattice[self.bendIndices[1]+1:]:
            ##    if isinstance(el.Length,numbers.Number):
            ##        length2 += el.Length
            ##    if isinstance(el.S, numbers.Number) and isinstance(el.Length, numbers.Number):
            ##        edgeList2.append(el.S - el.Length / 2)
            ##        edgeList2.append(el.S + el.Length / 2)
            ##print(length1,edgeList1)
            ##if length1>self.trackLength or length2>self.trackLength:
            ##    print('Total length greater than track length!!!')
            ##    sys.exit()
            ##temp=0-1E-10
            ##for item in edgeList1:
            ##    if item<temp:
            ##        print('Edges of elements cannot overlap')
            ##        sys.exit()
            ##for item in edgeList2:
            ##    if item<temp:
            ##        print('Edges of elements cannot overlap')
            ##        sys.exit()
    def update_Element_Lengths(self):

        #solve element length and position one track at a time. A track is the sequence of elements between two bends.
        #user must start with the first element being a bend and use only 2 bends

        for el in self.lattice: #manipulate lens elements only
            if el.elType=='LENS': #adjust lens lengths
                if el.index==self.bendIndices[0]+1 or el.index==self.bendIndices[1]+1:#if the lens is right after the bend
                    if el.S==None: #if position is unspecified
                        el.S=el.Length/2
                elif el.index == self.bendIndices[1] - 1 or el.index == self.numElements - 1: #if the lens is right
                        #before the bend
                    if el.S==None: #if position is unspecified
                        el.S=self.trackLength-el.Length/2
        for el in self.lattice: #manipulate drift elements only
            if el.elType=='DRIFT': #-----------adjust the drift lengths
                if el.index==self.bendIndices[0]+1 or el.index==self.bendIndices[1]+1: #edge case for drift right
                        # after bend. The lattice starts with first element as bend so there are two cases here for now
                    edgeR=self.lattice[el.index+1].S-self.lattice[el.index+1].Length/2 #the position of the next element edge reletaive
                        #to first bend end
                    el.Length=edgeR #distance from beginning of drift to edge of element
                elif el.index==self.bendIndices[1]-1: #if the drift is right before the bend
                    edgeL = self.lattice[self.bendIndices[1]-2].S+self.lattice[self.bendIndices[1]-2].Length/2
                                # the distance of the element edge from the beggining of the bend
                    el.Length=self.trackLength-edgeL #distance from previous element end to beginning of bend

                elif el.index==self.numElements-1: #if the drift is right before the end
                    edgeL = self.lattice[-2].S + self.lattice[-2].Length / 2
                    el.Length=self.trackLength-edgeL
                else:
                    edgeL=self.lattice[el.index-1].S+self.lattice[el.index-1].Length/2 #position of prev el edge
                    edgeR = self.lattice[el.index + 1].S-self.lattice[el.index+1].Length/2 #position of next el edge
                    el.Length=edgeR-edgeL
        for el in self.lattice:
            if type(el.Length)==float:
                if el.Length<0:
                    raise Exception("ELEMENT LENGTH IS BEING SET TO NEGATIVE VALUE")


    def begin_Lattice(self): #must be called before making lattice
        self.began=True

    def end_Lattice(self):
        #must be called after ending lattice. Prepares functions that will be used later
        if self.catchErrors==True:
            self.catch_Errors()
        self.update_Element_Lengths()

        for el in self.lattice:
            if el.deferred==True:
                el.update()

        self.MTot = self.compute_M_Total() #sympy version of full transfer function



        self.MTotFunc = sym.lambdify(self.sympyVarList, self.MTot, 'numpy') #MTot function that takes arguments
        self.injector.MFunc=sym.lambdify(self.sympyVarList,self.injector.M,'numpy')


        #this loop does 2 things
        # 1:make  an array function where each entry is the length of the corresping optic.
        #2: make an array function each entry is the sum of the lengths of the preceding optics and that optic. ie, the total length at that point
        temp1 = []#temporary array for case 1
        temp2 = [] #temporary aray for case 2
        for i in range(len(self.lattice)):
            temp1.append(self.lattice[i].Length)
            temp2.append(self.lattice[i].Length)
            for j in range(i): #do the sum up to that point
                temp2[i] += self.lattice[j].Length
            self.lattice[i].zFunc=sym.lambdify(self.sympyVarList,temp2[i]-self.lattice[i].Length/2)
        self.lengthListFunc = sym.lambdify(self.sympyVarList, temp1) #A function that returns a list where
                                                                        # each entry is length of that element

        self.totalLengthListFunc = sym.lambdify(self.sympyVarList, temp2) #each entry is an inclusive cumulative sum
                                                                            #of element lengths

        def tempFunc(*args):
        #takes in arguments and returns a list of transfer matrices
        #args: the variables defined by the user ie the sympyVarList
            tempList=[]
            for el in self.lattice:
                M_N=sym.lambdify(self.sympyVarList, el.M,'numpy')
                tempList.append(M_N(*args))
            return tempList
        self.MListFunc=tempFunc

    def compute_Resonance_Factor(self, tune,res):  # a factor, between 0 and 1, describing how close a tune is a to a given resonance
        # 0 is as far away as possible, 1 is exactly on resonance
        # the procedure is to see how close the tune is to an integer multiples of 1/res. ie a 2nd order resonance(res =2) can occure
        # when the tune is 1,1.5,2,2.5 etc and is maximally far off resonance when the tuen is 1.25,1.75,2.25 etc
        # tune: the given tune to analyze
        # res: the order of resonance interested in. 1,2,3,4 etc. 1 is pure bend, 2 is pure focus, 3 is pure corrector etc.
        resFact = 1 / res  # What the remainder is compare to
        tuneRem = tune%1  # tune remainder, the integer value doens't matter
        tuneResFact =1-np.abs(2*(tuneRem-np.round(tuneRem/resFact)*resFact) / resFact)  # relative nearness to resonances
        return tuneResFact
    def compute_Tune(self,*args):
        #this method is slow, but very accruate because it used default sample size of compute_Beta_Of_z_Array
        #args: input values, sympyVariablesList
        x,y=self.compute_Beta_Of_Z_Array(*args)
        tune=np.trapz(np.power(y,-2),x=x)/(2*np.pi)
        return tune




    def _compute_Lattice_Function_From_M(self,M,funcName,axis):
        #since many latice functions, such as beta,alpha,gamma and eta are calculated in a very similiar was
        #this function saces space by resusing code
        #M: 5x5 transfer matrix, or 3x3 if using x axis or 2x2 if using x axis and no eta
        if axis==None: #can't put self as keyword arg
            axis=self.axis
        def lattice_Func_From_M_Reduced(Mat): #to save space. This simply computes beta over a given 2x2 matrix
            M11 = Mat[0, 0]
            M12 = Mat[0, 1]
            M21 = Mat[1, 0]
            M22 = Mat[1, 1]

            if funcName=='BETA':
                #print('here',2 * M12 / sym.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2))
                return 2 * M12 / sym.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2)
            if funcName=='ETA':
                M13 = Mat[0, 2]
                M23 = Mat[1, 2]
                extraFact = 2  # THIS IS A KEY DIFFERENCE BETWEEN NEUTRAL AND CHARGED PARTICLES!!!
                etaPrime=(M21*M13+M23*(1-M11))/(2-M11-M22)
                return extraFact * (M12*etaPrime+M13)/(1-M11)
            if funcName=='ALPHA':
                return ((M11-M22)/(2*M12))*2 * M12 / sym.sqrt(2 - M11 ** 2 - 2 * M12 * M21 - M22 ** 2)
        if axis=='x':
            return lattice_Func_From_M_Reduced(M[:3,:3])
        elif axis=='y':
            return lattice_Func_From_M_Reduced(M[3:5, 3:5])
        elif axis=='both':
            funcx= lattice_Func_From_M_Reduced(M[:3, :3])
            funcy=lattice_Func_From_M_Reduced(M[3:5, 3:5])
            return funcx,funcy

    def compute_Alpha_From_M(self, M,axis=None):
        if axis==None:
            return self._compute_Lattice_Function_From_M(M,'ALPHA',self.axis)
        else:
            return self._compute_Lattice_Function_From_M(M, 'ALPHA', axis)

    def compute_Eta_From_M(self, M,axis=None):
        if axis == None:
            return self._compute_Lattice_Function_From_M(M, 'Eta', self.axis)
        else:
            return self._compute_Lattice_Function_From_M(M, 'Eta', axis)
    def compute_Beta_From_M(self, M,axis=None):
        if axis == None:
            return self._compute_Lattice_Function_From_M(M, 'Beta', self.axis)
        else:
            return self._compute_Lattice_Function_From_M(M, 'Beta', axis)
    def compute_Beta_Of_Z_Array(self,*args,numpoints=1000,axis=None,elIndex=None,returZarr=True,zArr=None):
        return self._compute_Lattice_Function_Of_z_Array('BETA',numpoints,elIndex,returZarr,zArr,axis,*args)
    def compute_Eta_Of_Z_Array(self,*args,numpoints=1000,axis=None,elIndex=None,returZarr=True,zArr=None):
        return self._compute_Lattice_Function_Of_z_Array('ETA',numpoints,elIndex,returZarr,zArr,axis,*args)
    def compute_Alpha_Of_Z_Array(self,*args,numpoints=1000,axis=None,elIndex=None,returZarr=True,zArr=None):
        return self._compute_Lattice_Function_Of_z_Array('ALPHA',numpoints,elIndex,returZarr,zArr,axis,*args)




    def _compute_Lattice_Function_Of_z_Array(self,funcName,numPoints,elIndex,returnZarr,zArr,axis,*args):
       # computes lattice functions over entire lattice, or single element.
       # args: supplied arguments. this depends on the variables created by user if there are any. Order is critical
       # numPoints: number of points compute. Initially none because it has different behaviour wether the user chooses to
       # compute beta over a single element or the while thing
       # elIndex: which element to compute points at. if none compute over whole lattice
       if axis==None:
           axis=self.axis
       totalLengthList = self.totalLengthListFunc(*args)
       if elIndex == None:  # use entire lattice
           if np.any(zArr==None): #if user wants to use default zArr
                if numPoints == None:  # if user is wanting default value
                    numPoints = 500
                zArr = np.linspace(0, totalLengthList[-1], num=numPoints)
       else:  # compute beta array over specific element
           if zArr!=None: #catch an error
               raise Exception("You can't set z array and compute lattice function over single element")
           if numPoints == None:  # if user is wanting default value
               numPoints = 50
           if elIndex == 0:
               zArr = np.linspace(0, totalLengthList[0], num=numPoints)  # different rule for first element
           else:
               zArr = np.linspace(totalLengthList[elIndex - 1], totalLengthList[elIndex], num=numPoints)

       if axis == 'both':
           latFuncxArr = np.empty(zArr.shape)
           latFuncyArr = latFuncxArr.copy()
           i = 0
           for z in zArr:
               M = self.compute_M_Trans_At_z(z, *args)

               latFuncxArr[i], latFuncyArr[i] = self._compute_Lattice_Function_From_M(M,funcName,axis=axis)
               i += 1
           latFuncArrReturn = [np.abs(latFuncxArr), np.abs(latFuncyArr)]
       else:
           latFuncArr = np.empty(zArr.shape)
           i=0
           for z in zArr:
               M=self.compute_M_Trans_At_z(z, *args)
               latFuncArr[i]=self._compute_Lattice_Function_From_M(M,funcName,axis=axis)
               i+=1
           latFuncArrReturn = np.abs(latFuncArr)
       if returnZarr == True:
           return zArr, latFuncArrReturn
       else:
           return latFuncArrReturn







    def compute_M_Trans_At_z(self, z, *args):
        #TODO: speedup!!!
        totalLengthArray = np.asarray(self.totalLengthListFunc(*args))
        lengthArray = np.asarray(self.lengthListFunc(*args))
        temp = totalLengthArray - z
        index = int(np.argmax(temp >= 0)) #to prevent a typecast warning
        M = self.lattice[index].M_Funcz(totalLengthArray[index] - z,*args)  # starting matrix
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


if __name__ == '__main__':
    PLS = PeriodicLatticeSolver(200,.02,axis='both',catchErrors=False)
    PLS.begin_Lattice()
    PLS.add_Injector(1,1,1,1)
    PLS.add_Drift(1)
    PLS.end_Lattice()