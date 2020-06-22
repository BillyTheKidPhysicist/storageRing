import sympy as sym
import copy


#version 1.0. June 22, 2020. William Huntington



elTypeList=['BEND','LENS','DRIFT','COMBINER','INJECTOR']
class Element:

    def __init__(self, PLS,elType, args,defer=False):
        #PLS: Lattice object, could be periodic or linear
        #type :: element type such as bender, lens etc. must be in list elType
        #args: parameters for the element such as length, magnetic field etc.
        #S: the position of the element, based on its center. Can be None
        self.PLS=PLS
        self.deferred=defer
        self.S=None #element's center's position in its respective track
        self.zFunc=None #function that gives the element's center's position in lattice
        self.index=None #position of element in the lattice
        self.elType = elType  # type of element, DRIFT,FOCUS,BEND,COMBINER
        self.Length =   None  # Length of element
        self.Lo=None #distance from object to beginning of the lens. only for injector
        self.rb=None #bending radius, used in bending magnet
        self.Bp = None  # field strength at pole
        self.rp = None  # radius of bore, distance from center to pole face
        self.alpha=None #dipole term of multipole expansion
        self.beta=None #quadrupole term of expansion
        self.angle=None #the bending angle of the magnet
        if defer==True: #comment
            self.M=None
            self.M_Funcz=None
        else:
            self.fill_Variables(args)
            self.M = self.calc_M()  # calculate the matrix representing the element.

            if elType!='INJECTOR': #no need to do this for injector
                self.M_Funcz = self.calc_M_Of_z() #function that returns for given position in element. first argument of
                    #of function is position, second is unpacked arguments such as length or strength of elements
                #self.Force=self.generate_ForceFunc(PLS) #calculate force function


    def update(self):
        #only for drift elements. finishes the process if it was deferred. This is so that elements can be placed
        #and the length of the drift region determined after

        self.M = self.calc_M()  # calculate the matrix representing the element. The drift length is set so now it works
        self.M_Funcz = self.calc_M_Of_z()  # function that returns for given position in element. first argument of
        # of function is position, second is unpacked arguments such as length or strength of elements
        # self.Force=self.generate_ForceFunc(PLS) #calculate force function
    def fill_Variables(self,args):#fill variable
        #args: arguments provided by PLS
        if self.elType==elTypeList[0]: #bend magnet
            self.angle=args[0]
            self.alpha=args[1] #dipole coefficient
            self.beta=args[2]
            self.rb = 1/(self.beta * self.PLS.u0 / (self.PLS.m * self.PLS.v0 ** 2)) # the radius is derived from the bending power
            self.Length=args[0]*self.rb #angle*radius
            self.S=args[3]
        elif self.elType==elTypeList[1]: #lens
            self.Length=args[0]
            self.Bp=args[1]
            self.rp=args[2]
            self.S = args[3]
        elif self.elType==elTypeList[2]: #drift
            self.Length=args[0]
        elif self.elType==elTypeList[3]: #combiner
            self.Length=args[0]
            self.alpha=args[1]
            self.beta=args[2]
            self.S = args[3]
        elif self.elType==elTypeList[4]: #injector
            self.Length=args[0]
            self.Lo=args[1]
            self.Bp=args[2]
            self.rp=args[3]
        else:
            raise Exception('INVALID element NAME PROVIDED!')


    def calc_M_Of_z(self):
        # calculate a function that returns the transformation matrix for the element at a given z
        # where z is distance aPLSng optical axis
        Mnew = self.calc_M( L=sym.symbols('Delta_z'))
        tempList = [sym.symbols('Delta_z')]
        tempList.extend(self.PLS.sympyVarList)  # add the other variables which appear
        Mnew_Func = sym.lambdify(tempList, Mnew, 'numpy')  # make function version
        return Mnew_Func

    def calc_M(self,  L=None):
        # L: The length of the element. By default this is whatever the length property of the element is.
        #  it can be replaced with a variable to make M depend on the length of the element as is done in calc_M_Of_z

        M = sym.zeros(5, 5) #matrix is 5x5. ux,uxd,eta,uy,uyd
        if L==None: #because you can't use self in the args for a function
            L=self.Length
        if self.elType == elTypeList[0]:  # ------------------BEND---------------------------------------
            #combined function magnet with predominatly quadrupole and dipole terms
            # NOTE: the form of the potential here quadrupole does not have the 2. ie Vquad=beta*x*y
            #x component of matrix

            #x component of matrix
            r0 = self.rb
            k0=1/r0 #nominal bending radius. Only present in x direction

            kappaX = k0 ** 2
            phiX = sym.sqrt(kappaX) * L
            M[0,0] = sym.cos(phiX)
            M[0,1] = sym.sin(phiX) / sym.sqrt(kappaX)
            M[1,0] = -sym.sin(phiX) * sym.sqrt(kappaX)
            M[1,1]=sym.cos(phiX)
            M[0,2] = (1 - sym.cos(phiX))*k0 / kappaX # dispersion
            M[1,2] = sym.sin(phiX)*k0 / sym.sqrt(kappaX)
            M[2,2]=1
            #y component
            k= self.PLS.u0 * self.beta ** 2 / (self.alpha * self.PLS.m * self.PLS.v0 ** 2) #there is lensing present in y direction
                #for combind function magnets. See notes
            kappaY=k
            phiY = sym.sqrt(kappaY) * L
            M[3,3] = sym.cos(phiY)
            M[3,4] = sym.sin(phiY) / sym.sqrt(kappaY)
            M[4,3] = -sym.sqrt(kappaY) * sym.sin(phiY)
            M[4,4] = sym.cos(phiY)


        elif self.elType == elTypeList[1]:  #-----------------------------LENS-----------------------------------------------
            #same along both axis
            kappa = 2 * self.PLS.u0 * self.Bp / (self.PLS.m * self.PLS.v0 ** 2 * self.rp ** 2)
            #x component

            phi = sym.sqrt(kappa) * L
            M[0,0] = sym.cos(phi)
            M[0, 1] = sym.sin(phi) / sym.sqrt(kappa)
            M[1, 0] = -sym.sqrt(kappa) * sym.sin(phi)
            M[1, 1] = sym.cos(phi)
            M[2, 2] = 1
            #y component
            M[3,3]=copy.copy(M[0,0]) #this is more robust than just copy(), not all types can use copy and apparently
                                    #sympy floats can't
            M[3,4]=copy.copy(M[0,1])
            M[4,3]=copy.copy(M[1,0])
            M[4,4]=copy.copy(M[1,1])


        elif self.elType == elTypeList[2]:  # ----------------------DRIFT----------------
            #same along both axis
            M[0,0]=1
            M[0,1]=L
            M[1,1]=1
            M[2,2]=1
            M[3,3]=1
            M[3,4]=L
            M[4,4]=1


        elif self.elType == elTypeList[3]:  # ------------------------COMBINER------------------
            # combined function magnet with predominatly quadrupole and dipole terms
            # NOTE: the form of the potential here quadrupole does not have the 2. ie Vquad=beta*x*y
            #x component of matrix
            k0 = self.beta * self.PLS.u0 / (self.PLS.m * self.PLS.v0 ** 2)
            r0 = 1 / k0
            kappaX = k0 ** 2
            phiX = sym.sqrt(kappaX) * L
            M[0,0] = sym.cos(phiX)
            M[0,1] = sym.sin(phiX) / sym.sqrt(kappaX)
            M[1,0] = -sym.sin(phiX) * sym.sqrt(kappaX)
            M[1,1] = sym.cos(phiX)
            M[0,2] = (1 - sym.cos(phiX))*k0 / kappaX # dispersion
            M[1,2] = sym.sin(phiX)*k0/ sym.sqrt(kappaX) #dispersion
            M[2,2]=1

            #y component
            kappaY = self.PLS.u0 * self.beta ** 2 / (self.alpha * self.PLS.m * self.PLS.v0 ** 2)  # 'spring constant' of the magnet
            phiY = sym.sqrt(kappaY) * L
            M[3,3] = sym.cos(phiY)
            M[3,4] = sym.sin(phiY) / sym.sqrt(kappaY)
            M[4,3] = -sym.sqrt(kappaY) * sym.sin(phiY)
            M[4,4] = sym.cos(phiY)

        elif self.elType==elTypeList[4]:  #-------------------INJECTOR---------------
            lensArgs=[self.Length, self.Bp, self.rp,None]
            MLens=Element(self.PLS, 'LENS',args=lensArgs).M
            MLo=Element(self.PLS,'DRIFT',args=[self.Lo,None]).M
            K = 2 * self.PLS.u0 * self.Bp / (self.PLS.m * self.PLS.v0 ** 2 * self.rp ** 2)
            phi = sym.sqrt(K) * L
            Li=(sym.sqrt(K)*self.Lo*sym.cos(phi)+sym.sin(phi))/(K*self.Lo*sym.sin(phi)-sym.sqrt(K)*sym.cos(phi))
            MLi=Element(self.PLS,'DRIFT',args=[Li,None]).M
            M=sym.simplify(MLi@MLens@MLo)

        return M
