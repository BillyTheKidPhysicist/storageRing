
import sympy as sym
import copy
import sympy.utilities.autowrap as symWrap
import sys
import numpy as np

#version 1.0. June 22, 2020. William Huntington




class Element:

    def __init__(self, PLS,elType, args,defer=False,injectorComponent=False):
        #PLS: Lattice object, could be periodic or linear
        #type :: element type such as bender, lens etc. must be in list elType
        #args: parameters for the element such as length, magnetic field etc.
        #S: the position of the element, based on its center. Can be None
        self.PLS=PLS
        self.deferred=defer
        self.S=None #element's center's position in its respective track
        self.index=None #position of element in the lattice
        self.elType = elType  # type of element, DRIFT,FOCUS,BEND,COMBINER
        self.Length =   None  # Length of element
        self.Lo=None #distance from object to beginning of the lens. only for injector
        self.r0=None #bending radius, used in bending magnet and combiner
        self.Bp = None  # field strength at pole
        self.rp = None  # radius of bore, distance from center to pole face
        self.alpha=None #dipole term of multipole expansion
        self.beta=None #quadrupole term of expansion
        self.angle=None #the bending angle of the magnet
        self.Li=None #to hold the sympy expression for the image distance of the injector
        self.rpFunc=None #to hold the function that returns rp for the bender
        self.rtFunc = None  # radius of vacuum tube in element.
        self.elTypeList = ['BEND', 'LENS', 'DRIFT', 'COMBINER', 'INJECTOR']
        self.zVar=None #sympy object to hold z
        self.M = None
        self.Mz= None #matrix with length replaced by sympy z object
        self.M_Funcz = None #returns transfer matric for the element as a function of position in the element
        self.M_Func=None #returns total transfer matrix for the element
        self.apxFunc=None #apeture in x plane
        self.apyFunc=None #apeture in y plane
        self.fill_Variables(args)
        if injectorComponent==True or elType=='INJECTOR': #if the component is part of the injector only compute the matrix and do it
                #it without having to update
            self.M = self.calc_M()  # calculate the matrix representing the element.



    def fill_Params_And_Functions(self):
        # first argument of function is position, second is unpacked arguments such as length
        # or strength of elements
        #only for drift elements. finishes the process if it was deferred. This is so that elements can be placed
        #and the length of the drift region determined after
        if self.elType=='BEND': #this is so that a undetermined value, None, can be entered for the bending angle by the
            #user. Then those angles can be massaged to fit constraints and then that new angle can be used to compute the
            #rest of the elements parameters and functions
            self.Length=self.angle*self.r0
        self.zVar = sym.symbols('z', real=True, positive=True)
        self.Mz = self.calc_M(L=self.zVar)
        self.M = self.Mz.subs(self.zVar, self.Length)
        tempList = [self.zVar]
        tempList.extend(self.PLS.sympyVarList)  # add the other variables which appear
        self.M_Funcz=symWrap.autowrap(self.Mz,args=tempList)
        self.M_Func = symWrap.autowrap(self.M, args=self.PLS.sympyVarList)
        if self.elType=='LENS':
            self.rpFunc = symWrap.autowrap(self.rp, args=self.PLS.sympyVarList)
        if self.elType=='BEND':
            self.rpFunc = symWrap.autowrap(self.rp, args=self.PLS.sympyVarList)
            self.rtFunc = symWrap.autowrap(self.rp / 2,args=self.PLS.sympyVarList)  # vacuum tube radius is half of bore radius
        # first argument of function is position, second is unpacked arguments such as length
        # or strength of elements
    def fill_Variables(self,args):#fill variable
        #args: arguments provided by PLS
        if self.elType==self.elTypeList[0]: #bend magnet
            self.angle=args[0]
            self.r0 = args[1]
            self.Bp=args[2]
            self.S=args[3]
        elif self.elType==self.elTypeList[1]: #lens
            self.Length=args[0]
            self.Bp=args[1]
            self.rp=args[2]
            self.S = args[3]
        elif self.elType==self.elTypeList[2]: #drift
            self.Length=args[0]
        elif self.elType==self.elTypeList[3]: #combiner
            self.Length=args[0]
            self.alpha=args[1]
            self.beta=args[2]
            self.S = args[3]
            self.r0=args[4]
        elif self.elType==self.elTypeList[4]: #injector
            self.Length=args[0]
            self.Lo=args[1]
            self.Bp=args[2]
            self.rp=args[3]
        else:
            raise Exception('INVALID element NAME PROVIDED!')




    def calc_M(self,  L=None):
        # L: The length of the element. By default this is whatever the length property of the element is.
        #  it can be replaced with a variable to make M depend on the length of the element
        # TODO: remove the need to come through here twice.
        M = sym.zeros(5, 5) #matrix is 5x5. ux,eta,uy
        if L==None: #because you can't use self in the args for a function
            L=self.Length
        if self.elType == self.elTypeList[0]:  # ------------------BEND---------------------------------------
            #combined function magnet with predominatly quadrupole and dipole terms
            # NOTE: the form of the potential here quadrupole does not have the 2. ie Vquad=beta*x*y
            #x component of matrix
            k0=1/self.r0
            self.rp=self.r0*(self.PLS.u0*self.Bp/(2*.5*self.PLS.m * self.PLS.v0 ** 2)) #from the theory of what the maximum bore is
            k =2 * self.PLS.u0 * self.Bp / (self.PLS.m * self.PLS.v0 ** 2 * self.rp ** 2)
            kappa=k0**2+k
            phi = sym.sqrt(kappa) * L
            M[0, 0] = sym.cos(phi)
            M[0, 1] = sym.sin(phi) / sym.sqrt(kappa)
            M[1, 0] = -sym.sqrt(kappa) * sym.sin(phi)
            M[1, 1] = sym.cos(phi)
            M[2, 2] = 1
            M[0,2] = (1 - sym.cos(phi))*k0 / kappa # dispersion
            M[1,2] = sym.sin(phi)*k0 / sym.sqrt(kappa)



            # y component
            kappa = k
            phi = sym.sqrt(kappa) * L
            M[3, 3] = sym.cos(phi)
            M[3, 4] = sym.sin(phi) / sym.sqrt(kappa)
            M[4, 3] = -sym.sqrt(kappa) * sym.sin(phi)
            M[4, 4] = sym.cos(phi)










            #the following is for when I was using the combined function idea
            ##x component of matrix
            #r0 = self.rb
            #k0=1/r0 #nominal bending radius. Only present in x direction
#
            #kappaX = k0 ** 2
            #phiX = sym.sqrt(kappaX) * L
            #M[0,0] = sym.cos(phiX)
            #M[0,1] = sym.sin(phiX) / sym.sqrt(kappaX)
            #M[1,0] = -sym.sin(phiX) * sym.sqrt(kappaX)
            #M[1,1]=sym.cos(phiX)
            #M[0,2] = (1 - sym.cos(phiX))*k0 / kappaX # dispersion
            #M[1,2] = sym.sin(phiX)*k0 / sym.sqrt(kappaX)
            #M[2,2]=1
            ##y component
            #k= self.PLS.u0 * self.beta ** 2 / (self.alpha * self.PLS.m * self.PLS.v0 ** 2) #there is lensing present in y direction
            #    #for combind function magnets. See notes
            #kappaY=k
            #phiY = sym.sqrt(kappaY) * L
            #M[3,3] = sym.cos(phiY)
            #M[3,4] = sym.sin(phiY) / sym.sqrt(kappaY)
            #M[4,3] = -sym.sqrt(kappaY) * sym.sin(phiY)
            #M[4,4] = sym.cos(phiY)


        elif self.elType == self.elTypeList[1]:  #-----------------------------LENS-----------------------------------------------
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


        elif self.elType == self.elTypeList[2]:  # ----------------------DRIFT----------------
            #same along both axis
            M[0,0]=1
            M[0,1]=L
            M[1,1]=1
            M[2,2]=1
            M[3,3]=1
            M[3,4]=L
            M[4,4]=1


        elif self.elType == self.elTypeList[3]:  # ------------------------COMBINER------------------
            # combined function magnet with predominatly quadrupole and dipole terms
            # NOTE: the form of the potential here quadrupole does not have the 2. ie Vquad=beta*x*y
            #x component of matrix
            k0 = self.beta * self.PLS.u0 / (self.PLS.m * self.PLS.v0 ** 2)
            self.r0 = 1 / k0
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

        elif self.elType==self.elTypeList[4]:  #-------------------INJECTOR---------------
            #this all checks out, I've triple checked it now
            lensArgs=[self.Length, self.Bp, self.rp,None]
            MLens=Element(self.PLS, 'LENS',args=lensArgs,injectorComponent=True).M
            MLo=Element(self.PLS,'DRIFT',args=[self.Lo,None],injectorComponent=True).M
            K = 2 * self.PLS.u0 * self.Bp / (self.PLS.m * self.PLS.v0 ** 2 * self.rp ** 2)
            phi = sym.sqrt(K) * L
            self.Li=(sym.sqrt(K)*self.Lo*sym.cos(phi)+sym.sin(phi))/(K*self.Lo*sym.sin(phi)-sym.sqrt(K)*sym.cos(phi))
            MLi = Element(self.PLS, 'DRIFT', args=[self.Li, None], injectorComponent=True).M

            M=MLi@MLens@MLo
            #print(lensArgs[2].subs(lensArgs[1],.45).subs(lensArgs[0],.124).subs(self.Lo,.2))
            #print(M.subs(lensArgs[2],.0208).subs(lensArgs[1],.45).subs(lensArgs[0],.124).subs(self.Lo,.2))
            #print(MLi.subs(lensArgs[2],.0235).subs(lensArgs[1],.45).subs(lensArgs[0],.124).subs(self.Lo,.2))
            #print(self.Li.subs(lensArgs[2],.0235).subs(lensArgs[1],.45).subs(lensArgs[0],.124).subs(self.Lo,.2))
            #sys.exit()
        return sym.simplify(M)
