import sympy as sym
import sys

#-added else statement to catch no magnet name
#-added paraemter axis to track which axis of betatron oscillations
#-added if to change transfer matrix depending on which axis
#-improved error handling 

magType=['BEND','LENS','DRIFT','COMBINER']
class Magnet:

    def __init__(self, LO, axis,type, args):
        #LO: Lattice object, could be periodic or linear
        #type :: magnet type such as bender, lens etc. must be in list magType
        #axis: wether to look at physics along x or y axis. Matters for combiner and bender. must be 'x' or 'y'
                #x is horizontal
        #args: parameters for the magnet such as length, magnetic field etc.
        self.type = type  # type of magnet, DRIFT,FOCUS,BEND,COMBINER
        self.axis=axis #x or y direction
        self.Length = None  # Length of magnet
        self.rb=None #bending radius, used in bending magnet
        self.Bp = None  # field strength at pole
        self.rp = None  # radius of bore, distance from center to pole face
        self.Bdd=None #second derivative of field, used in bending magnet
        self.fill_Variables(args)

        self.M = self.calc_M(LO)  # calculate the matrix representing the element.
        self.M_Funcz = self.calc_M_Of_z(LO)
    def fill_Variables(self,args):#fill variable
        #args: arguments provided by LO
        if self.type==magType[0]: #bender
            self.Length=args[0]*args[1] #angle*radius
            self.rb=args[1]
            self.Bdd=args[2]
        elif self.type==magType[1]: #lens
            self.Length=args[0]
            self.Bp=args[1]
            self.rp=args[2]
        elif self.type==magType[2]: #drift
            self.Length=args[0]
        elif self.type==magType[3]: #combiner
            self.Length=args[0]
        else:
            raise Exception('INVALID MAGNET NAME PROVIDED!')

    def calc_M_Of_z(self,LO):  # calculate a function that returns the transformation matrix for the element at a given z
        # where z is distance along optical axis
        # LL: outer Linear Lattice class
        Mnew = self.calc_M(LO, L=sym.symbols('Delta_z'))
        tempList = [sym.symbols('Delta_z')]
        tempList.extend(LO.sympyVariableList)  # add the other variables which appear
        Mnew_Func = sym.lambdify(tempList, Mnew, 'numpy')  # make function version
        return Mnew_Func

    def calc_M(self, LO, L=None): # PL: outer Linear Lattice class
        # L: By default L is the user provided length, but it can tbe changed to generate a matrix that depends
        # on an arbitray length. This is used to calculate beta/alpha/eta etc as a function of z
        if L==None: #because you can't use self in the args for a function
            L=self.Length
        if self.type == magType[0]:  # ------------------BEND---------------------------------------
            # Will probably be a quadrupole cut in half, this introduces
            # a gradient in the force in additself.ion to the bending force. The total norm of the field is written
            # as B0=.5*Bdd*r^2, where Bdd is the second derivative of the field.thus k=Bdd*u0/m*v^2. needs to be looked
            #at again
            if self.axis=='x': #along x there is bending and thus some focusing
                k0 = 1 / self.rb
                kappa = k0 ** 2
                phi = sym.sqrt(kappa) * L
                C = sym.cos(phi)
                S = sym.sin(phi) / sym.sqrt(kappa)
                Cd = -sym.sin(phi) * sym.sqrt(kappa)
                Sd = sym.cos(phi)
                D = (1 - sym.cos(phi)) / (self.rb * kappa)  # dispersion
                Dd = (sym.sin(phi)) / (self.rb * sym.sqrt(kappa))
                M= sym.Matrix([[C, S, D], [Cd, Sd, Dd], [0, 0, 1]])
            else: #if y. along y it is a drift section
                M = sym.Matrix([[1, L, 0], [0, 1, 0], [0, 0, 1]])
        elif self.type == magType[1]:  #-----------------------------LENS-----------------------------------------------
            #same along both axis
            kappa = 2 * LO.u0 * self.Bp / (LO.m * LO.v0 ** 2 * self.rp ** 2)  # 'spring constant' of the magnet
            phi = sym.sqrt(kappa) * L
            C = sym.cos(phi)
            S = sym.sin(phi) / sym.sqrt(kappa)
            Cd = -sym.sqrt(kappa) * sym.sin(phi)
            Sd = sym.cos(phi)
            M= sym.Matrix([[C, S, 0], [Cd, Sd, 0], [0, 0, 1]])
        elif self.type == magType[2]:  # ----------------------DRIFT----------------
            #same along both axis
            M= sym.Matrix([[1, L, 0], [0, 1, 0], [0, 0, 1]])
        elif self.type == magType[3]:  # ------------------------COMBINER------------------
            alpha=17.4 #quadrupole parameter, T/m
            B0=1.01 #dipole parameter, T
            if self.axis=='x':
                k0=alpha*LO.u0/(LO.m*LO.v0**2)
                r0=1/k0
                kappa = k0 ** 2
                phi = sym.sqrt(kappa) * L
                C = sym.cos(phi)
                S = sym.sin(phi) / sym.sqrt(kappa)
                Cd = -sym.sin(phi) * sym.sqrt(kappa)
                Sd = sym.cos(phi)
                D = (1 - sym.cos(phi)) / (r0 * kappa)  # dispersion
                Dd = (sym.sin(phi)) / (r0 * sym.sqrt(kappa))
                M = sym.Matrix([[C, S, D], [Cd, Sd, Dd], [0, 0, 1]])
            else: # if y. along y it is a lens
                kappa = LO.u0 * alpha**2 / (B0*LO.m * LO.v0**2 )  # 'spring constant' of the magnet
                phi = sym.sqrt(kappa) * L
                C = sym.cos(phi)
                S = sym.sin(phi) / sym.sqrt(kappa)
                Cd = -sym.sqrt(kappa) * sym.sin(phi)
                Sd = sym.cos(phi)
                M= sym.Matrix([[C, S, 0], [Cd, Sd, 0], [0, 0, 1]])
        else:
            raise Exception("ERROR WITH MAGNET TYPE SELECTION")
        if LO.type=="LINEAR":
            return M[:2,:2]
        else:
            return M
