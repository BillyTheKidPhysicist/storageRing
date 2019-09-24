import sympy as sym
import sys


magType=['BEND','LENS','DRIFT','COMBINER']
class Magnet:

    def __init__(self, LO, type, args):
        #LO: Lattice object, could be periodic or linear
        self.type = type  # type of magnet, DRIFT,FOCUS,BEND,COMBINER
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
            print('INVALID MAGNET NAME PROVIDED!')
            sys.exit()

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
        if self.type == magType[0]:  # BENDING
            # Will probably be a quadrupole cut in half, this introduces
            # a gradient in the force in additself.ion to the bending force. The total norm of the field is written
            # as B0=.5*Bdd*r^2, where Bdd is the second derivative of the field.thus k=Bdd*u0/m*v^2
            k = LO.u0 * self.Bdd / (LO.m * LO.v0 ** 2)
            k0 = 1 / self.rb
            kappa = k + k0 ** 2
            phi = sym.sqrt(kappa) * L
            C = sym.cos(phi)
            S = sym.sin(phi) / sym.sqrt(kappa)
            Cd = -sym.sin(phi) * sym.sqrt(kappa)
            Sd = sym.cos(phi)
            D = (1 - sym.cos(phi)) / (self.rb * kappa)  # dispersion
            Dd = (sym.sin(phi)) / (self.rb * sym.sqrt(kappa))
            M= sym.Matrix([[C, S, D], [Cd, Sd, Dd], [0, 0, 1]])
        if self.type == magType[1]:  #LENS
            kappa = 2 * LO.u0 * self.Bp / (LO.m * LO.v0 ** 2 * self.rp ** 2)  # 'spring constant' of the magnet
            phi = sym.sqrt(kappa) * L
            C = sym.cos(phi)
            S = sym.sin(phi) / sym.sqrt(kappa)
            Cd = -sym.sqrt(kappa) * sym.sin(phi)
            Sd = sym.cos(phi)
            M= sym.Matrix([[C, S, 0], [Cd, Sd, 0], [0, 0, 1]])
        if self.type == magType[2]:  # DRIFT
            M= sym.Matrix([[1, L, 0], [0, 1, 0], [0, 0, 1]])
        if self.type == magType[3]:  # COMBINER
            # ie Stern Gerlacht magnet. Made by Collin. Have yet to characterise fully
            # as of now it's basically a square bender, which seems to be mostly true based on Comsol
            # except it is square. Rather than add the matrices to account for that, which is small
            # I will wait till we characterize it in a general way
            # VERY temporarily, it's a drift region
            M= sym.Matrix([[1, L, 0], [0, 1, 0], [0, 0, 1]])


        if LO.type=="LINEAR":
            return M[:2,:2]
        else:
            return M
