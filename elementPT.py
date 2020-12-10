import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import sympy as sym
from numba import jit
import numba
import sys

class Element:
    # Class to represent the lattice element such as a drift/lens/bender/combiner.
    # each element type has its own reference frame, as well as attributes, which I will described below
    # Lens and Drift: Input is centered at origin and points to the 'west' with the ouput pointing towards the 'east'
    # Bender: output is facing south and aligned with y=0. The center of the bender is at the origin. Input is at some
        # positive angle relative to the output. A pi/2 bend would have the input aligned with x=0 for example
    # combiner: the output is at the origin, and the input is towards the east, but pointing a bit up at north. Note that
        # the input/beginning is considered to be at the origin. This doesn't really make sense and sould be changed
    # Bender, Segemented: This bending section is composed of discrete elements. It is modeled by considering a unit
        #cell and transforming the particle's coordinates into that frame to compute the force. The unit cell is
        # half a magnet at y=0, x=rb. An imaginary plane is at some angle to y=0 that defines the unit cell angle. Say the
        # unit cell angle is phi, and the particle is at angle theta, then the particle's angle in the unit cell is
        # theta-phi*(theta//phi)
    #Bender, simulated, segmented: Particle is traced using the force from a COMSOL simulation of the magnetic field. The
        #bore radius, bending radius and UCAngle is computed from the supplied field by finding the
        #maximum value in the y direction.
    # TODO: SWITCH THIS
    def __init__(self, args, type, PT):
        self.args = args
        self.type = type  # type of element. Options are 'BENDER', 'DRIFT', 'LENS_IDEAL', 'COMBINER'
        self.PT = PT  # particle tracer object
        self.Bp = None  # field strength at bore of element, T
        self.c1 = None  # dipole component of combiner, T
        self.c2 = None  # quadrupole component of combiner, T/m
        self.rp = None  # bore of element, m
        self.L = None  # length of element (or for vacuum tube), (for segmented this is length of 1 segment). This can also
            #include gaps. Not necesarily the length of the magnetic material. m
        self.Lo = None  # length of orbit inside element. This is different for bending, m
        self.rb = None  # 'bending' radius of magnet. actual bending radius of atoms is slightly different cause they
        # ride on the outside edge, m
        self.yokeWidth=None #Thickness of the yoke, but also refers to the thickness of the permanent magnets, m
        self.ucAng=None #the angle that the unit cell makes with the origin. This is for the segmented bender. It is
            #modeled as a unit cell and the particle's coordinates are rotated into it's reference frame, rad
        self.numMagnets=None #the number of magnets. Keep in mind for a segmented bender two units cells make a total
            #of 1 magnet with each unit cell having half a magnet.
        self.space=None #extra space added to the length of the magnet in each direction to make up for width
            #of the yoke. The total increase in width is TWICE this value
        self.BGradXFunc=None #Fit functions for the provided field gradients data
        self.BGradyFunc=None #Fit functions for the provided field gradients data
        self.BGradzFunc=None #Fit functions for the provided field gradients data


        self.r0 = None  # center of element (for bender this is at bending radius center),vector, m
        self.ro = None  # bending radius of orbit. Includes trajectory offset, m
        self.ang = 0  # Angle that the particles are bent, either bender or combiner. this is the change in  angle in
        # polar coordinates
        self.r1 = None  # position vector of beginning of element in lab frame, m
        self.r2 = None  # position vector of ending of element in lab frame, m
        self.r1El = None  # position vector of beginning of element in element frame, m
        self.r2El = None  # position vector of ending of element in element frame, m

        self.ne = None  # normal vector to end of element
        self.nb = None  # normal vector to beginning of element
        self.theta = None  # angle from horizontal of element. zero degrees is to the right, or 'east', in polar coordinates
        self.ap = None  # size of apeture. For now the same in both dimensions and vacuum tubes are square
        self.SO = None  # shapely object used to find if particle is inside
        self.index = None  # the index of the element in the lattice
        self.K = None  # spring constant for magnets
        self.rOffset = 0  # the offset of the particle trajectory in a bending magnet, m
        self.ROut = None  # rotation matrix so values don't need to be calculated over and over. This is the rotation
        # matrix OUT of the element frame
        self.RIn = None  # rotation matrix so values don't need to be calculated over and over. This is the rotation
        # matrix IN to the element frame
        self.inputOffset = None  # for the combiner. Incoming particles enter the combiner with an offset relative to its
        # geometric center. A positive value is more corresponds to moved up the y axis in the combiner's regerence
        # frame.
        self.LFunc = None  # for the combiner. The length along the trajector that the particle has traveled. This length
        # is referring to the nominal trajectory, not the actual distance of the particle
        self.distFunc = None  # The transerse distance from the nominal trajectory of the particle.
        self.cFact = None  # factor in the function y=c*x**2. This is used for finding the trajectory of the particle
        # in the combiner.
        self.trajLength = None  # total length of trajectory, m. This is for combiner because it is not trivial like
        # benders or lens or drifts

        self.data = None  # 6xn array containing the numeric values of a magnetic field. n is the number of data points.
        # each row of the data must be in the format x,y,z,Bx',By',Bz'. x,y,z are in the element's frame. Bi' is the
        # gradient in the i direction
        self.forceFitFunc = None  # an interpolation that the force in the x,y,z direction at a given point

        self.unpack_Args_And_Fill_Params()

    def unpack_Args_And_Fill_Params(self):
        if self.type == 'LENS_IDEAL':

            self.Bp = self.args[0]
            self.rp = self.args[1]
            self.L = self.args[2]
            self.Lo=self.args[2]
            self.ap = self.args[3]
            self.K = (2 * self.Bp * self.PT.u0 / self.rp ** 2)  # reduced force
        elif self.type == 'DRIFT':
            self.L = self.args[0]
            self.Lo = self.args[0]
            self.ap = self.args[1]
        elif self.type == 'BENDER_IDEAL':
            self.Bp = self.args[0]
            self.rb = self.args[1]
            self.rp = self.args[2]
            self.ang = self.args[3]
            self.ap = self.args[4]
            self.K = (2 * self.Bp * self.PT.u0 / self.rp ** 2)  # reduced force
            self.rOffset = np.sqrt(self.rb ** 2 / 4 + self.PT.m * self.PT.v0Nominal ** 2 / self.K) - self.rb / 2  # this method does not
                # account for reduced speed in the bender from energy conservation
            # self.rOffset=np.sqrt(self.rb**2/16+self.PT.m*self.PT.v0Nominal**2/(2*self.K))-self.rb/4 #this acounts for reduced
            # energy
            self.ro = self.rb + self.rOffset
            self.L = self.ang * self.rb
            self.Lo = self.ang * self.ro
        elif self.type=="BENDER_SIM_SEGMENTED":
            sys.exit()
            pass
        elif self.type=="BENDER_IDEAL_SEGMENTED":
            self.L=self.args[0]
            self.Bp = self.args[1]
            self.rb=self.args[2]
            self.rp=self.args[3]
            self.yokeWidth=self.args[4]
            self.numMagnets=self.args[5] #number of magnets which is half the number of unit cells.
            self.ap=self.args[6]
            self.space=self.args[7]
            self.L=self.L+self.space*2 #add extra space
            self.K = (2 * self.Bp * self.PT.u0 / self.rp ** 2)  # reduced force
            self.rOffset = np.sqrt(self.rb ** 2 / 4 + self.PT.m * self.PT.v0Nominal ** 2 / self.K) - self.rb / 2  # this method does not
                # account for reduced speed in the bender from energy conservation
            self.ro = self.rb + self.rOffset
            #compute the angle that the unit cell makes as well as total bent angle
            D = self.rb - self.rp - self.yokeWidth
            self.ucAng = np.arctan(self.L / (2 * D))
            self.ang=2*self.numMagnets * self.ucAng #number of units cells times bending angle of 1 cell
            self.Lo = self.ang * self.ro
        elif self.type == 'COMBINER':
            self.L = self.args[0]
            self.ap = self.args[1]
            self.c1 = self.args[2]
            self.c2 = self.args[3]
            if self.inputOffset is not None:
                # if self.inputOffset is none, then this feature is not being used. This will happen so that the combiner element
                # can be used to predict what the trajectory looks like inside before final use.
                # solve for LFunc and distFunc. These equation are very big so I use sympy to handle them
                x1, x, y1, c = sym.symbols('x1 x y1 c', real=True, positive=True, nonzero=True)
                dist = sym.sqrt((x1 - x) ** 2 + (y1 - self.cFact * x ** 2) ** 2)
                func = x1 - x + 2 * self.cFact * x * (y1 - self.cFact * x ** 2)
                x0 = sym.simplify(
                    sym.solve(func, x)[0])  # REMEMBER, THE CORRECT ANSWER CAN CHANGE POSITION WITH DIFFERENT
                # INPUT
                # NEED TO NAME FUNCTIONS DIFFERENTLY EACH TIME. LAMBDA EVALUATES A FUNCTION IN IT'S LAST STATE, SO MULTIPLE
                # TEMPORARY FUNCTIONS WITH THE SAME NAME INTERFERE WITH EACH OTHER
                tempFunc1 = sym.lambdify([x1, y1], dist.subs(x, x0))
                self.distFunc = lambda x1, y1: np.real(
                    tempFunc1(x1 + 0J, y1))  # need input to be complex to avoid error on
                # roots of negative numbers. There is a tiny imaginary component from numerical imprecision, so I take
                # only the real
                tempFunc2 = sym.lambdify([x1, y1], sym.integrate(sym.sqrt(1 + (2 * self.cFact * x) ** 2), (x, 0, x0)))
                self.LFunc = lambda x1, y1: np.real(tempFunc2(x1 + 0J, y1))

                self.trajLength = sym.integrate(sym.sqrt(1 + (2 * self.cFact * x) ** 2), (x, 0, self.L)).subs(x, self.L)
        elif self.type == 'LENS_SIM':
            self.data = self.args[0]
            print('here')
            Fx = spi.LinearNDInterpolator(self.data[:, :3], self.data[:, 3] * self.PT.u0)
            print('here')
            Fy = spi.LinearNDInterpolator(self.data[:, :3], self.data[:, 4] * self.PT.u0)
            print('here')
            Fz = spi.LinearNDInterpolator(self.data[:, :3], self.data[:, 5] * self.PT.u0)
            print(Fx(0, .1, .1))
        else:
            raise Exception('No proper element name provided')

    def transform_Lab_Coords_Into_Orbit_Frame(self, q, cumulativeLength):
        q = self.transform_Lab_Coords_Into_Element_Frame(q)
        qo = self.transform_Element_Coords_Into_Orbit_Frame(q)
        qo[0] = qo[0] + cumulativeLength
        return qo

    def transform_Lab_Coords_Into_Element_Frame(self, q):
        # q: particle coords in x and y plane. numpy array
        # self: element object to transform to
        # get the coordinates of the particle in the element's frame. See element class for description
        qNew = q.copy()  # CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        if self.type == 'DRIFT' or self.type == 'LENS_IDEAL':
            qNew[0] = qNew[0] - self.r1[0]
            qNew[1] = qNew[1] - self.r1[1]
        elif self.type == 'BENDER_IDEAL' or self.type == 'BENDER_IDEAL_SEGMENTED':
            qNew[:2] = qNew[:2] - self.r0
        elif self.type == 'COMBINER':
            qNew[:2] = qNew[:2] - self.r2
        qNewx = qNew[0]
        qNewy = qNew[1]
        qNew[0] = qNewx * self.RIn[0, 0] + qNewy * self.RIn[0, 1]
        qNew[1] = qNewx * self.RIn[1, 0] + qNewy * self.RIn[1, 1]
        return qNew

    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        # This returns the nominal orbit in the element's reference frame.
        # q: particles position in the element's reference frame
        # The description for each element is given below.
        qo = q.copy()
        if self.type == 'LENS_IDEAL' or self.type == 'DRIFT':
            pass
        elif self.type == 'BENDER_IDEAL' or self.type == 'BENDER_IDEAL_SEGMENTED':
            qo = q.copy()
            phi = self.ang - np.arctan2(q[1] + 1e-10, q[0])  # angle swept out by particle in trajectory. This is zero
            # when the particle first enters
            ds = self.ro * phi
            qos = ds
            qox = np.sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
            qo[0] = qos
            qo[1] = qox
        elif self.type == 'COMBINER':
            if qo[0] > self.L:
                dr = self.r2El - qo[:2]
                rot = np.asarray([[np.cos(-self.ang), -np.sin(-self.ang)], [np.sin(-self.ang), np.cos(-self.ang)]])
                qo[:2] = rot @ dr
            else:
                qo[0] = self.trajLength - self.LFunc(q[0], q[1]) + np.sin(self.ang) * (self.ap - self.inputOffset)
                qo[1] = self.distFunc(q[0], q[1])  # TODO: FUCKING FIX THIS....
                raise Exception('doesnt quite work right')
        return qo

    def force(self, q):
        # force at point q in element frame
        F = np.zeros(3)  # force vector starts out as zero
        if self.type == 'DRIFT':  # no force from drift region
            pass
        elif self.type == 'BENDER_IDEAL':
            r = np.sqrt(q[0] ** 2 + q[1] ** 2)  # radius in x y frame
            F0 = -self.K * (r - self.rb)  # force in x y plane
            phi = np.arctan2(q[1], q[0])
            F[0] = np.cos(phi) * F0
            F[1] = np.sin(phi) * F0
            F[2] = -self.K * q[2]
        elif self.type == 'LENS_IDEAL':
            # note: for the perfect lens, in it's frame, there is never force in the x direction
            F[1] = -self.K * q[1]
            F[2] = -self.K * q[2]
        elif self.type == 'COMBINER':
            if q[0] < self.L:
                B0 = np.sqrt((self.c2 * q[2]) ** 2 + (self.c1 + self.c2 * q[1]) ** 2)
                F[1] = self.PT.u0 * self.c2 * (self.c1 + self.c2 * q[1]) / B0
                F[2] = self.PT.u0 * self.c2 ** 2 * q[2] / B0
        elif self.type=='BENDER_IDEAL_SEGMENTED':
            quc=self.transform_Element_Into_Unit_Cell_Frame(q) #get unit cell coords
            if quc[1]<self.L/2: #if particle is inside the magnet region
                F[0]=-self.K*(quc[0]-self.rb)
                F[2]=-self.K*quc[2]
                F = self.transform_Unit_Cell_Vector_Into_Element_Frame(F, q, quc)
            else: #if instead it's in the sliver that isn't inside the magnet
                pass
            #print(quc[1],F[0])

        return F
    def transform_Element_Into_Unit_Cell_Frame(self,q):
        if self.type == 'BENDER_IDEAL_SEGMENTED':
            qNew=q.copy()
            phi=self.ang-np.arctan2(q[1],q[0])

            r=np.sqrt(q[0]**2+q[1]**2)
            theta=phi-self.ucAng*(phi//self.ucAng)
            qNew[0]=r*np.cos(theta) #cartesian coords in unit cell frame
            qNew[1]=r*np.sin(theta) #cartesian coords in unit cell frame
            return qNew
        else:
            raise Exception('not implemented')


    def transform_Unit_Cell_Vector_Into_Element_Frame(self,vec,q,quc):
        vecNew=vec.copy()
        rotAngle=-(np.arctan2(quc[1],quc[0])-np.arctan2(q[1],q[0]))
        vecx=vecNew[0]
        vecy=vecNew[1]
        vecNew[0] = vecx * np.cos(rotAngle) - vecy *np.sin(rotAngle)
        vecNew[1] = vecx*np.sin(rotAngle) +vecy*np.cos(rotAngle)
        return vecNew


    def transform_Vector_Out_Of_Element_Frame(self,vec):
        # rotation matrix is 3x3 to account for z axis
        vecNew=vec.copy()
        vecx = vecNew[0]
        vecy = vecNew[1]
        vecNew[0] = vecx * self.ROut[0, 0] + vecy * self.ROut[0, 1]
        vecNew[1] = vecx * self.ROut[1, 0] + vecy * self.ROut[1, 1]
        return vecNew

    def transform_Lab_Momentum_Into_Orbit_Frame(self, q, p):
        # TODO: CONSOLIDATE THIS WITH GET_POSITION
        pNew = p.copy()
        pNew[0] = p[0] * self.RIn[0, 0] + p[1] * self.RIn[0, 1]
        pNew[1] = p[0] * self.RIn[1, 0] + p[1] * self.RIn[1, 1]
        if self.type == 'BENDER_IDEAL' or self.type == 'BENDER_IDEAL_SEGMENTED':  # need to use a change of vectors from cartesian to polar for bender
            q = self.transform_Lab_Coords_Into_Element_Frame(q)
            pNew = p.copy()
            sDot = (q[0] * pNew[1] - q[1] * pNew[0]) / np.sqrt((q[0] ** 2 + q[1] ** 2))
            rDot = (q[0] * pNew[0] + q[1] * pNew[1]) / np.sqrt((q[0] ** 2 + q[1] ** 2))
            po = np.asarray([sDot, rDot, pNew[2]])
            return po
        elif self.type == 'LENS_IDEAL' or self.type == 'DRIFT':
            return pNew
        if self.type == 'COMBINER':
            raise Exception('NOT YET IMPLEMENTED')


