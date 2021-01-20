import numpy as np
from interp3d import interp_3d
import scipy.interpolate as spi
import pandas as pd
import numpy.linalg as npl
import sys
import matplotlib.pyplot as plt

#Notes:
#--begining and ending of elements refer to the geometric sense in the lattice. For example, the beginning of the lens
#is the part encountered going clockwise, and the end is the part exited going clockwise for a particle being traced

class Element:
    def __init__(self):
        self.theta=None #angle that describes an element's rotation in the xy plane.
        #SEE EACH ELEMENT FOR MORE DETAILS
        #-Straight elements like lenses and drifts: theta=0 is the element's input at the origin and the output pointing
        #east. for theta=90 the output is pointing up.
        #-Bending (unsegmented) elements without caps: at theta=0 the outlet is at (bending radius,0) pointing south with the input
        # at some angle counterclockwise. a 180 degree bender would have the inlet at (-bending radius,0) pointing south.
        # force is a continuous function of r and theta, ie a revolved cross section of a hexapole
        #-Bending (unsegmented) elements with caps: same as without caps, but keep in mind that the cap on the output would be BELOW
        #y=0
        #-Segmented bending elements (with and without caps): very similiar to ideal bender, but force is not a continuous
        #function of theta and r. It is instead a series of discrete magnets which are represented as a unit cell. A
        #full magnet would be modeled as two unit cells, instead of a single unit cell, to exploit symmetry and thus
        #save memory. Half the time the symetry is exploited by using a simple rotation, the other half of the time the
        #symmetry requires a reflection, then rotation.
        # combiner: This is is the element that bends the two beams together. The logic is a bit tricky. It's geometry is
        # modeled as a straight section, a simple square, with a segment coming of at the particle in put at an angle. The
        # angle is decided by tracing particles through the combiner and finding the bending angle.

        #- simulated models: There are simulated versions of the above elements that are for the most part the same except
        #the force function, which calls to a method I got from stack exchange, which is a cython 3d version of scipy's
        #linear nd interpolater. This method gives the same results (except at the edges, where the logic fails, but scipy
        #seems to still give reasonable ansers) and take about 5us per evaluatoin instead of 200us.


        self.nb=None #normal vector to beginning (clockwise sense) of element.
        self.ne=None #normal vector to end (clockwise sense) of element
        self.r0=None #coordinates of center of bender, minus any caps
        self.ROut=None #2d matrix to rotate a vector out of the element's reference frame
        self.RIn = None #2d matrix to rotate a vector into the element's reference frame
        self.r1=None #3D coordinates of beginning (clockwise sense) of element
        self.r2=None #3D coordinates of ending (clockwise sense) of element
        self.SO = None #the shapely object for the element. These are used for plotting, and for finding if the coordinates
        #are inside an element that can't be found with simple geometry
        self.ang=0 #bending angle of the element. 0 for lenses and drifts
        self.Lm=None #hard edge length of magnet along line through the bore
        self.L=None #length of magnet along line through the bore
        self.K=None #'spring constant' of element. For some this comes from comsol fields.
        self.rOffset=None #the offset to the bending radius because of centrifugal force. There are two versions, one that
        #accounts for reduced speed from energy conservation for actual fields, and one that doesn't
        self.Lo=None #length of orbit for particle. For lenses and drifts this is the same as the length. This is a nominal
        #value because for segmented benders the path length is not simple to compute
        self.ro=None #bending radius of orbit, ie rb + rOffset.
        self.index=None #elements position in lattice
        self.cap=False #wether there is a cap or not present on the element. Cap simulates fringe fields
        self.comsolExtraSpace=.1e-3 #extra space in comsol files to exported grids. this can be used to find dimensions
        self.apz=None #apeture in the z direction. all but the combiner is symmetric, so there apz is the same as ap
    def transform_Lab_Coords_Into_Orbit_Frame(self, q, cumulativeLength):
        #Change the lab coordinates into the particle's orbit frame.
        q = self.transform_Lab_Coords_Into_Element_Frame(q) #first change lab coords into element frame
        qo = self.transform_Element_Coords_Into_Orbit_Frame(q) #then element frame into orbit frame
        qo[0] = qo[0] +cumulativeLength #orbit frame is in the element's frame, so the preceding length needs to be
        #accounted for
        return qo
    def transform_Lab_Coords_Into_Element_Frame(self,q):
        #this is overwridden by all other elements
        pass
    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        #for straight elements element and orbit frame are identical
        return q
    def transform_Lab_Frame_Vector_Into_Element_Frame(self,vec):
        vecNew=vec.copy()
        vecx=vecNew[0];vecy=vecNew[1]
        vecNew[0] = vecx * self.RIn[0, 0] + vecy * self.RIn[0, 1]
        vecNew[1] = vecx * self.RIn[1, 0] + vecy * self.RIn[1, 1]
        return vecNew
    def transform_Element_Frame_Vector_To_Lab_Frame(self, vec):
        # rotate vector out of element frame into lab frame
        #vec: vector in
        vecNew=vec.copy()#copy input vector to not modify the original
        vecx = vecNew[0];vecy = vecNew[1]
        vecNew[0] = vecx * self.ROut[0, 0] + vecy * self.ROut[0, 1]
        vecNew[1] = vecx * self.ROut[1, 0] + vecy * self.ROut[1, 1]
        return vecNew
    def set_Length(self,L):
        #this is used typically for setting constraints.
        self.L=L
        self.Lo=L

class Lens_Ideal(Element):
    def __init__(self,PTL,L,Bp,rp,ap,fillParams=True):
        super().__init__()
        self.PTL=PTL
        self.Bp = Bp
        self.rp = rp
        self.L = L
        self.ap = ap
        self.type='STRAIGHT'
        if fillParams==True:
            self.fill_Params()
    def fill_Params(self):
        self.K = (2 * self.Bp * self.PTL.u0 / self.rp ** 2)
        if self.L is not None:
            self.Lo=self.L
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew = q.copy()  # CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        qNew[0] = qNew[0] - self.r1[0]
        qNew[1] = qNew[1] - self.r1[1]
        qNew=self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def force(self,q):
        F = np.zeros(3)  # force vector starts out as zero
        # note: for the perfect lens, in it's frame, there is never force in the x direction
        F[1] = -self.K * q[1]
        F[2] = -self.K * q[2]
        return F

    def is_Coord_Inside(self, q):
        # check with fast geometric arguments if the particle is inside the element. This won't necesarily work for all
        # elements. If True is retured, the particle is inside. If False is returned, it is defintely outside. If none is
        # returned, it is unknown
        # q: coordinates to test
        if np.abs(q[2]) > self.ap:
            return False
        elif np.abs(q[1]) > self.ap:
            return False
        elif q[0] < 0 or q[0] > self.L:
            return False
        else:
            return True

class Drift(Lens_Ideal):
    def __init__(self,PTL,L,ap):
        super().__init__(PTL,L,0,np.inf,ap)
    def force(self,q):
        return np.zeros(3)

class Bender_Ideal(Element):
    def __init__(self,PTL,ang,Bp,rp,rb,ap,fillParams=True):
        super().__init__()
        self.PTL=PTL
        self.ang=ang
        self.Bp = Bp
        self.rp = rp
        self.ap = ap
        self.rb=rb
        self.type='BEND'
        self.rOffsetFunc=None
        self.segmented=False
        self.capped=False
        self.Lcap=0

        if fillParams==True:
            self.fill_Params()
    def fill_Params(self):
        self.K = (2 * self.Bp * self.PTL.u0 / self.rp ** 2)
        self.rOffsetFunc=lambda rb: np.sqrt(rb ** 2 / 4 + self.PTL.m * self.PTL.v0Nominal ** 2 / self.K) -rb / 2
        self.rOffset = self.rOffsetFunc(self.rb)
        self.ro=self.rb+self.rOffset
        if self.ang is not None: #calculation is being delayed until constraints are solved
            self.L  = self.rb * self.ang
            self.Lo = self.ro * self.ang



    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew=q-self.r0
        qNew=self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew

    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        qo = q.copy()
        phi = self.ang - np.arctan2(q[1], q[0])  # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = np.sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
        qo[0] = qos
        qo[1] = qox
        return qo
    def force(self, q):
        # force at point q in element frame
        #q: particle's position in element frame
        F = np.zeros(3)  # force vector starts out as zero
        r = np.sqrt(q[0] ** 2 + q[1] ** 2)  # radius in x y frame
        F0 = -self.K * (r - self.rb)  # force in x y plane
        phi = np.arctan2(q[1], q[0])
        F[0] = np.cos(phi) * F0
        F[1] = np.sin(phi) * F0
        F[2] = -self.K * q[2]
        return F
    def is_Coord_Inside(self,q):
        if np.abs(q[2]) > self.ap:  # if clipping in z direction
            return False
        phi = np.arctan2(q[1], q[0])
        if (phi > self.ang and phi < 2 * np.pi) or phi < 0:  # if outside bender's angle range
            return False
        r = np.sqrt(q[0] ** 2 + q[1] ** 2)
        if r < self.rb - self.ap or r > self.rb + self.ap:
            return False
        return True

class Combiner_Ideal(Element):
    def __init__(self,PTL,Lm,c1,c2,ap,fillsParams=True):
        super().__init__()
        self.PTL=PTL
        self.ap = ap
        self.Lm=Lm
        self.Lb=self.Lm
        self.La=None
        self.c1=c1
        self.c2=c2
        self.type='COMBINER'
        self.inputOffset=None
        if fillsParams==True:
            self.fill_Params()
    def fill_Params(self):
        inputAngle, inputOffset = self.compute_Input_Angle_And_Offset(self.Lm)
        self.apz = self.ap / 2
        self.ang = inputAngle
        self.inputOffset = inputOffset
        self.La = self.ap * np.sin(self.ang)
        self.L = self.La * np.cos(self.ang) + self.Lb #TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING
        self.Lo = self.L

    def compute_Input_Angle_And_Offset(self, limit,h=1e-6):
        # this computes the output angle and offset for a combiner magnet.
        # NOTE: for the ideal combiner this gives slightly inaccurate results because of lack of conservation of energy!
        # todo: make proper edge handling
        q = np.asarray([0, 0, 0])
        p = self.PTL.m * np.asarray([self.PTL.v0Nominal, 0, 0])
        #xList=[]
        #yList=[]
        while True:
            F = self.force(q)
            a = F / self.PTL.m
            q_n = q + (p / self.PTL.m) * h + .5 * a * h ** 2
            F_n = self.force(q_n)
            a_n = F_n / self.PTL.m  # accselferation new or accselferation sub n+1
            p_n = p + self.PTL.m * .5 * (a + a_n) * h
            if q_n[0] > limit:  # if overshot, go back and walk up to the edge assuming no force
                dr = limit - q[0]
                dt = dr / (p[0] / self.PTL.m)
                q = q + (p / self.PTL.m) * dt
                break
            #xList.append(q[0])
            #yList.append(F[0])
            q = q_n
            p = p_n

        #plt.plot(xList,yList)
        #plt.show()

        outputAngle = np.arctan2(p[1], p[0])
        outputOffset = q[1]
        return outputAngle, outputOffset
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew=q.copy()
        qNew = qNew - self.r2
        qNew=self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        # TODO: FIX THIS
        qo=q.copy()
        qo[0] = self.L - qo[0]
        qo[1] = 0#qo[1]
        return qo
    def force(self, q):
        # force at point q in element frame
        #q: particle's position in element frame
        F = np.zeros(3)  # force vector starts out as zero
        if q[0] < self.Lb:
            B0 = np.sqrt((self.c2 * q[2]) ** 2 + (self.c1 + self.c2 * q[1]) ** 2)
            F[1] = self.PTL.u0 * self.c2 * (self.c1 + self.c2 * q[1]) / B0
            F[2] = self.PTL.u0 * self.c2 ** 2 * q[2] / B0

        return F
    def is_Coord_Inside(self,q):
        if np.abs(q[2]) > self.ap:
            return False
        elif q[0] < self.Lb and q[0] > 0:  # particle is in the straight section that passes through the combiner
            if np.abs(q[1]) < self.ap:
                return True
        else:  # particle is in the bent section leading into combiner
            # TODO: ADD THIS LOGIc
            return None

class CombinerSim(Combiner_Ideal):
    def __init__(self,PTL,combinerFile):
        super().__init__(PTL,.18,None,None,None,fillsParams=False)
        self.space = 4 * 1.1E-2  # extra space past the hard edge on either end to account for fringe fields
        self.data=None
        self.combinerFile=combinerFile
        self.FxFunc=None
        self.FyFunc=None
        self.FzFunc=None
        self.fill_Params()
    def fill_Params(self):

        self.data = np.asarray(pd.read_csv(self.combinerFile, delim_whitespace=True, header=None))
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input
        xArr = np.unique(self.data[:, 0])
        yArr = np.unique(self.data[:, 1])
        zArr = np.unique(self.data[:, 2])
        BGradx = self.data[:, 3]
        BGrady = self.data[:, 4]
        BGradz = self.data[:, 5]
        numx = xArr.shape[0]
        numy = yArr.shape[0]
        numz = zArr.shape[0]
        self.ap = (yArr.max() - yArr.min() - 2 * self.comsolExtraSpace) / 2.0
        self.apz = (zArr.max() - zArr.min() - 2 * self.comsolExtraSpace) / 2.0

        BGradxMatrix = BGradx.reshape((numz, numy, numx))
        BGradyMatrix = BGrady.reshape((numz, numy, numx))
        BGradzMatrix = BGradz.reshape((numz, numy, numx))

        BGradxMatrix = np.ascontiguousarray(BGradxMatrix)
        BGradyMatrix = np.ascontiguousarray(BGradyMatrix)
        BGradzMatrix = np.ascontiguousarray(BGradzMatrix)
        #
        tempx = interp_3d.Interp3D(-self.PTL.u0 * BGradxMatrix, zArr, yArr, xArr)
        tempy = interp_3d.Interp3D(-self.PTL.u0 * BGradyMatrix, zArr, yArr, xArr)
        tempz = interp_3d.Interp3D(-self.PTL.u0 * BGradzMatrix, zArr, yArr, xArr)

        self.FxFunc = lambda x, y, z: tempx((z, y, x))
        self.FyFunc = lambda x, y, z: tempy((z, y, x))
        self.FzFunc = lambda x, y, z: tempz((z, y, x))
        inputAngle, inputOffset = self.compute_Input_Angle_And_Offset(self.Lm+2*self.space)
        self.ang = inputAngle
        self.inputOffset=inputOffset-np.tan(inputAngle) * self.space  # the input offset is measure at the end of the hard
        # edge

        # the inlet length needs to be long enough to extend past the fringe fields
        # TODO: MAKE EXACT, now it overshoots
        self.La = self.space + np.tan(self.ang) * self.ap
        self.Lo = self.La + self.Lb
        self.L = self.Lo
    def force(self,q):
        F=np.zeros(3)
        F[0] = self.FxFunc(*q)
        F[1] = self.FyFunc(*q)
        F[2] = self.FzFunc(*q)
        return F

class BenderIdealSegmented(Bender_Ideal):
    def __init__(self, PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space, ap,fillParams=True):
        super().__init__(PTL,None,Bp,rp,rb,ap,fillParams=False)
        self.numMagnets = numMagnets
        self.Lm = Lm
        self.space = space
        self.yokeWidth = yokeWidth
        self.ucAng = None
        self.segmented = True
        self.cap = False
        self.ap = ap
        self.RIn_Ang = None
        self.Lseg=None
        self.M_uc=None #matrix for reflection used in exploting segmented symmetry. This is 'inside' a single magnet element
        self.M_ang=None #matrix for reflection used in exploting segmented symmetry. This is reflecting out from the unit cell
        if fillParams==True:
            self.fill_Params()

    def fill_Params(self):
        super().fill_Params()
        self.Lseg=self.Lm+2*self.space
        if self.numMagnets is not None:
            self.ucAng = np.arctan((self.Lm / 2 + self.space) / (self.rb - self.yokeWidth - self.rp))
            self.ang = 2 * self.ucAng * self.numMagnets
            self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
            self.Lo = self.ro * self.ang
            m = np.tan(self.ucAng)
            self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew = q - self.r0
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew

    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        qo = q.copy()
        phi = self.ang - np.arctan2(q[1], q[0])  # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = np.sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
        qo[0] = qos
        qo[1] = qox
        return qo

    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame
        F = np.zeros(3)  # force vector starts out as zero
        quc = self.transform_Element_Coords_Into_Unit_Cell_Frame(q)  # get unit cell coords
        if quc[1] < self.Lm / 2:  # if particle is inside the magnet region
            F[0] = -self.K * (quc[0] - self.rb)
            F[2] = -self.K * quc[2]
            F = self.transform_Unit_Cell_Force_Into_Element_Frame(F, q)  # transform unit cell coordinates into
            # element frame
        return F

    def transform_Unit_Cell_Force_Into_Element_Frame(self, F, q):
        # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
        # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
        # or leaving the element interface as mirror images of each other.
        # F: Force to be rotated out of unit cell frame
        # q: particle's position in the element frame where the force is acting
        FNew = F.copy()  # copy input vector to not modify the original
        phi = np.arctan2(q[1], q[0])  # the anglular displacement from output of bender to the particle. I use
        # output instead of input because the unit cell is conceptually located at the output so it's easier to visualize
        cellNum = int(phi // self.ucAng) + 1  # cell number that particle is in, starts at one
        if cellNum % 2 == 1:  # if odd number cell. Then the unit cell only needs to be rotated into that position
            rotAngle = 2 * (cellNum // 2) * self.ucAng
        else: #otherwise it needs to be reflected. This is the algorithm for reflections
            Fx = FNew[0]
            Fy = FNew[1]
            FNew[0] = self.M_uc[0, 0] * Fx + self.M_uc[0, 1] * Fy
            FNew[1] = self.M_uc[1, 0] * Fx + self.M_uc[1, 1] * Fy
            rotAngle = 2 * ((cellNum - 1) // 2) * self.ucAng
        Fx = FNew[0]
        Fy = FNew[1]
        FNew[0] = Fx * np.cos(rotAngle) - Fy * np.sin(rotAngle)
        FNew[1] = Fx * np.sin(rotAngle) + Fy * np.cos(rotAngle)
        return FNew
    def transform_Element_Coords_Into_Unit_Cell_Frame(self,q):
        #As particle leaves unit cell, it does not start back over at the beginning, instead is turns around so to speak
        #and goes the other, then turns around again and so on. This is how the symmetry of the unit cell is exploited.
        #q: particle coords in element frame
        # returnUCFirstOrLast: return 'FIRST' or 'LAST' if the coords are in the first or last unit cell. This is typically
        # used for including unit cell fringe fields
        qNew=q.copy()
        phi=self.ang-np.arctan2(q[1],q[0])
        revs=int(phi//self.ucAng) #number of revolutions through unit cell
        if revs%2==0: #if even
            theta = phi - self.ucAng * revs
        else: #if odd
            theta = self.ucAng-(phi - self.ucAng * revs)
        r=np.sqrt(q[0]**2+q[1]**2)
        qNew[0]=r*np.cos(theta) #cartesian coords in unit cell frame
        qNew[1]=r*np.sin(theta) #cartesian coords in unit cell frame

        return qNew

    def is_Coord_Inside(self, q):
        if np.abs(q[2]) > self.ap:  # if clipping in z direction
            return False
        phi = np.arctan2(q[1], q[0])
        if phi < 0:  # constraint to between zero and 2pi
            phi += 2 * np.pi
        if phi < self.ang:
            r = np.sqrt(q[0] ** 2 + q[1] ** 2)
            if r < self.rb - self.ap or r > self.rb + self.ap:
                return False
        else:
            return False

        return True

class BenderIdealSegmentedWithCap(BenderIdealSegmented):
    def __init__(self,PTL,numMagnets,Lm,Lcap,Bp,rp,rb,yokeWidth,space,ap,fillParams=True):
        super().__init__(PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space, ap,fillParams=False)
        self.Lcap = Lcap
        self.cap=True
        if fillParams==True:
            self.fill_Params()
    def fill_Params(self):
        super().fill_Params()
        if self.numMagnets is not None:
            if self.ang>3*np.pi/2 or self.ang<np.pi/2 or self.Lcap>self.rb-self.rp*2: #this is done so that finding where the particle is inside the
                #bender is not a huge chore. There is almost no chance it would have this shape anyways. Changing this
                #would affect force, orbit coordinates and isinside at least
                raise Exception('DIMENSIONS OF BENDER ARE OUTSIDE OF ACCEPTABLE BOUNDS')
            self.Lo = self.ro * self.ang+2*self.Lcap


    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        qo = q.copy()
        angle = np.arctan2(qo[1], qo[0])
        if angle<0: #to use full 2pi with arctan2
            angle+=2*np.pi
        if angle>self.ang: #if particle is outside of the bending segment so it could be in the caps, or elsewhere
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap): #If inside the cap on
                #the eastward side
                qo[0] = self.Lcap + self.ang * self.ro + (-q[1])
                qo[1] = q[0] - self.ro
            qTest=q.copy()
            qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
            qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
            if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0):
                qo[0]=self.Lcap-qTest[1]
                qo[1]=qTest[0]-self.ro
        else:
            phi = self.ang - np.arctan2(q[1], q[0])  # angle swept out by particle in trajectory. This is zero
            # when the particle first enters
            ds = self.ro * phi+self.Lcap
            qos = ds
            qox = np.sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
            qo[0] = qos
            qo[1] = qox
        return qo

    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame
        F = np.zeros(3)  # force vector starts out as zero
        phi = np.arctan2(q[1], q[0])
        if phi<0: #constraint to between zero and 2pi
            phi+=2*np.pi
        if phi<self.ang:
            return super().force(q)
        elif phi>self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap): #If inside the cap on
                #the eastward side
                F[0]=-self.K*(q[0]-self.rb)
            qTest=q.copy()
            qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
            qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]

            if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0):
                forcex = -self.K * (qTest[0] - self.rb)
                F[0]=self.RIn_Ang[0,0]*forcex
                F[1]=-self.RIn_Ang[1,0]*forcex
        return F

    def is_Coord_Inside(self, q):
        if np.abs(q[2]) > self.ap:  # if clipping in z direction
            return False
        phi = np.arctan2(q[1], q[0])
        if phi<0: #constraint to between zero and 2pi
            phi+=2*np.pi
        if phi<self.ang:
            r = np.sqrt(q[0] ** 2 + q[1] ** 2)
            if self.rb+self.ap>r>self.rb-self.ap:
                return True
        if phi>self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap): #If inside the cap on
                #the eastward side
                return True
            qTest=q.copy()
            qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
            qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
            if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0):
                return True
        return False

class BenderSimSegmented(BenderIdealSegmented):
    def __init__(self, PTL, fileName,numMagnets, Lm, Bp, rp, rb, yokeWidth, space, ap,fillParams=True):
        super().__init__( PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space, ap,fillParams=False)
        self.fileName=fileName
        sys.exit()

class BenderSimSegmentedWithCap(BenderIdealSegmentedWithCap):
    def __init__(self,PTL, fileSeg,fileCap,fileInternalFringe,Lm,numMagnets,rb,extraSpace,yokeWidth,ap):
        super().__init__(PTL,numMagnets,Lm,None,None,None,rb,yokeWidth,extraSpace,ap,fillParams=False)
        self.PTL=PTL
        self.fileSeg=fileSeg
        self.fileCap=fileCap
        self.fileInternalFringe=fileInternalFringe
        self.numMagnets=numMagnets
        self.extraSpace=extraSpace
        self.yokeWidth=yokeWidth
        self.ap=ap
        self.ucAng=None
        self.dataSeg=None
        self.dataCap=None
        self.dataInternalFringe=None
        self.FxFunc_Seg=None
        self.FyFunc_Seg=None
        self.FzFunc_Seg=None
        self.FxFunc_Cap=None
        self.FyFunc_Cap=None
        self.FzFunc_Cap=None
        self.FxFunc_Internal_Fringe = None
        self.FyFunc_Internal_Fringe = None
        self.FzFunc_Internal_Fringe = None
        self.fill_Params()
    def fill_Params(self):
        self.Lseg=self.Lm+self.space*2
        if self.dataSeg is None:
            self.fill_Force_Func_Seg()
            self.rp=(self.dataSeg[:,0].max()-self.dataSeg[:,0].min()-2*self.comsolExtraSpace)/2
            if self.ap is None:
                self.ap=self.rp*.9
            elif self.ap>self.rp:
                raise Exception('APETURE IS LARGER THAN BORE')
            self.K = self.compute_K()
            rOffsetFact=1.00125 #emperical factor that reduces amplitude of off orbit oscillations. An approximation.
            self.rOffsetFunc = lambda rb:  rOffsetFact*np.sqrt(rb ** 2 / 16 + self.PTL.m * self.PTL.v0Nominal ** 2 / (2 * self.K)) - rb / 4  # this
            # acounts for reduced energy
        if self.dataCap is None:
            self.fill_Force_Func_Cap()
            self.Lcap=self.dataCap[:,2].max()-self.dataCap[:,2].min()-self.comsolExtraSpace*2
        if self.dataInternalFringe is None:
            self.fill_Force_Func_Internal_Fringe()
        if self.numMagnets is not None:
            D = self.rb - self.rp - self.yokeWidth
            self.ucAng = np.arctan(self.Lseg / (2 * D))
            self.ang = 2 * self.numMagnets * self.ucAng
            self.rOffset=self.rOffsetFunc(self.rb)
            self.ro=self.rb+self.rOffset
            self.L=self.ang*self.rb
            self.Lo=self.ang*self.ro+2*self.Lcap
            self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
            m = np.tan(self.ucAng)
            self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2) #reflection matrix
            m = np.tan(self.ang / 2)
            self.M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2) #reflection matrix
    def fill_Force_Func_Cap(self):
        self.dataCap = np.asarray(pd.read_csv(self.fileCap, delim_whitespace=True, header=None))
        xArr = np.unique(self.dataCap[:, 0])
        yArr = np.unique(self.dataCap[:, 1])
        zArr = np.flip(np.unique(self.dataCap[:, 2]))
        BGradx = self.dataCap[:, 3]
        BGrady = self.dataCap[:, 4]
        BGradz = self.dataCap[:, 5]
        numx = xArr.shape[0]
        numy = yArr.shape[0]
        numz = zArr.shape[0]


        BGradxMatrix = BGradx.reshape((numz, numy, numx))
        BGradyMatrix = BGrady.reshape((numz, numy, numx))
        BGradzMatrix = BGradz.reshape((numz, numy, numx))
        # plt.imshow(BGradyMatrix[0,:,:])
        # plt.show()

        BGradxMatrix = np.ascontiguousarray(BGradxMatrix)
        BGradyMatrix = np.ascontiguousarray(BGradyMatrix)
        BGradzMatrix = np.ascontiguousarray(BGradzMatrix)
        #
        tempx = interp_3d.Interp3D(-self.PTL.u0 * BGradxMatrix, zArr, yArr, xArr)
        tempy = interp_3d.Interp3D(-self.PTL.u0 * BGradyMatrix, zArr, yArr, xArr)
        tempz = interp_3d.Interp3D(-self.PTL.u0 * BGradzMatrix, zArr, yArr, xArr)
        self.FxFunc_Cap = lambda x, y, z:  tempx((y, -z, x))
        self.FyFunc_Cap = lambda x, y, z: tempz((y, -z, x)) #todo: THIS NEEDS TO BE TESTED MORE!!
        self.FzFunc_Cap = lambda x, y, z: -tempy((y, -z, x))

    def fill_Force_Func_Internal_Fringe(self):
        self.dataInternalFringe = np.asarray(pd.read_csv(self.fileInternalFringe, delim_whitespace=True, header=None))
        xArr = np.unique(self.dataInternalFringe[:, 0])
        yArr = np.unique(self.dataInternalFringe[:, 1])
        zArr = np.flip(np.unique(self.dataInternalFringe[:, 2]))
        BGradx = self.dataInternalFringe[:, 3]
        BGrady = self.dataInternalFringe[:, 4]
        BGradz = self.dataInternalFringe[:, 5]
        numx = xArr.shape[0]
        numy = yArr.shape[0]
        numz = zArr.shape[0]

        BGradxMatrix = BGradx.reshape((numz, numy, numx))
        BGradyMatrix = BGrady.reshape((numz, numy, numx))
        BGradzMatrix = BGradz.reshape((numz, numy, numx))
        # plt.imshow(BGradyMatrix[0,:,:])
        # plt.show()

        BGradxMatrix = np.ascontiguousarray(BGradxMatrix)
        BGradyMatrix = np.ascontiguousarray(BGradyMatrix)
        BGradzMatrix = np.ascontiguousarray(BGradzMatrix)
        #
        tempx = interp_3d.Interp3D(-self.PTL.u0 * BGradxMatrix, zArr, yArr, xArr)
        tempy = interp_3d.Interp3D(-self.PTL.u0 * BGradyMatrix, zArr, yArr, xArr)
        tempz = interp_3d.Interp3D(-self.PTL.u0 * BGradzMatrix, zArr, yArr, xArr)
        self.FxFunc_Internal_Fringe = lambda x, y, z: tempx((y, -z, x))
        self.FyFunc_Internal_Fringe = lambda x, y, z: tempz((y, -z, x))  # todo: THIS NEEDS TO BE TESTED MORE!!
        self.FzFunc_Internal_Fringe = lambda x, y, z: -tempy((y, -z, x))



        #xPlot=np.linspace(0,.029)
        #temp=[]
        #for x in xPlot:
        #   temp.append(self.FzFunc_Internal_Fringe(1e-3,x,1e-3))
        #plt.plot(xPlot,temp)
        #plt.show()
        #sys.exit()
    def fill_Force_Func_Seg(self):
        self.dataSeg = np.asarray(pd.read_csv(self.fileSeg, delim_whitespace=True, header=None))
        xArr = np.unique(self.dataSeg[:, 0])
        yArr = np.unique(self.dataSeg[:, 1])
        zArr = np.unique(self.dataSeg[:, 2])
        BGradx = self.dataSeg[:, 3]
        BGrady = self.dataSeg[:, 4]
        BGradz = self.dataSeg[:, 5]
        numx = xArr.shape[0]
        numy = yArr.shape[0]
        numz = zArr.shape[0]

        BGradxMatrix = BGradx.reshape((numz, numy, numx))
        BGradyMatrix = BGrady.reshape((numz, numy, numx))
        BGradzMatrix = BGradz.reshape((numz, numy, numx))
        # plt.imshow(BGradyMatrix[0,:,:])
        # plt.show()

        BGradxMatrix = np.ascontiguousarray(BGradxMatrix)
        BGradyMatrix = np.ascontiguousarray(BGradyMatrix)
        BGradzMatrix = np.ascontiguousarray(BGradzMatrix)
        #
        tempx = interp_3d.Interp3D(-self.PTL.u0 * BGradxMatrix, zArr, yArr, xArr)
        tempy = interp_3d.Interp3D(-self.PTL.u0 * BGradyMatrix, zArr, yArr, xArr)
        tempz = interp_3d.Interp3D(-self.PTL.u0 * BGradzMatrix, zArr, yArr, xArr)
        self.FxFunc_Seg = lambda x, y, z: tempx((y, -z, x))
        self.FyFunc_Seg = lambda x, y, z: tempz((y, -z, x))
        self.FzFunc_Seg = lambda x, y, z: -tempy((y, -z, x))

    def compute_K(self):
        #use the fit to the gradient of the magnetic field to find the k value in F=-k*x
        xFit=np.linspace(-self.rp/2,self.rp/2,num=10000)+self.dataSeg[:,0].mean()
        yFit=[]
        for x in xFit:
            yFit.append(self.FxFunc_Seg(x,0,0))
        xFit=xFit-self.dataSeg[:,0].mean()
        K = -np.polyfit(xFit, yFit, 1)[0] #fit to a line y=m*x+b, and only use the m component
        K0=12037000
        if .99*K0<K<1.01*K0:
            K=K0
        else:
            raise Exception('K VALUE FALLS OUTSIDE ACCEPTABLE BOUND')
        return K
    def force(self, q):
        # force at point q in element frame
        # q: particle's position in element frame
        F = np.zeros(3)  # force vector starts out as zero
        phi = np.arctan2(q[1], q[0])
        if phi<0: #constraint to between zero and 2pi
            phi+=2*np.pi
        if phi<self.ang: #if particle is inside bending angle region
            revs = int((self.ang-phi) // self.ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position='FIRST'
            elif revs == self.numMagnets * 2 - 1 or revs == self.numMagnets * 2 - 2:
                position='LAST'
            else:
                position='INNER'
            if position == 'INNER':
                quc = self.transform_Element_Coords_Into_Unit_Cell_Frame(q)  # get unit cell coords
                F[0] = self.FxFunc_Seg(*quc)
                F[1] = self.FyFunc_Seg(*quc)
                F[2] = self.FzFunc_Seg(*quc)
                F = self.transform_Unit_Cell_Force_Into_Element_Frame(F, q)  # transform unit cell coordinates into
                    # element frame
            elif position =='FIRST' or position == 'LAST':
                F=self.force_First_And_Last(q,position)
            else:
                raise Exception('UNSOPPORTED PARTICLE POSITION')
        elif phi>self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < q[0] < self.rb + self.ap) and (0 > q[1] > -self.Lcap): #If inside the cap on
                #westward side
                x,y,z=q.copy()
                x=x-self.rb
                F[0] = self.FxFunc_Cap(x,y,z)
                F[1] = self.FyFunc_Cap(x,y,z)
                F[2] = self.FzFunc_Cap(x,y,z)

            qTest=q.copy()
            qTest[0] = self.RIn_Ang[0, 0] * q[0] + self.RIn_Ang[0, 1] * q[1]
            qTest[1] = self.RIn_Ang[1, 0] * q[0] + self.RIn_Ang[1, 1] * q[1]
            if (self.rb - self.ap < qTest[0] < self.rb + self.ap) and (self.Lcap > qTest[1] > 0):
                x,y,z=qTest
                x=x-self.rb
                y=-y
                F[0]=self.FxFunc_Cap(x,y,z)
                F[1]=self.FyFunc_Cap(x,y,z)
                F[2]=self.FzFunc_Cap(x,y,z)
                Fx = F[0]
                Fy = F[1]
                F[0] = self.M_ang[0, 0] * Fx + self.M_ang[0, 1] * Fy
                F[1] = self.M_ang[1, 0] * Fx + self.M_ang[1, 1] * Fy
                qTest[0]+=-self.rb

        return F
    def force_First_And_Last(self,q,position):
        F = np.zeros(3)
        qNew=q.copy()
        if position=='FIRST':
            qx = qNew[0]
            qy = qNew[1]
            qNew[0] = self.M_ang[0, 0] * qx + self.M_ang[0, 1] * qy
            qNew[1] = self.M_ang[1, 0] * qx + self.M_ang[1, 1] * qy

            qNew[0]=qNew[0]-self.rb

            F[0] = self.FxFunc_Internal_Fringe(*qNew)
            F[1] = self.FyFunc_Internal_Fringe(*qNew)
            F[2] = self.FzFunc_Internal_Fringe(*qNew)

            Fx = F[0]
            Fy = F[1]
            F[0] = self.M_ang[0, 0] * Fx + self.M_ang[0, 1] * Fy
            F[1] = self.M_ang[1, 0] * Fx + self.M_ang[1, 1] * Fy
        if position=='LAST':
            qNew[0]=qNew[0]-self.rb
            F[0] = self.FxFunc_Internal_Fringe(*qNew)
            F[1] = self.FyFunc_Internal_Fringe(*qNew)
            F[2] = self.FzFunc_Internal_Fringe(*qNew)

        return F

class LensSimWithCaps(Lens_Ideal):
    def __init__(self, PTL, file2D, file3D, L, ap):
        super().__init__(PTL, None, None, None, None,fillParams=False)
        self.PTL=PTL
        self.file2D=file2D
        self.file3D=file3D
        self.L=L
        self.ap=ap
        self.Lcap=None
        self.Linner=None
        self.data2D=None
        self.data3D=None
        self.FxFunc_Cap=None
        self.FyFunc_Cap = None
        self.FzFunc_Cap = None
        self.FxFunc_Inner=None
        self.FyFunc_Inner = None
        self.FzFunc_Inner = None
        self.forceFact=1.0
        self.fill_Params()
    def fill_Params(self):
        if self.data3D is None:
            self.data3D = np.asarray(pd.read_csv(self.file3D, delim_whitespace=True, header=None))
            self.fill_Force_Func_Cap()
            self.Lcap = self.data3D[:,2].max() - self.data3D[:,2].min() - 2 * self.comsolExtraSpace
        if self.data2D is None:
            self.data2D = np.asarray(pd.read_csv(self.file2D, delim_whitespace=True, header=None))
            self.fill_Force_Func_2D()
            self.rp=(self.data2D[:,0].max()-self.data2D[:,0].min()-2*self.comsolExtraSpace)/2
            if self.ap is None:
                self.ap=.9*self.rp
        if self.L is not None:
            self.set_Length(self.L)

    def set_Length(self,L):
        self.L=L
        self.Linner=L-2*self.Lcap
        if self.Linner < 0:
            raise Exception('LENSES IS TOO SHORT TO ACCOMODATE FRINGE FIELDS')
        self.Lo = self.L
    def fill_Force_Func_Cap(self):
        xArr = np.unique(self.data3D[:, 0])
        yArr = np.unique(self.data3D[:, 1])
        zArr = np.unique(self.data3D[:, 2])
        BGradx = self.data3D[:, 3]
        BGrady = self.data3D[:, 4]
        BGradz = self.data3D[:, 5]
        numx = xArr.shape[0]
        numy = yArr.shape[0]
        numz = zArr.shape[0]
        BGradxMatrix = BGradx.reshape((numz, numy, numx))
        BGradyMatrix = BGrady.reshape((numz, numy, numx))
        BGradzMatrix = BGradz.reshape((numz, numy, numx))
        BGradxMatrix = np.ascontiguousarray(BGradxMatrix)
        BGradyMatrix = np.ascontiguousarray(BGradyMatrix)
        BGradzMatrix = np.ascontiguousarray(BGradzMatrix)
        #
        tempx = interp_3d.Interp3D(-self.PTL.u0 * BGradxMatrix, zArr, yArr, xArr)
        tempy = interp_3d.Interp3D(-self.PTL.u0 * BGradyMatrix, zArr, yArr, xArr)
        tempz = interp_3d.Interp3D(-self.PTL.u0 * BGradzMatrix, zArr, yArr, xArr)
        self.FxFunc_Cap = lambda x, y, z: tempz((x,y ,-z))
        self.FyFunc_Cap = lambda x, y, z: tempy((x,y ,-z))
        self.FzFunc_Cap = lambda x, y, z: -tempx((x,y ,-z))

        #xPlot=np.linspace(0,L)
        #yPlot=[]
        #for x in xPlot:
        #    yPlot.append(self.FxFunc_Cap(x,.003,.003))#tempy((x,.003,.003)))
        #plt.plot(xPlot,yPlot)
        #plt.show()
#
    def fill_Force_Func_2D(self):
        tempx = spi.LinearNDInterpolator(self.data2D[:, :2], -self.data2D[:, 2] * self.PTL.u0)
        tempy = spi.LinearNDInterpolator(self.data2D[:, :2], -self.data2D[:, 3] * self.PTL.u0)
        self.FxFunc_Inner = lambda x, y, z: 0.0
        self.FyFunc_Inner = lambda x, y, z: tempy(-z, y)
        self.FzFunc_Inner = lambda x, y, z: -tempx(-z, y)
    def force(self,q):
        F=np.zeros(3)
        if q[0]<self.Lcap:
            x,y,z=q
            x=self.Lcap-x
            F[0]= -self.forceFact * self.FxFunc_Cap(x, y, z)
            F[1]= self.forceFact * self.FyFunc_Cap(x, y, z)
            F[2]= self.forceFact * self.FzFunc_Cap(x, y, z)
        elif self.Lcap<q[0]<self.L-self.Lcap:
            F[0]= self.forceFact * self.FxFunc_Inner(*q)
            F[1]= self.forceFact * self.FyFunc_Inner(*q)
            F[2]= self.forceFact * self.FzFunc_Inner(*q)
        elif self.L-self.Lcap<q[0]<self.L:
            x,y,z=q
            x=x-(self.Linner+self.Lcap)
            F[0]= self.forceFact * self.FxFunc_Cap(x, y, z)
            F[1]= self.forceFact * self.FyFunc_Cap(x, y, z)
            F[2]= self.forceFact * self.FzFunc_Cap(x, y, z)
        return F