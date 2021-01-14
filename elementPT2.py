import numpy as np
import sys

class Element:
    def __init__(self):
        self.theta=None
        self.nb=None
        self.r0=None
        self.ROut=None
        self.Rin = None
        self.r1=None
        self.r2=None
        self.SO = None
        self.ROut=None
        self.RIn = None
        self.ang=None
        self.Lm=None #hard edge length of magnet along line through the bore
        self.L=None #length of magnet along line through the bore
        self.K=None
        self.rOffset=None
        self.Lo=None
        self.ro=None
        self.index=None #elements position in lattice
        self.cap=None
    def transform_Lab_Coords_Into_Orbit_Frame(self, q, cumulativeLength):
        #Change the lab coordinates into the particle's orbit frame.
        q = self.transform_Lab_Coords_Into_Element_Frame(q) #first change lab coords into element frame
        qo = self.transform_Element_Coords_Into_Orbit_Frame(q) #then element frame into orbit frame
        qo[0] = qo[0] + cumulativeLength #orbit frame is in the element's frame, so the preceding length needs to be
        #accounted for
        return qo

    def transform_Lab_Coords_Into_Element_Frame(self, q):
        pass
    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        return q.copy()
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

class Lens_Ideal(Element):
    def __init__(self,PTL,Bp,rp,L,ap):
        super().__init__()
        self.PTL=PTL
        self.Bp = Bp
        self.rp = rp
        self.L = L
        self.Lo=L
        self.ap = ap
        self.type='STRAIGHT'
        self.K = (2 * self.Bp * self.PTL.u0 / self.rp ** 2)
        self.ang=0
    def fill_Params(self):
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


class Bender_Ideal(Element):
    def __init__(self,PTL,ang,Bp,rp,rb,ap):
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
        self.fill_Params()
    def fill_Params(self):
        self.K = (2 * self.Bp * self.PTL.u0 / self.rp ** 2)
        self.rOffset = np.sqrt(self.rb ** 2 / 4 + self.PTL.m * self.PTL.v0Nominal ** 2 / self.K) - self.rb / 2
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
    def __init__(self,PTL,Lm,c1,c2,ap):
        super().__init__()
        self.PTL=PTL
        self.ap = ap
        self.Lm=Lm
        self.Lb=self.Lm
        self.c1=c1
        self.c2=c2
        self.type='COMBINER'
        self.fill_Params()
    def fill_Params(self):
        inputAngle, inputOffset = self.compute_Input_Angle_And_Offset(self.Lm)
        self.ang = inputAngle
        self.inputOffset = inputOffset
        self.La = self.ap * np.sin(self.ang)
        self.L = self.La * np.cos(self.ang) + self.Lb #TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING
        self.Lo = self.L

    def compute_Input_Angle_And_Offset(self, limit,h=10e-6):
        # this computes the output angle and offset for a combiner magnet
        # todo: make proper edge handling
        q = np.asarray([0, 0, 0])
        p = self.PTL.m * np.asarray([self.PTL.v0Nominal, 0, 0])
        # xList=[]
        # yList=[]
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
            # xList.append(q[0])
            # yList.append(npl.norm(F))
            q = q_n
            p = p_n
        # plt.plot(xList,yList)
        # plt.show()
        outputAngle = np.arctan2(p[1], p[0])
        outputOffset = q[1]
        return outputAngle, outputOffset
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        qNew=q.copy()
        qNew[:2] = qNew[:2] - self.r2
        qNew=self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew
    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        # TODO: FIX THIS
        qo=q.copy()
        qo[0] = self.L - qo[0]
        qo[1] = qo[1] - self.inputOffset
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


class BenderIdealSegmented(Element):
    def __init__(self, PTL, numMagnets, Lm, Bp, rp, rb, yokeWidth, space, ap,fillParams=True):
        super().__init__()
        self.PTL = PTL
        self.numMagnets = numMagnets
        self.Lm = Lm
        self.Bp = Bp
        self.rp = rp
        self.space = space
        self.rb = rb
        self.yokeWidth = yokeWidth
        self.ucAng = None
        self.type = 'BEND'
        self.segmented = True
        self.cap = False
        self.ap = ap
        self.RIn_Ang = None
        if fillParams==True:
            self.fill_Params()

    def fill_Params(self):
        self.K = (2 * self.Bp * self.PTL.u0 / self.rp ** 2)
        self.rOffset = np.sqrt(self.rb ** 2 / 4 + self.PTL.m * self.PTL.v0Nominal ** 2 / self.K) - self.rb / 2
        self.ro = (self.rb + self.rOffset)
        if self.numMagnets is not None:
            self.ucAng = np.arctan((self.Lm / 2 + self.space) / (self.rb - self.yokeWidth - self.rp))
            self.ang = 2 * self.ucAng * self.numMagnets
            self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
            self.Lo = self.ro * self.ang
            print(self.ang,self.rOffset)

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
        # q: particle's position in the unit cell where the force is acting
        FNew = F.copy()  # copy input vector to not modify the original
        phi = np.arctan2(q[1], q[0])  # the anglular displacement from output of bender to the particle. I use
        # output instead of input because the unit cell is conceptually located at the output so it's easier to visualize
        cellNum = int(phi // self.ucAng) + 1  # cell number that particle is in, starts at one
        if cellNum % 2 == 1:  # if odd number cell. Then the unit cell only needs to be rotated into that position
            rotAngle = 2 * (cellNum // 2) * self.ucAng
        else:
            m = np.tan(self.ucAng)
            M = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)
            Fx = FNew[0]
            Fy = FNew[1]
            FNew[0] = M[0, 0] * Fx + M[0, 1] * Fy
            FNew[1] = M[1, 0] * Fx + M[1, 1] * Fy
            rotAngle = 2 * ((cellNum - 1) // 2) * self.ucAng
        Fx = FNew[0]
        Fy = FNew[1]
        FNew[0] = Fx * np.cos(rotAngle) - Fy * np.sin(rotAngle)
        FNew[1] = Fx * np.sin(rotAngle) + Fy * np.cos(rotAngle)
        return FNew
    def transform_Element_Coords_Into_Unit_Cell_Frame(self,q):
        #As particle leaves unit cell, it does not start back over at the beginning, instead is turns around so to speak
        #and goes the other, then turns around again and so on. This is how the symmetry of the unit cell is exploited.
        qNew=q.copy()
        phi=self.ang-np.arctan2(q[1],q[0])
        revs=int(phi//self.ucAng) #number of revolutions through unit cell
        if revs%2==0: #if even
            theta = phi - self.ucAng * revs
            pass
        else: #if odd
            theta = self.ucAng-(phi - self.ucAng * revs)
            pass
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


    def transform_Element_Coords_Into_Orbit_Frame(self, q):
        qo = q.copy()
        angle = np.arctan2(qo[1], qo[0])
        if angle<0: #to use full 2pi with arctan2
            angle+=2*np.pi
        if angle>self.ang:
            if q[0]>self.rb: #particle is in the eastward cap
                qo[0] = self.Lcap + self.ang * self.ro + (-q[1])
                qo[1]=q[0]-self.ro
            else: #particle is in the westward cap
                qox=self.RIn_Ang[0,0]*q[0]+self.RIn_Ang[0,1]*q[1]
                qoy=self.RIn_Ang[1,0]*q[0]+self.RIn_Ang[1,1]*q[1]
                qo[0]=self.Lcap-qoy
                qo[1]=qox-self.ro
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
            if r < self.rb - self.ap or r > self.rb + self.ap:
                return False
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
    