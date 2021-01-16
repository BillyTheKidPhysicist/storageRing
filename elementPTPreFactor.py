import time
import numpy as np
import scipy.interpolate as spi
import pandas as pd
import numpy.linalg as npl
import sys
from interp3d import interp_3d
import matplotlib.pyplot as plt

#TODO: BREAK UP ELEMENTS INTO THEIR OWN CLASSES? USE POLYMORPHISM OR OTHER METHOD TO SIMPLIFY?
#todo: combine elements with more than one into one element
class Element:
    # Class to represent the lattice element such as a drift/lens/bender/combiner.
    # each element type has its own reference frame, as well as attributes, which I will described below. Note that
    # the details of the exported data from COMSOL must be carefully adhered to. For now, the way to do that is to just modify
    # the comsol files that produce the field values.
    # LENS_IDEAL: simple hard edge model of lens. Field is modeled as Bp*r**2/rp**2, where Bp is field at pole and rp
    # is radius of bore. Reference frame is with the input centered at the origin and output pointing at 0 degree
    # in the x,y plane
    # LENS_SIM_CAP: A short stub of a hexapole lens. The length of this stub is just enough that the magnetic field has
    # reached 99% of it inside the magnet, and fallen to 1% outside. These elements are used to efficiently simulate
    # a real lens. The inner portion's field is sampled from a transverse fit, whereas the ends are sampled from
    # these caps. As long as the caps are long enough, the inner region is mostly represented by a 2d slice. It seems
    # like a cap with a length of 3-4 boreradius is sufficient. Reference frame is with input centered at origin and output
    # facing 0 degree. NOTE!! the transform from lab to element must account for wether the element is at the input
    # or output because the element is not symmetric along its length
    #DRIFT: element free region. reference frame same as lens
    # BENDER_IDEAL: Simple hardedge model of ideal bender. Field is modeled as Bp*dr**2/rp**2 where dr=r-rb, where
    # r is the particles orbit relative to the center of the bending segment (not center of the bore), and rb is the
    # bending radius. reference frame is output is facing south and aligned with y=0 at (rb,0) where rb is bending radiu.
    # The center of the bender is at the origin. Input is at some positive angle relative to the output. A pi/2 bend
    # would have the input aligned with x=0 at position (0,rb) for example
    #BENDER_SIM_SEGMENTED This bending section is composed of discrete elements. It is modeled by considering a unit
    #cell and transforming the particle's coordinates into that frame to compute the force. The unit cell is
    # half a magnet at y=0, x=rb. An imaginary plane is at some angle to x,y=0,0 that defines the unit cell angle. Say the
    # unit cell angle is phi, and the particle is at angle theta, then the particle's angle in the unit cell is
    # theta-phi*(theta//phi). The modeled data must be in the format that comsol exports a grid as, and the model
    #must be pointing in the positive z direction and be located at x=rb
    # BENDER_SIM_SEGMENTED_CAP: same idea as lens cap. Reference frame element input at origin pointing south.
    # COMBINER_IDEAL: combiner modeled as a combination of quadrupole and dipole. The outlet is facing 180 degrees and
    # the inlet as at some angle in the upper right quadrant. The vacuum tube looks like a rectangular shape with another
    # recangular shape (the inlet) attached at some angle

    def __init__(self, args, type, PT):
        self.args = args
        self.type = type  # type of element. Options are 'BENDER', 'DRIFT', 'LENS_IDEAL', 'COMBINER'
        self.PT = PT  # particle tracer object
        self.Bp = None  # field strength at bore of element, T
        self.c1 = None  # dipole component of combiner, T
        self.c2 = None  # quadrupole component of combiner, T/m
        self.rp = None  # bore of element, m
        self.L = None  #length of element (or for vacuum tube).Not necesarily the length of the magnetic material. m
        self.Lo = None  #length of orbit inside element. This is different for bending, m
        self.Lseg=None #length of segment. Really, this is the length at the inner edge of the yoke, the edge closest to
        #the center of the bending ring.
        self.Lm=None #hard edge length of magnet in a segment
        self.rb = None  #'bending' radius of magnet. actual bending radius of atoms is slightly different cause they
        # ride on the outside edge, m
        self.yokeWidth=None #Thickness of the yoke, but also refers to the thickness of the permanent magnets, m
        self.ucAng=None #the angle that the unit cell makes with the origin. This is for the segmented bender. It is
            #modeled as a unit cell and the particle's coordinates are rotated into it's reference frame, rad
        self.numMagnets=None #the number of magnets. Keep in mind for a segmented bender two units cells make a total
            #of 1 magnet with each unit cell having half a magnet.
        self.space=None #extra space added to the length of the magnet in each direction. With segments this is to account for the
        #fact that you can't perfectly line them up. with the combiner this is to account for the extra space added for
        #fringe fields at each end
        self.FxFunc=None #Fit functions for froce from comsol data
        self.FyFunc=None #Fit functions for froce from comsol data
        self.FzFunc=None #Fit functions for froce from comsol data
        self.position=None #for bender 'caps' wether it's at the inlet or out of the bender


        self.r0 = None  # center of element (for bender this is at bending radius center),vector, m
        self.ro = None  # bending radius of orbit. Includes trajectory offset, m
        self.ang = 0  # Angle that the particles are bent, either bender or combiner. this is the change in  angle in
        # polar coordinates for a particle.
        self.r1 = None  # position vector of beginning of element in lab frame, m
        self.r2 = None  # position vector of ending of element in lab frame, m
        self.r1El = None  # position vector of beginning of element in element frame, m
        self.r2El = None  # position vector of ending of element in element frame, m

        self.ne = None  # normal vector to input of element. 2D, only in xy plane
        self.nb = None  # normal vector to output of element. 2D, only in xy plane
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
        self.La=None #inlet length of combiner. This is for the section of the combiner that particles enter into that is bent
        #relative to the combienr
        self.Lb=None #the length of the portion of the vaccuum tube that goes through the combiner. This is where the field
        #primarily is
        self.inputOffset = None  # for the combiner. Incoming particles enter the combiner with an offset relative to its
        # geometric center. A positive value corresponds to a trajectory that is shifted up from the center line
        #of y=0. This picture is most clear when imagining a the ideal case of a hard edge combiner, but works with
        # the real field model as well
        self.LFunc = None  # for the combiner. The length along the trajector that the particle has traveled. This length
        # is referring to the nominal trajectory, not the actual distance of the particle
        self.distFunc = None  # The transerse distance from the nominal trajectory of the particle.
        self.cFact = None  # factor in the function y=c*x**2. This is used for finding the trajectory of the particle
        # in the combiner.
        self.trajLength = None  # total length of trajectory, m. This is for combiner because it is not trivial like
        # benders or lens or drifts
        self.extraSpace = .0001  # extraspace added in each dimension past the vacuum tube. This is so the interpolation works
        # correctly, but the specific value is used here to infer the dimensions of the vacuum tub. This value
        # is added in each dimension at the beginning and end when exporting a grid from comsol. THIS MUST BE ADDED
        self.data = None  # array containing the numeric values of a magnetic field. can be for 2d or 3d data. 
        self.forceFitFunc = None  # an interpolation that the force in the x,y,z direction at a given point
        self.forceFact=1 #factor that multiplies field values to simulate
        self.edgeFact=None #ratio of fringe field length to bore radius
        self.width=None #width of combiner vacuum tube

        #TODO: THIS SYSTEM DOESN'T MAKE SENSE ANYMORE
        self.unpack_Args()

        self.fill_Params_And_Functions()
    def unpack_Args(self):
        # unload the user supplied arguments into the elements parameters. Also, load data and compute parameters
        if self.type == 'LENS_IDEAL':
            self.Bp = self.args[0]
            self.rp = self.args[1]
            self.L = self.args[2]
            self.ap = self.args[3]
        elif self.type == 'LENS_SIM_CAP':
            self.data=np.asarray(pd.read_csv(self.args[0],delim_whitespace=True,header=None))
            apFrac = self.args[1] #fraction size of apeture
            self.rp = self.data[:, 0].max() - self.extraSpace
            self.ap=self.rp*apFrac
            self.position = self.args[2] #wether inlet or outlet
        elif self.type == "LENS_SIM_TRANSVERSE":
            self.data=np.asarray(pd.read_csv(self.args[0],delim_whitespace=True,header=None))
            self.L = self.args[1]
            self.rp=self.data[:,0].max()-self.extraSpace
            self.ap=self.args[2]*self.rp
            self.edgeFact=self.args[3]
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
        elif self.type == "BENDER_SIM_SEGMENTED":

            self.data=np.asarray(pd.read_csv(self.args[0],delim_whitespace=True,header=None))
            self.rb = self.args[3]
            self.space = self.args[4]
            self.Lseg = self.args[1] + 2 * self.space  # total length is hard magnet length plus 2 times extra space
            self.rp = self.args[2]
            self.yokeWidth = self.args[5]
            self.numMagnets = self.args[6]
            self.ap = self.args[7]
        elif self.type == "BENDER_SIM_SEGMENTED_CAP":
            self.data=np.asarray(pd.read_csv(self.args[0],delim_whitespace=True,header=None))
            self.L = self.args[1]
            self.rOffset = self.args[2]
            self.rp = self.args[3]
            self.ap = self.args[4]
            self.position = self.args[5]
        elif self.type == "BENDER_IDEAL_SEGMENTED":
            self.Lm = self.args[0]
            self.Bp = self.args[1]
            self.rb = self.args[2]
            self.rp = self.args[3]
            self.ap=self.rp*.9
            self.yokeWidth = self.args[4]
            self.numMagnets = self.args[5]  # number of magnets which is half the number of unit cells
            self.space = self.args[6]
            self.Lseg=self.Lm+2*self.space
        elif self.type == 'COMBINER_IDEAL':
            self.Lb = self.args[0]
            self.ang = self.args[1]
            self.inputOffset = self.args[2]
            self.ap = self.args[3]
            self.c1 = self.args[4]
            self.c2 = self.args[5]
        elif self.type == 'LENS_SIM':
            self.data=np.asarray(pd.read_csv(self.args[0],delim_whitespace=True,header=None))
        elif self.type=='COMBINER_SIM':
            self.data=np.asarray(pd.read_csv(self.args[0],delim_whitespace=True,header=None))
        else:
            raise Exception('No proper element name provided')

    def fill_Params_And_Functions(self):
        #unload the user supplied arguments into the elements parameters. Also, load data and compute parameters
        if self.type == 'LENS_IDEAL':
            self.Lo=self.L
            self.K = (2 * self.Bp * self.PT.u0 / self.rp ** 2)
        elif self.type=='LENS_SIM_CAP':

            xArr = np.unique(self.data[:, 0])
            yArr = np.unique(self.data[:, 1])
            zArr = np.unique(self.data[:, 2])
            BGradx = self.data[:, 3]
            BGrady = self.data[:, 4]
            BGradz = self.data[:, 5]
            numx = xArr.shape[0]
            numy = yArr.shape[0]
            numz = zArr.shape[0]
            #need to shift the data
            self.L=zArr.max()-self.extraSpace-(zArr.min()+self.extraSpace)
            self.edgeFact=self.L/self.rp
            self.Lo=self.L
            zArr=zArr-(np.min(zArr)+self.extraSpace) #shift down to zeroinclude additional space from comsol for overshooting the rgion
            sys.exit()


            BGradxMatrix = BGradx.reshape((numz, numy, numx))
            BGradyMatrix = BGrady.reshape((numz, numy, numx))
            BGradzMatrix = BGradz.reshape((numz, numy, numx))

            BGradxMatrix = np.ascontiguousarray(BGradxMatrix)
            BGradyMatrix = np.ascontiguousarray(BGradyMatrix)
            BGradzMatrix = np.ascontiguousarray(BGradzMatrix)
            #
            tempx = interp_3d.Interp3D(-self.PT.u0 * BGradxMatrix, zArr, yArr, xArr)
            tempy = interp_3d.Interp3D(-self.PT.u0 * BGradyMatrix, zArr, yArr, xArr)
            tempz = interp_3d.Interp3D(-self.PT.u0 * BGradzMatrix, zArr, yArr, xArr)

            self.FxFunc = lambda x, y, z: -tempz((self.L-x, y, z))
            self.FyFunc = lambda x, y, z: tempy((self.L-x, y, z))
            self.FzFunc = lambda x, y, z: tempx((self.L-x, y, z))


        elif self.type=="LENS_SIM_TRANSVERSE":
            if self.L is not None:
                self.Lo = self.L
            tempx=spi.LinearNDInterpolator(self.data[:,:2],-self.data[:,2]*self.PT.u0)
            tempy = spi.LinearNDInterpolator(self.data[:, :2], -self.data[:, 3]*self.PT.u0)
            self.FyFunc = lambda x, y, z: tempy(-z,y)
            self.FzFunc = lambda x, y, z: -tempx(-z,y)
            self.FxFunc = lambda x, y, z: 0.0

        elif self.type == 'DRIFT':
            pass
        elif self.type == 'BENDER_IDEAL':
            self.K = (2 * self.Bp * self.PT.u0 / self.rp ** 2)  # reduced force
            if self.rb is not None:
                self.rOffset =self.compute_rOffset()  # this method does not
                # account for reduced speed in the bender from energy conservation
                self.ro = self.rb + self.rOffset
            if self.ang is not None:
                self.L = self.ang * self.rb
                self.Lo = self.ang * self.ro
        elif self.type=="BENDER_SIM_SEGMENTED":
            xArr = np.unique(self.data[:, 0])
            yArr = np.unique(self.data[:, 1])
            zArr = np.unique(self.data[:, 2])
            BGradx=self.data[:,3]
            BGrady=self.data[:,4]
            BGradz=self.data[:,5]
            numx = xArr.shape[0]
            numy = yArr.shape[0]
            numz = zArr.shape[0]

            BGradxMatrix = BGradx.reshape((numz, numy, numx))
            BGradyMatrix = BGrady.reshape((numz, numy, numx))
            BGradzMatrix = BGradz.reshape((numz, numy, numx))
            #plt.imshow(BGradyMatrix[0,:,:])
            #plt.show()


            BGradxMatrix=np.ascontiguousarray(BGradxMatrix)
            BGradyMatrix=np.ascontiguousarray(BGradyMatrix)
            BGradzMatrix=np.ascontiguousarray(BGradzMatrix)
#
            tempx = interp_3d.Interp3D(-self.PT.u0*BGradxMatrix, zArr, yArr, xArr)
            tempy = interp_3d.Interp3D(-self.PT.u0*BGradyMatrix, zArr, yArr, xArr)
            tempz = interp_3d.Interp3D(-self.PT.u0*BGradzMatrix, zArr, yArr, xArr)
            self.FxFunc=lambda x,y,z:tempx((y,-z,x))
            self.FyFunc = lambda x, y, z: tempz((y, -z, x))
            self.FzFunc = lambda x, y, z: -tempy((y, -z, x))

            self.K = self.compute_K_Value()
            if self.rb is not None and self.numMagnets is not None:
                D = self.rb - self.rp - self.yokeWidth
                self.ucAng = np.arctan(self.Lseg / (2 * D))
                self.ang = 2 * self.numMagnets * self.ucAng  # number of units cells times bending angle of 1 cel
                self.rOffset =self.compute_rOffset()
                self.ro = self.rb + self.rOffset
                self.Lo = self.ang * self.ro
                self.L=self.rb*self.ang

        elif self.type == "BENDER_SIM_SEGMENTED_CAP":
            self.Lo=self.L
            xArr = np.unique(self.data[:, 0])
            yArr = np.unique(self.data[:, 1])
            zArr = np.unique(self.data[:, 2])
            BGradx=self.data[:,3]
            BGrady=self.data[:,4]
            BGradz=self.data[:,5]
            numx = xArr.shape[0]
            numy = yArr.shape[0]
            numz = zArr.shape[0]
            zArr=np.flip(zArr)

            BGradxMatrix = BGradx.reshape((numz, numy, numx))
            BGradyMatrix = BGrady.reshape((numz, numy, numx))
            BGradzMatrix = BGradz.reshape((numz, numy, numx))


            BGradxMatrix = np.ascontiguousarray(BGradxMatrix)
            BGradyMatrix = np.ascontiguousarray(BGradyMatrix)
            BGradzMatrix = np.ascontiguousarray(BGradzMatrix)
            #
            tempx = interp_3d.Interp3D(-self.PT.u0*BGradxMatrix, zArr, yArr, xArr)
            tempy = interp_3d.Interp3D(-self.PT.u0*BGradyMatrix, zArr, yArr, xArr)
            tempz = interp_3d.Interp3D(-self.PT.u0*BGradzMatrix, zArr, yArr, xArr)

            self.FxFunc = lambda x,y,z:tempz((x-self.L,y,z))
            self.FyFunc = lambda x, y, z: tempy((x-self.L,y,z))
            self.FzFunc = lambda x, y, z: -tempx((x-self.L,y,z))

        elif self.type=="BENDER_IDEAL_SEGMENTED":
            self.Lseg=self.Lm+self.space*2 #add extra space
            self.K = (2 * self.Bp * self.PT.u0 / self.rp ** 2)  # reduced force
            if self.rb is not None and self.numMagnets is not None:
                self.rOffset = self.compute_rOffset()  # this method does not
                # account for reduced speed in the bender from energy conservation
                self.ro = self.rb + self.rOffset
                #compute the angle that the unit cell makes as well as total bent angle
                D = self.rb - self.rp - self.yokeWidth
                self.ucAng = np.arctan(self.Lseg / (2 * D))
                self.ang=2*self.numMagnets * self.ucAng #number of units cells times bending angle of 1 cell
                self.Lo = self.ang * self.ro
                self.L=self.ang*self.rb
        elif self.type == 'COMBINER_IDEAL':
            inputAngle, inputOffset = self.compute_Input_Angle_And_Offset(1, 200)
            self.ang=inputAngle
            self.inputOffset=inputOffset
            self.La = self.ap * np.sin(self.ang)
            self.L=self.La*np.cos(self.ang)+self.Lb
            self.Lo=self.L
        elif self.type == 'LENS_SIM':
            self.Fx = spi.LinearNDInterpolator(self.data[:, :3], self.data[:, 3] * self.PT.u0)
            self.Fy = spi.LinearNDInterpolator(self.data[:, :3], self.data[:, 4] * self.PT.u0)
            self.Fz = spi.LinearNDInterpolator(self.data[:, :3], self.data[:, 5] * self.PT.u0)
        elif self.type=='COMBINER_SIM':
            self.space=4*1.1E-2 #extra space past the hard edge on either end to account for fringe fields
            self.Lm=.18
            self.Lb=self.space+self.Lm #the combiner vacuum tube will go from a short distance from the ouput right up
            #to the hard edge of the input
            xArr = np.unique(self.data[:, 0])
            yArr = np.unique(self.data[:, 1])
            zArr = np.unique(self.data[:, 2])
            BGradx=self.data[:,3]
            BGrady=self.data[:,4]
            BGradz=self.data[:,5]
            numx = xArr.shape[0]
            numy = yArr.shape[0]
            numz = zArr.shape[0]
            self.ap = (yArr.max() - yArr.min() - 2 * self.extraSpace)/2.0
            BGradxMatrix = BGradx.reshape((numz, numy, numx))
            BGradyMatrix = BGrady.reshape((numz, numy, numx))
            BGradzMatrix = BGradz.reshape((numz, numy, numx))

            BGradxMatrix = np.ascontiguousarray(BGradxMatrix)
            BGradyMatrix = np.ascontiguousarray(BGradyMatrix)
            BGradzMatrix = np.ascontiguousarray(BGradzMatrix)
            #
            tempx = interp_3d.Interp3D(-self.PT.u0 * BGradxMatrix, zArr, yArr, xArr)
            tempy = interp_3d.Interp3D(-self.PT.u0 * BGradyMatrix, zArr, yArr, xArr)
            tempz = interp_3d.Interp3D(-self.PT.u0 * BGradzMatrix, zArr, yArr, xArr)

            self.FxFunc = lambda x, y, z: tempx((z,y,x))
            self.FyFunc = lambda x, y, z: tempy((z,y,x))
            self.FzFunc = lambda x, y, z: tempz((z,y,x))
            inputAngle,inputOffset=self.compute_Input_Angle_And_Offset(1,200)
            self.ang=inputAngle
            self.inputOffset=inputOffset-np.tan(inputAngle)*self.space #the input offset is measure at the end of the hard
            #edge

            #the inlet length needs to be long enough to extend past the fringe fields
            #TODO: MAKE EXACT, now it overshoots
            self.La=self.space+np.tan(self.ang)*self.ap
            self.Lo=self.La+self.Lb
            self.L=self.Lo
            #plot along x
            #xPlot=np.linspace(0,self.Lm+2*self.space)
            #y0=0
            #z0=0
            #plotList=[]
            #for x in xPlot:
            #    plotList.append(self.FxFunc(x,y0,z0))
            #plt.plot(xPlot,plotList)
            #plt.show()
        else:
            raise Exception('No proper element name provided')
    def compute_rOffset(self,rb=None,K=None):
        if K is None:
            K=self.K
        if rb is None:
            rb=self.rb
        if self.type=="BENDER_IDEAL_SEGMENTED" or self.type=='BENDER_IDEAL':
            return  np.sqrt(rb ** 2 / 4 + self.PT.m * self.PT.v0Nominal ** 2 / K) - rb / 2 #does not account for reduced
                #energy
        elif self.type=="BENDER_SIM_SEGMENTED":
            #r1=np.sqrt(rb ** 2 / 4 + self.PT.m * self.PT.v0Nominal ** 2 / K) - rb / 2
            #r2=np.sqrt(self.rb ** 2 / 16 + self.PT.m * self.PT.v0Nominal ** 2 / (2 * self.K)) - self.rb / 4
            #print(r1,r2)
            #sys.exit()
            return np.sqrt(rb ** 2 / 16 + self.PT.m * self.PT.v0Nominal ** 2 / (2 * K)) - rb / 4  # this
            # acounts for reduced energy
        else:
            raise Exception("NOT IMPLEMENTED")
    def compute_Input_Angle_And_Offset(self,m,v0, h=10e-6):
        # this computes the output angle and offset for a combiner magnet
        #todo: make proper edge handling
        q = np.asarray([0, 0, 0])
        p = m * np.asarray([v0, 0, 0])
        #xList=[]
        #yList=[]
        if self.type=="COMBINER_IDEAL":
            limit=self.Lb
        elif self.type=='COMBINER_SIM':
            limit=self.Lm+2*self.space
        else:
            raise Exception('NOT IMPLEMENTED')
        while True:
            F = self.force(q)
            a = F / m
            q_n = q + (p / m) * h + .5 * a * h ** 2
            F_n = self.force(q_n)
            a_n = F_n / m  # accselferation new or accselferation sub n+1
            p_n = p + m * .5 * (a + a_n) * h
            if q_n[0] > limit:  # if overshot, go back and walk up to the edge assuming no force
                dr = limit - q[0]
                dt = dr / (p[0] / m)
                q = q + (p / m) * dt
                break
            #xList.append(q[0])
            #yList.append(npl.norm(F))
            q = q_n
            p = p_n
        #plt.plot(xList,yList)
        #plt.show()
        outputAngle = np.arctan2(p[1], p[0])
        outputOffset = q[1]
        return outputAngle, outputOffset
    def transform_Lab_Coords_Into_Orbit_Frame(self, q, cumulativeLength):
        #Change the lab coordinates into the particle's orbit frame.
        q = self.transform_Lab_Coords_Into_Element_Frame(q) #first change lab coords into element frame

        qo = self.transform_Element_Coords_Into_Orbit_Frame(q) #then element frame into orbit frame
        qo[0] = qo[0] + cumulativeLength #orbit frame is in the element's frame, so the preceding length needs to be
            #accounted for
        return qo
    def compute_K_Value(self):
        #use the fit to the gradient of the magnetic field to find the k value in F=-k*x
        xFit=np.linspace(-self.rp/2,self.rp/2,num=10000)+self.data[:,0].mean()
        yFit=[]
        for x in xFit:
            yFit.append(self.FxFunc(x,0,0))
        xFit=xFit-self.data[:,0].mean()
        K = -np.polyfit(xFit, yFit, 1)[0] #fit to a line y=m*x+b, and only use the m component
        return K
    def transform_Lab_Coords_Into_Element_Frame(self, q):
        # q: particle coords in x and y plane. numpy array
        # self: element object to transform to
        # get the coordinates of the particle in the element's frame. See element class for description
        qNew = q.copy()  # CAREFUL ABOUT EDITING THINGS YOU DON'T WANT TO EDIT!!!! Need to copy
        if self.type == 'DRIFT' or self.type == 'LENS_IDEAL' or self.type=="LENS_SIM_TRANSVERSE":
            qNew[0] = qNew[0] - self.r1[0]
            qNew[1] = qNew[1] - self.r1[1]
        elif self.type=="BENDER_SIM_SEGMENTED_CAP" or self.type=='LENS_SIM_CAP':
            if self.position=='INLET':
                r=self.r1
            elif self.position=='OUTLET':
                r=self.r2
            else:
                raise Exception('NOT IMPLEMENTED')
            qNew[0]=qNew[0]-r[0]
            qNew[1]=qNew[1]-r[1]
        elif self.type == 'BENDER_IDEAL' or self.type == 'BENDER_IDEAL_SEGMENTED' or self.type=='BENDER_SIM_SEGMENTED':
            qNew[:2] = qNew[:2] - self.r0
        elif self.type == 'COMBINER_IDEAL' or self.type=='COMBINER_SIM':
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
        if self.type == 'LENS_IDEAL' or self.type == 'DRIFT' or self.type=="LENS_SIM_TRANSVERSE":
            pass
        elif self.type=="BENDER_SIM_SEGMENTED_CAP" or self.type=='LENS_SIM_CAP':
            if self.position=='INLET':
                qo[1] = qo[1]-self.rOffset
            elif self.position=='OUTLET':
                qo[1] = -qo[1] -self.rOffset
                qo[0]=self.L-q[0]

        elif self.type == 'BENDER_IDEAL' or self.type == 'BENDER_IDEAL_SEGMENTED' or self.type=='BENDER_SIM_SEGMENTED':
            qo = q.copy()
            phi = self.ang - np.arctan2(q[1], q[0])  # angle swept out by particle in trajectory. This is zero
            # when the particle first enters
            ds = self.ro * phi
            qos = ds
            qox = np.sqrt(q[0] ** 2 + q[1] ** 2) - self.ro
            qo[0] = qos
            qo[1] = qox
        elif self.type == 'COMBINER_IDEAL' or self.type=='COMBINER_SIM':
            #TODO: FIX THIS
            qo[0]=self.L-qo[0]
            qo[1]=qo[1]-self.inputOffset

        return qo

    def force(self, q):
        # force at point q in element frame
        #q: particle's position in element frame
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
        elif self.type == 'COMBINER_IDEAL' or self.type=='COMBINER_IDEAL':
            if q[0] < self.Lb:
                B0 = np.sqrt((self.c2 * q[2]) ** 2 + (self.c1 + self.c2 * q[1]) ** 2)
                F[1] = self.PT.u0 * self.c2 * (self.c1 + self.c2 * q[1]) / B0
                F[2] = self.PT.u0 * self.c2 ** 2 * q[2] / B0
        elif self.type=='COMBINER_SIM':
            F[0] = self.FxFunc(*q)
            F[1] = self.FyFunc(*q)
            F[2] = self.FzFunc(*q)
            #print(F, q[1], q[0],q[0]-self.Lb, self.type)
        elif self.type=='BENDER_IDEAL_SEGMENTED':
            #This is a little tricky. The coordinate in the element frame must be transformed into the unit cell frame and
            #then the force is computed from that. That force is then transformed back into the unit cell frame
            quc=self.transform_Element_Into_Unit_Cell_Frame(q) #get unit cell coords
            if quc[1]<self.Lm/2: #if particle is inside the magnet region
                F[0]=-self.K*(quc[0]-self.rb)
                F[2]=-self.K*quc[2]
                F = self.transform_Unit_Cell_Force_Into_Element_Frame(F, q, quc) #transform unit cell coordinates into
                #element frame
        elif self.type=='BENDER_SIM_SEGMENTED':
            quc = self.transform_Element_Into_Unit_Cell_Frame(q)  # get unit cell coords
            qucCopy=quc.copy()
            F[0]=self.FxFunc(*qucCopy)
            F[1]=self.FyFunc(*qucCopy) #element is rotated 90 degrees so they are swapped
            F[2]=self.FzFunc(*qucCopy) #element is rotated 90 degrees so they are swapped
            F = self.transform_Unit_Cell_Force_Into_Element_Frame(F, q,quc)#transform unit cell coordinates into
        elif self.type=="BENDER_SIM_SEGMENTED_CAP" or self.type=="LENS_SIM_TRANSVERSE" or self.type=='LENS_SIM_CAP':

            F[0] = self.FxFunc(*q)
            F[1] = self.FyFunc(*q)
            F[2] = self.FzFunc(*q)

        else:
            raise Exception('not yet implemented')
        F=self.forceFact*F
        return F
    def transform_Element_Into_Unit_Cell_Frame(self,q):
        #As particle leaves unit cell, it does not start back over at the beginning, instead is turns around so to speak
        #and goes the other, then turns around again and so on. This is how the symmetry of the unit cell is exploited.
        if self.type == 'BENDER_IDEAL_SEGMENTED' or self.type == 'BENDER_SIM_SEGMENTED':
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
        else:
            raise Exception('not implemented')


    def transform_Unit_Cell_Force_Into_Element_Frame(self, F, q,quc):
        #transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
        #that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
        # or leaving the element interface as mirror images of each other.
        #F: Force to be rotated out of unit cell frame
        #q: particle's position in the unit cell where the force is acting
        FNew=F.copy() #copy input vector to not modify the original
        phi =np.arctan2(q[1], q[0]) #the anglular displacement from output of bender to the particle. I use
        #output instead of input because the unit cell is conceptually located at the output so it's easier to visualize
        cellNum = int(phi // self.ucAng)+1 #cell number that particle is in, starts at one




        if cellNum%2==1: #if odd number cell. Then the unit cell only needs to be rotated into that position
            rotAngle = 2 * (cellNum // 2) * self.ucAng
        else:
            m = np.tan(self.ucAng)
            M = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)
            Fx = FNew[0]
            Fy = FNew[1]
            FNew[0]=M[0,0]*Fx+M[0,1]*Fy
            FNew[1]=M[1,0]*Fx+M[1,1]*Fy
            rotAngle = 2 * ((cellNum-1) // 2) * self.ucAng

        Fx = FNew[0]
        Fy = FNew[1]
        FNew[0] = Fx * np.cos(rotAngle) - Fy *np.sin(rotAngle)
        FNew[1] = Fx*np.sin(rotAngle) +Fy*np.cos(rotAngle)

        return FNew

    def transform_Element_Frame_Vector_To_Lab_Frame(self, vec):
        # rotate vector out of element frame into lab frame
        #vec: vector in
        vecNew=vec.copy()#copy input vector to not modify the original
        vecx = vecNew[0]
        vecy = vecNew[1]
        vecNew[0] = vecx * self.ROut[0, 0] + vecy * self.ROut[0, 1]
        vecNew[1] = vecx * self.ROut[1, 0] + vecy * self.ROut[1, 1]
        return vecNew

    def transform_Lab_Momentum_Into_Orbit_Frame(self, q, p):
        # rotate or transform momentum lab fame to orbit frame
        #q: particle's position in lab frame
        #p: particle's momentum in lab frame
        pNew = p.copy()#copy input vector to not modify the original
        #rotate momentum into element frame
        pNew[0] = p[0] * self.RIn[0, 0] + p[1] * self.RIn[0, 1]
        pNew[1] = p[0] * self.RIn[1, 0] + p[1] * self.RIn[1, 1]
        if self.type == 'BENDER_IDEAL' or self.type == 'BENDER_IDEAL_SEGMENTED' or self.type == 'BENDER_SIM_SEGMENTED':
            # need to use a change of vectors from cartesian to polar for bender
            q = self.transform_Lab_Coords_Into_Element_Frame(q)
            pNew = p.copy()
            sDot = (q[0] * pNew[1] - q[1] * pNew[0]) / np.sqrt((q[0] ** 2 + q[1] ** 2))
            rDot = (q[0] * pNew[0] + q[1] * pNew[1]) / np.sqrt((q[0] ** 2 + q[1] ** 2))
            po = np.asarray([sDot, rDot, pNew[2]])
            return po
        elif self.type == 'LENS_IDEAL' or self.type == 'DRIFT' or self.type=="BENDER_SIM_SEGMENTED_CAP" or self.type == 'LENS_SIM_TRANSVERSE'\
                or self.type=='LENS_SIM_CAP':
            return pNew
        else:
            raise Exception('NOT YET IMPLEMENTED')
    def get_Potential_Energy(self,q):
        #compute potential energy of element. For ideal elements it's a simple formula, for simulated elements the magnet
        #field must be sampled
        #q: lab frame coordinate
        PE=0
        qel = self.transform_Lab_Coords_Into_Element_Frame(q) #transform to element frame
        if self.type == 'LENS_IDEAL':
            r = np.sqrt(qel[1] ** 2 + q[2] ** 2)
            B = self.Bp * r ** 2 / self.rp ** 2
            PE = self.PT.u0 * B
        elif self.type == 'DRIFT':
            pass
        elif self.type == 'BENDER_IDEAL':
            deltar = np.sqrt(qel[0] ** 2 + qel[1] ** 2) - self.rb
            B = self.Bp * deltar ** 2 / self.rp ** 2
            PE = self.PT.u0 * B
        elif self.type=='BENDER_IDEAL_SEGMENTED':
            #magnetic field is zero if particle is in the empty sliver of ideal element
            quc=self.transform_Element_Into_Unit_Cell_Frame(qel)
            if quc[1]<self.Lseg/2:
                deltar=np.sqrt(quc[0]**2+quc[2]**2)-self.rb
                B = self.Bp * deltar ** 2 / self.rp ** 2
                PE = self.PT.u0 * B
        else:
            raise Exception('not implemented')
        return PE
    def is_Coord_Inside(self,q):
        #check with fast geometric arguments if the particle is inside the element. This won't necesarily work for all
        #elements. If True is retured, the particle is inside. If False is returned, it is defintely outside. If none is
        #returned, it is unknown
        #q: coordinates to test
        if self.type=='LENS_IDEAL' or self.type=='DRIFT' or self.type=="BENDER_SIM_SEGMENTED_CAP" or self.type=="LENS_SIM_TRANSVERSE"\
                or self.type=='LENS_SIM_CAP':
            if np.abs(q[2])>self.ap:
                return False
            elif np.abs(q[1])>self.ap:
                return False
            elif q[0]<0 or q[0]>self.L:
                return False
            else:
                return True
        elif self.type=='BENDER_IDEAL' or self.type=='BENDER_IDEAL_SEGMENTED' or self.type=='BENDER_SIM_SEGMENTED':
            if np.abs(q[2])>self.ap: #if clipping in z direction
                return False
            phi=np.arctan2(q[1],q[0])
            if (phi>self.ang and phi<2*np.pi) or phi<0: #if outside bender's angle range
                return False

            r=np.sqrt(q[0]**2+q[1]**2)
            if r<self.rb-self.ap or r>self.rb+self.ap:
                return False
            return True
        elif self.type=='COMBINER_IDEAL' or self.type=='COMBINER_SIM':
            if np.abs(q[2])>self.ap:
                return False
            elif q[0]<self.Lb and q[0]>0: #particle is in the straight section that passes through the combiner
                if np.abs(q[1])<self.ap:
                    return True
            else: #particle is in the bent section leading into combiner
                #TODO: ADD THIS LOGIc
                return None
        else:
            raise Exception('not implemented')
