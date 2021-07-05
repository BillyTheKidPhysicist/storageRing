from profilehooks import profile
import time
from injectionOptimizer import ApetureOptimizer
from particleTracerLattice import ParticleTracerLattice
from SwarmTracer import SwarmTracer
import scipy.interpolate as spi
import numpy as np
from ParaWell import ParaWell
import matplotlib.pyplot as plt
from ParticleClass import Swarm,Particle
import scipy.optimize as spo
import scipy.special as sps
import poisson_disc
import scipy.signal as spsig

class Compactor(ApetureOptimizer):
    def __init__(self,h=1e-5):
        self.X={"Lo1":None,"Lm1":None,"Bp1":None,"rp1":None,'Lsep1':None,'Lm2':None,'Bp2':None,'rp2':None,'Lsep2':None,
                'Lm3':None,'rp3':None,'Lsep3':None,'Lm4':None,'rp4':None,'sigma':None}
        self.fringeFrac=3.0
        self.v0Nominal = 200.0
        self.lattice=None
        self.h=h
        self.swarmInitial = None
        self.helper=ParaWell()
    def build_Lattice(self):
        #sigma here moves the element upwards
        self.lattice = ParticleTracerLattice(self.v0Nominal)
        self.lattice.add_Drift(self.X["Lo1"]-self.X['rp1']*self.fringeFrac,ap=.05) #need to update this after adding the first lens
        # self.lattice.add_Lens_Ideal(self.X["Lm2"],-1*self.X['Bp2'],self.X['rp1'])
        self.lattice.add_Lens_Sim_With_Caps(None,None,self.fringeFrac,None)#"optimizerData_Full.txt"
        self.lattice.add_Drift(self.X['Lsep1'],ap=.25) #need to update this after adding the first lens
        # self.lattice.add_Lens_Sim_With_Caps('lens2D_Injection_Short.txt','lens3D_Injection_Short.txt',self.fringeFrac,L=self.X['Lm2'],
        #                                     rp=self.X['rp2'],ap=.8*self.X['rp2'])
        # self.lattice.add_Drift(self.X['Lsep2'],ap=.02) #need to update this after adding the first lens
        # self.lattice.add_Lens_Sim_With_Caps('lens2D_Injection_Short.txt','lens3D_Injection_Short.txt',self.fringeFrac,L=self.X['Lm3'],
        #                                     rp=self.X['rp3'],ap=.8*self.X['rp3'])
        # self.lattice.add_Drift(self.X['Lsep3'],ap=.02) #need to update this after adding the first lens
        # self.lattice.add_Bump_Lens_Ideal(self.X['Lm3'],1.0,self.X['rp4'],self.X['sigma'],ap=.8*self.X['rp4'])
        # self.lattice.add_Drift(.25,ap=.05)
        # self.lattice.elList[1].ap=self.lattice.elList[1].rp*self.X['ap1']
        # self.lattice.end_Lattice(enforceClosedLattice=False, latticeType='injector', surpressWarning=True,trackPotential=True)
    def updateX_With_List(self,args):
        #unpack args into the lattice paramters dictionary
        Lm1,Bp1,rp1=args
        self.X['Lm1']=Lm1
        self.X['rp1']=rp1
        self.X['Bp1']=Bp1
    def trace_Through_Compactor(self,parallel=False):
        swarmTracer = SwarmTracer(self.lattice)
        swarm = swarmTracer.trace_Swarm_Through_Lattice(self.swarmInitial, self.h, 1.0, fastMode=False,parallel=parallel)
        #now 'unclip' particles that got clipped at the last element. The last element represents free space so any
        #clipping there is not part of the model now and is considered not clipped cause I assume all that is captured.
        #Also go through and assign interpolated functions
        for particle in swarm:
            if particle.currentElIndex+1==len(self.lattice.elList): #in the last element, so it survived
                particle.clipped=False
            particle.yInterp=spi.interp1d(particle.qArr[:,0],particle.qArr[:,1],bounds_error=False,fill_value=np.nan)
            particle.zInterp=spi.interp1d(particle.qArr[:,0],particle.qArr[:,2],bounds_error=False,fill_value=np.nan)
        return swarm
    def get_Width_Along_Lattice(self,swarm,numPoints,elStart=0,fraction=.9,offset=0.0):
        if not 0<fraction<1.0:
            raise Exception('fraction must be number between 0 and 1')
        zArr=np.linspace(self.lattice.elList[elStart].r1[0]+1e-2+offset,self.lattice.elList[-1].r2[0]-1e-2,num=numPoints)
        rList=[]
        for particle in swarm:
                qyArr = particle.yInterp(zArr)
                qzArr = particle.zInterp(zArr)
                r=np.sqrt(qyArr**2+qzArr**2)
                if np.isnan(r[-1])==False:
                    rList.append(r)

        rArr=np.asarray(rList).T
        rArrSorted=np.sort(rArr,axis=1)
        cutoffIndex = int(fraction * rArr.shape[1])
        widthArr=rArrSorted[:,cutoffIndex]
        return zArr,widthArr
    def initialize_Test_Swarm1(self):
        print('using test swarm 1')
        swarm=Swarm()
        qyArr=np.linspace(0,5.0e-3,num=50)#np.asarray([0.0,.001,.003,.004,])
        qyArr=np.append(qyArr,-qyArr[1:])
        pyArr=np.linspace(-15.0,15.0,num=21)
        pxArr=np.linspace(-1,1,num=3)
        #colorList=['blue','green','orange','red','purple','brown']
        colorList=['red']
        i=0
        for qy in qyArr:
            for py in pyArr:
                for px in pxArr:
                    particle=Particle(qi=np.asarray([1e-10,qy,0]),pi=np.asarray([self.v0Nominal+px,py,0]))
                    # print(particle.p)
                    particle.color=colorList[i]
                    swarm.particles.append(particle)
            i+=1
            if i==len(colorList): #start over
                i=0
        self.swarmInitial=swarm
    def initialize_Test_Swarm2(self,thetaArr=None,numPhiParticlesMax=100,thetaMax=10/200,numTheta=10,useSymmetry=False):
        #A swarm originationg from the origin but spread out in phase space
        # print('using test swarm 2')

        swarm=Swarm()
        if thetaArr is None:
            thetaArr=np.linspace(thetaMax/5,thetaMax,num=numTheta)
        pxArr=[0]#np.linspace(-1,1,num=3)
        colorList=['blue','green','orange']
        for theta in thetaArr:
            numPhi=int(numPhiParticlesMax*theta/thetaMax)
            phiArr=np.linspace(0,np.pi*2,num=numPhi,endpoint=False)
            if useSymmetry==True:
                phiArr=phiArr[phiArr<np.pi/6] #exploit the 6 fold symmetry
            for phi in phiArr:
                i=0
                for pxDelta in pxArr:
                    py=self.v0Nominal*np.sin(theta)*np.sin(phi)
                    pz=self.v0Nominal*np.sin(theta)*np.cos(phi)
                    px=self.v0Nominal*np.cos(theta)+pxDelta
                    particle=Particle(qi=np.asarray([1e-10,0,0]),pi=np.asarray([px,py,pz]))
                    particle.color=colorList[i]
                    swarm.particles.append(particle)
                    i+=1
        self.swarmInitial=swarm
    def initialize_Test_Swarm3(self,rMax=.001,pTransMax=15.0,numParticles=1000,T=.001,seedDistance=0.0):
        #A swarm spread out in phase space uniformally
        #rMax: Max radius of particle
        #pTransMax: Maximimum transverse momentum
        np.random.seed(42) #reliable seed
        # print('using swarm 3')
        sigma=91*np.sqrt(T/7.0) #sigma for given temperature
        numParticles=numParticles

        radius=(.45/numParticles)**.5 #volume of 4d sphere goes as r**4
        samplesArrPos=np.asarray(poisson_disc.Bridson_sampling(np.asarray([1,1]),radius))
        samplesArrVel=np.asarray(poisson_disc.Bridson_sampling(np.asarray([1,1]),radius))
        samplesArrPos=2*(samplesArrPos-.5)
        samplesArrVel=2*(samplesArrVel-.5)

        samplesArrPos=samplesArrPos*rMax
        samplesArrVel=samplesArrVel*pTransMax

        #remove samples that violate constraints
        samplesPosList=[] #list to rebuild the array with
        for sample in samplesArrPos:
            qy,qz=sample
            if np.sqrt(qy**2+qz**2)<=rMax:
                samplesPosList.append(sample)
        samplesVelList=[] #list to rebuild the array
        for sample in samplesArrVel:
            py,pz=sample
            if np.sqrt(py**2+pz**2)<pTransMax:
                samplesVelList.append(sample)
        samplesArrPos=np.asarray(samplesPosList)
        samplesArrVel=np.asarray(samplesVelList)

        minLength=min(samplesArrVel.shape[0],samplesArrPos.shape[0])
        samplesArrVel=samplesArrVel[:minLength]
        samplesArrPos=samplesArrPos[:minLength]
        samplesArr=np.column_stack((samplesArrPos,samplesArrVel))



        swarm=Swarm()
        for sample in samplesArr:
            qy,qz,py,pz=sample
            qx=seedDistance+1e-10
            qy=(py/self.v0Nominal)*seedDistance+qy
            qz=(pz/self.v0Nominal)*seedDistance+qz
            px=self.v0Nominal+np.random.normal(scale=sigma) #add temperature distribution
            py+=np.random.normal(scale=sigma)
            pz+=np.random.normal(scale=sigma)
            swarm.add_Particle(qi=np.asarray([qx,qy,qz]),pi=np.asarray([px,py,pz]))

        self.swarmInitial=swarm
        np.random.seed(int(time.time()))
    # @profile()
    def find_Image_Plane(self,parallel=False,thetaArr=None,numPhiParticlesMax=100,thetaMax=None,numParticles=1000,numTheta=10):
        if thetaMax is None:
            L=self.lattice.elList[0].L+self.lattice.elList[1].fringeFrac*self.lattice.elList[1].rp  #length from lens to
            #source.
            thetaMax=self.lattice.elList[1].ap/L
        self.initialize_Test_Swarm2(thetaArr=thetaArr,numPhiParticlesMax=numPhiParticlesMax,thetaMax=thetaMax
                                    ,numTheta=numTheta,useSymmetry=True)
        # self.initialize_Test_Swarm3(pTransMax=.95*self.v0Nominal*thetaMax,numParticles=numParticles)
        swarm = self.trace_Through_Compactor(parallel=parallel)
        zArr,widthArr=self.get_Width_Along_Lattice(swarm,250,elStart=2,offset=.25)
        widthArr=spsig.savgol_filter(widthArr,7,0)
        minWidth=widthArr[np.argmin(widthArr)]
        zFocal=zArr[np.argmin(widthArr)]
        # self.lattice.show_Lattice(swarm=swarm, showTraceLines=True, showMarkers=False, traceLineAlpha=.1,
        #                           trueAspectRatio=False)
        return zFocal,minWidth
    def quantify_Spherical_Abberation(self):
        L=self.lattice.elList[0].L+self.lattice.elList[1].fringeFrac*self.lattice.elList[1].rp #length from lens to
        #source.
        thetaMax=self.lattice.elList[1].ap/L
        thetaTestArr=np.linspace(thetaMax/5,.9*thetaMax,num=10)
        temp=[]
        for theta in thetaTestArr:
            temp.append(self.find_Image_Plane(thetaArr=[theta],numPhiParticlesMax=250,thetaMax=theta)[0])
        variance=np.std(np.asarray(temp))
        plt.plot(temp)
        plt.show()
        print(temp,variance)

    def get_Width_At_Point(self,swarm,z0,fraction=.9):
        qList=[]
        for particle in swarm:
            if particle.q[0] > z0:
                qy = particle.yInterp(z0)
                qz = particle.zInterp(z0)
                qList.append([particle.q[0], qy, qz])
        width = self.get_Frac_Width(np.asarray(qList), fraction)
        return width
    def analyze(self,numParticles=1000,numPoints=25,aperture=5e-3,plot=False):
        #find magnification
        self.build_Lattice()
        # zFocal=self.find_Image_Plane()
        # print('image distance',zFocal-self.lattice.elList[1].r2[0])
        self.initialize_Observed_Swarm(numParticles=numParticles,aperture=aperture)
        self.initialize_Test_Swarm1()
        swarm=self.trace_Through_Compactor(parallel=True)
        print(swarm.survival_Bool())
        initialSpotSize=self.get_Width_At_Point(swarm,1e-6)
        # spotSize=self.get_Width_At_Point(swarm,zFocal)
        # print(np.round(1e3*initialSpotSize,1),np.round(1e3*spotSize))
        if plot==True:
            # for particle in swarm:
            #     z0=self.lattice.elList[0].r1[0]+1e-6
            #     r=np.sqrt(particle.yInterp(z0)**2+particle.zInterp(z0)**2)
            #     if r<.005:
            #         particle.color='purple'
            self.lattice.show_Lattice(swarm=swarm, showTraceLines=True, showMarkers=False, traceLineAlpha=.1,
                                      trueAspectRatio=False)
            zArr, widthArr = self.get_Width_Along_Lattice(swarm, numPoints)
            plt.plot(zArr,widthArr)
            plt.axvline(x=self.lattice.elList[1].r1[0],c='black')
            plt.axvline(x=self.lattice.elList[1].r2[0],c='black')
            plt.grid()
            plt.show()
    def voigt(self,r,a,b,sigma,gamma):
        V0=sps.voigt_profile(0,sigma,gamma)
        V=sps.voigt_profile(r,sigma,gamma)
        return a*V/V0+b
    def qGauss(self,r, a, q, sigma):
        eq = np.abs((1 + (1 - q) * (-(r / sigma) ** 2))) ** (1 / (1 - q))
        return a * eq
    def analyze2(self,numParticles=1000):
        self.build_Lattice()
        # xFocal = self.find_Image_Plane()
        # print(xFocal-self.lattice.elList[1].r2[0])
        self.initialize_Test_Swarm3(numParticles=numParticles,rMax=1.5e-3,pTransMax=self.X['PTMax'],T=1e-3,seedDistance=0.0)
        print('-----------aperture: ',self.X['ap1'])
        swarm = self.trace_Through_Compactor(parallel=True)
        zArr, spotSizeArr = self.get_Width_Along_Lattice(swarm, 1000, elStart=2)

        plt.close('all')
        plt.title("Spot size vs position")
        plt.plot(1e2*(zArr+.7-self.X['Lo1']),1e3*spotSizeArr)
        plt.ylabel('90% spot size, mm')
        plt.xlabel('Distance from nozzle, cm')
        plt.grid()
        # plt.show()
        # plt.savefig(str(100*self.X['ap1'])+'%'+'_plot_90.png')
        start=1.2#self.lattice.elList[2].r1[0]
        stop=2.0#self.lattice.elList[2].r2[0]-1e-6
        xAnalysisPositions=np.linspace(start,stop,num=40)
        fwhmList=[]
        for x in xAnalysisPositions:
            # print('-----position-----: ',x)
            posList=[]
            for particle in swarm:
                if particle.q[0]>x:
                    qy=particle.yInterp(x)
                    qz=particle.zInterp(x)
                    radius=np.sqrt(qy**2+qz**2)
                    posList.append(radius)
                    # if np.abs(qz)<.002:
                    #     posList.append(qy)
            cutOff=.035
            posArr=np.asarray(posList)
            posArr=posArr[np.abs(posArr)<cutOff]
            probeArr,binArr=np.histogram(posArr,bins=50)
            binArr+=(binArr[1]-binArr[0])/2
            binArr=binArr[:-1]

            #get rid of element near zero
            binArr=binArr[1:]
            probeArr=probeArr[1:]
            probeArr=probeArr/np.abs(binArr)
            probeArr=np.append(np.flip(probeArr),probeArr)
            binArr=np.append(-np.flip(binArr),binArr)


            bounds=[(0.0,1.0001,0.0),(np.inf,2.999,np.inf)]

            guess=[probeArr.max()-probeArr.min(),1.5,5.0]

            params=spo.curve_fit(self.qGauss,binArr,probeArr,p0=guess,bounds=bounds)[0]
            xDense=np.linspace(binArr[0],binArr[-1],num=1000)
            yDense=self.qGauss(xDense,*params)
            HM=(yDense.max()-0.0)/2
            FWHM=np.abs(2*xDense[np.argwhere(yDense>HM)[0]][0])

            fwhmList.append(FWHM)

            # plt.plot(xDense,self.qGauss(xDense,*params),c='r')
            # plt.scatter(binArr,probeArr)
            # plt.show()
        fwhmArr=np.asarray(fwhmList)

        for x in [xAnalysisPositions[np.argmin(fwhmArr)]]:
            # print('-----position-----: ',x)
            posList = []
            for particle in swarm:
                if particle.q[0] > x:
                    qy = particle.yInterp(x)
                    qz = particle.zInterp(x)
                    radius = np.sqrt(qy ** 2 + qz ** 2)
                    posList.append(radius)
                    # if np.abs(qz)<.002:
                    #     posList.append(qy)
            cutOff = .0175
            pos0Arr = np.asarray(posList)
            posArr = pos0Arr[np.abs(pos0Arr) < cutOff]
            probeArr, binArr = np.histogram(posArr, bins=50)
            binArr += (binArr[1] - binArr[0]) / 2
            binArr = binArr[:-1]

            # get rid of element near zero
            binArr=binArr[1:]
            probeArr=probeArr[1:]
            probeArr = probeArr / np.abs(binArr)
            probeArr = np.append(np.flip(probeArr), probeArr)
            binArr = np.append(-np.flip(binArr), binArr)

            bounds = [(0.0, 1.0001, 0.0), (np.inf, 2.999, np.inf)]

            guess = [probeArr.max() - probeArr.min(), 1.5, 5.0]

            params = spo.curve_fit(self.qGauss, binArr, probeArr, p0=guess, bounds=bounds)[0]
            xDense = np.linspace(binArr[0], binArr[-1], num=1000)
            yDense = self.qGauss(xDense, *params)
            HM = (yDense.max() - 0.0) / 2
            FWHM = np.abs(2 * xDense[np.argwhere(yDense > HM)[0]][0])


            survivedIntoLens=0 #count the number of particles that made it into at least the lens entrance
            for particle in swarm:
                if particle.q[0]>self.lattice.elList[1].r1[0]:
                    survivedIntoLens+=1

            intensityFWHM=self.X['ap1']**2*np.sum(np.abs(pos0Arr)<(FWHM/2))/survivedIntoLens



            plt.close('all')
            plt.plot(xDense,self.qGauss(xDense,*params),c='r')
            plt.scatter(binArr,probeArr)
            plt.savefig(str(100 * self.X['ap1']) + '_Spot.png')
            # plt.show()


        print('fwhm min: ',1e3*np.min(fwhmArr),'mm')
        print('spot size min: ', 1e3*np.min(spotSizeArr),'mm')
        print("fwhm min location :",1e2*(xAnalysisPositions[np.argmin(fwhmArr)]+.7-self.X['Lo1']),'cm')
        print('spot size min location: ',1e2*(zArr[np.argmin(spotSizeArr)]+.7-self.X['Lo1']),'cm')
        print('intensity FWHM: ',str(intensityFWHM))





        plt.close('all')
        plt.title('FWHM of profile vs distance')
        plt.scatter(100*(xAnalysisPositions+.7-self.X['Lo1']),fwhmArr*1e3)
        plt.xlabel("Distance from nozzle, cm")
        plt.ylabel("FWHM, mm")
        plt.grid()
        # plt.show()
        # plt.savefig(str(100 * self.X['ap1']) + '_plot_fwhm.png')
        # self.lattice.show_Lattice(swarm=swarm, showTraceLines=True, showMarkers=False, traceLineAlpha=.1,
        #                           trueAspectRatio=False)
        # return zArr[np.argmin(spotSizeArr)],np.min(spotSizeArr)


# compactor=Compactor(h=1e-5)
# t=time.time()
# compactor.initialize_Test_Swarm3(numParticles=1000,T=0,pTransMax=5.0)
# compactor.X = {"Lo1": .5, "Lm1": None, "rp1": .05,'ap1':1.0, 'Lsep1': 1.75, 'Lm2': .171 ,'rp2': .0241,
#           'Lsep2': .12,'Lm3': .16, 'rp3': .0241,'Lsep3':.15,'Lm4':.16,'rp4':.04,'sigma':-.02}
# compactor.build_Lattice()
# print(compactor.find_Image_Plane())
#
# apArr=np.linspace(.2,1.0,num=9)
# spotSizeList=[]
# posList=[]
# for ap in apArr:
#
#     compactor = Compactor(h=1e-5)
#     compactor.X = {"Lo1": .6, "Lm1": .25, "rp1": .05, 'Lsep1': 1.75, 'Lm2': .171, 'rp2': .0241,
#                    'Lsep2': .12, 'Lm3': .16, 'rp3': .0241, 'Lsep3': .15, 'Lm4': .16, 'rp4': .04, 'sigma': -.02}
#     compactor.X['ap1']=ap
#     compactor.X['PTMax']=ap*15.0
#     compactor.analyze2(numParticles=30000)

