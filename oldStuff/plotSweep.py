import numpy as np
import time
import multiprocessing as mp
from periodicLatticeSolver import PeriodicLatticeSolver
import matplotlib.pyplot as plt
import sys
from interactivePlot import InteractivePlot


class Solution():
    #An object to hold onto the solution generated with the object PlotSweep. There will be many of these
    def __init__(self):
        self.index=None #the index of the solution corresponding to the index of the input data
        self.args=None #the sympy arguments that generated this solution
        self.tunex=None #tune in the x plane
        self.tuney=None #tune in the y plane
        self.tracex=None #trace of transfer matrix in x plane
        self.tracey=None #trace of transfer matrix in y plane
        self.beta=[None,None] #List to hold arrays of beta function in x and y plane
        self.envBeta=[None,None] #List of the arrays of the envelope of the beta function in the x and y plane.
        self.eta=None #to hold array of eta functions, only x plane of course
        self.zArr=None #to hold the zArr for this solution.
        self.emittancex=None # the emittance in the x plane. This is claculated using the injector
        self.emittancey = None # the emittance in the y plane. This is claculated using the injector
        self.resonanceFactx=None #the array of resonance factors in the x plane. closer to zero is better
        self.resonanceFacty=None #the array of resonance factors in the x plane. closer to zero is better
        self.lengthList=[] #List to hold the length of each element in the lattice
        self.totalLengthList=[] #List to hold the cumulative length of each element in the list
        self.injec_LoOp = None #the optimal object length of the injection system. Distance from collector focus to beginnig
                #of shaper lens
        self.injec_LmOp=None #the optimal magnet length of the injection system
        self.injec_LiOp=None #the optimal image length. The distance from the end of the magnet to the focus at the combiner
        self.injec_Mag=None #injector magnification for optimal lengths



class PlotSweep():
    def __init__(self,PLS,numStepsArgs=5):
        self.PLS=PLS
        self.inputDataArr=[] #hold the data that is worked on to find the solutions. shape of (x1,x2) where x1 is number of 
                #configurations to test and x2 is the number of arguments. Does not have any arguments for the injector
        self.numStepsArgs=numStepsArgs #number of steps to create the n dimensional grid of configurations to try. For a valid solution
            #each point would get loaded into Solution.args
        self.solList=[] #the solutions after postprocessing. This is to preserve the orignial solutions
        self.solList0=[] #list of valid solutions. Configurations with trace greater than 2 are discarded
        self.numPointsZ = 300 #number of points along the z axis. Less is faster, but looks rougher and can cause spikes
            #in beta to not be captured when trimming solutions that exceed an apeture
        self.numConfigs=None #number of configurations to solve, ie the number of points in the n dimensional grid. Most
            #of these won't actually be solved because the trace will be too large
        self.globalEnvTrim=None #remove solutions in post processing who's envelope exceeds this value anywhere
        self.benderEnvTrim=None #remove solutions in post processing who's envelope exceeds this value in the bending
                #section
        self.resTrimList=None #List of resonance to trim in post processing. Any solution who's first, second, or third etc.
                #resonance exceeds the corresponding value is trimmed
        self.make_Input_Data() #construct the input data.

    def make_Input_Data(self):
        #constructs an n dimensional even grid of configurations to test

        tempList=[]# hold the array solutions
        for item in self.PLS.VOList:
            x=np.linspace(item.varMin,item.varMax,num=self.numStepsArgs)
            tempList.append(x)
        tempList=np.meshgrid(*tempList) #construct the meshgrid. See numpy docs for info
        for i in range(len(tempList)):
            tempList[i]=tempList[i].flatten()
        self.inputDataArr=np.asarray(tempList)
        self.inputDataArr = self.inputDataArr.T #now I have an array who's rows represent a point in the space of configs
                #and each column is input values to the lattice
        self.enforce_Constraints()
        self.numConfigs=self.numStepsArgs**len(self.PLS.VOList) #number of points in the space

    def enforce_Constraints(self):
        None

    def generate_Plots(self):
        print("Generating and saving plots")
        for sol in self.solList:
            name=str(int(sol.index))
            self.plot_Solution(sol,save=True,folderPath='results/',fileName=name)
        print('DONE')
    def plot_Solution(self,sol,save=False,folderPath=None,fileName=None):
        #TODO: PLOT APETURES



        fig,axBeta=plt.subplots(2,figsize=(15,8)) #make canvas for plotting and axis object to manipulate
                #self.ax is a list with 2 objects, the first for x axis and the second for y axis
        axEta=axBeta[0].twinx() #matplotlib axis object for x dimension. It uses the same axis obviously
        titleString='Particle envelope in x and y plane. '+str(np.round(sol.args,4))
        fig.suptitle(titleString,fontsize=24) #the big title over all the little plots
        axBeta[0].set_title('--x plane--',fontsize=16)  #title the x axis plot
        axBeta[0].set_ylabel('Beta envelope, mm')
        axEta.set_ylabel('Dispersion envelope,mm')
        axBeta[1].set_title('--y plane--',fontsize=16)  #title the y axis plot
        axBeta[1].set_ylabel('Beta envelope, mm')
        axBeta[1].set_xlabel('Distance along ideal trajectory, m')

        axBeta[0].grid() #make grid lines for x axis beta subplot
        axBeta[1].grid() #make grid lines for y axis beta subplot
        #axEta.grid() #make grid lines for x axis beta subplot
        envEta=sol.eta*self.PLS.delta
        axBeta[0].plot(sol.z,1000*sol.envBeta[0]) #plot the envelope in the x plane
        axBeta[1].plot(sol.z, 1000*sol.envBeta[1]) #plot the envelope in the y plane
        axEta.plot(sol.z, 1000*envEta,linestyle=':') #plot the eta envelope, only the x plane of course

        temp='abs(Trace) :'
        axBeta[0].text(0,1.05,temp+str(np.round(sol.tracex,2)),transform=axBeta[0].transAxes) #put trace in the top left
                #of top plot (x)
        axBeta[1].text(0,1.05,temp+str(np.round(sol.tracey,2)),transform=axBeta[1].transAxes) #put trace in the top left
                #of bottom plot (y)

        temp='Emittance: '
        textx=temp+str(np.round(sol.emittancex,6)) #emittance is typically 1E-5, so need to round to many digits
        texty=temp+str(np.round(sol.emittancey,6))
        axBeta[0].text(.15,1.05,textx,transform=axBeta[0].transAxes) #put trace in the top left
                #of top plot (x)
        axBeta[1].text(.15,1.05,texty,transform=axBeta[1].transAxes)#put trace in the top left
                #of top plot (x)


        textx = 'Resonance Factors: '+str(np.round(100*sol.resonanceFactx).astype(int))
        texty = 'Resonance Factors: '+str(np.round(100*sol.resonanceFacty).astype(int))
        axBeta[0].text(.75,1.05,textx,transform=axBeta[0].transAxes) #put resonance factors in the top left
                #of top plot (x)
        axBeta[1].text(.75,1.05,texty,transform=axBeta[1].transAxes)#put resonance factors in the top left
                #of top plot (x)

        Li=sol.injec_LiOp #optimal image distance for injector
        Lm=sol.injec_LmOp #optimal magnet distance for injector
        Lo=sol.injec_LoOp #optimal object distance for injector
        text='Lo,Lm,Li: ['+str(np.round(Lo*100,1))+','+str(np.round(Lm*100,1))+','+str(np.round(Li*100,1))+'] cm'
        axBeta[0].text(.75, 1.1, text, transform=axBeta[0].transAxes) #place in top right of top plot, only the top
            #plot


        rp=self.PLS.injector.rpFunc(Lo,Lm) #Bore radius of the shaper magnet
        text='Shaper rp: '+str(np.round(rp*1000,1))+'mm'
        axBeta[0].text(.75, 1.15, text, transform=axBeta[0].transAxes) #place in top right of only top magnet

        mag=sol.injec_Mag #injector magnification of optimal solution
        text='Injec Mag: '+str(np.round(mag,2))
        axBeta[0].text(.87, 1.15, text, transform=axBeta[0].transAxes) #place in top right of top plot only


        for i in range(self.PLS.numElements):
            #go through each element and place lines at its boundaries and text objects towards the top with the elements name
            xEnd=sol.totalLengthList[i] #boundary of current element
            axBeta[0].axvline(x=xEnd,c='black',linestyle=':') #line placed at boundary for top (x) plo
            axBeta[1].axvline(x=xEnd,c='black',linestyle=':') #line placed at boundary for bottom (y) plot


            #draw lines at at apetures. Only draws the apeture if it fits to prevent wasting resources
            el=self.PLS.latticeElementList[i]
            apx = el.apxFunc(*sol.args) #x apeture
            apy = el.apyFunc(*sol.args) #y apeture
            if apx<axBeta[0].get_ylim()[1] or apy<axBeta[0].get_ylim()[1]: #if any apeture fits on the graph
                if i==0: #if the first element
                    start=0
                else:
                    start=sol.totalLengthList[i-1]
                end=sol.totalLengthList[i]
                if apy<axBeta[0].get_ylim()[1]: #if apeture x fits on the graph
                    axBeta[0].hlines(apx * 1000, start, end, color='r', linewidth=1)
                if apy<axBeta[1].get_ylim()[1]: #if apeture y fits on the graph
                    axBeta[1].hlines(apy * 1000, start, end, color='r', )


        if save==True:
            plt.savefig(folderPath+'plot_'+fileName)
            plt.close()
        else:
            plt.show()

    def sweep(self):
        t1=time.time()
        processes = mp.cpu_count()
        print("working on "+str(int(self.numConfigs)) +' configurations across '+str(processes) +' processes')
        orderArr=np.arange(0,self.inputDataArr.shape[0]) #array of indices where each index corresponds to a row in
            #self.inputDataArr
        splitSize=int(orderArr.shape[0]/processes) #the size of the chunks that will be fed to each process
        splitIndices=np.arange(0,self.inputDataArr.shape[0],splitSize)+splitSize #the indices to split the input data into
                #chunks along
        mpInputData=np.column_stack((orderArr,self.inputDataArr)) #combine input arguments with index. The solutions will
            #will get shuffle so this keeps track of who went where
        np.random.shuffle(mpInputData) #shuffle the input arguments. Otherwise there can be regions of solutions and some
                #processes will have much more work
        mpInputList=np.split(mpInputData,splitIndices,axis=0) #split the input arguments into chunks
        manager=mp.Manager() #multiprocessing manager
        resultList=manager.list() #this is the list object that will collect the solutions
        jobs=[] #To hold each process
        for item in mpInputList:
            p = mp.Process(target=self.parallel_Helper,args=(item,resultList))
            p.start()
            jobs.append(p)

        for proc in jobs:
            proc.join()
        self.solList0=list(resultList)
        self.solList=list(resultList)
        print(str(len(self.solList0)),'solutions found in '+str(int((time.time()-t1)))+' seconds')
    def post_Process(self,globalEnvTrim=None,benderEnvTrim=None,resTrimList=None,apetureMult=np.inf):
        self.resTrimList=resTrimList
        self.solList=self.solList0.copy() #the solution list for post processing. I want to preserve the original solutions
        self.trim_Trace()
        self.trim_ResonanceFactors()
        self.globalEnvTrim=globalEnvTrim
        self.benderEnvTrim=benderEnvTrim
        self.trim_Env()
        self.trim_Apeture(mult=apetureMult)
        self.sort_Sol()
        print('Finished! '+str(len(self.solList))+' solutions remain')
    def trim_ResonanceFactors(self):
        #solutions with a resonance factor greater the corresponding trim amount are removed
        if self.resTrimList!=None:
            resTrimArr=np.append(np.asarray(self.resTrimList), np.asarray(self.resTrimList))
            temp=[] #list to hold solutions that satisfy constraint
            print("removing solutions with a resonance factors greater than " + str(resTrimArr[:3]) +','+str(resTrimArr[3:]))
            for sol in self.solList:
                facts=np.append(sol.resonanceFactx,sol.resonanceFactx)
                cond=np.all(facts<resTrimArr) #true if every rosonance factor is less than the trim value
                if cond==True:
                    temp.append(sol)
            print(str(len(self.solList) - len(temp)) + ' solutions removed')
            self.solList=temp #the new trimmed solution list
    def trim_Apeture(self,mult=1):
        print('Removing solutions who\'s envelope exceeds the apeture times a factor of ' +str(mult))
        temp=[]
        for sol in self.solList:
            clipped=False
            for i in range(len(self.PLS.latticeElementList)):
                el=self.PLS.latticeElementList[i]
                if i==0:
                    ind1=0
                else:
                    ind1=int(sol.totalLengthList[i-1]*self.numPointsZ/sol.totalLengthList[-1])+1 #overshoot beginning
                ind2=int(sol.totalLengthList[i]*self.numPointsZ/sol.totalLengthList[-1])-1 #undershoot ending
                if ind2>ind1+1: #sometimes the element is very short
                    elMaxx = np.max(sol.envBeta[0][ind1:ind2])
                    elMaxy = np.max(sol.envBeta[1][ind1:ind2])
                    if elMaxx>mult*el.apxFunc(*sol.args) or elMaxy>mult*el.apyFunc(*sol.args):
                        clipped=True
                        break
            if clipped==False:
                temp.append(sol)
        print(str(len(self.solList) - len(temp)) + ' solutions removed')
        self.solList=temp

    def trim_Env(self):
        #trim solutions with values that exceed certain apetures
        if self.globalEnvTrim!=None:
            #if any value anywhere exceed a ceratain limit, self. globalEnvTrim
            trim = self.globalEnvTrim / 1000 #convert to meters
            print("removing solutions with a global betatron envelope greater than "+str(trim*1000)+' mm')
            temp=[]
            for sol in self.solList:
                if np.max(sol.envBeta[0])<trim and np.max(sol.envBeta[0])<trim: #both planes need to satisfy
                    temp.append(sol)
            print(str(len(self.solList)-len(temp))+' solutions removed')
            self.solList=temp
        if self.benderEnvTrim!=None:
            #If a value only inside a bender exceeds self.benderEnvTrim
            trim = self.benderEnvTrim / 1000 #convert to meters
            print("removing solutions with a bender betatron envelope greater than " + str(trim*1000) + ' mm')
            temp=[]

            for sol in self.solList:
                #extract the envelopes from the benders in the x and y plane
                num=sol.z.shape[0]
                Ltotal=sol.z[-1]
                ind1=0 #first element is a bender always
                ind2=int(sol.totalLengthList[0]*num/Ltotal-1) #undershoot the ending
                ind3=int(sol.totalLengthList[self.PLS.benderIndices[1]-1]*num/Ltotal+1) #overshoot the beginnig
                ind4 = int(sol.totalLengthList[self.PLS.benderIndices[1]] * num / Ltotal -1) #undershoot the ending
                bend1Maxx=np.max(sol.envBeta[0][ind1:ind2])
                bend1Maxy = np.max(sol.envBeta[1][ind1:ind2])
                bend2Maxx = np.max(sol.envBeta[0][ind3:ind4])
                bend2Maxy = np.max(sol.envBeta[1][ind3:ind4])
                if bend1Maxx<trim and bend1Maxy<trim and bend2Maxx<trim and bend2Maxy<trim:
                    temp.append(sol)
            print(str(len(self.solList) - len(temp)) + ' solutions removed')
            self.solList=temp
    def trim_Trace(self):
        print("Removing solutions with trace greater than 1.9")
        traceLim=1.9
        temp=[]
        for sol in self.solList:
            if sol.tracex<=traceLim and sol.tracey<traceLim:
                temp.append(sol)
        print(str(len(self.solList)-len(temp))+' solutions removed')
        self.solList=temp
    def sort_Sol(self):
        #take a value of the envelope squared and use that to weight. This punishes large excursions.
        print('Sorting solutions by weighted envelope in ascending order')
        temp=[]
        for sol in self.solList:
            weight=np.sum(np.sqrt((sol.beta[0]*sol.emittancex)**2+(sol.beta[1]*sol.emittancey)**2))
            temp.append(weight)
        sortArr=np.asarray(temp)
        temp=[]
        for index in np.argsort(sortArr):
            temp.append(self.solList[index])
        self.solList=temp
    def find_Beta_And_Alpha_Injection(self,sol):
        #finds the value of beta at the injection point
        #value of alpha is ambigous so need to find it with slope...
        z0=sol.totalLengthList[self.PLS.combinerIndex]
        beta=self.PLS.compute_Beta_At_Z(z0,sol.args)
        beta1=self.PLS.compute_Beta_At_Z(z0-1E-3,sol.args)
        beta2 = self.PLS.compute_Beta_At_Z(z0+1E-3, sol.args)
        slopex=(beta2[0]-beta1[0])/2E-3
        slopey=(beta2[1]-beta1[1])/2E-3
        alpha=[-slopex/2,-slopey/2]
        return beta+alpha #combine the two lists
    def compute_Emittance(self,sol):
        #TODO: MAKE RUN FASTER IF EXACT LENGTHS ARE GIVEN
        if self.PLS.combinerIndex==None: #if there is no combiner, don't worry about injecting obviously
            return
        else: #don't bother if there is no stability
            MArr=np.linspace(3,7,num=1000)
            injector = self.PLS.injector
            xf=injector.xi*MArr
            xfd=injector.xdi/MArr
            betax, betay, alphax, alphay = self.find_Beta_And_Alpha_Injection(sol)

            epsxArr = (xf ** 2 + (betax * xfd) ** 2 + (alphax * xf) ** 2 + 2 * alphax * xfd * xf * betax) / betax
            epsyArr = (xf ** 2 + (betay * xfd) ** 2 + (alphay * xf) ** 2 + 2 * alphay * xfd * xf * betay) / betay

            epsArr = np.sqrt(
                epsxArr ** 2 + 2 * epsyArr ** 2)  # minimize the quadrature of them. The y axis is weighted more

            minIndex = np.argmin(epsArr)
            sol.injec_LoOp =1
            sol.injec_LmOp =1
            sol.injec_LiOp =1
            sol.injec_Mag = MArr[minIndex]#injector.MFunc(*argArr[:, minIndex])[0, 0]
            return epsxArr[minIndex], epsyArr[minIndex]



            '''
            injector=self.PLS.injector
            numPoints=100
            LoArr = np.linspace(injector.LoMin, injector.LoMax, num=numPoints)
            LmArr = np.linspace(injector.LmMin, injector.LmMax, num=numPoints)
            argList = np.meshgrid(LoArr,LmArr)
            argList[0]=argList[0].flatten()
            argList[1] = argList[1].flatten()
            argArr = np.asarray(argList)
            LiArr = injector.LiFunc(*argArr)

            trimIndices = np.logical_and(LiArr > injector.LiMin, LiArr < injector.LiMax)
            if trimIndices.sum()==0:
                raise Exception('NO CONFIGURATION FOUND FOR INJECTOR THAT SATISFIES CONTRAINTS')
            argArr = argArr[:, trimIndices]
            LiArr = LiArr[trimIndices]

            LtArr = np.sum(argArr, axis=0) + LiArr
            trimIndices = np.logical_and(LtArr < injector.LtMax,LtArr>injector.LtMin)
            if trimIndices.sum()==0:
                raise Exception('NO CONFIGURATION FOUND FOR INJECTOR THAT SATISFIES CONTRAINTS')

            argArr = argArr[:, trimIndices]
            betax,betay,alphax,alphay=self.find_Beta_And_Alpha_Injection(sol)
            epsxArr = injector.epsFunc(betax,alphax, *argArr)
            epsyArr = injector.epsFunc(betay,alphay, *argArr)
            epsArr=np.sqrt(epsxArr**2+2*epsyArr**2) #minimize the quadrature of them. The y axis is weighted more
            minIndex = np.argmin(epsArr)
            sol.injec_LoOp = argArr[:, minIndex][0]
            sol.injec_LmOp=argArr[:,minIndex][1]
            sol.injec_LiOp=injector.LiFunc(*argArr[:,minIndex])
            sol.injec_Mag=injector.MFunc(*argArr[:,minIndex])[0,0]
            return epsxArr[minIndex], epsyArr[minIndex]
            '''

    def parallel_Helper(self,*args):
        inputData=args[0]
        mpList=args[-1]
        for item in inputData:
            results=self.compute_Solution(item[1:])
            if len(results)==1: #unstable configuration
                None
            else:
                sol=Solution()
                sol.args=item[1:]
                sol.index=item[0]
                sol.z=results[0]
                sol.beta=results[1]
                sol.eta=results[2]
                sol.tracex=results[3]
                sol.tracey = results[4]

                sol.tunex = np.trapz(np.power(sol.beta[0], -1), x=sol.z) / (2 * np.pi)
                sol.tuney = np.trapz(np.power(sol.beta[1], -1), x=sol.z) / (2 * np.pi)
                sol.lengthList=self.PLS.lengthListFunc(*item[1:])
                sol.totalLengthList=self.PLS.totalLengthListFunc(*item[1:])
                sol.emittancex,sol.emittancey=self.compute_Emittance(sol)
                sol.envBeta[0] = np.sqrt(sol.beta[0] * sol.emittancex)
                sol.envBeta[1] = np.sqrt(sol.beta[1] * sol.emittancey)
                sol.resonanceFactx = self.PLS.compute_Resonance_Factor(sol.tunex, np.arange(3) + 1)
                sol.resonanceFacty = self.PLS.compute_Resonance_Factor(sol.tuney, np.arange(3) + 1)
                mpList.append(sol)
    def compute_Solution(self,args):
        M=self.PLS.compute_MTot(args)
        tracex=np.abs(np.trace(M[:2,:2]))
        tracey = np.abs(np.trace(M[3:, 3:]))
        if tracex>=2 or tracey>=2:
            return [None]
        else:
            zArr,beta=self.PLS.compute_Beta_Of_Z_Array(args,numpoints=self.numPointsZ)
            eta = np.zeros(zArr.shape)#self.PLS.compute_Eta_Of_Z_Array(*args, numpoints=self.numPointsZ,returZarr=False)
            return [zArr,beta,eta,tracex,tracey]



        #print(type(mpInputData[0,0]))
        #for i in range(self.inputDataArr.shape[0]):
'''       #    args=self.inputDataArr[i]

combinerLength=.2
r0=1

PLS = PeriodicLatticeSolver(200,.03,'both')
Bp1=PLS.Variable('Bp1',varMin=.001,varMax=.45)
Bp2=PLS.Variable('Bp2',varMin=.001,varMax=.45)
Bp3=PLS.Variable('Bp3',varMin=.001,varMax=.45)
Bp4=PLS.Variable('Bp4',varMin=.001,varMax=.45)

Lm1=PLS.Variable('Lm1',varMin=.001,varMax=.5)
Lm2=PLS.Variable('Lm2',varMin=.001,varMax=.75)
Lm3=PLS.Variable('Lm3',varMin=.001,varMax=.75)
Lm4=PLS.Variable('Lm4',varMin=.001,varMax=.4)



Lt=2
#Scombiner=PLS.Variable('Scomb',varMin=.4,varMax=Lt-.4)
#Lt=PLS.Variable('Lt',varMin=1.35,varMax=2.1)
#pos=Lt/2-combinerLength/2

PLS.begin_Lattice()

PLS.set_Track_Length(Lt)

PLS.add_Injector(.45,.002,10/200)

rp=0.025
PLS.add_Bend(np.pi, r0, .45)
PLS.add_Drift(L=.03)
PLS.add_Lens(Lm4, Bp4, rp)
PLS.add_Drift()
PLS.add_Combiner(L=combinerLength,S=1.1)
PLS.add_Drift()
PLS.add_Lens(Lm1, Bp1, rp)
PLS.add_Drift(L=.03)

PLS.add_Bend(np.pi, r0, .45)
PLS.add_Drift(L=.03)
PLS.add_Lens(Lm2, Bp2, rp)
PLS.add_Drift()
PLS.add_Lens(Lm3, Bp3, rp)
PLS.add_Drift(L=.03)

PLS.end_Lattice()
'''
