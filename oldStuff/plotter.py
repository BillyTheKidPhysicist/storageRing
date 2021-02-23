import matplotlib.pyplot as plt
import numpy as np
from periodicLatticeSolver import PeriodicLatticeSolver
import sys
class Plotter:
    def __init__(self,PLS):
        self.PLS=PLS
    def plot(self,sol=None,args=None,numPoints=250,folderPath=None,fileName=None):
        if (np.any(args!=None) and sol!=None) or (np.any(args==None) and sol==None):
            raise Exception('YOU NEED TO PROVIDE ARGUMENTS OR A MINIMIZER OBJECT')
        if sol!=None:
            args=sol.args[:-3] #only keep lattice parameters
            zArr=sol.zArr
            env=sol.env
            eta=sol.eta
            emittance=sol.emittance
            tracex=sol.tracex
            tracey=sol.tracey
            resonanceFactx=sol.resonanceFactx
            resonanceFacty = sol.resonanceFacty
            totalLengthList=sol.totalLengthList
            mag=sol.mag
            Lo=sol.Lo
            Lm=sol.Lm
            Li=sol.Li
            sOffset=sol.sOffset
        else:
            print('args not supported yet')
            sys.exit()
        #else:
        #    zArr, beta = self.PLS.compute_Beta_Of_Z_Array(args, numpoints=numPoints)
        #    M = self.PLS.MTotFunc(*args)
        #    tracex = np.abs(np.trace(M[:2, :2]))
        #    tracey = np.abs(np.trace(M[3:, 3:]))
        #    tunex = np.trapz(np.power(beta[0], -1), x=zArr) / (2 * np.pi)
        #    tuney = np.trapz(np.power(beta[1], -1), x=zArr) / (2 * np.pi)
        #    resonanceFactx = self.PLS.compute_Resonance_Factor(tunex, np.arange(3) + 1)
        #    resonanceFacty = self.PLS.compute_Resonance_Factor(tuney, np.arange(3) + 1)
        #    totalLengthList = self.PLS.totalLengthListFunc(*args)
#
        #    emittance,temp = list(self.compute_Emittance(args,totalLengthList, returnAll=True))
        #    mag, Lo, Lm, Li=temp
        #    envBeta = [np.sqrt(emittance[0] * beta[0]), np.sqrt(emittance[0] * beta[1])]
        #    eta = self.PLS.compute_Eta_Of_Z_Array(args, numpoints=numPoints, returZarr=False)




        fig,axBeta=plt.subplots(2,figsize=(15,8)) #make canvas for plotting and axis object to manipulate
                #self.ax is a list with 2 objects, the first for x axis and the second for y axis
        axEta=axBeta[0].twinx() #matplotlib axis object for x dimension. It uses the same axis obviously
        titleString='Particle envelope in x and y plane.'
        print(str(np.round(np.asarray(args),4)))
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
        envEta=eta*self.PLS.delta
        axBeta[0].plot(zArr,1000*env[0]) #plot the envelope in the x plane
        axBeta[1].plot(zArr, 1000*env[1]) #plot the envelope in the y plane
        axEta.plot(zArr, 1000*envEta,linestyle=':') #plot the eta envelope, only the x plane of course

        temp='abs(Trace) :'
        axBeta[0].text(0,1.05,temp+str(np.round(tracex,2)),transform=axBeta[0].transAxes) #put trace in the top left
                #of top plot (x)
        axBeta[1].text(0,1.05,temp+str(np.round(tracey,2)),transform=axBeta[1].transAxes) #put trace in the top left
                #of bottom plot (y)

        temp='Emittance: '
        textx=temp+str(np.round(emittance[0],6)) #emittance is typically 1E-5, so need to round to many digits
        texty=temp+str(np.round(emittance[1],6))
        axBeta[0].text(.15,1.05,textx,transform=axBeta[0].transAxes) #put trace in the top left
                #of top plot (x)
        axBeta[1].text(.15,1.05,texty,transform=axBeta[1].transAxes)#put trace in the top left
                #of top plot (x)


        textx = 'Resonance Factors: '+str(np.round(100*resonanceFactx).astype(int))
        texty = 'Resonance Factors: '+str(np.round(100*resonanceFacty).astype(int))
        axBeta[0].text(.75,1.05,textx,transform=axBeta[0].transAxes) #put resonance factors in the top left
                #of top plot (x)
        axBeta[1].text(.75,1.05,texty,transform=axBeta[1].transAxes)#put resonance factors in the top left
                #of top plot (x)


        text='Lo,Lm,Li: ['+str(np.round(Lo*100,1))+','+str(np.round(Lm*100,1))+','+str(np.round(Li*100,1))+'] cm'
        axBeta[0].text(.75, 1.1, text, transform=axBeta[0].transAxes) #place in top right of top plot, only the top
            #plot

        rp=self.PLS.injector.rpFunc(Lo,Lm,sOffset) #Bore radius of the shaper magnet
        text='Shaper rp: '+str(np.round(rp*1000,1))+'mm'
        axBeta[0].text(.75, 1.15, text, transform=axBeta[0].transAxes) #place in top right of only top magnet

        text='Injec Mag: '+str(np.round(mag,2))
        axBeta[0].text(.87, 1.15, text, transform=axBeta[0].transAxes) #place in top right of top plot only

        for i in range(self.PLS.numElements):
            #go through each element and place lines at its boundaries and text objects towards the top with the elements name
            xEnd=totalLengthList[i] #boundary of current element
            axBeta[0].axvline(x=xEnd,c='black',linestyle=':') #line placed at boundary for top (x) plo
            axBeta[1].axvline(x=xEnd,c='black',linestyle=':') #line placed at boundary for bottom (y) plot


            #draw lines at at apetures. Only draws the apeture if it fits to prevent wasting resources
            el=self.PLS.latticeElementList[i]
            apx = el.apxFuncL(*args) #x apeture
            apy = el.apyFunc(*args) #y apeture
            if 1000*apx<axBeta[0].get_ylim()[1]/1000 or 1000*apy<axBeta[0].get_ylim()[1]: #if any apeture fits on the graph. convert to mm
                if i==0: #if the first element
                    start=0
                else:
                    start=totalLengthList[i-1]
                end=totalLengthList[i]
                if 1000*apy<axBeta[0].get_ylim()[1]: #if apeture x fits on the graph
                    axBeta[0].hlines(apx * 1000, start, end, color='r', linewidth=1)
                if 1000*apy<axBeta[1].get_ylim()[1]: #if apeture y fits on the graph
                    axBeta[1].hlines(apy * 1000, start, end, color='r', )


        if fileName!=None:
            if folderPath==None:
                folderPath=''
            plt.savefig(folderPath+'plot_'+fileName)
            plt.close()
        else:
            plt.show()

    def compute_Emittance(self, args,totalLengthList,returnAll=False):
        # Computes the paramters that optimize the emittance. This is done with brute force, but the function evaluates
        #very fast so it works fine
        MArr = np.linspace(.5, 10, num=1000)
        injector = self.PLS.injector
        xf = injector.xi * MArr
        xfd = injector.xdi / MArr
        betax, betay, alphax, alphay = self.find_Beta_And_Alpha_Injection(args,totalLengthList)
        epsxArr = (xf ** 2 + (betax * xfd) ** 2 + (alphax * xf) ** 2 + 2 * alphax * xfd * xf * betax) / betax
        epsyArr = (xf ** 2 + (betay * xfd) ** 2 + (alphay * xf) ** 2 + 2 * alphay * xfd * xf * betay) / betay
        epsArr=np.sqrt(epsxArr**2 +2*epsyArr**2)  # minimize the quadrature of them. The y axis is weighted more
        minIndex = np.argmin(epsArr)
        if returnAll==True:
            emittance=[epsxArr[minIndex],epsyArr[minIndex]]
            mag=MArr[minIndex]
            return emittance,mag
        else:
            return epsxArr[minIndex], epsyArr[minIndex]
    def find_Beta_And_Alpha_Injection(self,args,totalLengthList):
        #finds the value of beta at the injection point
        #sign of alpha is ambigous so need to find it with slope of beta unfortunately. This compares very favourably
        #with the analytic solution
        z0=totalLengthList[self.PLS.combinerIndex]
        beta1=self.PLS.compute_Beta_At_Z(z0-1E-3,args) #beta value of the 'left' side
        beta2 = self.PLS.compute_Beta_At_Z(z0+1E-3, args) #beta value on the 'right' side
        beta=[(beta1[0]+beta2[0])/2,(beta1[1]+beta2[1])/2] #Save resources and find beta by averaging
        slopex=(beta2[0]-beta1[0])/2E-3 #slope of beta in x direction
        slopey=(beta2[1]-beta1[1])/2E-3 #slope of beta in y direction
        alpha=[-slopex/2,-slopey/2] #Remember, alpha=-(dbeta/dz)/2
        return beta + alpha  # combine the two lists
