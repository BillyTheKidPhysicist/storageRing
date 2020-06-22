import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from periodicLatticeSolver import PeriodicLatticeSolver
import functools

#version 1.0. June 22, 2020. William Huntington

class interactivePlot():
    def __init__(self,PLS):
        self.PLS=PLS
        self.master=tk.Tk()
        self.z=None #hold position data along z (longitudinal) axis
        self.envEta=None #to hold the array of the dispersion lattice function envelope
        self.eta=None #to hold the array of the dispersion lattice function. There is only dispersion in 1 dimension
        self.beta=[None,None] #A list that holds the arrays of beta functions
        self.envBeta=[None,None] #list to hold the envelope arrays from beta oscillations
        self.master.geometry('750x750') #set window size
        self.fig,self.axBeta=plt.subplots(2,figsize=(15,8)) #make canvas for plotting and axis object to manipulate
                #self.ax is a list with 2 objects, the first for x axis and the second for y axis
        self.axEta=self.axBeta[0].twinx() #matplotlib axis object for x dimension. It uses the same axis obviously

        self.betaLinex=None #matplotlib object for the data in the x beta plane of the ring
        self.betaLiney=None #matplotlib object for the data in the y beta plane of the ring
        self.etaLinex=None #matplotlib object for the data in the x eta plane of the ring
        self.etaLiney=None #matplotlib object for the data in the y eta plane of the ring


        self.stablex=None #whether the current configuration of optics produces stable orbits, x plane of ring
        self.stabley=None #whether the current configuration of optics produces stable orbits, Y plane of ring
        self.sliderList = [] #holds tkinter slider objects
        self.q=[] #a list of current values of parameters of the optics such as length, Bp, etc
        self.elNameTextListx=[] #list of element names (matplotlib text objects) in plot of x plane
        self.elNameTextListy = [] #list of element names (matplotlib text objects) in plot of y plane
        self.lineListx=[] #list of matplotlib line objects in plot of x plane. Lines are used to separate elements
        self.lineListy = [] #list of matplotlib line objects in plot of y plane. Lines are used to separate elements
        self.stabilityTextList=[] #A list of the trace of transfer matrix. trace in [x plane, y plane]
        self.numPoints=250 #number of points to use along an axis in model
        self.lengthList=[] #to hold values of element lengths after each slider event
        self.totalLengthList=[]#to hold values of cumulative(inclusive) element lengths after each slider event
        self.tracexAbs=None #absolute value of trace over x dimension transfer matrix
        self.traceyAbs=None #absolute value of trace over y dimension transfer matrix
        self.tunex=None #tune in the x direction
        self.tuney=None #tune in the y direction
        self.emittanceTextList=[] #to hold the text objects for displaying the emmittance

        self.resonanceTextList=[] #a list to hold the two text objects to display resonance factors. one for x and y
                        #dimension plots
        self.resonanceNumber=3 #how many resonances to include, ie first,second, third etc.
        self.resonanceNumArray=np.arange(self.resonanceNumber)+1 #array holding resonances to evaluate such as
                                                                #first,second, third etc.
        self.resonanceFactorList=[np.zeros(self.resonanceNumber),np.zeros(self.resonanceNumber)] #list to hold two 
                                        # numpy arrays of resonance factors
        self.envBetaXMax=None #maximum value of beta envelope in x dimension, mm
        self.envBetaYMax = None #maximum value of beta envelope in y dimension, mm
        self.envBetaXMin = None #minimum value of beta envelope in x dimension, mm
        self.envBetaYMin = None #minimum value of beta envelope in y dimension, mm
        self.betaEnvXBorder=None #size of spacing in the y axis between top and bottom for graph of x dimension
        self.betaEnvyBorder=None #size of spacing in the y axis between top and bottom for graph of y dimension
        
        self.envEtaXMax=None #maximum value of eta envelope in x dimension, mm
        self.envEtaXMin = None #minimum value of eta envelope in x dimension, mm
        self.etaEnvxBorder=None #size of spacing in the y axis between top and bottom for graph of x dimension


        sliderID=0 #ID of each slider. Corresponds to which variable and it's location is associated lists such as
                #sliderlist and q

        for item in PLS.VOList:
            sliderSteps=100 #number of steps between min and max of slider
            resolution=(item.varMax-item.varMin)/sliderSteps # increment size of the slider
            wrapper=functools.partial(self.update_Plot,sliderID) #This is so that each slider passes a ID number
                    #to the function update_Plot when moved.
            slider=tk.Scale(self.master,command=wrapper,from_=item.varMin,to=item.varMax,resolution=resolution,length=500)
            slider.set(item.varInit) #set the slider to an initial value. Either provided by user or half of min+max
            slider.grid(column=sliderID,row=0) #place slider
            label=tk.Label(text=item.symbol) #label slider
            label.grid(column=sliderID,row=1) #place slider
            self.sliderList.append(slider) #add slider to list of slider objects
            sliderID+=1 #update value to next slider ID
        self.q=self.get_Slider_Values() #initially fill self.q
    def get_Slider_Values(self):
        #grab the values from each slider
        tempList=[] #a temporary list to hold the values
        for item in self.sliderList:
            tempList.append(float(item.get())) #returned item is a string, needs to be casted to float
        return tempList
    def initial_Plot(self):
        self.totalLengthList = self.PLS.totalLengthListFunc(*self.q)  # List of the total length up to that the end of that
        # element for each element
        self.lengthList = self.PLS.lengthListFunc(*self.q)  # List of length of each element.
        #this method is first called to create the plot and plot objects. It is not called again
        self.fig.suptitle('Particle envelope in x and y plane',fontsize=24) #the big title over all the little plots
        self.axBeta[0].set_title('--x plane--',fontsize=16)  #title the x axis plot
        self.axBeta[0].set_ylabel('Beta envelope, mm')
        self.axEta.set_ylabel('Dispersion envelope,mm')
        self.axBeta[1].set_title('--y plane--',fontsize=16)  #title the y axis plot
        self.axBeta[1].set_ylabel('Beta envelope, mm')
        self.axBeta[1].set_xlabel('Distance along ideal trajectory, m')


        #first create straight line in case solution is unstable, and no line can be generated. These lines will only
        #be overwritten if there is a stable solution. After being overwritten, the last stable solution will instead
        #be used. Also set emittance to 1 with same reasoning.
        self.envBeta[0] = np.linspace(0,1,num=self.numPoints)
        self.envBeta[1] = np.linspace(0,1,num=self.numPoints)
        self.envEta = np.linspace(0,1,num=self.numPoints)
        self.eta=np.linspace(0,1,num=self.numPoints)
        self.beta[0]=np.linspace(0,1,num=self.numPoints)
        self.beta[1]=np.linspace(0,1,num=self.numPoints)
        self.PLS.emittancex=1
        self.PLS.emittancey=1
        #---------------



        self.make_Plot_Data(*self.q) #create the data (and tune and determine stability)
        self.set_Y_Ranges() #set range in matplotlib graph for y axis. A little tricky to get the words and lines to
                #show up nicely, so it takes some work



        self.axBeta[0].grid() #make grid lines for x axis beta subplot
        self.axBeta[1].grid() #make grid lines for y axis beta subplot
        self.axEta.grid() #make grid lines for x axis beta subplot

        self.betaLinex=self.axBeta[0].plot(self.z, 1000*self.envBeta[0])[0] #plot the data for x beta axis,
                                                                # and save the line object for later. Convert to mm
        self.betaLiney=self.axBeta[1].plot(self.z, 1000*self.envBeta[1])[0]#plot the data for y beta axis,
                                                                    # and save the line object for later. Convert to mm
        self.etaLinex=self.axEta.plot(self.z, 1000*self.envEta, linestyle=':')[0]#plot the data for x eta axis,
                                                                    # and save the line object for later. Convert to mm


        if self.stablex == False: #if x is unstable make the lines red
            self.betaLinex.set_color('r')
            self.etaLinex.set_color('r')
        if self.stabley == False: #if y is unstable make the line red
            self.betaLiney.set_color('r')







        #put the current value of the trace of the transfer matrix in the top left of both plots
        temp='abs(Trace) :'
        self.stabilityTextList.append(self.axBeta[0].text(0,1.05,temp+str(np.round(self.tracexAbs,2)),transform=self.axBeta[0].transAxes))
        self.stabilityTextList.append(self.axBeta[1].text(0,1.05,temp+str(np.round(self.traceyAbs,2)),transform=self.axBeta[1].transAxes))
        #---------------------

        #place resonance factor text on the top right of each plot. The factors are represented as percents here
        textx = 'Resonance Factors: '+str(np.round(100*self.resonanceFactorList[0]).astype(int))
        texty = 'Resonance Factors: '+str(np.round(100*self.resonanceFactorList[1]).astype(int))
        self.resonanceTextList.append(self.axBeta[0].text(.75,1.05,textx,transform=self.axBeta[0].transAxes)) #add text object
                #to list to be used for later
        self.resonanceTextList.append(self.axBeta[1].text(.75,1.05,texty,transform=self.axBeta[1].transAxes)) #add text object
                #to list to be used for later

        temp='Emittance: '
        textx=temp+str(np.round(self.PLS.emittancex,5))
        texty=temp+str(np.round(self.PLS.emittancey,5))
        self.emittanceTextList.append(self.axBeta[0].text(.15,1.05,textx,transform=self.axBeta[0].transAxes))
        self.emittanceTextList.append(self.axBeta[1].text(.15,1.05,texty,transform=self.axBeta[1].transAxes))



        for i in range(self.PLS.numElements):
            #go through each element and place lines at its boundaries and text objects towards the top with the elements name
            xEnd=self.totalLengthList[i] #boundary of current element
            linex=self.axBeta[0].axvline(x=xEnd,c='black',linestyle=':') #line placed at boundary
            self.lineListx.append(linex) #store the line object so it can be updated later

            liney=self.axBeta[1].axvline(x=xEnd,c='black',linestyle=':') #line placed at boundary
            self.lineListy.append(liney) #store the line object so it can be updated later


            #create and place names of elements above the element
            textx=self.axBeta[0].text(xEnd-self.lengthList[i]/2,self.envBetaXMax+1.5*self.betaEnvXBorder,self.PLS.lattice[i].elType,rotation=90)
            self.elNameTextListx.append(textx) #store text object to manipulate later
            texty=self.axBeta[1].text(xEnd-self.lengthList[i]/2,self.envBetaYMax+1.5*self.betaEnvyBorder,self.PLS.lattice[i].elType,rotation=90)
            self.elNameTextListy.append(texty) #store text object to manipulate later

        plt.show(block=False)
    def update_Plot(self,sliderID,val):
        val=float(val) #val comes in as a string
        if val!=self.q[sliderID]: #to circumvent some annoying behaviour with setting scale value. It seems to call this
                            #for every slider that was set to a value besides it's default initial value based
                            #on the range of the slider. This behaviour only occurs once for each slider before mainloop
            self.update_q_And_Lengths(sliderID,val) #fill self.q, the list that holds current slider values.
            self.make_Plot_Data(*self.q) #generate data to be plotted by compute self.beta, self.eta etc.




            self.set_Y_Ranges()#set range in matplotlib graph for y axis. A little tricky to get the words and lines to
                #show up nicely, so it takes some work

            self.betaLinex.set_ydata(1000*self.envBeta[0])  # plot the new plot beta y data in the x plane of the ring, convert to mm
            self.betaLiney.set_ydata(1000*self.envBeta[1]) #plot the new plot beta y data in the y plane of the ring, convert to mm
            self.etaLinex.set_ydata(1000 * self.envEta) #plot the new eta y data in the x plane



            #change line colors depending on stability
            if self.stablex==True:
                self.betaLinex.set_color('C0') #blue for stable
                self.betaLinex.set_color('C0')  # blue for stable
            else:
                self.betaLinex.set_color('r') #red for unstable
            if self.stabley==True:
                self.betaLiney.set_color('C0')
            else:
                self.betaLiney.set_color('r')

            #update the text of the resonance factors in the top right of the plot
            textx = 'Resonance Factors: ' + str(np.round(100 * self.resonanceFactorList[0]).astype(int))
            texty = 'Resonance Factors: ' + str(np.round(100 * self.resonanceFactorList[1]).astype(int))
            self.resonanceTextList[0].set_text(textx) #update the text
            self.resonanceTextList[1].set_text(texty) #update the text



            #update the trace text at the top left of the plot
            temp = 'abs(Trace) :'
            self.stabilityTextList[0].set_text(temp+str(np.round(self.tracexAbs,2))) #update trace value
            self.stabilityTextList[1].set_text(temp+str(np.round(self.traceyAbs,2))) #update trace value

            temp = 'Emittance: '
            textx = temp + str(np.round(self.PLS.emittancex, 5))
            texty = temp + str(np.round(self.PLS.emittancey, 5))
            self.emittanceTextList[0].set_text(textx)
            self.emittanceTextList[1].set_text(texty)



            #loop through the lines and text objects for each element and update
            for i in range(self.PLS.numElements):
                xEnd=self.totalLengthList[i] #the ending of the current element
                self.lineListx[i].set_data(([xEnd,xEnd],[0,1])) #move the line
                self.elNameTextListx[i].set_position((xEnd-self.lengthList[i]/2,self.envBetaXMax+1.5*self.betaEnvXBorder))

                self.lineListy[i].set_data(([xEnd,xEnd],[0,1])) #move the line
                self.elNameTextListy[i].set_position((xEnd-self.lengthList[i]/2,self.envBetaYMax+1.5*self.betaEnvyBorder))

            self.fig.canvas.draw() #render it!
            plt.pause(.01) #otherwise the text doesn't move!
    def make_Plot_Data(self,*args):
        #this generates points on the z axis and the corresponding envelope/beta/eta values. It also find which solutions are
        #stable or not and only return stable ones, then labels them. It also finds the tune
        M=self.PLS.MTotFunc(*args) #transfer matrix of the entire lattice.
        self.tracexAbs=np.abs(np.trace(M[:2,:2,])) #absolute value of the trace for the x dimension. Stable if less than 2
        self.traceyAbs=np.abs(np.trace(M[3:,3:])) #absolute value of the trace for the y dimension. Stable if less than 2
        self.make_Zarr()
        if self.tracexAbs<=2 and self.traceyAbs<=2 : #if solution is stable in both directions
            self.stablex=True #set stability in x dimension
            self.stabley=True #set stability in y dimension
            self.beta=self.PLS.compute_Beta_Of_Z_Array(*args,zArr=self.make_Zarr(self.numPoints),returZarr=False) #compute
                    #beta function and the z values they occur at
            self.eta=self.PLS.compute_Eta_Of_Z_Array(*args,zArr=self.z,returZarr=False,axis='x')
            self.envEta=self.eta*self.PLS.delta #dispersion offset from achromatic particles
            self.compute_Emittance()
            self.envBeta[0]=np.sqrt(self.PLS.emittancex*self.beta[0]) #compute the beta oscillation envelope for x dimension
            self.envBeta[1]=np.sqrt(self.PLS.emittancey*self.beta[1]) #compute the beta oscillation envelope for y dimension
            self.tunex=np.trapz(np.power(self.beta[0],-2),x=self.z)/(2*np.pi) #compute the tune in the x direction
            self.tuney=np.trapz(np.power(self.beta[1],-2),x=self.z)/(2*np.pi) #compute the tune in the x direction
            self.resonanceFactorList[0]=self.PLS.compute_Resonance_Factor(self.tunex,self.resonanceNumArray) #fill
                            #array with values for nearness to resonance for up to self.numResonance resonances
            self.resonanceFactorList[1] = self.PLS.compute_Resonance_Factor(self.tuney, self.resonanceNumArray) #y dimension
                                                                        #resonances
        elif self.tracexAbs<=2: #if the solution is stable in x direction
            self.stablex=True #set stability in x dimension
            self.stabley=False #set stability in y dimension
            self.beta[0]=self.PLS.compute_Beta_Of_Z_Array(*args,zArr=self.z,returZarr=False,axis='x') #compute beta array
            self.eta=self.PLS.compute_Eta_Of_Z_Array(*args,zArr=self.z,returZarr=False,axis='x')
            self.envEta=self.eta*self.PLS.delta #dispersion offset from achromatic particles
            self.compute_Emittance()
            self.envBeta[0]=np.sqrt(self.PLS.emittancex*self.beta[0])
            self.tunex=np.trapz(np.power(self.beta[0],-2),x=self.z)/(2*np.pi) #compute tune value
            self.resonanceFactorList[0] = self.PLS.compute_Resonance_Factor(self.tunex, self.resonanceNumArray) #fill
                        # array with values for nearness to resonance for up to self.numResonance resonances. x dimension
        elif self.traceyAbs<=2: #if solution is stable in y direction
            self.stablex = False #set stability in x dimension
            self.stabley = True #set stability in y dimension
            self.beta[1] = self.PLS.compute_Beta_Of_Z_Array(*args, zArr=self.z,returZarr=False, axis='y') #compute beta array
            self.compute_Emittance()
            self.envBeta[1]=np.sqrt(self.PLS.emittancey*self.beta[1]) #compute betatron oscillation envelope
            self.tuney=np.trapz(np.power(self.beta[1],-2),x=self.z)/(2*np.pi) #compute tune value
            self.resonanceFactorList[1]=self.PLS.compute_Resonance_Factor(self.tuney,self.resonanceNumArray)# fill
                            # array with values for nearness to resonance for up to self.numResonance resonances. y dimension
        else:
            self.stablex=False
            self.stabley=False
            print('unstable solution')
    def make_Zarr(self):
        #ideally this would create points in the z array that are more closely spaced in certain elements, and
        #further spaced in others to save computing time. As of now, it's not worth figuring out how to get this to work.
        # this will probably be necesary when doing optimizing
        totalLengthArray = self.PLS.totalLengthListFunc(*self.q)
        self.z = np.linspace(0, totalLengthArray[-1], num=self.numPoints)


    def compute_Emittance(self):
        #compute the emmittance for the x and y dimension
        xi,xdi=self.PLS.injector.compute_Xf_Xdf_RMS(*self.q)
        dzi=self.lengthList[self.PLS.combinerIndex]/self.totalLengthList[-1] #length of combiner element divided by total
                #lattice length
        zi=self.totalLengthList[self.PLS.combinerIndex-1]/self.totalLengthList[-1] #length up to beginning of combiner,
                #divided by total lattice length
        numPointsComb=int(self.numPoints*dzi) #number of points in combiner
        indexComb=int(zi*self.numPoints) #first index of zArr inside combiner
        if self.stablex==True: #only bother computing emittance for stable orbits
            betax = np.min(self.beta[0][indexComb:indexComb + numPointsComb]) #find the injection point, which is assumed
                    #to be a point of minimum beta
            self.PLS.emittancex=(xi**2+(betax*xdi)**2)/betax
        if self.stabley==True:
            betay = np.min(self.beta[1][indexComb:indexComb + numPointsComb]) #see betax above
            self.PLS.emittancey = (xi ** 2 + (betay * xdi) ** 2)/betay



    def update_q_And_Lengths(self,sliderID,val):
        #update self.q, the vector that stores the current element parameters. Also recompute and update values in length
        #lists
        #sliderID: the id number of the slider, and thus the position of the value in self.q it is associated with
        #val: the value to change in self.q
        self.q[sliderID]=val
        self.totalLengthList=self.PLS.totalLengthListFunc(*self.q)
        self.lengthList=self.PLS.lengthListFunc(*self.q)

    def set_Y_Ranges(self):
        self.envBetaXMax=1000*self.envBeta[0].max() #max value of x axis envelope, mm
        self.envBetaXMin=1000*self.envBeta[0].min() #min value of x axis envelope, mm
        self.betaEnvXBorder=(self.envBetaXMax-self.envBetaXMin)/10 #1/10th of the plot scale. helps with setting the y range
        self.axBeta[0].set_ylim(self.envBetaXMin-self.betaEnvXBorder/2,self.envBetaXMax+2*self.betaEnvXBorder) #set plot x range of data. I don't use the default
                                                                    #so I can make room for text and such

        self.envBetaYMax=1000*self.envBeta[1].max() #max value of y axis envelope, mm
        self.envBetaYMin=1000*self.envBeta[1].min() #min value of y axis envelope, mm
        self.betaEnvyBorder=(self.envBetaYMax-self.envBetaYMin)/10 #1/10th of the plot scale. helps with setting the y range
        self.axBeta[1].set_ylim(self.envBetaYMin-self.betaEnvyBorder/2,self.envBetaYMax+2*self.betaEnvyBorder) #set plot y range of data. I don't use the default
                                                                    #so I can make room for text and such


        self.envEtaXMax=1000*self.envEta.max() #max value of x axis envelope, mm
        self.envEtaXMin=1000*self.envEta.min() #min value of x axis envelope, mm
        self.etaEnvxBorder=(self.envEtaXMax-self.envEtaXMin)/10 #1/10th of the plot scale. helps with setting the y range
        self.axEta.set_ylim(self.envEtaXMin-self.etaEnvxBorder/2,self.envEtaXMax+2*self.etaEnvxBorder) #set plot x range of data. I don't use the default
                                                                    #so I can make room for text and such


    def launch(self):
        self.initial_Plot()
        self.master.mainloop()

def main():
    PLS = PeriodicLatticeSolver(200,.03)
    L=PLS.Variable('L',varMin=.01,varMax=.5)
    Lo=PLS.Variable('Lo',varMin=.01,varMax=1)
    Bp=1
    rp=.05
    Lm1 = PLS.Variable('Lm1', varMin=0.01, varMax=.2)
    Lm2 = PLS.Variable('Lm2', varMin=0.01, varMax=.2)
    PLS.begin_Lattice()
    PLS.add_Injector(L, Lo, Bp, rp,xiRMS=.001,xdiRMS=5/200)
    PLS.add_Bend(np.pi, 1, 50)
    PLS.add_Lens(Lm1, 1, .05)
    PLS.add_Drift()
    PLS.add_Combiner(S=.5)
    PLS.add_Drift()
    PLS.add_Lens(Lm1, 1, .05)


    PLS.add_Bend(np.pi, 1, 50)
    PLS.add_Lens(Lm2, 1, .05)
    PLS.add_Drift()
    PLS.add_Lens(Lm2, 1, .05)
    PLS.set_Track_Length(1)

    PLS.end_Lattice()
    app=interactivePlot(PLS)
    app.launch()


if __name__ == '__main__':
    main()
