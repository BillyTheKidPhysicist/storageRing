import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from periodicLatticeSolver import PeriodicLatticeSolver
import functools
import time
import sys

#version 1.0. June 22, 2020. William Huntington


class InteractivePlot:
    def __init__(self,PLS):
        self.PLS=PLS
        self.master=tk.Tk()
        self.eps=1E-5 #zero will make the program crash often because of things like val**2/val for example. This is clearly
        #zero with limits, but can be Nan with computer think
        self.z=None #hold position data along z (longitudinal) axis
        self.envEta=None #to hold the array of the dispersion lattice function envelope
        self.eta=None #to hold the array of the dispersion lattice function. There is only dispersion in 1 dimension
        self.beta=[None,None] #A list that holds the arrays of beta functions
        self.envBeta=[None,None] #list to hold the envelope arrays from beta oscillations
        self.master.geometry('400x750') #set window size
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
        self.traceList=[] #A list of the trace of transfer matrix. trace in [x plane, y plane]
        self.injectorText=None #to hold the text object that displays the injector parameters
        self.shaperText=None #to hold text object that displays radius of shaping magnet
        self.numPoints=1000 #number of points to use along an axis in model
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
        #self.autoMinEnabled=True
        #self.autoMinCheckBox=tk.Checkbutton(self.master,command=self.enable_Auto_Min_Emittance)
        #self.autoMinCheckBox.grid(column=len(self.PLS.VOList),row=2)
        for item in PLS.VOList:
            sliderSteps=101 #number of steps between min and max of slider
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
        self.q=self.get_Slider_Values() #fill self.q
    def get_Slider_Values(self):
        #grab the values from each slider
        tempList=[] #a temporary list to hold the values
        for item in self.sliderList:
            tempList.append(float(item.get())+self.eps) #returned item is a string, needs to be casted to float
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
        self.make_Plot_Data(self.q, initial=True)  # create the data (and tune and determine stability)



        print('---------------------------------------------------------')
        self.make_Zarr()
        xi=-1e-6
        xdi=0
        beta=self.beta[0][0]
        alpha=(self.beta[0][1]-self.beta[0][0])/(self.z[1]-self.z[0])
        eps = (xi ** 2 + (beta * xdi) ** 2 + (alpha * xi) ** 2 + 2 * alpha * xdi * xi * beta) / beta
        self.PLS.emittancex=eps
        print(eps)

        M=np.eye(2)
        tempList= self.PLS.latticeElementList[1:]
        tempList.append(self.PLS.latticeElementList[0])
        for el in tempList:
            print('-----',el.elType,'-----')
            M=el.M_Func(*self.q)[:2,:2]@M
            print(el.M_Func(*self.q)[:2,:2])
            #print(M)
            print(1e3*(M[0,0]*xi+M[0,1]*xdi))
            print((200*(M[1,0]*xi+M[1,1]*xdi)))

        self.make_Plot_Data(self.q, initial=True)  # create the data (and tune and determine stability)






        self.set_Y_Ranges() #set range in matplotlib graph for y axis. A little tricky to get the words and lines to
                #show up nicely, so it takes some work

        self.axBeta[0].grid() #make grid lines for x axis beta subplot
        self.axBeta[1].grid() #make grid lines for y axis beta subplot

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
        self.traceList.append(self.axBeta[0].text(0,1.05,temp+str(np.round(self.tracexAbs,2)),transform=self.axBeta[0].transAxes))
        self.traceList.append(self.axBeta[1].text(0,1.05,temp+str(np.round(self.traceyAbs,2)),transform=self.axBeta[1].transAxes))
        #---------------------

        #place resonance factor text on the top right of each plot. The factors are represented as percents here
        textx = 'Resonance Factors: '+str(np.round(100*self.resonanceFactorList[0]).astype(int))
        texty = 'Resonance Factors: '+str(np.round(100*self.resonanceFactorList[1]).astype(int))
        self.resonanceTextList.append(self.axBeta[0].text(.75,1.05,textx,transform=self.axBeta[0].transAxes)) #add text object
                #to list to be used for later
        self.resonanceTextList.append(self.axBeta[1].text(.75,1.05,texty,transform=self.axBeta[1].transAxes)) #add text object
                #to list to be used for later



        #Li=self.PLS.injector.LiOp
        #Lm=self.PLS.injector.LmOp
        #Lo=self.PLS.injector.LoOp
        #text='Lo,Lm,Li: ['+str(np.round(Lo*100,1))+','+str(np.round(Lm*100,1))+','+str(np.round(Li*100,1))+'] cm'
        #self.injectorText=self.axBeta[0].text(.75, 1.1, text, transform=self.axBeta[0].transAxes)

        #rp=self.PLS.injector.rpFunc(Lm,Lo)
        #text='Shaper rp: '+str(np.round(rp*1000,1))+'mm'
        #self.shaperText=self.axBeta[0].text(.75, 1.15, text, transform=self.axBeta[0].transAxes)


        temp='Emittance: '
        textx=temp+str(np.round(self.PLS.emittancex,6))
        texty=temp+str(np.round(self.PLS.emittancey,6))
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
            textx=self.axBeta[0].text(xEnd - self.lengthList[i] / 2, self.envBetaXMax + 1.5 * self.betaEnvXBorder, self.PLS.latticeElementList[i].elType, rotation=90)
            self.elNameTextListx.append(textx) #store text object to manipulate later
            texty=self.axBeta[1].text(xEnd - self.lengthList[i] / 2, self.envBetaYMax + 1.5 * self.betaEnvyBorder, self.PLS.latticeElementList[i].elType, rotation=90)
            self.elNameTextListy.append(texty) #store text object to manipulate later

        plt.show(block=False)




    #def enable_Auto_Min_Emittance(self):
    #    self.autoMinEnabled=not self.autoMinEnabled
    #    if self.autoMinEnabled==True:
    #        print('Emittance auto minimization enabled')
    #    else:
    #        print('Emittance auto minimization disabled')
    def update_Plot(self,sliderID,val,initial=True):


        val=float(val)+self.eps#val comes in as a string
        if val!=self.q[sliderID]: #to circumvent some annoying behaviour with setting scale value. It seems to call this
                            #for every slider that was set to a value besides it's default initial value based
                            #on the range of the slider. This behaviour only occurs once for each slider before mainloop.
                            #program will crash without this.


            self.update_q_And_Length_Lists(sliderID,val) #fill self.q, the list that holds current slider values.
            self.make_Plot_Data(self.q,sliderID=sliderID) #generate data to be plotted by compute self.beta, self.eta etc.

            self.set_Y_Ranges()#set range in matplotlib graph for y axis. A little tricky to get the words and lines to
                #show up nicely, so it takes some work

            self.betaLinex.set_ydata(1000*self.envBeta[0])  # plot the new plot beta y data in the x plane of the ring, convert to mm
            self.betaLinex.set_xdata(self.z)
            self.betaLiney.set_ydata(1000*self.envBeta[1]) #plot the new plot beta y data in the y plane of the ring, convert to mm
            self.betaLiney.set_xdata(self.z)
            self.etaLinex.set_ydata(1000 * self.envEta) #plot the new eta y data in the x plane
            self.etaLinex.set_xdata(self.z)



            #change line colors depending on stability
            if self.stablex==True:
                self.betaLinex.set_color('C0') #blue for stable
                self.etaLinex.set_color('C0')  # blue for stable
            else:
                self.betaLinex.set_color('r') #red for unstable
                self.etaLinex.set_color('r')  # red for unstable
            if self.stabley==True:
                self.betaLiney.set_color('C0')
            else:
                self.betaLiney.set_color('r')

            #update the text of the resonance factors in the top right of the plot
            textx = 'Resonance Factors: ' + str(np.round(100 * self.resonanceFactorList[0]).astype(int))
            texty = 'Resonance Factors: ' + str(np.round(100 * self.resonanceFactorList[1]).astype(int))
            self.resonanceTextList[0].set_text(textx) #update the text
            self.resonanceTextList[1].set_text(texty) #update the text


            #Li = self.PLS.injector.LiOp
            #Lm = self.PLS.injector.LmOp
            #Lo = self.PLS.injector.LoOp
            #text = 'Lo,Lm,Li: [' + str(np.round(Lo * 100, 1)) + ',' + str(np.round(Lm * 100, 1)) + ',' + str(np.round(Li * 100, 1)) + '] cm'
            #self.injectorText.set_text(text)
#
            #rp = self.PLS.injector.rpFunc(Lm, Lo)
            #text = 'Shaper rp: ' + str(np.round(rp * 1000, 1)) + 'mm'
            #self.shaperText.set_text(text)





            #update the trace text at the top left of the plot
            temp = 'abs(Trace) :'
            self.traceList[0].set_text(temp+str(np.round(self.tracexAbs,2))) #update trace value
            self.traceList[1].set_text(temp+str(np.round(self.traceyAbs,2))) #update trace value

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
            #plt.pause(.05) #otherwise the text doesn't move!

    def make_Plot_Data(self,args,initial=False,sliderID=None):
        #this generates points on the z axis and the corresponding envelope/beta/eta values. It also find which solutions are
        #stable or not and only return stable ones, then labels them. It also finds the tune

        if True: #only bother computing beta and eta if the variable changed
                                #by the slider affects them
            M=self.PLS.MTotFunc(*args) #transfer matrix of the entire lattice.
            self.tracexAbs=np.abs(np.trace(M[:2,:2,])) #absolute value of the trace for the x dimension. Stable if less than 2
            self.traceyAbs=np.abs(np.trace(M[3:,3:])) #absolute value of the trace for the y dimension. Stable if less than 2
            self.make_Zarr()
            if self.tracexAbs<=2 and self.traceyAbs<=2 : #if solution is stable in both directions
                self.stablex=True #set stability in x dimension
                self.stabley=True #set stability in y dimension
                self.beta=self.PLS.compute_Beta_Of_Z_Array(args,zArr=self.z,returZarr=False) #compute
                        #beta function and the z values they occur at
                self.eta=self.PLS.compute_Eta_Of_Z_Array(args,zArr=self.z,returZarr=False)
                self.envEta=self.eta*self.PLS.delta #dispersion offset from achromatic particles
                self.envBeta[0]=np.sqrt(self.PLS.emittancex*self.beta[0]) #compute the beta oscillation envelope for x dimension
                self.envBeta[1]=np.sqrt(self.PLS.emittancey*self.beta[1]) #compute the beta oscillation envelope for y dimension
                self.tunex=np.trapz(np.power(self.beta[0],-1),x=self.z)/(2*np.pi) #compute the tune in the x direction
                self.tuney=np.trapz(np.power(self.beta[1],-1),x=self.z)/(2*np.pi) #compute the tune in the x direction
                self.resonanceFactorList[0]=self.PLS.compute_Resonance_Factor(self.tunex,self.resonanceNumArray) #fill
                                #array with values for nearness to resonance for up to self.numResonance resonances
                self.resonanceFactorList[1] = self.PLS.compute_Resonance_Factor(self.tuney, self.resonanceNumArray) #y dimension
                                                                            #resonances
            elif self.tracexAbs<=2: #if the solution is stable in x direction
                self.stablex=True #set stability in x dimension
                self.stabley=False #set stability in y dimension
                self.beta[0]=self.PLS.compute_Beta_Of_Z_Array(args,zArr=self.z,returZarr=False,axis='x') #compute beta array
                self.eta=self.PLS.compute_Eta_Of_Z_Array(args,zArr=self.z,returZarr=False)
                self.envEta=self.eta*self.PLS.delta #dispersion offset from achromatic particles
                self.envBeta[0]=np.sqrt(self.PLS.emittancex*self.beta[0])
                self.tunex=np.trapz(np.power(self.beta[0],-1),x=self.z)/(2*np.pi) #compute tune value
                self.resonanceFactorList[0] = self.PLS.compute_Resonance_Factor(self.tunex, self.resonanceNumArray) #fill
                            # array with values for nearness to resonance for up to self.numResonance resonances. x dimension
            elif self.traceyAbs<=2: #if solution is stable in y direction
                self.stablex = False #set stability in x dimension
                self.stabley = True #set stability in y dimension
                self.beta[1] = self.PLS.compute_Beta_Of_Z_Array(args, zArr=self.z,returZarr=False, axis='y') #compute beta array
                self.envBeta[1]=np.sqrt(self.PLS.emittancey*self.beta[1]) #compute betatron oscillation envelope
                self.tuney=np.trapz(np.power(self.beta[1],-1),x=self.z)/(2*np.pi) #compute tune value
                self.resonanceFactorList[1]=self.PLS.compute_Resonance_Factor(self.tuney,self.resonanceNumArray)# fill
                                # array with values for nearness to resonance for up to self.numResonance resonances. y dimension
            else:
                self.stablex=False
                self.stabley=False
                print('unstable solution')
        #print("BENDER VACUUM TUBE RADIUS: ",np.round(self.PLS.lattice[0].rpFunc(*self.q)*1000/2,1),'mm')

        #if Type=='INJECTOR' or initial==True and self.PLS.combinerIndex!=None:
        #    self.compute_Emittance_And_Update_Injector()
        #    if self.stablex==True:
        #        self.envBeta[0]=np.sqrt(self.PLS.emittancex*self.beta[0]) #compute the beta oscillation envelope for x dimension
        #    if self.stabley==True:
        #        self.envBeta[1]=np.sqrt(self.PLS.emittancey*self.beta[1]) #compute the beta oscillation envelope for y dimension
    def make_Zarr(self):
        #ideally this would create points in the z array that are more closely spaced in certain elements, and
        #further spaced in others to save computing time. As of now, it's not worth figuring out how to get this to work.
        # this will probably be necesary when doing optimizing
        self.z = np.linspace(0, self.totalLengthList[-1], num=self.numPoints)

    def find_Beta_And_Alpha_Injection(self,axis='both'):
        #finds the value of beta at the injection point
        #sign of alpha is ambigous,
        #finds the value of beta at the injection point
        #value of alpha is ambigous so need to find it with slope...
        z0=self.totalLengthList[self.PLS.combinerIndex]
        beta=self.PLS.compute_Beta_At_Z(z0,self.q)
        beta1=self.PLS.compute_Beta_At_Z(z0-1E-3,self.q)
        beta2 = self.PLS.compute_Beta_At_Z(z0+1E-3, self.q)
        slopex=(beta2[0]-beta1[0])/2E-3
        slopey=(beta2[1]-beta1[1])/2E-3
        alpha=[-slopex/2,-slopey/2]
        return beta+alpha #combine the two lists




    def update_q_And_Length_Lists(self,sliderID,val):
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
        self.axBeta[0].set_xlim(0, self.z[-1])

        self.envBetaYMax=1000*self.envBeta[1].max() #max value of y axis envelope, mm
        self.envBetaYMin=1000*self.envBeta[1].min() #min value of y axis envelope, mm
        self.betaEnvyBorder=(self.envBetaYMax-self.envBetaYMin)/10 #1/10th of the plot scale. helps with setting the y range
        self.axBeta[1].set_ylim(self.envBetaYMin-self.betaEnvyBorder/2,self.envBetaYMax+2*self.betaEnvyBorder) #set plot y range of data. I don't use the default
                                                                    #so I can make room for text and such
        self.axBeta[1].set_xlim(0, self.z[-1])

        self.envEtaXMax=1000*self.envEta.max() #max value of x axis envelope, mm
        self.envEtaXMin=1000*self.envEta.min() #min value of x axis envelope, mm
        self.etaEnvxBorder=(self.envEtaXMax-self.envEtaXMin)/10 #1/10th of the plot scale. helps with setting the y range
        self.axEta.set_ylim(self.envEtaXMin-self.etaEnvxBorder/2,self.envEtaXMax+2*self.etaEnvxBorder) #set plot x range of data. I don't use the default
                                                                    #so I can make room for text and such
        self.axEta.set_xlim(0, self.z[-1])

    def launch(self):
        print('launching')
        self.initial_Plot()
        self.master.mainloop()








PLS = PeriodicLatticeSolver(200, .02, axis='both')

L1=  PLS.Variable('L1', varMin=.35, varMax=.65,varInit=.49)
PLS.set_Track_Length(1)
#L2= PLS.Variable('L2', varMin=.01, varMax=.3,varInit=.1)
#L3 = PLS.Variable('L3', varMin=.01, varMax=.0825)
#L4= PLS.Variable('L4', varMin=.01, varMax=.3)

Bp1 = 1#PLS.Variable('Bp1', varMin=.8, varMax=1.2,varInit=1)
#Bp2 = PLS.Variable('Bp2', varMin=.8, varMax=1.2,varInit=1)
#Bp3 = PLS.Variable('Bp3', varMin=.01, varMax=.01)
#Bp4 = PLS.Variable('Bp4', varMin=.01, varMax=.45)

PLS.begin_Lattice()

PLS.add_Bend(np.pi, 1, 1,rp=.01)
PLS.add_Drift()
PLS.add_Lens(L1, 1, .01,S=.5)
PLS.add_Drift()


PLS.add_Bend(np.pi, 1, 1,rp=.01)
PLS.add_Drift()
PLS.add_Lens(L1, 1, .01,S=.5)
PLS.add_Drift()


PLS.end_Lattice()
app=InteractivePlot(PLS)
app.launch()