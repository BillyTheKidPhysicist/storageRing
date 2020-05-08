import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import time
import tkinter as tk
from periodicLatticeSolver import PeriodicLatticeSolver
import functools
import sys
import sympy as sym

#PLS=PeriodicLatticeSolver('both',v0=200,T=.025)
#PLS.begin_Lattice()
#Lm1=PLS.Variable('Lm1')
#Lm2=PLS.Variable('Lm2')
#PLS.add_Drift(Lm1)
#PLS.add_Lens(25,1,1)
#PLS.add_Drift(Lm2)
#PLS.end_Lattice()
#
#Lm1=1
#Lm2Arr=np.linspace(.5,1.5,num=5)
#x, y = PLS.compute_Beta_Of_Z_Array(Lm1, Lm2Arr[0])
#fig=plt.figure()
#plt.ylim(0,10)
#a=plt.plot(x, y[0])[0]
#
#plt.show(block=False)
#plt.pause(.1)
#
#for Lm2 in Lm2Arr:
#    x,y=PLS.compute_Beta_Of_Z_Array(Lm1,Lm2)
#    a.set_ydata(y[0])
#    fig.canvas.draw()
#    plt.pause(.01)

class interactivePlot():
    def __init__(self,PLS):
        self.PLS=PLS
        self.master=tk.Tk()
        self.z=None #hold position data along z (longitudinal) axis
        self.beta=[None,None] #A list that holds the array of values for the betatron function at each point in self.z.
                                #Both x and y betatron values are stored

        self.master.geometry('750x750')
        self.fig,self.ax=plt.subplots(2,figsize=(15,8))
        #self.fig=plt.figure(figsize=(15,8))
        self.linex=None #matplotlib object for the data in the x plane of the ring
        self.liney=None #matplotlib object for the data in the y plane of the ring
        self.stablex=None #whether the current configuration of optics produces stable orbits, x plane of ring
        self.stabley=None #whether the current configuration of optics produces stable orbits, Y plane of ring
        self.sliderList = [] #holds tkinter slider objects
        self.q=[] #a list of current values of parameters of the optics such as length, Bp, etc
        self.elNameTextListx=[] #list of element names (matplotlib text objects) in plot of x plane
        self.elNameTextListy = [] #list of element names (matplotlib text objects) in plot of y plane
        self.lineListx=[] #list of matplotlib line objects in plot of x plane. Lines are use separate elements
        self.lineListy = [] #list of matplotlib line objects in plot of y plane. Lines are use separate elements
        self.stabilityTextList=[] #A list of the trace of transfer matrix. [trace in x plane, y plane]
        self.numPoints=250 #number of points to use along a axis in model
        self.lengthList=[] #to hold values of element lengths after each slider event
        self.totalLengthList=[]#to hold values of cumulative element lengths after each slider event
        self.tracexAbs=None #absolute value of trace over x matrix
        self.traceyAbs=None #absolute value of trace over y matrix
        sliderID=0 #ID of each slider. Corresponds to which variable and it's location is associated lists such as
                #sliderlist and q

        for item in PLS.VOList:
            resolution=(item.varMax-item.varMin)/100
            wrapper=functools.partial(self.update_Plot,sliderID)
            slider=tk.Scale(self.master,command=wrapper,from_=item.varMin,to=item.varMax,resolution=resolution,length=500)
            slider.set(item.varInit)
            slider.grid(column=sliderID,row=0)
            label=tk.Label(text=item.symbol)
            label.grid(column=sliderID,row=1)
            self.sliderList.append(slider)
            sliderID+=1
        self.q=self.get_Slider_Values()
    def get_Slider_Values(self):
        tempList=[]
        for item in self.sliderList:
            tempList.append(float(item.get()))
        return tempList
    def initial_Plot(self):
        #this method is first called to create the plot and plot objects

        #first create straight line in case solution is unstable, and no line can be generated
        self.beta[0]=np.linspace(0,1,num=self.numPoints)
        self.beta[1] = np.linspace(0, 1, num=self.numPoints) #
        #-----------



        self.make_Plot_Data(*self.q)

        betaxMax=self.beta[0].max()
        betaxMin=self.beta[0].min()
        deltaBetax=(betaxMax-betaxMin)/10
        self.ax[0].set_ylim(betaxMin-deltaBetax/2,betaxMax+2*deltaBetax) #set plot y range of data. I don't use the default
                                                                    #so I can make room for text and such

        betayMax=self.beta[1].max()
        betayMin=self.beta[1].min()
        deltaBetay=(betayMax-betayMin)/10
        self.ax[1].set_ylim(betayMin-deltaBetay/2,betayMax+2*deltaBetay) #set plot y range of data. I don't use the default
                                                                    #so I can make room for text and such

        self.ax[0].grid()
        self.ax[1].grid()
        self.linex=self.ax[0].plot(self.z, self.beta[0])[0]
        self.liney=self.ax[1].plot(self.z, self.beta[1])[0]
        if self.stablex == False:
            self.linex.set_color('r')
        if self.stabley == False:
            self.liney.set_color('r')


        #self.ax[0].axvline(x=0,c='black',linestyle=':')
        self.totalLengthList=self.PLS.totalLengthListFunc(*self.q)
        self.lengthList=self.PLS.lengthListFunc(*self.q)

        #put the current value of the trace of the transfer matrix in the top left of both plots
        temp='absolute value of trace :'
        self.stabilityTextList.append(self.ax[0].text(0,1.05,temp+str(np.round(self.tracexAbs,2)),transform=self.ax[0].transAxes))
        self.stabilityTextList.append(self.ax[1].text(0,1.05,temp+str(np.round(self.traceyAbs,2)),transform=self.ax[1].transAxes))
        #---------------------


        for i in range(self.PLS.numElements):
            #go through each element and place lines at its boundaries and text objects towards the top with the elements name
            xEnd=self.totalLengthList[i] #boundary of current element
            linex=self.ax[0].axvline(x=xEnd,c='black',linestyle=':') #line placed at boundary
            self.lineListx.append(linex) #store the line object so it can be updated later

            liney=self.ax[1].axvline(x=xEnd,c='black',linestyle=':') #line placed at boundary
            self.lineListy.append(liney) #store the line object so it can be updated later


            #create and place names of elements above the element
            textx=self.ax[0].text(xEnd-self.lengthList[i]/2,betaxMax+1.5*deltaBetax,self.PLS.lattice[i].elType,rotation=90)
            self.elNameTextListx.append(textx) #store text object to manipulate later
            texty=self.ax[1].text(xEnd-self.lengthList[i]/2,betayMax+1.5*deltaBetay,self.PLS.lattice[i].elType,rotation=90)
            self.elNameTextListy.append(texty) #store text object to manipulate later

        plt.show(block=False)
    def update_Plot(self,sliderID,val):
        val=float(val)
        if val!=self.q[sliderID]: #to circumvent some annoying behavious with setting scale value. It seems to call this
                            #for every slider that was set to a value besides it's default initial value based
                            #on the range of the slider. This behaviour only occurs once for each slider before mainloop
            self.update_q_And_Lengths(sliderID,val)
            self.make_Plot_Data(*self.q)
            betaxMax = self.beta[0].max()
            betaxMin = self.beta[0].min()
            deltaBetax = (betaxMax - betaxMin) / 10
            self.ax[0].set_ylim(betaxMin - deltaBetax / 2, betaxMax + 2 * deltaBetax) #set plot y range of data. I don't use the default
                                                                    #so I can make room for text and such
            self.linex.set_ydata(self.beta[0]) #plot the new plot y data in the x plane of the ring

            betayMax = self.beta[1].max()
            betayMin = self.beta[1].min()
            deltaBetay = (betayMax - betayMin) / 10
            self.ax[1].set_ylim(betayMin - deltaBetay / 2, betayMax + 2 * deltaBetay) #set plot y range of data. I don't use the default
                                                                    #so I can make room for text and such
            self.liney.set_ydata(self.beta[1]) #plot the new plot y data in the y plane of the ring

            if self.stablex==False:
                self.linex.set_color('r')
            else:
                self.linex.set_color('C0')
            if self.stabley==False:
                self.liney.set_color('r')
            else:
                self.liney.set_color('C0')

            temp = 'absolute value of trace :'
            self.stabilityTextList[0].set_text(temp+str(np.round(self.tracexAbs,2))) #update trace value
            self.stabilityTextList[1].set_text(temp+str(np.round(self.traceyAbs,2))) #update trace value
            for i in range(self.PLS.numElements):
                xEnd=self.totalLengthList[i]
                self.lineListx[i].set_data(([xEnd,xEnd],[0,1]))
                self.elNameTextListx[i].set_position((xEnd-self.lengthList[i]/2,betaxMax+1.5*deltaBetax))

                self.lineListy[i].set_data(([xEnd,xEnd],[0,1]))
                self.elNameTextListy[i].set_position((xEnd-self.lengthList[i]/2,betayMax+1.5*deltaBetay))

            self.fig.canvas.draw()
            plt.pause(.01) #otherwise the text doesn't move!
    def make_Plot_Data(self,*args):
        #this generates points on the z axis and the corresponding beta function value. It also find which solutions are
        #stable or not and only return stable ones, then labels them
        M=self.PLS.M_Tot_N(*args) #transfer matrix of the entire lattice.
        self.tracexAbs=np.abs(np.trace(M[:2,:2,]))
        self.traceyAbs=np.abs(np.trace(M[3:,3:]))
        if self.tracexAbs<=2 and self.traceyAbs<=2 :
            self.z,self.beta=self.PLS.compute_Beta_Of_Z_Array(*args,zArr=self.make_Zarr(self.numPoints))
            self.stablex=True
            self.stabley=True
        elif self.tracexAbs<=2:
            self.z,self.beta[0]=self.PLS.compute_Beta_Of_Z_Array(*args,zArr=self.make_Zarr(self.numPoints),axis='x')
            self.stablex=True
            self.stabley=False
        elif self.traceyAbs<=2:
            self.z, self.beta[1] = self.PLS.compute_Beta_Of_Z_Array(*args, zArr=self.make_Zarr(self.numPoints), axis='y')
            self.stablex=False
            self.stabley=True
        else:
            print('unstable solution')
            self.stablex=False
            self.stabley=False
    def make_Zarr(self,numPoints):
        #lengthList=self.PLS.len
        totalLengthArray = self.PLS.totalLengthListFunc(*self.q)

        zArr = np.linspace(0, totalLengthArray[-1], num=numPoints)
        return zArr


    def update_q_And_Lengths(self,sliderID,val):
        #update self.q, the vector that stores the current element parameters. Also recompute and update values in length
        #lists
        #sliderID: the id number of the slider, and thus the position of the value in self.q it is associated with
        #val: the value to change in self.q
        self.q[sliderID]=val
        self.totalLengthList=self.PLS.totalLengthListFunc(*self.q)
        self.lengthList=self.PLS.lengthListFunc(*self.q)

    def launch(self):
        self.initial_Plot()
        self.master.mainloop()

def main():
    PLS = PeriodicLatticeSolver('both', v0=200, T=.025)
    Lm1 = PLS.Variable('Lm1', varMin=0.01, varMax=.2)
    Lm2 = PLS.Variable('Lm2', varMin=0.01, varMax=.2)
    #Lm3 = PLS.Variable('Lm3', varMin=0.01, varMax=.2)
    #Lm4 = PLS.Variable('Lm4', varMin=0.01, varMax=.2)
    PLS.begin_Lattice()

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
