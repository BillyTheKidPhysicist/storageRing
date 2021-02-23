import sympy as sym
import numpy.linalg as npl
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib

import time



def plot_Stability_Regions_2D(self, xMin, xMax, yMin, yMax, numPoints=500):
    plt.figure(figsize=(8, 8))  # required to plot multiple plots
    plotData = self.compute_Stability_Grid(xMin, xMax, yMin, yMax, numPoints=numPoints)
    plotData = np.transpose(plotData)
    plotData = np.flip(plotData, axis=0)
    plt.imshow(plotData, extent=[xMin, xMax, yMin, yMax], aspect=(xMax - xMin) / (yMax - yMin))
    titleString = "Stability regions, yellow regions are stable \n"
    if self.axis == 'both':
        titleString += 'both axes'
    elif self.axis == 'y':
        titleString += 'y axis'
    elif self.axis == 'x':
        titleString += 'x axis'
    plt.title(titleString)
    plt.grid()
    plt.xlabel(self.sympyStringList[0])
    plt.ylabel(self.sympyStringList[1])
    plt.show(block=False)
    plt.pause(.1)


def plot_Beta_And_Eta(self, *args):
    self._1D_Plot_Helper("BETA AND ETA", *args)


def plot_Envelope(self, *args, emittance=1):
    self._1D_Plot_Helper("ENVELOPE", *args, emittance=emittance)


def _1D_Plot_Helper(self, plotType, *args,
                    emittance=None):  # used to plot different functions witout repetativeness of code
    fig, ax1 = plt.subplots(figsize=(10, 5))
    totalLengthArray = self.totalLengthListFunc(*args)
    zArr, y1 = self.compute_Beta_Of_z_Array(*args)  # compute beta array
    zArr, y2 = self.compute_Eta_Of_z_Array(*args)  # compute periodic dispersion array
    y1 = y1[0]
    y2 = y2[0]
    tune = np.trapz(1 / y1, x=zArr) / (2 * np.pi)

    if plotType == "BETA AND ETA":
        y2 = y2 * 1000  # dispersion shift is the periodic dispersion times the velocity shift. convert to mm
        ax1Name = 'Beta'
        ax1yLable = 'Beta, m^2'
        ax2Name = 'Eta'
        ax2yLable = 'Eta, mm'
        xLable = 'Nominal trajectory distance,m'
        titleString = 'Beta and Eta versus trajectory'
        ax2 = ax1.twinx()
        ax1.plot(zArr, y1, c='black', label=ax1Name)
        ax2.plot(zArr, y2, c='red', alpha=1, linestyle=':', label=ax2Name)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc=4)
        ax1.set_xlabel(xLable)
        ax1.set_ylabel(ax1yLable, color='black')
        ax2.set_ylabel(ax2yLable, color='black')
    if plotType == "ENVELOPE":
        y2 = y2 * 1000 * self.delta  # dispersion shift is the periodic dispersion times the velocity shift. convert to mm

        y1 = np.sqrt(emittance * y1) * 1000
        ax1Name = 'Betatron envelope'
        ax1yLable = 'Betatron envelope, mm'
        ax2Name = 'Dispersion'
        ax2yLable = 'Dispersion, mm'
        xLable = 'Nominal trajectory distance,m'
        titleString = 'Betatron oscillation envelope vs dispersion'
        titleString += "\n Emittance is " + str(np.round(emittance, 6)) + ". Delta is " + str(
            self.delta) + '. Total tune is ' + str(np.round(tune, 2)) + '.'
        titleString += " Time of flight ~" + str(int(1000 * zArr[-1] / self.v0)) + " ms"
        ax2 = ax1.twinx()
        ax1.plot(zArr, y1, c='black', label=ax1Name)
        ax2.plot(zArr, y2, c='red', alpha=1, linestyle=':', label=ax2Name)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc=4)
        ax1.set_xlabel(xLable)
        ax1.set_ylabel(ax1yLable, color='black')
        ax2.set_ylabel(ax2yLable, color='black')

    for i in range(self.numElements):
        if i == 0:
            center = totalLengthArray[0] / 2
        else:
            center = (totalLengthArray[i - 1] + totalLengthArray[i]) / 2
        plt.axvline(totalLengthArray[i], c='black', linestyle=':')
        ax1.text(center, np.max(y1) * .9, self.latticeElementList[i].magType, rotation=45)
    plt.title(titleString)
    plt.draw()
    plt.show(block=False)


def compute_Stability_Grid(self, xMin, xMax, yMin, yMax, numPoints=500):
    # This returns a grid of stable points. The top left point on the grid is 0,0. Going down rows
    # is going up in x and going across columns is going up in y.
    # TO GET 0,0 ON THE BOTTOM LEFT AND NORMAL BEHAVIOUR, THE OUTPUT MUST BE TRANSPOSED AND FLIPPED ABOUT AXIS=0
    # using both axes only returns stable for the case where both x and y dimensions are stable

    # split up the transfer matrix depending on desired axis
    if self.axis == 'y':  # look at only y axis, lower right hand side of matrix
        newMatrix = self.M_Tot[3:, 3:].copy()
        dim = 3  # dimension of transfer matrix
    elif self.axis == 'x':  # look at only x axis, upper ledt==ft hand side of matrix
        newMatrix = self.M_Tot[:3, :3].copy()
        dim = 3  # dimension of transfer matrix
    else:  # use full matrix.
        newMatrix = self.M_Tot.copy()
        dim = 5  # dimension of transfer matrix

    x = np.linspace(xMin, xMax, num=numPoints)
    y = np.linspace(yMin, yMax, num=numPoints)
    inputValues = np.meshgrid(x, y)  # make grid of input values for parameters

    # this is a hack i have for getting the lamdfiy matrix function to return the correct format. What I want is an
    # array of arrays where the outer array holds the transfer matrices. it would have dimension of (numpoints**2,dim,dim)
    # When I feed a list of values to use in lamdbify to make this array of arrays it will return the correct
    # dimension only if every entry of the (dim,dim) dimension matrix has a variable to be filled. It an entry doesn't
    # then it would be extened to length numpoints**2 which leads to dimension mismatch and the result can't be
    # cast as a numpy ndarray. So I must add a dummy variable to every entry that will be replaced by zero.

    dummy = sym.symbols('dummy')  # dummy variable
    for i in range(dim):
        for j in range(dim):
            newMatrix[i, j] += dummy  # add dummy variable to every entry in transfer matrix
    newArgs = self.sympyVariableList.copy()  # make a list of the original sympy arguments to be used such as magnet length etc.
    newArgs.extend([dummy])  # add to the original variables the dummy variable
    matrixFunc = sym.lambdify(newArgs, newMatrix, 'numpy')  # create matrix function from symbolic matrix
    dummyArg = np.zeros(numPoints ** 2).reshape(numPoints,
                                                numPoints)  # format the dummy arg correctly. Also its value is zero
    inputValues.extend([dummyArg])  # add dummyArg array of arrays to list of input values
    matrixArr = matrixFunc(*inputValues)  # generate array of transfer matrics. The * is to unpack correctly

    eigValsArr = np.abs(
        npl.eigvals(matrixArr.flatten(order='F').reshape(numPoints ** 2, dim, dim)))  # some tricky shaping
    results = ~np.any(eigValsArr > 1 + 1E-12,
                      axis=1)  # any eigenvalue greater than 1 represents instability. Add a little
    # number for good practice
    stabilityGrid = results.reshape((numPoints, numPoints))
    return stabilityGrid


def compute_Resonance_Factor(self, tune,
                             res):  # a factor, between 0 and 1, describing how close a tune is a to a given resonance
    # 0 is as far away as possible, 1 is exactly on resonance
    # the precedure is to see how close the tune is to an integer multiples of 1/res. ie a 2nd order resonance(res =2) can occure
    # when the tune is 1,1.5,2,2.5 etc and is maximally far off resonance when the tuen is 1.25,1.75,2.25 etc
    # tune: the given tune to analyze
    # res: the order of resonance interested in. 1,2,3,4 etc. 1 is pure bend, 2 is pure focus, 3 is pure corrector etc.
    resFact = 1 / res  # What the remainder is compare to
    tuneRem = tune - tune.astype(int)  # tune remainder, the integer value doens't matter
    tuneResFactArr = 1 - np.abs(
        2 * (tuneRem - np.round(tuneRem / resFact) * resFact) / resFact)  # relative nearness to resonances
    return tuneResFactArr


def plot_Beta_Min_2D(self, xMin, xMax, yMin, yMax, elementIndex, useLogScale=False, numPoints=100, trim=2.5):
    t1 = time.time()
    self._2D_Plot_Parallel_Helper(xMin, xMax, yMin, yMax, 'BETA MIN', elementIndex, useLogScale, numPoints, trim, None)
    print("plot time: ", np.round(time.time() - t1, 1))


def plot_Dispersion_Min_2D(self, xMin, xMax, yMin, yMax, elementIndex=None, useLogScale=False, numPoints=100, trim=100):
    t1 = time.time()
    self._2D_Plot_Parallel_Helper(xMin, xMax, yMin, yMax, 'DISPERSION MIN', elementIndex, useLogScale, numPoints, trim,
                                  None)
    print("plot time: ", np.round(time.time() - t1, 1))


def plot_Tune_2D(self, xMin, xMax, yMin, yMax, useLogScale=False, numPoints=100):
    t1 = time.time()
    self._2D_Plot_Parallel_Helper(xMin, xMax, yMin, yMax, 'TUNE', None, useLogScale, numPoints, None, None)
    print("plot time: ", np.round(time.time() - t1, 1))


def plot_Resonance_2D(self, xMin, xMax, yMin, yMax, resonance=1, numPoints=100, useLogScale=False):
    t1 = time.time()
    self._2D_Plot_Parallel_Helper(xMin, xMax, yMin, yMax, 'RESONANCE', None, useLogScale, numPoints, None, resonance)
    print("plot time: ", np.round(time.time() - t1, 1))


def plot_Dispersion_Max_2D(self, xMin, xMax, yMin, yMax, elementIndex=None, useLogScale=False, numPoints=100,
                           trim=None):
    t1 = time.time()
    self._2D_Plot_Parallel_Helper(xMin, xMax, yMin, yMax, 'DISPERSION MAX', elementIndex, useLogScale, numPoints, trim,
                                  None)
    print("plot time: ", np.round(time.time() - t1, 1))


def _2D_Plot_Parallel_Helper(self, xMin, xMax, yMin, yMax, output, elementIndex, useLogScale, numPoints, trim,
                             resonance):
    plotData = self._compute_Grid_Parallel(xMin, xMax, yMin, yMax, output, elementIndex=elementIndex,
                                           numPoints=numPoints)

    if type(plotData) != tuple:
        plotData = [plotData]
    else:
        plotData = list(plotData)
    for i in range(len(plotData)):
        plotData[i] = np.transpose(plotData[i])  # need to reorient the data so that 0,0 is bottom left
        plotData[i] = np.flip(plotData[i], axis=0)  # need to reorient the data so that 0,0 is bottom left
    if trim != None:  # if plot values are clipped pegged above some value
        titleExtra = ' \n Data clipped at a values greater than ' + str(trim)
    else:
        titleExtra = ''
    if output == 'BETA MIN':
        if trim != None:
            for i in range(len(plotData)):
                plotData[i][plotData[i] >= trim] = trim  # trim values
        cmap = matplotlib.cm.inferno_r  # Can be any colormap that you want after the cay
        title = 'Beta,m.'
    if output == "DISPERSION MAX" or output == 'DISPERSION MAX':
        for i in range(len(plotData)):
            plotData[i] = plotData[i] * 1000  # convert to mm
            if trim != None:
                plotData[i][plotData[i] >= trim] = trim  # trim values
        cmap = matplotlib.cm.inferno_r  # Can be any colormap that you want after the cm
        if output == 'DISPERSION MAX':
            title = 'Disperison Maximum, mm.'
        else:
            title = 'Disperison Maximum, mm.'
    if output == "TUNE":
        cmap = matplotlib.cm.inferno  # Can be any colormap that you want after the cm
        title = 'tune'
    if output == 'RESONANCE':
        for i in range(len(plotData)):
            plotData[i] = self.compute_Resonance_Factor(plotData[i], resonance)  # compute the tune
        cmap = matplotlib.cm.inferno  # Can be any colormap that you want after the cm
        title = 'Resonances of order ' + str(resonance) + '. \n 1.0 indicates exactly on resonance, 0.0 off'

    if self.axis == 'both':
        fig, a = plt.subplots(1, 2, figsize=(15, 8))  # required to plot multiple plots
        a[0].grid()
        a[1].grid()
        a[0].set_xlabel(self.sympyStringList[0])
        a[1].set_xlabel(self.sympyStringList[0])
        a[0].set_ylabel(self.sympyStringList[1])
        a[1].set_ylabel(self.sympyStringList[1])
        fig.suptitle(title + titleExtra)
        a[0].title.set_text('x dimension')
        a[1].title.set_text('y dimension')

        cmap.set_bad(color='grey')  # set the colors of values in masked array
        if np.max(np.nan_to_num(plotData[0])) > np.max(np.nan_to_num(plotData[1])):
            vmax = np.max(np.nan_to_num(plotData[0]))
        else:
            vmax = np.max(np.nan_to_num(plotData[1]))
        masked_arrayx = np.ma.masked_where(plotData[0] == np.nan, plotData[0])
        masked_arrayy = np.ma.masked_where(plotData[1] == np.nan, plotData[1])
        map1 = a[0].imshow(masked_arrayx, cmap=cmap, vmin=0, vmax=vmax, extent=[xMin, xMax, yMin, yMax],
                           aspect=(xMax - xMin) / (yMax - yMin))
        a[1].imshow(masked_arrayy, cmap=cmap, vmin=0, vmax=vmax, extent=[xMin, xMax, yMin, yMax],
                    aspect=(xMax - xMin) / (yMax - yMin))
        cbar_ax = fig.add_axes([0.01, 0.15, 0.05, 0.7])
        fig.colorbar(map1, cax=cbar_ax)
        plt.show(block=False)
    else:

        plotData = plotData[0]
        if useLogScale == True:
            plotData = np.log10(plotData)
        masked_array = np.ma.masked_where(plotData == np.nan,
                                          plotData)  # a way of marking ceratin values to have different colors
        plt.subplots(figsize=(8, 8))  # required to plot multiple plots
        plt.grid()
        cmap.set_bad(color='grey')  # set the colors of values in masked array
        plt.title(title + titleExtra)
        plt.imshow(masked_array, cmap=cmap, extent=[xMin, xMax, yMin, yMax], aspect=(xMax - xMin) / (yMax - yMin))
        plt.colorbar()
        plt.xlabel(self.sympyStringList[0])
        plt.ylabel(self.sympyStringList[1])
        plt.show()


def _compute_Grid_Parallel(self, xMin, xMax, yMin, yMax, output, elementIndex=None, numPoints=100):
    stableCoords = []
    gridPosList = []
    x = np.linspace(xMin, xMax, num=numPoints)
    y = np.linspace(yMin, yMax, num=numPoints)
    stableGrid = self.compute_Stability_Grid(xMin, xMax, yMin, yMax,
                                             numPoints=numPoints)  # grid of stable solutions to compute tune for
    outputGrid = np.zeros((numPoints, numPoints))

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if (stableGrid[i, j] == True):
                stableCoords.append([x[i], y[j]])
                gridPosList.append([i, j])
            else:
                outputGrid[i, j] = np.nan  # no tune
    processes = mp.cpu_count() - 1
    jobSize = int(len(stableCoords) / processes) + 1  # in case rounding down. will make a small difference
    manager = mp.Manager()
    resultList = manager.list()
    jobs = []

    loop = True
    i = 0
    while loop == True:
        arg1List = []
        arg2List = []
        for j in range(jobSize):
            if i == len(stableCoords):
                loop = False
                break
            arg1List.append(stableCoords[i])
            arg2List.append(gridPosList[i])
            i += 1
        p = mp.Process(target=self._parallel_Helper, args=(arg1List, arg2List, elementIndex, resultList, output))
        p.start()
        jobs.append(p)
    for proc in jobs:
        proc.join()
    if self.axis == 'both':
        outputGridx = outputGrid.copy()
        outputGridy = outputGrid.copy()
        for item in resultList:
            i, j = item[0]
            outputGridx[i, j] = item[1]
            outputGridy[i, j] = item[2]
        return outputGridx, outputGridy
    else:
        for item in resultList:
            i, j = item[0]
            outputGrid[i, j] = item[1]
        return outputGrid

