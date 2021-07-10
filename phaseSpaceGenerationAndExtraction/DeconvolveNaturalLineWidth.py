import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import scipy.linalg as spl
from scipy.sparse.linalg import gmres
import scipy.optimize as spo
import numba
import scipy.signal as sps
import scipy.interpolate as spi
from DataAnalysis import DataAnalyzer

def generate_Curve(coef,size,order=2):
  numKnotsTotal=coef.shape[0]+order+1
  numKnotsInterior=coef.shape[0]-order
  sep=(size+1)/numKnotsInterior
  knots=np.linspace(0,(numKnotsTotal-1)*sep,numKnotsTotal)
  knots-=knots[order]
  func=spi.BSpline(knots,coef,order,extrapolate=False)
  vals=func(np.arange(0,size))
  return vals

def generate_MConv(a,b):
    M=np.eye(a.shape[0])
    for i in range(M.shape[0]): # go through rows
      if i<(b.shape[0]-1)//2:
        startA=0
        startB=((b.shape[0]-1)//2+1)-1-i
        endA=i+(b.shape[0]-1)//2+1
        endB=b.shape[0]
        M[i,startA:endA]=b[startB:endB]
      elif M.shape[0]-i<=(b.shape[0]-1)//2:
        startA=i-(b.shape[0]-1)//2
        endA=-1
        startB=0
        endB=(M.shape[0]-i)+(b.shape[0]-1)//2
        M[i,startA:]=b[startB:endB]
      else:
        startB=0
        endB=b.shape[0]
        startA=i-(b.shape[0]-1)//2
        endA=i+(b.shape[0]-1)//2+1
        M[i,startA:endA]=b[startB:endB]
    return M


def deconvolve(a,deltaF):
    #a: signal to deconvolve natural linewidht from
    #deltaF: frequency range over which a resides
    peakSig=a.max() #to properly scale the final result
    x=np.arange(a.shape[0])
    xDense=np.linspace(x[0],x[-1],num=251)
    a=a/a.max()
    # plt.scatter(x,a)



    dataAnalyzer=DataAnalyzer()
    b=dataAnalyzer.multi_Voigt((xDense-(xDense.min()+xDense.max())/2)*deltaF/xDense.max(),0,1,0,1e-10)
    b=b/b.max()
    # b=b[75:-75]
    aFunc=spi.Rbf(x,a,smooth=0.0)#interp1d(x,a)
    a=aFunc(xDense)
    # plt.plot(xDense,a)

    window=2*int((a.shape[0]/20)//2)+1
    for i in range(25):
        a = sps.savgol_filter(a, window, 2)
    a=a/a.max()

    # plt.plot(xDense,a)
    # plt.plot(b)
    # plt.show()



    M=generate_MConv(a,b)
    M=M/M.max()
    # test=np.asarray([1,1,1,1,2,0,1,1,0,1])
    # plt.plot(generate_Curve(test,a.shape[0]))
    # plt.show()

    @numba.njit(numba.float64(numba.float64[:]))
    def cost_Inner(c):
        cost=0

        cost+=10*np.abs(np.sum(c[c<0])) #sum up the negative values
        # c=c-c.min()
        c=c/c.max()
        aNew=M@c
        aNew=aNew/aNew.max()
        cost+=np.sum((aNew-a)**2)
        cost0=1.0
        #find any peaks in the data, besides the one single peak
        j=0
        for i in range(0, c.shape[0] - 3):  # xclude the end
            probe = c[i:i + 3]
            if np.argmax(probe)==1: #center should not be peak, except at the actual peak
                if j!=0: #There should be at least one peak, so ingore the fir time
                    cost+=cost0 #add a small penalty
                j=1
        return cost

    def cost(args):
        #args: the deviation from the original array
        c=generate_Curve(args,a.shape[0])
        # cFunc=spi.interp1d(np.linspace(0,a.shape[0],args.shape[0]),args)
        # c=cFunc(np.arange(0,a.shape[0]))
        return cost_Inner(c)

    bounds=[]

    # for i in range(a.shape[0]):
    #     deltaLower=a[i] #the signal can be totally compensated, but not go below zero
    #     deltayMaxFact=.25 #Factor for setting upper bound. Much more efficient than adding same amount everywhere
    #     deltaUpper=deltayMaxFact*a[i]+.1
    #     bounds.append((-deltaLower,deltaUpper))
    for i in range(25):
        bounds.append((-1.0,1.0))
    bestSol=None
    for i in range(5):
        sol=spo.differential_evolution(cost,bounds,maxiter=5000,disp=False,mutation=(.5,1.5),recombination=.25,popsize=1,tol=0,workers=1,polish=True)
        if sol.fun<1.0:
            bestSol=sol
            break
        else:
            if bestSol is None:
                bestSol=sol
            elif sol.fun<bestSol.fun:
                bestSol=sol
    # cFunc = spi.interp1d(np.linspace(0, a.shape[0], bestSol.x.shape[0]), bestSol.x)
    # c = cFunc(np.arange(0, a.shape[0]))
    c=generate_Curve(bestSol.x,a.shape[0])
    c=c*peakSig/c.max()
    # plt.plot(c/c.max())
    # window = 2 * int((a.shape[0] / 10) // 2) + 1
    # c=sps.savgol_filter(c,window,2)
    # j = 0
    # for i in range(0, c.shape[0] - 3):  # xclude the end
    #     probe = c[i:i + 3]
    #     if np.argmax(probe) == 1:  # center should not be peak, except at the actual peak
    #         if j != 0:  # There should be at least one peak, so ingore the fir time
    #             print('fail')
    #         j = 1
    # plt.plot(c)
    # c=sps.savgol_filter(c,window,2)
    # c=sps.savgol_filter(c,window,2)

    # plt.plot(c/c.max())

    # aNew=M@c
    # aNew=aNew/aNew.max()
    # plt.plot(a)
    # plt.plot(aNew)
    # aNew=aNew/aNew.max()
    # print(np.sum((a-aNew)**2))

    # plt.show()

    cFunc=spi.Rbf(xDense,c)
    cDownSample=cFunc(x)
    return cDownSample

