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


def deconvolve(a):
    peakSig=a.max() #to properly scale the final result
    x=np.arange(a.shape[0])
    xDense=np.linspace(x[0],x[-1],num=101)
    a=a/a.max()


    dataAnalyzer=DataAnalyzer()
    b=dataAnalyzer.multi_Voigt((xDense-(xDense.min()+xDense.max())/2)*230/xDense.max(),0,1,0,1e-10)
    b=b/b.max()
    # b=b[50:-50]
    aFunc=spi.Rbf(x,a,smooth=10)#interp1d(x,a)
    a=aFunc(xDense)

    a = sps.savgol_filter(a, 15, 2)
    a=a/a.max()



    M=generate_MConv(a,b)
    M=M/M.max()
    # test=np.asarray([1,1,1,1,2,0,1,1,0,1])
    # plt.plot(generate_Curve(test,a.shape[0]))
    # plt.show()



    # @numba.njit(numba.float64(numba.float64[:]))
    def cost(args):
        #args: the deviation from the original array
        c=generate_Curve(args,a.shape[0])
        # c=c-c.min()
        c=c/c.max()
        aNew=M@c
        aNew=aNew/aNew.max()
        cost=np.sum((aNew-a)**2)
        return cost


    bounds=[]

    # for i in range(a.shape[0]):
    #     deltaLower=a[i] #the signal can be totally compensated, but not go below zero
    #     deltayMaxFact=.25 #Factor for setting upper bound. Much more efficient than adding same amount everywhere
    #     deltaUpper=deltayMaxFact*a[i]+.1
    #     bounds.append((-deltaLower,deltaUpper))
    for i in range(25):
        bounds.append((-.1,1))
    sol=spo.differential_evolution(cost,bounds,maxiter=1000,disp=False,popsize=1,tol=0,workers=1,polish=False)


    c=generate_Curve(sol.x,a.shape[0])
    c=sps.savgol_filter(c,15,2)
    c=c*peakSig/c.max()



    #
    # aNew=M@c
    # aNew=aNew/aNew.max()
    # plt.plot(a)
    # plt.plot(aNew)
    # print(np.sum((a-aNew)**2))
    # plt.plot(c)
    # plt.show()

    cFunc=spi.interp1d(xDense,c)
    cDownSample=cFunc(x)
    print(np.argmax(a),np.argmax(c))
    return cDownSample

