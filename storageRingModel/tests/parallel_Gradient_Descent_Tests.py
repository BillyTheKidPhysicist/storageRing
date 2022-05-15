import numpy as np
from parallel_Gradient_Descent import gradient_Descent,GradientOptimizer
def test1():
    def func(X):
        return np.linalg.norm(X) / 2 + np.sin(X[0]) ** 2 * np.sin(X[1] * 3) ** 2
    Xi=[8.0,8.0]
    x,f=gradient_Descent(func,Xi,.05,1000,momentumFact=0.9,gradMethod='central',parallel=False,
                         Plot=False,disp=False,maxStepSize=.15)
    x0=np.asarray([-0.0033649623312369302 ,1.8697646129320236e-11])
    f0=0.00033280573104994056
    assert np.all(x0==x)
    assert f==f0
def test2():
    tolF=5e-5
    tolC=5e-9
    xArr=np.linspace(0,np.pi,100)
    yDifAnalytic=np.cos(xArr)
    def func(X):
        return np.sin(X[0])
    opt=GradientOptimizer(func,[0],1,1,1,False,False,False,'central',100e-6,1,'adam')
    yDifNumericF=[]
    yDifNumericC=[]
    for x in xArr:
        F0F,gradF=opt._forward_Difference_And_F0([x])
        F0C,gradC=opt._central_Difference_And_F0([x])
        yDifNumericF.append(gradF[0])
        yDifNumericC.append(gradC[0])
    yDifNumericF=np.asarray(yDifNumericF)
    yDifNumericC=np.asarray(yDifNumericC)
    assert np.all(np.abs(yDifNumericF-yDifAnalytic)<tolF)
    assert np.all(np.abs(yDifNumericC-yDifAnalytic)<tolC)