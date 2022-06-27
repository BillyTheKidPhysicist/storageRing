from helperTools import *
def change_Value_Along_Dimension(xi,dim,delta,bounds):
    xNew=np.array(xi)
    xNew[dim]+=delta
    xUpper=np.array(bounds)[:,1]
    xLower=np.array(bounds)[:,0]
    xNew=np.clip(xNew,a_min=xLower,a_max=xUpper)
    return xNew
def get_Search_Params(xi,delta,bounds):
    paramsList=[]
    for i in range(len(xi)):
        for shift in [-delta,delta]:
            paramsList.append(change_Value_Along_Dimension(xi,i,shift,bounds))
    return paramsList

def find_Optimal_Value(func,xi,delta,bounds,processes):
    assert len(bounds) == len(xi)
    paramsList=get_Search_Params(xi,delta,bounds)
    results=tool_Parallel_Process(func,paramsList,processes=processes)
    return paramsList[np.argmin(results)],np.min(results)

def line_Search(costFunc,xi,deltaInitial,bounds,costInitial=None,processes=-1):
    xOpt,delta=xi,deltaInitial
    costOpt=costInitial if costInitial is not None else costFunc(xi)
    i,maxiters=0,10
    while i<maxiters:
        i+=1
        print('results',costOpt, delta, xOpt)
        xOptNew, costOptNew = find_Optimal_Value(costFunc,xOpt, delta,bounds,processes)
        if costOptNew >= costOpt:
            delta *= .75
        else:
            xOpt = xOptNew
            costOpt = costOptNew
    return xOpt,costOpt
