from ParaWell import ParaWell
import numpy as np
import scipy.interpolate as spi
def generate_New_Coords(oldCoords,stepSizex,stepSizey,stepSizez):
    numStepsx=int((oldCoords[:,0].max()-oldCoords[:,0].min())/stepSizex)+1
    numStepsy=int((oldCoords[:,1].max()-oldCoords[:,1].min())/stepSizey)+1
    numStepsz=int((oldCoords[:,2].max()-oldCoords[:,2].min())/stepSizez)+1
    xArr=np.linspace(oldCoords[:,0].min(),oldCoords[:,0].max(),numStepsx)
    yArr=np.linspace(oldCoords[:,1].min(),oldCoords[:,1].max(),numStepsy)
    zArr=np.linspace(oldCoords[:,2].min(),oldCoords[:,2].max(),numStepsz)
    newCoords=np.asarray(np.meshgrid(xArr,yArr,zArr)).T.reshape(-1,3)
    return newCoords
def generate_Trace_Data_From_Shitty_Comsol_Data(data,stepSizex=1e-3,stepSizey=1e-3,stepSizez=1e-3):
    #take data, an array of coordinates and magnetic field norms, and return an array of the coordinates, gradient
    #and magnetic field norm
    #data: (m,4) array where m is number of points and each row is (x,y,z,B0)
    #stepsize x y and z: The step size in the return data
    #return (m,7) array where each row is (x,y,z,Bgradx,Bgrady,Bgradz,B0)
    coordsArr=data[:,:3]
    B0FuncTemp=spi.LinearNDInterpolator(coordsArr,data[:,-1])
    newCoords=generate_New_Coords(coordsArr,stepSizex,stepSizey,stepSizez)
    B0Func=lambda x:B0FuncTemp(x)[0]
    dx=1e-6 #stepsize
    def derivative(x0,i):
        #compute derivative at x0 with central difference. If a nan is returned because out of bounds, use forward
        #or backward difference
        x2=x0.copy()
        x2[i]+=dx
        x1 = x0.copy()
        x1[i] -= dx
        B20=B0Func(x2)
        B10=B0Func(x1)
        if np.isnan(B20)==False and np.isnan(B10)==False:
            return (B20-B10)/(2*dx)
        elif np.isnan(B20)==False and np.isnan(B10)==True:
            B0=B0Func(x0)
            return (B20-B0)/dx
        elif np.isnan(B20)==True and np.isnan(B10)==False:
            B0=B0Func(x0)
            return (B0-B10)/dx
        else:
            raise Exception('Issue with data')
    def gradient_And_New_B0(coord):
        dBx = derivative(coord, 0)
        dBy = derivative(coord, 1)
        dBz = derivative(coord, 2)
        B0=B0Func(coord)
        return [dBx,dBy,dBz,B0]
    newFieldValsList=ParaWell().parallel_Chunk_Problem(gradient_And_New_B0,newCoords,onlyReturnResults=True)#
    # =np.asarray(newFieldValsList)
    newData=np.column_stack((newCoords,newFieldValsList))
    return newData
data=np.loadtxt('rawComsolCombinerData.txt')
newData=generate_Trace_Data_From_Shitty_Comsol_Data(data)
print(newData.shape)
np.savetxt('combinerData.txt',newData)