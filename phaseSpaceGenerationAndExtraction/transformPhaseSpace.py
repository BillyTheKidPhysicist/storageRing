import numpy as np
import matplotlib.pyplot as plt


def generate_2DPLot(particleCoords,x,yMax=.1,pyMax=10):
    #generate a plot of the phase space for a cloud of particles initially sitting at x=0
    #particleCoords: (n,6) array of n particles, and their respective phase coordinates
    #x: the location to generate the plot at
    #yMax: maximum transverse x position
    #pymax: maximum transverse momentum

    yList=[]
    pyList=[]
    for coord in particleCoords:
        x0, y0, z0, px0, py0, pz0 = coord
        dt = (x - x0) / px0
        y = y0 + py0 * dt
        if yMax>y>-yMax and pyMax>py0>-pyMax:
            yList.append(y)
            pyList.append(py0)




    image=np.histogram2d(yList,pyList,30)[0]
    image=np.transpose(image)
    image=np.flip(image,axis=0)
    # plt.plot(np.sum(hist,axis=0))
    # plt.show()
    print(image.sum())
    extent=[min(yList),max(yList),min(pyList),max(pyList)]
    aspect=(extent[1]-extent[0])/(extent[3]-extent[2])
    plt.title('Phase space plot of far field data')
    plt.ylabel("velocity, m/s")
    plt.xlabel('Position, m')
    plt.imshow(image,extent=extent,aspect=aspect)
    plt.show()


runName='run42Far'
particleCoords=np.loadtxt(runName+'PhaseSpaceParticleCloudOriginal.dat')
generate_2DPLot(particleCoords,0)
generate_2DPLot(particleCoords,-.2)
# generate_2DPLot(particleCoords,.4)
# generate_2DPLot(particleCoords,.8)
# generate_2DPLot(particleCoords,2)


