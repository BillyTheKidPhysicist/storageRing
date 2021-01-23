
#@staticmethod
#@numba.njit(numba.float64[:](numba.float64[:] ,numba.float64[: ,:]))
#def transform_Lab_Frame_Vector_Into_Element_Frame_NUMBA(vecNew ,RIn):
#    vec x =vecNew[0]
#    vec y =vecNew[1]
#    vecNew[0] = vecx * RIn[0, 0] + vecy * RIn[0, 1]
#    vecNew[1] = vecx * RIn[1, 0] + vecy * RIn[1, 1]
#    return vecNew


#def wrap(args):
#  q0=args[0]
#  v0=args[1]
#  h=args[2]
#  T=args[3]
#  q, p, qo, po, particleOutside = test.trace(q0, v0,h,T)
#  return (q0,q[-1])
#temp=multi_particleTrace()
#multi_Trace(wrap,argsList)

#
# q0=np.asarray([-1e-10,0,0])
# v0=np.asarray([-200.0,0,0])
#
# Lto=(2*L1+2*Lcap+2*1.15*np.pi)
# Lt=Lto*.0001
#
# dt=5e-6
# print('start, total time',np.round(Lt/200,3),' s')
# t=time.time()
# q, p, qo, po, particleOutside = test.trace(q0, v0,dt, Lt/200)
# # print(particleOutside,time.time()-t)
# test.show_Lattice(particleCoords=q[-1])
# plt.plot(qo[:,0][::10],test.test[::10])
# plt.show()
#
# def speed():
#     test.trace(q0, v0, dt, Lt / 200)
# t=time.time()
# #cProfile.run('speed()')
# #q, p, qo, po, particleOutside = test.trace(q0, v0,dt, Lt/200)
# #print(time.time()-t)
# #print(q[-1])
# # [-1.64437006  0.23518299  0.        ]
# #time: 4 sec
#
# # print(particleOutside)
#dataSteps=5
#q=q[::dataSteps]
#p=p[::dataSteps]
#qo=qo[::dataSteps]
# # print(q[-1])
# # #
##----------------find envelope----------------
#qoFunc=spi.interp1d(qo[:,0],qo[:,1])
#revs=int(qo[-1,0]/test.totalLength)
#print(revs)
#sArr=np.linspace(qo[0][0],test.totalLength,num=10000)
#envList=[]
#for s0 in sArr:
#    samps=np.arange(0,revs)*test.totalLength+s0
#    env=np.max(np.abs(qoFunc(samps)))
#    envList.append(env)
#plt.plot(sArr,envList)
#plt.grid()
#plt.show()
#
#
#
# #
# # #plt.plot(test.EList)
# # #plt.show()
# #
# #
plt.plot(qo[:,0]/test.totalLength,qo[:,1])
plt.grid()
plt.show()
# #print(test.EList)
# #v0Arr=np.sum(p**2,axis=1)
# #plt.plot(qo[:,0],qo[:,2])
# #plt.grid()
# #plt.show()
# #plt.plot(qo[:,0],test.TList)
# #plt.grid()
# #plt.show()
# #plt.plot(qo[:,0],test.VList)
# #plt.grid()
# #plt.show()
# #plt.plot(qo[:,0],test.EList)
# #plt.grid()
# #plt.show()
# #test.show_Lattice()
# #
#
# #q0=np.asarray([1,.01,0])
# #args=[.025,1,.01,1,.05,100]
# #PT=particleTracer(200)
# #el=Element(args,"BENDER_IDEAL_SEGMENTED",PT)
# #print(el.transform_Element_Into_Unit_Cell_Frame(q0))#






#
# #-------------------------------------------------------
# LlensArr=np.linspace(.1,.4,num=50)
# survivalList=[]
# for Llens in LlensArr:
#     rBend=1.0
#
#     lattice = ParticleTracerLattice(200.0)
#     particleTracer = ParticleTracer(lattice)
#     lattice.add_Lens_Ideal(Llens,1,.015)
#     lattice.add_Combiner_Ideal()
#     lattice.add_Lens_Ideal(Llens,1,.015)
#     lattice.add_Bender_Ideal(None,rBend,1.0,.01)
#     lattice.add_Lens_Ideal(None,1,.01)
#     lattice.add_Bender_Ideal(None,rBend,1.0,.01)
#     lattice.end_Lattice()
#
#     q0=np.asarray([-1e-10,0e-3,0])
#     v0=np.asarray([-200.0,0,0])
#
#     Lt=lattice.totalLength*100
#
#     dt=10e-6
#     print(Llens,Lt)
#     q, p, qo, po, particleOutside = particleTracer.trace(q0, v0,dt, Lt/200)
#     #plt.plot(qo[:,0],qo[:,1])
#     #plt.show()
#     survival=qo[-1,0]/lattice.totalLength
#     survivalList.append(survival)
#     print(particleOutside,survival)
# plt.plot(LlensArr,survivalList)
# plt.show()
#plt.plot(qo[:,0],qo[:,1])
#plt.show()
##
#lattice.show_Lattice(particleCoords=q[-1])







'''