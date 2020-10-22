import numpy as np
import matplotlib.pyplot as plt

class Element:
    def __init__(self,args,type):
        self.type=type
        self.Bp=None
        self.rp=None
        self.L=None

        self.unpack_Args(args)
    def unpack_Args(self,args):
        if self.type=="LENS":
            self.Bp=args[0]
            self.rp=args[1]
            self.L=args[2]
        if self.type=='DRIFT':
            self.L=args[0]
class particleTracer:
    def __init__(self):
        self.m_Actual = 1.16503E-26  # mass of lithium 7, SI
        self.u0_Actual = 9.274009E-24  # bohr magneton, SI
        #In the equation F=u0*B0'=m*a, m can be changed to one with the following sub: m=m_Actual*m_Adjust where m_Adjust
        # is 1. Then F=B0'*u0/m_Actual=B0'*u0_Adjust=m_Adjust*a
        self.m=1 #adjusted value
        self.u0=self.u0_Actual/self.m_Actual

        self.kb = .38064852E-23  # boltzman constant, SI
        self.q=np.zeros(3) #contains the particles current position coordinates
        self.p=np.zeros(3) #contains the particles current momentum. m is set to 1 so this is the same
            #as velocity
        self.h=None
        self.revs=None

        self.lattice=[]
    def add_Lens(self,Bp,rp,L):
        args=[Bp,rp,L]
        el=Element(args,'LENS')
        self.lattice.append(el)
    def add_Drift(self,L):
        args=[L]
        el=Element(args,'DRIFT')
        self.lattice.append(el)
    def trace(self,qi,vi,revs,h):
        self.q[0]=qi[0]
        self.q[1]=qi[1]
        self.q[2]=qi[2]
        self.p[0]=self.m*vi[0]
        self.p[1]=self.m*vi[1]
        self.p[2]=self.m*vi[2]
        self.revs=revs
        qList=[]
        pList=[]

        loop=True
        i=0
        while(loop==True):
            qList.append(self.q)
            pList.append(self.p)
            self.time_Step(h)
            #loop=self.loop_Check()
            i+=1
            if i==100:
                loop=False
        qArr=np.asarray(qList)
        pArr=np.asarray(pList)
        return qArr,pArr
    def time_Step(self,h):
        q=self.q #q old or q sub n
        p=self.p #p old or p sub n
        a=self.force(q)/self.m #acceleration old or acceleration sub n
        q_n=q+(p/self.m)*h+.5*a*h**2 #q new or q sub n+1
        a_n=self.force(q_n)/self.m #acceleration new or acceleration sub n+1
        p_n=p+self.m*.5*(a+a_n)*h
        self.q=q_n
        self.p=p_n

    def loop_Check(self):
        z=self.q[2]
        if z>2:
            return False
        else:
            return True
    def force(self,q):
        F=np.zeros(3)
        F[0]=-q[0]
        F[1]=-q[1]
        F[2]=-q[2]
        return F
    def which_Element(self,q):
        pass

test=particleTracer()
test.add_Lens(1,.01,.5)
test.add_Drift(1)