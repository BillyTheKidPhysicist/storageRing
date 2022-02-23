from HalbachLensClass import HalbachLens,Layer,Sphere
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import time
class GeneticLens(HalbachLens):
    def __init__(self,DNA_List,sphereEndSymmetry=True):
        #DNA: list of dictionaries to construct lens. each entry in the list corresponds to a single layer. Layers
        #are assumed to be stacked on top of each other in the order they are entered. No support for multiple
        #concentric layers. arguments in first list entry are:
        #[radius, magnet width,length]
        #lens is centered at z=0
        self.DNA_List=DNA_List
        self.shimDNA_List=[]
        self.layerDNA_List=[]
        self.unpack_DNA_List()
        assert all(len(DNA)==4  for DNA in self.layerDNA_List)
        self.numLayers=len(self.layerDNA_List)
        self.length=sum([DNA['length'] for DNA in self.layerDNA_List])
        self.theta=None
        self.zArrLayers=self.make_zArr_Layers()
        self.layerList=[] #genetic lens is modeled as a list of layers and shimming component
        self.sphereList=[]
        self.sphereEndSymmetry=sphereEndSymmetry
        self.build()
    def unpack_DNA_List(self):
        for DNA in self.DNA_List:
            if DNA['component']=='shim':
                self.shimDNA_List.append(DNA)
            elif DNA['component']=='layer':
                self.layerDNA_List.append(DNA)
            else: raise ValueError
    def build(self):
        for DNA,z in zip(self.layerDNA_List,self.zArrLayers):
            layer=Layer(z,DNA['width'],DNA['length'],DNA['rp'])
            self.layerList.append(layer)
        for DNA in self.shimDNA_List:
            sphere=Sphere(DNA['radius'])
            sphere.position_Sphere(r=DNA['r'],phi=DNA['phi'],z=DNA['z'])
            sphere.orient(DNA['theta'],DNA['psi'])
            self.sphereList.append(sphere)

    def make_zArr_Layers(self):
        if self.numLayers==0:
            return []
        elif self.numLayers==1:
            return np.zeros(1)
        else:
            L_List=[DNA['length'] for DNA in self.layerDNA_List]
            L_Cumsum=np.cumsum(L_List)
            zList=[]
            for L,DNA in zip(L_Cumsum,self.DNA_List):
                zList.append(L-DNA['length']/2)
            zArr=np.asarray(zList)-self.length/2
            return zArr
    def _radius_Maxima(self,which):
        radiusList=[]
        for DNA in self.layerDNA_List:
            if isinstance(DNA['rp'],Iterable):
                radiusList.extend(DNA['rp'])
            else: radiusList.append(DNA['rp'])
        if which=='max':
            return max(radiusList)
        elif which=='min':
            return min(radiusList)
    def maximum_Radius(self):
        return self._radius_Maxima('max')
    def minimum_Radius(self):
        return self._radius_Maxima('min')
    def B_Vec(self,r):
        #r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        #Returns a either a (N,3) or (3) array, whichever matches the shape of the r array
        if len(r.shape)==1:
            rEval=np.asarray([r])
        else:
            rEval=r.copy()
        rEval=self._transform_r(rEval)
        BArr=np.zeros(rEval.shape)
        for layer in self.layerList:
            BArr+=layer.B(rEval)
        for sphere in self.sphereList:
            BArr+=sphere.B_Shim(rEval,planeSymmetry=self.sphereEndSymmetry)
        BArr=self._transform_Vector(BArr)
        if len(r.shape)==1:
            return BArr[0]
        else:
            return BArr
    def is_Geometry_Valid(self):
        for sphere in self.sphereList:
            for layer in self.layerList:
                if self._does_Sphere_Overlap_With_Layer(sphere,layer)==True:
                    return False
        return True
    def _does_Sphere_Overlap_With_Layer(self,sphere:Sphere,layer:Layer):
        # enforce geometric constraints of spheres and layer disk
        if sphere.z - sphere.radius > layer.z+layer.length/2:  # bottom of spher is above top of layer, so it's clear
            return False
        elif sphere.z+ sphere.radius < layer.z-layer.length/2:  #top of sphere is below bottom of layer, so it's clear
            return False
        elif sphere.r-sphere.radius>min(layer.rp)+layer.width: #sphere is outside the outer edge of the layer
            return False
        elif sphere.r+sphere.radius<min(layer.rp): #sphere is outside the outer edge of the layer
            return False
        elif layer.z-layer.length/2<sphere.z<layer.z+layer.length/2 and min(layer.rp)<sphere.r<min(layer.rp)+\
                layer.width: #sphere is inside the layer
            return True
        else: #intermediate region. Can be rolling off the edge so to speak
            return True
class TestHelper:
    def __init__(self):
        self.rp, self.magWidth, self.length = .05, .0254, .1
        self.numericTol=1e-14 #same approach should be this accurate on different machines
        self.algoTol=1e-9 #different approaches should give the same result, but algorithms might differ
        xArr = np.linspace(-self.rp * .5, self.rp * .5, 11)
        zArr = np.linspace(-self.length, self.length, 30)
        self.coords = np.asarray(np.meshgrid(xArr, xArr, zArr)).T.reshape(-1, 3)
    def run_Tests(self):
        self.test1()
        self.test2()
        print("Passed")
    def test1(self):
        #Test that a lens of one layer with all the magnets having the same symmetry gives the same
        #values as another lens made of slices with multiple magnets exploting symmetry but with the same valuues.
        DNA_List1=[{'component':'layer','rp':(self.rp,),'width':self.magWidth,'length':self.length}]
        magnetSymmetry, numlayer = 4, 10
        DNA_List2 = [{'component': 'layer', 'rp': (self.rp,)*magnetSymmetry, 'width': self.magWidth,
                      'length': self.length/numlayer}]*numlayer
        lens1,lens2=GeneticLens(DNA_List1),GeneticLens(DNA_List2)
        BNormGrad1,BNormGrad2=lens1.BNorm_Gradient(self.coords),lens2.BNorm_Gradient(self.coords)
        RMS1,RMS2=np.std(BNormGrad1),np.std(BNormGrad2)
        absMean1,absMean2=np.mean(np.abs(BNormGrad1)),np.mean(np.abs(BNormGrad2))
        RMS1_0,RMS2_0=5.051401935138513 ,5.051401935258421
        absMean1_0,absMean2_0=3.180307805389812 ,3.180307805708118
        self.assert_Fields_And_Length(RMS1,RMS2,absMean1,absMean2,RMS1_0,RMS2_0,absMean1_0,absMean2_0,lens1,lens2)
    def test2(self):
        #Test that the same configuration of shims gives the same result even if done differently by using or
        #not using symmetry and rotations
        DNA_List1=[{'component':'shim','radius':.01,'r':self.rp,'phi':0.1,'z':self.length/2,'theta':np.pi/4,
                    'psi':np.pi/3}]
        DNA_List2=[{'component':'shim','radius':.01,'r':self.rp,'phi':0.1,'z':self.length/2,'theta':np.pi/4,'psi':np.pi/3},
                   {'component':'shim','radius':.01,'r':self.rp,'phi':0.1,'z':-self.length/2,'theta':3*np.pi/4,
                    'psi':np.pi/3}]
        lens1, lens2 = GeneticLens(DNA_List1), GeneticLens(DNA_List2,sphereEndSymmetry=False)
        BNormGrad1,BNormGrad2=lens1.BNorm_Gradient(self.coords),lens2.BNorm_Gradient(self.coords)
        RMS1,RMS2=np.std(BNormGrad1),np.std(BNormGrad2)
        absMean1,absMean2=np.mean(np.abs(BNormGrad1)),np.mean(np.abs(BNormGrad2))
        RMS1_0,RMS2_0=1.259261565490003, 1.2592615654896586
        absMean1_0,absMean2_0=0.6293849561810528 ,0.6293849561809091
        self.assert_Fields_And_Length(RMS1,RMS2,absMean1,absMean2,RMS1_0,RMS2_0,absMean1_0,absMean2_0,lens1,lens2)
    def assert_Fields_And_Length(self,RMS1,RMS2,absMean1,absMean2,RMS1_0,RMS2_0,absMean1_0,absMean2_0,lens1,lens2):
        assert abs(RMS1 - RMS1_0) < self.numericTol and abs(RMS2 - RMS2_0) < self.numericTol
        assert abs(absMean1 - absMean1_0) < self.numericTol and abs(absMean2 - absMean2_0) < self.numericTol
        assert abs(RMS1 - RMS2) < self.algoTol and abs(absMean1 - absMean2) < self.algoTol
        assert abs(lens1.length - lens2.length) < self.numericTol
def test():
    TestHelper().run_Tests()
# print(time.time()-t)

# _test_Single_Layers()
# DNA_List=[{'component':'shim','radius':.01,'r':0.05,'phi':0.0,'z':.00,'theta':-np.pi/2,'psi':np.pi}]
# lens=GeneticLens(DNA_List,sphereEndSymmetry=False)
# # print(lens.is_Geometry_Valid())
# #
# rp=.03
# xArr=np.linspace(-rp,rp,50)
# coords=np.asarray(np.meshgrid(xArr,xArr,0.0)).T.reshape(-1,3)
# BVec=lens.B_Vec(coords)
# # #
# # # # plt.quiver(coords[:,0],coords[:,1],BVec[:,0],BVec[:,1])
# # # # plt.gca().set_aspect('equal')
# # # # plt.show()
# image=lens.BNorm(coords).reshape(len(xArr),len(xArr))
# print(np.sum(image)) #455.17114875612043
# # plt.imshow(image)
# # plt.show()
