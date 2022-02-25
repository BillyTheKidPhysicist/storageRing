import numpy as np
from geneticLensClass import GeneticLens
class testHelper:
    def __init__(self):
        self.rp, self.magWidth, self.length = .05, .0254, .1
        self.numericTol=1e-14 #same approach should be this accurate on different machines
        self.algoTol=1e-9 #different approaches should give the same result, but algorithms might differ
        xArr = np.linspace(-self.rp * .5, self.rp * .5, 11)
        zArr = np.linspace(-self.length, self.length, 30)
        self.coords = np.asarray(np.meshgrid(xArr, xArr, zArr)).T.reshape(-1, 3)
    def run_tests(self):
        self.test1()
        self.test2()
        print("Passed")
    def test1(self):
        #test that a lens of one layer with all the magnets having the same symmetry gives the same
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
        assert lens1.geometry_Frac_Overlap() + lens2.geometry_Frac_Overlap() == 0.0
        assert abs(np.mean(BNormGrad1)) < self.algoTol and abs(np.mean(BNormGrad2)) < self.algoTol
        self.assert_Fields_And_Length(RMS1,RMS2,absMean1,absMean2,RMS1_0,RMS2_0,absMean1_0,absMean2_0,lens1,lens2)
    def test2(self):
        #test that the same configuration of shims gives the same result even if done differently by using or
        #not using symmetry and rotations
        DNA_List1=[{'component':'shim','radius':.01,'r':self.rp,'phi':0.1,'z':self.length/2,'theta':np.pi/4,
                    'psi':np.pi/3,'planeSymmetry':True}]
        DNA_List2=[{'component':'shim','radius':.01,'r':self.rp,'phi':0.1,'z':self.length/2,'theta':np.pi/4,
                    'psi':np.pi/3,'planeSymmetry':False},
                   {'component':'shim','radius':.01,'r':self.rp,'phi':0.1,'z':-self.length/2,'theta':3*np.pi/4,
                    'psi':np.pi/3,'planeSymmetry':False}]
        lens1, lens2 = GeneticLens(DNA_List1), GeneticLens(DNA_List2)
        BNormGrad1,BNormGrad2=lens1.BNorm_Gradient(self.coords),lens2.BNorm_Gradient(self.coords)
        RMS1,RMS2=np.std(BNormGrad1),np.std(BNormGrad2)
        absMean1,absMean2=np.mean(np.abs(BNormGrad1)),np.mean(np.abs(BNormGrad2))
        RMS1_0,RMS2_0=1.2592615654857615, 1.2592615654863557
        absMean1_0,absMean2_0=0.6293849561807945 ,0.6293849561808934
        assert lens1.geometry_Frac_Overlap()+lens2.geometry_Frac_Overlap()==0.0
        self.assert_Fields_And_Length(RMS1,RMS2,absMean1,absMean2,RMS1_0,RMS2_0,absMean1_0,absMean2_0,lens1,lens2)
    def assert_Fields_And_Length(self,RMS1,RMS2,absMean1,absMean2,RMS1_0,RMS2_0,absMean1_0,absMean2_0,lens1,lens2):
        assert abs(RMS1 - RMS1_0) < self.numericTol and abs(RMS2 - RMS2_0) < self.numericTol
        assert abs(absMean1 - absMean1_0) < self.numericTol and abs(absMean2 - absMean2_0) < self.numericTol
        assert abs(RMS1 - RMS2) < self.algoTol and abs(absMean1 - absMean2) < self.algoTol
        assert abs(lens1.length - lens2.length) < self.numericTol
def test():
    testHelper().run_tests()
