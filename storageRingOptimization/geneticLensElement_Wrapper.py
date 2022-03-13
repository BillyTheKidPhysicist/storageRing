raise NotImplementedError
#under construction


from HalbachLensClass import HalbachLens,Layer,Sphere,RectangularPrism
from shapely.geometry import Polygon,Point
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
class GeneticLens(HalbachLens):
    def __init__(self,DNA_List):
        #DNA: list of dictionaries to construct lens. each entry in the list corresponds to a single layer. Layers
        #are assumed to be stacked on top of each other in the order they are entered. No support for multiple
        #concentric layers. arguments in first list entry are:
        #[radius, magnet width,length]
        #lens is centered at z=0
        assert isinstance(DNA_List,list)
        assert all('component' in DNA for DNA in DNA_List)
        self.DNA_List=DNA_List
        self.shimDNA_List=[]
        self.layerDNA_List=[]
        self.unpack_DNA_List()
        self.numLayers=len(self.layerDNA_List)
        self.length=sum([DNA['length'] for DNA in self.layerDNA_List])
        self.theta=None
        self.zArrLayers=self.make_zArr_Layers()
        self.layerList=[] #genetic lens is modeled as a list of layers and shimming component
        self.shimList=[]
        self.cubeList=[]
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
            phase=DNA['phase'] if 'phase' in DNA else 0.0
            layer=Layer(z,DNA['width'],DNA['length'],DNA['rp'],phase=phase)
            self.layerList.append(layer)
        for DNA in self.shimDNA_List:
            if DNA['shape']=='cube':
                cube=RectangularPrism(DNA['diameter'],DNA['diameter'])
                cube.place(DNA['r'],DNA['phi'],DNA['z'],DNA['psi'])
                self.shimList.append(cube)
            elif DNA['shape']=='sphere':
                sphere=Sphere(DNA['diameter']/2.0)
                sphere.position_Sphere(r=DNA['r'],phi=DNA['phi'],z=DNA['z'])
                sphere.orient(DNA['theta'],DNA['psi'])
                self.shimList.append(sphere)
            else: raise ValueError

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
        for shim,shimDNA in zip(self.shimList,self.shimDNA_List):
            BArr+=shim.B_Shim(rEval,planeSymmetry=shimDNA['planeSymmetry'])
        BArr=self._transform_Vector(BArr)
        if len(r.shape)==1:
            return BArr[0]
        else:
            return BArr
    def is_Geometry_Valid(self):
        return False if self.geometry_Frac_Overlap()>1e-6 else True
    def geometry_Frac_Overlap(self):
        totalFrac=0
        for shim in self.shimList:
            for layer in self.layerList:
                if isinstance(shim,Sphere):
                    totalFrac+= self.sphere_Frac_Overlap_With_Layer(shim,layer)
                else:
                    totalFrac+=self.cube_Frac_Overlap_With_Layer(shim,layer)
        return totalFrac
    def make_Layer_Profile(self,layer:Layer):
        z0 = layer.z
        L = layer.length
        rp = min(layer.rp)
        width = layer.width
        profile = Polygon([(rp, z0 - L / 2), (rp + width, z0 - L / 2), (rp + width, z0 + L / 2), (rp, z0 + L / 2)])
        return profile
    def make_Cube_Profile_Radial(self,cube:RectangularPrism):
        #approximately the profile. Will be different because the cube can be rotated about itself
        assert isinstance(cube,RectangularPrism)
        width=cube.width 
        r=cube.r
        z=cube.z
        profile=Polygon([(r-width/2,z-width/2),(r+width/2,z-width/2),(r+width/2,z+width/2),
                             (r-width/2,z+width/2)])
        return profile
    def cube_Frac_Overlap_With_Layer(self, cube:RectangularPrism, layer: Layer):
        assert len(layer.rp) == 1  # this does not work for goofy shapes of layers yet
        assert isinstance(cube,RectangularPrism) and isinstance(layer, Layer)
        cubeProfile=self.make_Cube_Profile_Radial(cube)
        layerProfile=self.make_Layer_Profile(layer)
        overlapFrac = layerProfile.intersection(cubeProfile).area / cubeProfile.area
        return overlapFrac

    def sphere_Frac_Overlap_With_Layer(self,sphere:Sphere,layer:Layer):
        assert len(layer.rp)==1 # this does not work for goofy shapes of layers yet
        assert isinstance(sphere,Sphere) and isinstance(layer,Layer)
        r = sphere.r
        z = sphere.z
        radius = sphere.radius
        z0 = layer.z
        L = layer.length
        rp = min(layer.rp)
        width = layer.width
        layerProfile = Polygon([(rp, z0 - L / 2), (rp + width, z0 - L / 2), (rp + width, z0 + L / 2), (rp, z0 + L / 2)])
        sphere = Point(r, z).buffer(radius)
        overlapFrac = layerProfile.intersection(sphere).area / sphere.area
        return overlapFrac