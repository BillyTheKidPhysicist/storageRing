import sys
import time
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
import matplotlib.pyplot as plt
import sympy as sym
from periodicLatticeSolver import PeriodicLatticeSolver
from shapely.ops import unary_union as union
import numpy as np


class FloorPlan:
    def __init__(self, PLS, parametersOnly=False):
        self.PLS = PLS
        self.objects = None
        self.combinerLength = self.PLS.latticeElementList[self.PLS.combinerIndex].Length
        self.parameters = None  # to hold the parameters used in the last build

        self.combinerWidth = .3
        rCombiner = self.PLS.latticeElementList[self.PLS.combinerIndex].r0
        self.combinerInputAngle = np.arcsin(self.combinerLength / rCombiner) * 180 / np.pi #Half angle seperation
            #between 2 incoming beams, deg
        self.combinerInputSep = 2 * self.combinerLength ** 2 / (2 * rCombiner)
        self.bendingRadius = None
        self.rOuterBender = .15
        self.benderPoints = 50
        self.TL1 = None  # tracklength 1
        self.TL2 = None  # trackLength 2
        self.wallSpacing=.3 #Minimum distance between wall and any component of the ring. meters
        self.focusToWallDistance=4.4 #distance from focus of the collector magnet to the wall along the nozzle axis, meters

        self.Lm4 = None
        self.rOuter4 = None
        self.Lm3 = None
        self.rOuter3 = None
        self.Lm1 = None
        self.rOuter1 = None
        self.Lm2 = None
        self.rOuter2 = None
        self.rOuterRatio = 6.5  # outer dimension of lens to bore
        self.airGap1 = .01  # extra lens spacing for coils
        self.airGap2 = .01  # extra lens spacing for coils
        self.airGap3 = .01  # extra lens spacing for coils
        self.airGap4 = .01  # extra lens spacing for coils
        self.coilGapRatio=2 #ratio between required coil spacing and bore radius

        # injector stuff
        self.LiMin = self.combinerLength + .2
        self.LiMax = 3
        self.LmMin = .025
        self.LmMax = .5
        self.LoMin = .05
        self.LoMax = .2
        self.Lo = None
        self.Lm = None
        self.Li = None

        self.rOuterShaper = .15  # shaper magnet yoke radius
        self.rVacuumTube = .02

        # total floorplan stuff
        self.distanceFromObjectMax = 4.2  # maximum distance from object of injector to wall
        self.combinerLen1Sep = .2  # 20 cm between output of combiner and lens1

        self.combiner = None
        self.lensShaper = None
        self.bender1 = None
        self.bender2 = None
        self.lens1 = None
        self.lens2 = None
        self.lens3 = None
        self.lens4 = None
        self.lens4VacuumTube = None
        self.shaperVacuumTube = None

        # make floorplan list func
        self.floorPlanArgListFunc = None
        if parametersOnly == False:
            if self.PLS.combinerIndex != None:
                temp = []
                temp.append(self.PLS.trackLength - (self.PLS.latticeElementList[self.PLS.combinerIndex].S + self.PLS.latticeElementList[
                    self.PLS.combinerIndex].Length / 2))  #
                # tracklength 1
                temp.append(self.PLS.latticeElementList[self.PLS.combinerIndex].S + self.PLS.latticeElementList[
                    self.PLS.combinerIndex].Length / 2)  # tracklength 2
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[3]].Length)  # lens lengths
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[0]].Length)  # lens lengths
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[1]].Length)  # lens lengths
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[2]].Length)  # lens lengths

                temp.append(self.PLS.latticeElementList[self.PLS.benderIndices[0]].r0)  # bender radius, same for both as of now
                temp.append(self.PLS.latticeElementList[self.PLS.combinerIndex].Length)  # length of combiner
                temp.append(self.PLS.injector.Lo)
                temp.append(self.PLS.injector.Lm)
                temp.append(self.PLS.injector.Li)

                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[1]].rp * self.rOuterRatio)
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[2]].rp * self.rOuterRatio)
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[3]].rp * self.rOuterRatio)
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[0]].rp * self.rOuterRatio)

                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[3]].rp * self.coilGapRatio)
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[0]].rp * self.coilGapRatio)
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[1]].rp * self.coilGapRatio)
                temp.append(self.PLS.latticeElementList[self.PLS.lensIndices[2]].rp * self.coilGapRatio)


                args = self.PLS.sympyVarList.copy()
                args.extend(self.PLS.injector.sympyVarList)
                self.floorPlanArgListFunc = sym.lambdify(args, temp)

    def load_Paramters(self, args):
        self.parameters = self.floorPlanArgListFunc(*args)
        self.TL1 = self.parameters[0]
        self.TL2 = self.parameters[1]
        self.Lm1 = self.parameters[2]
        self.Lm2 = self.parameters[3]
        self.Lm3 = self.parameters[4]
        self.Lm4 = self.parameters[5]
        self.bendingRadius = self.parameters[6]
        self.combinerLength = self.parameters[7]
        self.Lo = self.parameters[8]
        self.Lm = self.parameters[9]
        self.Li = self.parameters[10]
        self.rOuter1 = self.parameters[11]
        self.rOuter2 = self.parameters[12]
        self.rOuter3 = self.parameters[13]
        self.rOuter4 = self.parameters[14]

        self.airGap1 = self.parameters[15]
        self.airGap2 = self.parameters[16]
        self.airGap3 = self.parameters[17]
        self.airGap4 = self.parameters[18]



    def build(self, args, reloadParams=True):
        if reloadParams == True:
            # don't bother refilling the parameters list if I already di
            self.load_Paramters(args)
        self.objects = []
        self.combiner = Polygon(
            [(0, 0), (0, self.combinerWidth), (self.combinerLength, self.combinerWidth), (self.combinerLength, 0)])
        self.combiner = translate(self.combiner, yoff=-self.combinerWidth / 2)
        self.objects.append(self.combiner)

        angle = 180
        angleArr = np.linspace(-angle / 2, angle / 2, num=self.benderPoints)
        x1 = (self.bendingRadius - self.rOuterBender) * np.cos(angleArr * np.pi / 180)
        y1 = (self.bendingRadius - self.rOuterBender) * np.sin(angleArr * np.pi / 180)
        angleArr = np.linspace(angle / 2, -angle / 2, num=self.benderPoints)
        x2 = (self.bendingRadius + self.rOuterBender) * np.cos(angleArr * np.pi / 180)
        y2 = (self.bendingRadius + self.rOuterBender) * np.sin(angleArr * np.pi / 180)
        x = np.append(x1, x2)
        y = np.append(y1, y2)

        self.bender2 = Polygon(np.column_stack((x[None].T, y[None].T)))
        self.lens4 = Polygon([(0, 0), (self.Lm4, 0), (self.Lm4, self.rOuter4 * 2), (0, self.rOuter4 * 2)])
        self.lens4 = translate(self.lens4, xoff=-self.Lm4 - self.airGap4, yoff=-self.rOuter4)
        self.lens3 = Polygon([(0, 0), (self.Lm3, 0), (self.Lm3, self.rOuter3 * 2), (0, self.rOuter3 * 2)])
        self.lens3 = translate(self.lens3, xoff=-self.Lm3 - self.airGap3, yoff=self.bendingRadius - self.rOuter3)
        self.lens4VacuumTube = Polygon(
            [(0, 0), (self.TL2 - self.combinerLength - self.Lm4 - self.airGap4, 0), (self.TL2 - self.combinerLength
                                                                                    - self.Lm4 - self.airGap4,
                                                                                    self.rVacuumTube * 2),
             (0, self.rVacuumTube * 2)])
        self.lens4VacuumTube = translate(self.lens4VacuumTube, xoff=self.combinerLength, yoff=-self.rVacuumTube)
        self.lens4VacuumTube = rotate(self.lens4VacuumTube, self.combinerInputAngle, origin=(self.combinerLength, 0))

        self.bender2 = translate(self.bender2, yoff=self.bendingRadius)
        self.lens3 = translate(self.lens3, yoff=self.bendingRadius)
        self.bender2 = rotate(self.bender2, self.combinerInputAngle, origin=(self.combinerLength, 0))
        self.lens3 = rotate(self.lens3, self.combinerInputAngle, origin=(self.combinerLength, 0))
        self.lens4 = rotate(self.lens4, self.combinerInputAngle, origin=(self.combinerLength, 0))
        self.bender2 = translate(self.bender2, yoff=self.combinerInputSep / 2)
        self.lens3 = translate(self.lens3, yoff=self.combinerInputSep / 2)
        self.lens4 = translate(self.lens4, yoff=self.combinerInputSep / 2)
        tempx = self.TL2 * np.cos(self.combinerInputAngle * np.pi / 180)
        tempy = self.TL2 * np.sin(self.combinerInputAngle * np.pi / 180)
        self.bender2 = translate(self.bender2, xoff=tempx, yoff=tempy)
        self.lens3 = translate(self.lens3, xoff=tempx, yoff=tempy)
        self.lens4 = translate(self.lens4, xoff=tempx, yoff=tempy)

        self.objects.append(self.bender2)
        self.objects.append(self.lens4)
        self.objects.append(self.lens3)
        self.objects.append(self.lens4VacuumTube)

        self.bender1 = Polygon(np.column_stack((x[None].T, y[None].T)))
        self.bender1 = rotate(self.bender1, 180, origin=(0, 0))
        self.bender1 = translate(self.bender1, yoff=self.bendingRadius, xoff=-self.TL1)
        self.lens1 = Polygon([(0, 0), (self.Lm1, 0), (self.Lm1, self.rOuter1 * 2), (0, self.rOuter1 * 2)])
        self.lens1 = translate(self.lens1, xoff=-self.TL1 + self.airGap1, yoff=-self.rOuter1)
        self.lens2 = Polygon([(0, 0), (self.Lm2, 0), (self.Lm2, self.rOuter2 * 2), (0, self.rOuter2 * 2)])
        self.lens2 = translate(self.lens2, xoff=-self.TL1 + self.airGap2, yoff=2 * self.bendingRadius - self.rOuter2)
        self.objects.append(self.bender1)
        self.objects.append(self.lens1)
        self.objects.append(self.lens2)

        self.lensShaper = Polygon([(0, 0), (self.Lm, 0), (self.Lm, self.rOuterShaper * 2), (0, self.rOuterShaper * 2)])
        self.lensShaper = translate(self.lensShaper, yoff=-self.rOuterShaper, xoff=self.combinerLength)
        self.lensShaper = rotate(self.lensShaper, -self.combinerInputAngle, origin=(self.combinerLength, 0))
        self.lensShaper = translate(self.lensShaper, yoff=-self.combinerInputSep / 2)
        tempx = (self.Li - self.combinerLength) * np.cos(self.combinerInputAngle * np.pi / 180)
        tempy = (self.Li - self.combinerLength) * np.sin(self.combinerInputAngle * np.pi / 180)
        self.lensShaper = translate(self.lensShaper, xoff=tempx, yoff=-tempy)
        self.shaperVacuumTube = Polygon(
            [(0, 0), (self.Li - self.combinerLength, 0), (self.Li - self.combinerLength, self.rVacuumTube * 2),
             (0, self.rVacuumTube * 2)])
        self.shaperVacuumTube = translate(self.shaperVacuumTube, yoff=-self.rVacuumTube, xoff=self.combinerLength)
        self.shaperVacuumTube = rotate(self.shaperVacuumTube, -self.combinerInputAngle, origin=(self.combinerLength, 0))
        self.shaperVacuumTube = translate(self.shaperVacuumTube, yoff=-self.combinerInputSep / 2)
        self.objects.append(self.lensShaper)
        self.objects.append(self.shaperVacuumTube)

        # numShapes=len(self.objects)
        # temp=np.zeros((numShapes,numShapes))
        # for i in range(numShapes):
        #  for j in range(i+1,numShapes):
        #    temp[i,j]=1
        # print(temp)

    def show_Floor_Plan(self, sol=None, args=None):
        if args != None:  # if someone provides values for arguments then build it, otherwise its already built
            self.build(args)
        if sol != None:
            self.build(sol.args)
        if (sol == None and args == None) or (sol != None and args != None):
            raise Exception('YOU CANNOT PROVIDE BOTH A SOLUTION AND ARGUMENTS TO PLOT')
        for i in range(len(self.objects)):
            plt.plot(*self.objects[i].exterior.xy)
        plt.xlim(-2, 2)
        plt.ylim(-1, 3)
        plt.show()

    def calculate_Cost(self, args=None, offset=4, areaWeight=100, lineWeight=1):
        # This method works fast by only building the floorplan, which takes a few ms, if all simple costs return zero
        # use fractional overlapping area, (area overlapping)/(total area)



        totalCost = 0
        if args is not None:
            # if no arguments are provided, just use the cpreviously built floorplan. otherwise rebuild
            self.load_Paramters(args)

        xMostLeft=(self.Lo+self.Lm+self.Li)*np.cos(self.combinerInputAngle*np.pi/180) #the most leftward point of the
                #bender. This is used to prevent the layout from impinging onto the wall and too keep enough seperation
        xMostLeft+=self.TL1+self.bendingRadius+self.rOuterBender
        if xMostLeft>(self.focusToWallDistance-self.wallSpacing):
            totalCost+=lineWeight*np.abs(xMostLeft-(self.focusToWallDistance-self.wallSpacing))
        if self.Li < self.LiMin:
            totalCost += lineWeight * np.abs(self.LiMin - self.Li)
        if self.Li > self.LiMax:
            totalCost += lineWeight * np.abs(self.Li - self.LiMax)
        if self.TL1 < 0:
            totalCost += lineWeight * np.abs(self.TL1)
        if self.TL2 < 0:
            totalCost += lineWeight * np.abs(self.TL2)
        if (self.Lm2 + self.Lm3 + self.airGap2+self.airGap3) > (self.TL1 + self.TL2):  # if the two lenses overlap
            totalCost += lineWeight * np.abs((self.Lm2 + self.Lm3 +self.airGap2+self.airGap3) - (self.TL1 + self.TL2))
        if (self.Li + self.Lm + self.Lo + self.TL1 + self.bendingRadius) > self.distanceFromObjectMax:
            totalCost += lineWeight * np.abs(
                (self.Li + self.Lm + self.Lo + self.TL1 + self.bendingRadius) - self.distanceFromObjectMax)
        if (self.TL1 - self.Lm1 - self.airGap1) < self.combinerLen1Sep:
            # if there is not enough room between the combiner output and next lens for optical pumping:
            totalCost += lineWeight * np.abs((self.TL1 - self.Lm1 - self.airGap1) - self.combinerLen1Sep)
        if (self.TL2 - self.Lm4 - self.combinerLength) < 0:
            # if lens 4 is impinging on the combiner
            totalCost += lineWeight * np.abs(self.TL2 - self.Lm4 - self.combinerLength)
        if totalCost < 1E-10:  # to prevent rounding errors that screwed me up before
            area = 0
            if args is not None:
                # if args are None then use the current floorplan
                self.build(args, reloadParams=False)
            area += self.lens1.intersection(self.combiner).area / (self.lens1.area + self.combiner.area)  # fractional area overlap
            # between lens1 and combiner
            area += self.lens4.intersection(self.lensShaper).area / (
                        self.lens1.area + self.lensShaper.area)  # farctional area overlap
            # between lens4 and shaper lens
            area += self.bender2.intersection(self.lensShaper).area / (
                        self.lensShaper.area + self.bender2.area)  # fractional area overlap
            # between shaper lens and bender 2
            area += self.shaperVacuumTube.intersection(self.lens4).area / (self.shaperVacuumTube.area + self.lens4.area)
            area += self.lens4VacuumTube.intersection(self.lensShaper).area / (self.lens4VacuumTube.area + self.lensShaper.area)
            area += self.shaperVacuumTube.intersection(self.bender2).area / (self.shaperVacuumTube.area + self.bender2.area)
            totalCost += area * areaWeight

        return totalCost + offset


