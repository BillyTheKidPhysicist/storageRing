"""
Functions to analyze whether a storage ring system fits into the room. The reference frame here has 0,0 as the location
of the existing focus with the door to the lab a positive x,y. The floor plan should be looked at from above (+z),
where the wall that is parallel to the hall and adjacent to the door is the "right wall" and the wall that is
perpindicular to the hall and alongside the optics table is the "bottom wall". These are the two walls that bound the
storage ring.
"""

from typing import Union

from shapely.affinity import translate, rotate
from shapely.geometry import box, LineString, Polygon

from helperTools import np, plt, itertools

shapelyObject = Union[box, LineString, Polygon]
shapelyList = list[shapelyObject]


def wall_Positions() -> tuple[float, float]:
    """Get x and y positions of wall. Reference frame is 0,0 at existing origin"""

    wallRight_x = 427e-2
    wallBottom_y = -400e-2
    return wallRight_x, wallBottom_y


def make_Walls_Right_And_Bottom() -> shapelyList:
    """Make shapely objects of the two walls bounding the storage ring"""
    wallRight_x, wallBottom_y = wall_Positions()
    wallRight = LineString([(wallRight_x, 0.0), (wallRight_x, wallBottom_y)])
    wallBottom = LineString([(0.0, wallBottom_y), (wallRight_x, wallBottom_y)])
    return [wallRight, wallBottom]


def make_Walls_And_Structures_In_Room() -> tuple[shapelyList, shapelyList]:
    """Make list of walls and structures (chamber and optics table) in lab. Only two walls are worth modeling, the 
    walls that are closest to the ring"""

    opticsTableEdge_x = -3e-2
    opticsTableEdge_y = -130e-2
    opticsTableWidth = 135e-2
    chamberWidth = 56e-2

    beamTubeWidth = 3e-2
    beamTubeLength = 10e-2
    chamberAndOpticsTableLength = 1.0  # exact value not important. Just for overlap algo
    beamTube = box(0, 0, beamTubeLength, beamTubeWidth)
    beamTube = translate(beamTube, -beamTubeLength, -beamTubeWidth / 2.0)

    chamber = box(0, 0, chamberAndOpticsTableLength, chamberWidth)
    chamber = translate(chamber, -chamberAndOpticsTableLength - beamTubeLength, -chamberWidth / 2)

    opticsTable = box(0, 0, chamberAndOpticsTableLength, opticsTableWidth)
    opticsTable = translate(opticsTable, -chamberAndOpticsTableLength + opticsTableEdge_x,
                            -opticsTableWidth + opticsTableEdge_y)
    structures = [beamTube, chamber, opticsTable]
    walls = make_Walls_Right_And_Bottom()
    return walls, structures


def make_Storage_Ring_System_Components(model) -> shapelyList:
    """Make list of shapely objects representing outer dimensions of magnets and vacuum tubes of storage ring system.
    """

    firstEl = model.latticeInjector.elList[0]
    r1Ring = model.convert_Pos_Injector_Frame_To_Ring_Frame(firstEl.r1)
    n1Ring = model.convert_Moment_Injector_Frame_To_Ring_Frame(firstEl.nb)
    r2BumperLab = np.array([1.1, -.1, 0.0])

    bumpTilt = -.08

    angle = np.arctan2(n1Ring[1], n1Ring[0])
    rotAngle = -np.pi - angle + bumpTilt
    components = []
    components_RingFrame = model.generate_Shapely_Object_List_Of_Floor_Plan('exterior')
    for component in components_RingFrame:
        component = translate(component, -r1Ring[0], -r1Ring[1])
        component = rotate(component, rotAngle, use_radians=True, origin=(0, 0))
        component = translate(component, r2BumperLab[0], r2BumperLab[1])
        components.append(component)
    return components


def does_Fit_In_Room(model) -> bool:
    """Check if the arrangement of elements in 'model' is valid. This tests wether any elements extend to the right of
    the rightmost wall or below the bottom wall, or if any elements overlap with the chamber or the table"""

    _, structures = make_Walls_And_Structures_In_Room()
    wallRight_x, wallBottom_y = wall_Positions()
    isInValid = False
    components = make_Storage_Ring_System_Components(model)
    for component in components:
        x, y = component.exterior.xy
        x, y = np.array(x), np.array(y)
        isInValid = np.any(x > wallRight_x) or np.any(y < wallBottom_y) or isInValid
        for structure in structures:
            isInValid = structure.intersects(component) or isInValid
    isValid = not isInValid
    return isValid


def plot_Floor_Plan(model):
    """Plot the floorplan of the lab (walls, outer dimensions of magnets and vacuum tubes, optics table and chamber)
    """

    components = make_Storage_Ring_System_Components(model)
    walls, structures = make_Walls_And_Structures_In_Room()
    for shape in itertools.chain(components, walls, structures):
        if type(shape) is Polygon:
            plt.plot(*shape.exterior.xy)
        else:
            plt.plot(*shape.xy, linewidth=5, c='black')
    plt.gca().set_aspect('equal')
    plt.xlabel("meter")
    plt.ylabel("meter")
    plt.show()
