"""
Functions to analyze whether a storage ring system fits into the room. The reference frame here has 0,0 as the location
of the existing focus with the door to the lab a positive x,y. The floor plan should be looked at from above (+z),
where the wall that is parallel to the hall and adjacent to the door is the "right wall" and the wall that is
perpindicular to the hall and alongside the optics table is the "bottom wall". These are the two walls that bound the
storage ring.
"""

import itertools
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import translate, rotate
from shapely.geometry import box, LineString, Polygon

ShapelyObject = Union[box, LineString, Polygon]


def wall_coordinates() -> tuple[float, float]:
    """Get x and y positions of wall. Reference frame is 0,0 at existing origin"""

    wall_right_x = 427e-2
    wall_bottom_y = -400e-2
    return wall_right_x, wall_bottom_y


def make_walls_right_and_bottom() -> list[ShapelyObject]:
    """Make shapely objects of the two walls bounding the storage ring"""
    wall_right_x, wall_bottom_y = wall_coordinates()
    wall_right = LineString([(wall_right_x, 0.0), (wall_right_x, wall_bottom_y)])
    wall_bottom = LineString([(0.0, wall_bottom_y), (wall_right_x, wall_bottom_y)])
    return [wall_right, wall_bottom]


def walls_and_structures_in_room() -> tuple[list[ShapelyObject], list[ShapelyObject]]:
    """Make list of walls and structures (chamber and optics table) in lab. Only two walls are worth modeling, the 
    walls that are closest to the ring"""

    optics_table_edge_x = -3e-2
    optics_table_edge_y = -130e-2
    optics_table_width = 135e-2
    chamber_width = 56e-2

    beam_tube_width = 3e-2
    beam_tube_length = 10e-2
    chamber_and_optics_table_length = 1.0  # exact value not important. Just for overlap algo
    beam_tube = box(0, 0, beam_tube_length, beam_tube_width)
    beam_tube = translate(beam_tube, -beam_tube_length, -beam_tube_width / 2.0)

    chamber = box(0, 0, chamber_and_optics_table_length, chamber_width)
    chamber = translate(chamber, -chamber_and_optics_table_length - beam_tube_length, -chamber_width / 2)

    optics_table = box(0, 0, chamber_and_optics_table_length, optics_table_width)
    optics_table = translate(optics_table, -chamber_and_optics_table_length + optics_table_edge_x,
                             -optics_table_width + optics_table_edge_y)
    structures = [beam_tube, chamber, optics_table]
    walls = make_walls_right_and_bottom()
    return walls, structures


def bumper_guess(component):
    r2_bumper_lab = np.array([1.1, -.1, 0.0])
    bump_tilt = -.08
    rot_angle = bump_tilt
    component = translate(component, r2_bumper_lab[0], r2_bumper_lab[1])
    component = rotate(component, rot_angle, use_radians=True, origin=(0, 0))
    return component


def storage_ring_system_components(model) -> list[ShapelyObject]:
    """Make list of shapely objects representing outer dimensions of magnets and vacuum tubes of storage ring system.
    """

    first_el = model.lattice_injector.elList[0]
    r1_ring = model.convert_position_injector_to_ring_frame(first_el.r1)
    n1_ring = model.convert_momentum_injector_to_ring_frame(first_el.nb)

    angle = np.arctan2(n1_ring[1], n1_ring[0])
    rot_angle = -np.pi - angle
    components = []
    components_ring_frame = model.floor_plan_shapes('exterior')
    for component in components_ring_frame:
        component = translate(component, -r1_ring[0], -r1_ring[1])
        component = rotate(component, rot_angle, use_radians=True, origin=(0, 0))
        component = component if model.has_bumper else bumper_guess(component)
        components.append(component)
    return components


def does_fit_in_room(model) -> bool:
    """Check if the arrangement of elements in 'model' is valid. This tests wether any elements extend to the right of
    the rightmost wall or below the bottom wall, or if any elements overlap with the chamber or the table"""

    _, structures = walls_and_structures_in_room()
    wall_right_x, wall_bottom_y = wall_coordinates()
    is_valid = True
    min_area_overlap_invalid = 1e-10  # I check are overlap to determine in a configuration is invalid, 
    # but I don't want false positives if there is effectively no overlap, but it isn't zero
    components = storage_ring_system_components(model)
    for component, structure in itertools.product(components, structures):
        x, y = component.exterior.xy
        x, y = np.array(x), np.array(y)
        is_out_side_wall = np.any(x > wall_right_x) or np.any(y < wall_bottom_y)
        is_overlapping = structure.intersection(component).area > min_area_overlap_invalid
        if is_out_side_wall or is_overlapping:
            is_valid = False
            break
    return is_valid


def plot_floor_plan_in_lab(model):
    """Plot the floorplan of the lab (walls, outer dimensions of magnets and vacuum tubes, optics table and chamber)
    """

    components = storage_ring_system_components(model)
    walls, structures = walls_and_structures_in_room()
    for shape in itertools.chain(components, walls, structures):
        if type(shape) is Polygon:
            plt.plot(*shape.exterior.xy)
        else:
            plt.plot(*shape.xy, linewidth=5, c='black')
    plt.gca().set_aspect('equal')
    plt.xlabel("meter")
    plt.ylabel("meter")
    plt.show()
