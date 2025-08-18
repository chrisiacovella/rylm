import math
import numpy as np


####################################################################################
# ICOSAHEDRON
####################################################################################
# Cartesian coordinates of the vertices of an icosahedron
# The icosahedron has 12 vertices, where factor is the value of the golden ratio.
# Note this includes a central point at the origin (0, 0, 0)

factor = (1 + math.sqrt(5)) / 2.0
icosahedron = np.array(
    [
        [0, 0, 0],  # Central point
        [factor, 1, 0],
        [factor, -1, 0],
        [-factor, -1, 0],
        [-factor, 1, 0],
        [1, 0, factor],
        [-1, 0, factor],
        [-1, 0, -factor],
        [1, 0, -factor],
        [0, factor, 1],
        [0, -factor, 1],
        [0, -factor, -1],
        [0, factor, -1],
    ]
)

# create a structure that is the icosahedron but with additional points placed outside of the initial shell
icosahedron_extended = np.array(
    [
        [0, 0, 0],  # Central point
        [factor, 1, 0],
        [factor, -1, 0],
        [-factor, -1, 0],
        [-factor, 1, 0],
        [1, 0, factor],
        [-1, 0, factor],
        [-1, 0, -factor],
        [1, 0, -factor],
        [0, factor, 1],
        [0, -factor, 1],
        [0, -factor, -1],
        [0, factor, -1],
        [2 * factor, 2 * factor, 2 * factor],  # Additional point
        [-2 * factor, -2 * factor, -2 * factor],  # Additional point
        [2 * factor, -2 * factor, 2 * factor],  # Additional point
        [-2 * factor, 2 * factor, -2 * factor],  # Additional point
    ]
)

tetrahedron = np.array(
    [
        [0, 0, 0],
        [1, 0, -1 / math.sqrt(2)],
        [0, 1, 1 / math.sqrt(2)],
        [-1, 0, -1 / math.sqrt(2)],
        [0, -1, 1 / math.sqrt(2)],
    ]
)

square_planar = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
    ]
)

octahedron = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)

trigonal_pyramidal = np.array(
    [
        [0, 0, 0],  # Central point
        [1, 0, -1 / math.sqrt(2)],
        [0, 1, 1 / math.sqrt(2)],
        [-1, 0, -1 / math.sqrt(2)],
    ]
)
square_pyramidal = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)

triangular_bipyramidal = np.array(
    [
        [0, 0, 0],  # Central point
        [1, 0, 0],
        [-0.5, math.sqrt(3) / 2, 0],
        [-0.5, -math.sqrt(3) / 2, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
)

triangular_planar = np.array(
    [
        [0, 0, 0],  # Central point
        [1, 0, 0],
        [-0.5, math.sqrt(3) / 2, 0],  # Point A
        [-0.5, -math.sqrt(3) / 2, 0],  # Point B
    ]
)

linear = np.array(
    [
        [0, 0, 0],  # Central point
        [1, 0, 0],  # Point A
        [-1, 0, 0],  # Point B
    ]
)
