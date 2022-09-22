"""Provided by Dr. Ortner from Austrian Silicon Labs"""

import magpylib as magpy
# pylint: disable=invalid-name, redefined-outer-name
import numpy as np
import copy
from magpylib import Collection

def mesh_cuboid(cuboid, nnn):
    """
    Splits Cuboid up into small Cuboid cells

    Parameters
    ----------
    cuboid: magpylib.Cuboid object
        input object to be discretized

    nnn: tuple, shape (3,), dtype=int
        discretization (nx,ny,nz)

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells
    """
    print(f'Meshing Cuboid with {nnn}')

    # load cuboid properties
    pos = cuboid.position
    rot = cuboid.orientation
    dim = cuboid.dimension
    mag = cuboid.magnetization0

    # secure input type
    nnn = np.array(nnn, dtype=int)

    # new dimension
    new_dim = dim / nnn

    # inside position grid
    xs, ys, zs = [np.linspace(d / 2 * (1 / n - 1), d / 2 * (1 - 1 / n), n) for d, n in zip(dim, nnn)]
    grid = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    grid = rot.apply(grid) + pos

    # create cells as magpylib objects and return Collection
    cells=[]
    for pp in grid:
        cube=magpy.magnet.Cuboid(mag, new_dim, pp, rot)
        cube.mur=copy.copy(cuboid.mur)
        cube.magnetization0=cuboid.magnetization0.copy()
        cells.append(cube)

    # cells = [magpy.magnet.Cuboid(mag, new_dim, pp, rot) for pp in grid]

    return magpy.Collection(cells)


def demag_tensor(collection, store=False, load=False):
    """
    Compute the demagnetization tensor T based on point matching (see Chadbec 2006)
    for n sources in the input collection.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    store: `False` or filename (str)
        Store T after computation as filename.npy.

    load: `False` or filename (str)
        Try to load T from filename.npy.

    Returns
    -------
    Demagnetization tensor: ndarray, shape (3,n,n,3)

    TODO: allow multi-point matching
    TODO: allow current sources
    TODO: allow external stray fields
    TODO: status bar when n>1000
    TODO: Speed up with direct interface for field computation
    TODO: Use newell formulas for cube-cube interactions
    """
    n = len(collection.sources_all)

    # load pre-computed tensor
    if isinstance(load, str):
        try:
            T = np.load(load + '.npy')
            print(' - load pre-computed demagnetization tensor')
            if n != T.shape[1]:
                raise ValueError('Loaded demag tensor is not of same shape as input collection')
            return T
        except FileNotFoundError:
            print(' - file not found')

    # compute cell positions
    pos = np.empty((n, 3))
    for i, src in enumerate(collection.sources_all):
        if isinstance(src, magpy.magnet.CylinderSegment):
            pos[i] = src.barycenter
        else:
            pos[i] = src.position
    # split up magnetizations

    coll3 = magpy.Collection()
    for unit_mag in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        for src in collection.sources_all:
            src.magnetization = src.orientation.inv().apply(unit_mag)  # ROTATION CHECK
            coll3.add(src.copy())
    # point matching field and demag tensor
    Hpoint = magpy.getH(coll3.sources, pos)  # shape (3n cells, n pos, 3 xyz)
    T = Hpoint.reshape(3, n, n, 3)  # shape (3 unit mag, n cells, n pos, 3 xyz)

    # store tensor
    if isinstance(store, str):
        np.save(store + '.npy', T)

    return T


def invert(matrix, solver):
    """
    Matrix inversion

    Parameters
    ----------
    matrix: np.array, shape (n,n)
        Input matrix to be inverted.

    solver: str
        Solver to be used. Must be one of (np.linalg.inv, ).

    Returns
    -------
    matrix_inverse: ndarray, shape (n,n)


    TODO implement and test different solver packages
    TODO check input matrix and auto-select correct solver (direct, iterative, ...)
    """

    if solver == 'np.linalg.inv':
        return np.linalg.inv(matrix)

    raise ValueError('Bad solver input.')


def apply_demag(
        collection_input,
        solver='np.linalg.inv',
        demag_store=False,
        demag_load=False,
):
    '''
    Computes the interaction between all collection magnets and fixes their magnetization.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    xi: array_like, shape (n,)
        Vector of n magnetic susceptibilities of the cells.

    solver: str, default='np.linalg.inv'
        Solver to be used. Must be one of (np.linalg.inv, ).

    demag_store: `False` or filename (str)
        Store demagnetization tensor T after computation as filename.npy.

    demag_load: `False` or filename (str)
        Try to load demagnetization tensor T from filename.npy.

    Returns
    -------
    None
    '''
    collection=Collection(collection_input)
    # in order for this to work with coils, I need to have the coils only apply a field, not be part of the matrix
    # system
    xi = [src.mur - 1.0 for src in collection.sources_all]
    n = len(collection.sources_all)

    # set up mr
    mag = [src.orientation.apply(src.magnetization0) for src in collection.sources_all]  # ROTATION CHECK
    mag = np.reshape(mag, (3 * n, 1), order='F')  # shape ii = x1, ... xn, y1, ... yn, z1, ... zn

    # set up S
    xi = np.array(xi)
    if len(xi) != n:
        raise ValueError('Apply_demag input collection and xi must have same length.')
    S = np.diag(np.tile(xi, 3))  # shape ii, jj

    # set up T
    T = demag_tensor(
        collection,
        store=demag_store,
        load=demag_load,
    )  # shape (3 mag unit, n cells, n positions, 3 Bxyz)

    T = T.swapaxes(0, 3)
    T = T * (4 * np.pi / 10)
    T = T.swapaxes(2, 3)
    T = np.reshape(T, (3 * n, 3 * n)).T  # shape ii, jj

    # set up and invert Q
    Q = np.eye(3 * n) - np.matmul(S, T)
    Q_inv = invert(matrix=Q, solver=solver)

    # determine new magnetization vectors
    mag_new = np.matmul(Q_inv, mag)
    mag_new = np.reshape(mag_new, (n, 3), order='F')
    # mag_new *= .4*np.pi

    for s, mag in zip(collection.sources_all, mag_new):
        s.magnetization = s.orientation.inv().apply(mag)  # ROTATION CHECK
    collection_input.parent=None