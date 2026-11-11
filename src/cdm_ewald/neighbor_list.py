import numpy as np
from scipy.spatial import cKDTree

def neighbor_pairs(
    positions,
    box,
    r_cut,
    r_hardcore=0.0,
):
    """
    Find neighboring pairs with 2D periodic boundary conditions.

    Parameters
    ----------
    positions : (N, 3)
        Particle positions in 3D.

    box : (2,)
        Periodic box lengths [Lx, Ly].
        Periodic in x and y, nonperiodic in z.

    r_cut : float
        Neighbor cutoff (3D Euclidean distance).

    r_hardcore : float
        Minimum allowed separation.

    Returns
    -------
    pairs : (M, 2)
        Valid neighbor pairs.

    overlaps : (K, 2)
        Pairs violating the hardcore distance.
    """

    positions = np.asarray(positions, dtype=float).copy()
    box = np.asarray(box, dtype=float)

    if np.any(r_cut >= box / 2):
        raise ValueError(
            "The cutoff must be smaller than half of the box"
        )

    # Wrap x and y only; leave z unchanged
    positions[:, :2] %= box

    # Infinite box length means no periodicity in z
    boxsize = np.array([box[0], box[1], np.inf])

    tree = cKDTree(positions, boxsize=boxsize)

    pairs = np.array(
        list(tree.query_pairs(r_cut)),
        dtype=int
    ).reshape(-1, 2)

    if len(pairs) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0, 2), dtype=int),
        )

    P1 = pairs[:, 0]
    P2 = pairs[:, 1]

    dr = positions[P1] - positions[P2]

    # Minimum image convention in x and y only
    dr[:, :2] -= box * np.round(dr[:, :2] / box)

    # Full 3D distance
    r = np.linalg.norm(dr, axis=1)

    overlap_mask = r < r_hardcore

    overlaps = pairs[overlap_mask]
    pairs = pairs[~overlap_mask]

    return pairs, overlaps