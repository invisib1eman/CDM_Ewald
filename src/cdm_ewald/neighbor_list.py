import numpy as np
from scipy.spatial import cKDTree

def neighbor_pairs(
    positions,
    box,
    r_cut,
    r_hardcore=0.0,
):
    """
    Parameters
    ----------
    positions : (N,2)
    box : (2,)
    r_cut : float
        Neighbor cutoff.
    r_hardcore : float
        Minimum allowed separation.

    Returns
    -------
    pairs : (M,2)
        Valid neighbor pairs.

    overlaps : (K,2)
        Pairs violating the hardcore distance.
    """
    if r_cut < box / 2:
        raise ValueError("The cutoff must be smaller than half of the box")

    positions = np.asarray(positions) % box

    tree = cKDTree(positions, boxsize=box)

    pairs = np.array(list(tree.query_pairs(r_cut)), dtype=int)

    if len(pairs) == 0:
        return (
            np.empty((0,2), dtype=int),
            np.empty((0,2), dtype=int),
        )

    P1 = pairs[:,0]
    P2 = pairs[:,1]

    dr = positions[P1] - positions[P2]
    dr -= box * np.round(dr / box)

    r = np.linalg.norm(dr, axis=1)

    overlap_mask = r < r_hardcore

    overlaps = pairs[overlap_mask]
    pairs = pairs[~overlap_mask]

    return pairs, overlaps