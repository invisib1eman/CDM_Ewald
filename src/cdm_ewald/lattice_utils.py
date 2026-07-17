import numpy as np


def check_lattice_basis(lattice_vectors):
    """
    Validate a 2D lattice basis.

    Accepts either

        (2, 2):
            [[a1x, a1y],
             [a2x, a2y]]

    or

        (2, 3):
            [[a1x, a1y, 0],
             [a2x, a2y, 0]]

    Returns
    -------
    direct_vectors : ndarray, shape (2, 2)
        The two lattice vectors in the xy plane.
    """
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)

    if lattice_vectors.shape == (2, 2):
        direct_vectors = lattice_vectors.copy()

    elif lattice_vectors.shape == (2, 3):
        if not np.allclose(lattice_vectors[:, 2], 0.0):
            raise ValueError(
                "The z-components of the lattice vectors must be zero."
            )

        direct_vectors = lattice_vectors[:, :2].copy()

    else:
        raise ValueError(
            "lattice_vectors must have shape (2, 2) or (2, 3)."
        )

    if np.linalg.matrix_rank(direct_vectors) < 2:
        raise ValueError(
            "The lattice vectors must be linearly independent."
        )

    return direct_vectors


def reciprocal_lattice_basis(lattice_vectors):
    """
    Return the reciprocal lattice basis.

    If the rows of A are the direct lattice vectors, the rows of B satisfy

        A @ B.T = 2*pi*I.

    Returns
    -------
    reciprocal_vectors : ndarray, shape (2, 2)
    """
    direct_vectors = check_lattice_basis(lattice_vectors)

    reciprocal_vectors = (
        2.0 * np.pi * np.linalg.inv(direct_vectors).T
    )

    return reciprocal_vectors


def lattice_area(lattice_vectors):
    """
    Return the area of the 2D lattice unit cell.
    """
    direct_vectors = check_lattice_basis(lattice_vectors)

    return abs(np.linalg.det(direct_vectors))


def nearest_lattice_length(lattice_vectors):
    """
    Return the length of the shorter basis vector.

    Note
    ----
    This is the shortest supplied basis-vector length, not necessarily the
    shortest nonzero lattice-vector length for an arbitrary non-reduced basis.
    """
    direct_vectors = check_lattice_basis(lattice_vectors)

    return np.min(
        np.linalg.norm(direct_vectors, axis=1)
    )


def sorted_lattice_vectors(lattice_vectors, cutoff_radius):
    """
    Generate all 2D lattice vectors within a cutoff radius.

    Parameters
    ----------
    lattice_vectors : array_like, shape (2, 2) or (2, 3)
        Direct lattice basis.
    cutoff_radius : float
        Maximum lattice-vector length.

    Returns
    -------
    vectors : ndarray, shape (num_vectors, 2)
        Lattice vectors sorted by increasing magnitude. The zero vector is
        included.
    """
    direct_vectors = check_lattice_basis(lattice_vectors)

    cutoff_radius = float(cutoff_radius)

    if cutoff_radius < 0:
        raise ValueError("cutoff_radius must be nonnegative.")

    # For integer coefficient vector q = [m, n],
    #
    #     ||q @ A|| >= sigma_min(A) ||q||,
    #
    # so coefficients outside this range cannot produce vectors inside
    # the cutoff.
    min_stretch = np.linalg.svd(
        direct_vectors,
        compute_uv=False,
    ).min()

    max_index = int(
        np.ceil(cutoff_radius / min_stretch)
    )

    vectors = []

    for m in range(-max_index, max_index + 1):
        for n in range(-max_index, max_index + 1):
            vector = (
                m * direct_vectors[0]
                + n * direct_vectors[1]
            )

            vector_norm = np.linalg.norm(vector)

            if vector_norm <= cutoff_radius + 1e-12:
                vectors.append((vector_norm, m, n, vector))

    # Including m and n makes ordering deterministic when multiple vectors
    # have the same length.
    vectors.sort(
        key=lambda item: (item[0], item[1], item[2])
    )

    return np.asarray(
        [vector for _, _, _, vector in vectors],
        dtype=float,
    )