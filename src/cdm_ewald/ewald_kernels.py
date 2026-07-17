import numpy as np
from scipy.special import erfc, erfcx, erf, erfi
from cdm_ewald.lattice_utils import check_lattice_basis, reciprocal_lattice_basis, lattice_area, nearest_lattice_length, sorted_lattice_vectors

SQRT_PI = np.sqrt(np.pi)

LIGHTLINE_TOL = 1e-12

def f_ewald(Q: float,
            k: float,
            z: float,
            xi: float
) -> complex:
    # define the wavenumber difference
    delta = k**2 - Q**2

    # light-line limit
    if abs(delta) < LIGHTLINE_TOL:
        return (
            np.exp(-(xi * z) ** 2)
            / (2 * xi * SQRT_PI)
            - 0.5 * z * erfc(xi * z)
        )
    if delta > 0:

        beta = np.sqrt(delta)

        return (
            1j
            * np.exp(-1j * beta * z)
            * (
                np.exp(2j * beta * z)
                * erfc(-1j * beta / (2 * xi) - xi * z)
                + erfc(-1j * beta / (2 * xi) + xi * z)
            )
            / (4 * beta)
        )

    else:

        gamma = np.sqrt(-delta)

        x1 = gamma / (2 * xi) - xi * z
        x2 = gamma / (2 * xi) + xi * z

        term1 = np.exp(-gamma * z) * erfc(x1)

        # stable version of exp(gamma*z)*erfc(x2)
        term2 = np.exp(gamma * z - x2 * x2) * erfcx(x2)

        return (term1 + term2) / (4 * gamma)


def d_fEwaldZ_dz(Q: float,
            k: float,
            z: float,
            xi: float
            ) -> complex:
    delta = k**2 - Q**2

    # light-line limit
    if abs(delta) < LIGHTLINE_TOL:
        return -0.5 * erf(xi * z)

    if delta > 0:

        beta = np.sqrt(delta)

        return (
            0.25
            * np.exp(-1j * beta * z)
            * (
                -np.exp(2j * beta * z)
                * erfc(-1j * beta / (2 * xi) - xi * z)
                + erfc(-1j * beta / (2 * xi) + xi * z)
            )
        )

    else:

        gamma = np.sqrt(-delta)

        x1 = gamma / (2 * xi) - xi * z
        x2 = gamma / (2 * xi) + xi * z

        term1 = -np.exp(-gamma * z) * erfc(x1)

        term2 = np.exp(gamma * z - x2 * x2) * erfcx(x2)

        return 0.25 * (term1 + term2)


def d2_fEwaldZ_dz2(Q: float,
            k: float,
            z: float,
            xi: float
            ) -> complex:
    delta = k**2 - Q**2

    gaussian = (
        xi
        * np.exp(
            (delta - 4 * xi**4 * z**2)
            / (4 * xi**2)
        )
        / SQRT_PI
    )

    # light-line limit
    if abs(delta) < LIGHTLINE_TOL:
        return (
            -xi
            * np.exp(-xi**2 * z**2)
            / SQRT_PI
        )

    if delta > 0:

        beta = np.sqrt(delta)

        return (
            -gaussian
            - 0.25j
            * np.exp(1j * beta * z)
            * beta
            * erfc(
                -1j * beta / (2 * xi)
                - xi * z
            )
            - 0.25j
            * np.exp(-1j * beta * z)
            * beta
            * erfc(
                -1j * beta / (2 * xi)
                + xi * z
            )
        )

    else:

        gamma = np.sqrt(-delta)

        x1 = gamma / (2 * xi) - xi * z
        x2 = gamma / (2 * xi) + xi * z

        term1 = (
            np.exp(-gamma * z)
            * gamma
            * erfc(x1)
        )

        term2 = (
            gamma
            * np.exp(gamma * z - x2 * x2)
            * erfcx(x2)
        )

        return 0.25 * (
            -4 * gaussian
            + term1
            + term2
        )
def reciprocal_tensor_kernel(Qx, Qy, dz, k, xi):
    """
    Full reciprocal-space dyadic Green tensor:

        G = (Del Del - Laplacian I) g(Q,z)

    Tensor:

        [[ Qy^2 g - gzz,   -QxQy g,      iQx gz ],
         [ -QxQy g,        Qx^2 g-gzz,   iQy gz ],
         [ iQx gz,         iQy gz,       Q^2 g  ]]
    """

    Q2 = Qx**2 + Qy**2
    Q = np.sqrt(Q2)

    g = f_ewald(Q, k, dz, xi)

    gz = d_fEwaldZ_dz(Q, k, dz, xi)

    gzz = d2_fEwaldZ_dz2(Q, k, dz, xi)

    G = np.zeros((3, 3), dtype=complex)

    # --------------------------------------------------------
    # diagonal terms
    # --------------------------------------------------------

    G[0, 0] = Qy**2 * g - gzz

    G[1, 1] = Qx**2 * g - gzz

    G[2, 2] = Q2 * g

    # --------------------------------------------------------
    # off-diagonal xy
    # --------------------------------------------------------

    G[0, 1] = -Qx * Qy * g
    G[1, 0] = -Qx * Qy * g

    # --------------------------------------------------------
    # xz yz couplings
    # --------------------------------------------------------

    G[0, 2] = 1j * Qx * gz
    G[2, 0] = 1j * Qx * gz

    G[1, 2] = 1j * Qy * gz
    G[2, 1] = 1j * Qy * gz

    return G

def real_space_tensor(rvec, k, xi):
    """
    Real-space short-range Ewald dipole tensor.

    Expression provided by user.

    G_real(r) =
        A(r) (I - rhat rhat)
        +
        B(r) (rhat rhat)

    epsilon_f = 1

    Parameters
    ----------
    rvec : ndarray shape (3,)
        displacement vector

    k : float
        wavenumber

    xi : float
        Ewald splitting parameter

    Returns
    -------
    G : ndarray shape (3,3)
        real-space dyadic tensor
    """

    r = np.linalg.norm(rvec)

    if r < 1e-14:
        return np.zeros((3, 3), dtype=complex)

    rhat = rvec / r

    rr = np.outer(rhat, rhat)

    I = np.eye(3)

    # --------------------------------------------------------
    # transverse coefficient
    # --------------------------------------------------------

    termT1 = (
        -np.exp(
            k**2 / (4 * xi**2)
            - r**2 * xi**2
        )
        * xi
        * (1 + 2 * r**2 * xi**2)
        / (2 * np.pi**1.5 * r**2)
    )

    termT2 = (
        np.exp(-1j * k * r)
        * (-1 - 1j * k * r + k**2 * r**2)
        * erfc(
            -(1j * k) / (2 * xi)
            + r * xi
        )
        / (8 * np.pi * r**3)
    )

    termT3 = (
        np.exp(1j * k * r)
        * (-1 + 1j * k * r + k**2 * r**2)
        * erfc(
            (1j * k) / (2 * xi)
            + r * xi
        )
        / (8 * np.pi * r**3)
    )

    A = termT1 + termT2 + termT3

    # --------------------------------------------------------
    # longitudinal coefficient
    # --------------------------------------------------------

    termL1 = (
        np.exp(
            k**2 / (4 * xi**2)
            - r**2 * xi**2
        )
        * xi
        / (np.pi**1.5 * r**2)
    )

    termL2 = (
        np.exp(-1j * k * r)
        * (1 + 1j * k * r)
        * erfc(
            -(1j * k) / (2 * xi)
            + r * xi
        )
        / (4 * np.pi * r**3)
    )

    termL3 = (
        np.exp(1j * k * r)
        * (1 - 1j * k * r)
        * erfc(
            (1j * k) / (2 * xi)
            + r * xi
        )
        / (4 * np.pi * r**3)
    )

    B = termL1 + termL2 + termL3

    # --------------------------------------------------------
    # tensor assembly
    # --------------------------------------------------------

    G = (
        A * (I - rr)
        +
        B * rr
    )

    return G

def self_term_scalar(k: float,
                     xi:float 
)-> complex:
    """
    Self-term coefficient.

    Parameters
    ----------
    k : float
        Wavenumber

    xi : float
        Ewald splitting parameter

    Returns
    -------
    complex
        Scalar self-term coefficient
    """

    numerator = (
        2
        * np.exp(k**2 / (4 * xi**2))
        * xi
        * (k**2 + 2 * xi**2)
        +
        1j
        * k**3
        * np.sqrt(np.pi)
        * (
            1
            + 1j * erfi(k / (2 * xi))
        )
    )

    denominator = (
        6
        * np.pi**1.5
    )

    return numerator / denominator


def f_ewald_vec(Q, k, z, xi):

    Q = np.asarray(Q)

    delta = k**2 - Q**2

    out = np.zeros_like(Q, dtype=complex)

    light = np.abs(delta) < LIGHTLINE_TOL

    prop = delta > LIGHTLINE_TOL

    evan = delta < -LIGHTLINE_TOL

    # light-line limit

    out[light] = (

        np.exp(-(xi * z) ** 2)

        / (2 * xi * SQRT_PI)

        - 0.5 * z * erfc(xi * z)

    )

    # propagating modes: delta > 0

    beta = np.sqrt(delta[prop])

    out[prop] = (

        1j

        * np.exp(-1j * beta * z)

        * (

            np.exp(2j * beta * z)

            * erfc(-1j * beta / (2 * xi) - xi * z)

            + erfc(-1j * beta / (2 * xi) + xi * z)

        )

        / (4 * beta)

    )

    # evanescent modes: delta < 0

    gamma = np.sqrt(-delta[evan])

    x1 = gamma / (2 * xi) - xi * z

    x2 = gamma / (2 * xi) + xi * z

    term1 = np.exp(-gamma * z) * erfc(x1)

    term2 = np.exp(gamma * z - x2 * x2) * erfcx(x2)

    out[evan] = (term1 + term2) / (4 * gamma)

    return out

def d_fEwaldZ_dz_vec(Q, k, z, xi):
    Q = np.asarray(Q)
    delta = k**2 - Q**2

    out = np.zeros_like(Q, dtype=complex)

    light = np.abs(delta) < LIGHTLINE_TOL
    prop = delta > LIGHTLINE_TOL
    evan = delta < -LIGHTLINE_TOL

    # light-line limit
    out[light] = -0.5 * erf(xi * z)

    # propagating modes
    beta = np.sqrt(delta[prop])

    out[prop] = (
        0.25
        * np.exp(-1j * beta * z)
        * (
            -np.exp(2j * beta * z)
            * erfc(-1j * beta / (2 * xi) - xi * z)
            + erfc(-1j * beta / (2 * xi) + xi * z)
        )
    )

    # evanescent modes
    gamma = np.sqrt(-delta[evan])

    x1 = gamma / (2 * xi) - xi * z
    x2 = gamma / (2 * xi) + xi * z

    term1 = -np.exp(-gamma * z) * erfc(x1)
    term2 = np.exp(gamma * z - x2 * x2) * erfcx(x2)

    out[evan] = 0.25 * (term1 + term2)

    return out
def d2_fEwaldZ_dz2_vec(Q, k, z, xi):
    Q = np.asarray(Q)
    delta = k**2 - Q**2

    out = np.zeros_like(Q, dtype=complex)

    light = np.abs(delta) < LIGHTLINE_TOL
    prop = delta > LIGHTLINE_TOL
    evan = delta < -LIGHTLINE_TOL

    gaussian = (
        xi
        * np.exp((delta - 4 * xi**4 * z**2) / (4 * xi**2))
        / SQRT_PI
    )

    # light-line limit
    out[light] = (
        -xi
        * np.exp(-xi**2 * z**2)
        / SQRT_PI
    )

    # propagating modes
    beta = np.sqrt(delta[prop])

    out[prop] = (
        -gaussian[prop]
        - 0.25j
        * np.exp(1j * beta * z)
        * beta
        * erfc(-1j * beta / (2 * xi) - xi * z)
        - 0.25j
        * np.exp(-1j * beta * z)
        * beta
        * erfc(-1j * beta / (2 * xi) + xi * z)
    )

    # evanescent modes
    gamma = np.sqrt(-delta[evan])

    x1 = gamma / (2 * xi) - xi * z
    x2 = gamma / (2 * xi) + xi * z

    term1 = np.exp(-gamma * z) * gamma * erfc(x1)
    term2 = gamma * np.exp(gamma * z - x2 * x2) * erfcx(x2)

    out[evan] = 0.25 * (
        -4 * gaussian[evan]
        + term1
        + term2
    )

    return out
def tensor_kernel_grid(Qx, Qy, dz, k, xi, mu):

    Q2 = Qx**2 + Qy**2

    Q = np.sqrt(Q2)

    g = f_ewald_vec(Q, k, dz, xi)

    gz = d_fEwaldZ_dz_vec(Q, k, dz, xi)

    gzz = d2_fEwaldZ_dz2_vec(Q, k, dz, xi)

    G = np.zeros(Q.shape + (3, 3), dtype=complex)

    G[..., 0, 0] = Qy**2 * g - gzz

    G[..., 1, 1] = Qx**2 * g - gzz

    G[..., 2, 2] = Q2 * g

    xy = -Qx * Qy * g

    G[..., 0, 1] = xy

    G[..., 1, 0] = xy

    xz = 1j * Qx * gz

    yz = 1j * Qy * gz

    G[..., 0, 2] = xz

    G[..., 2, 0] = xz

    G[..., 1, 2] = yz

    G[..., 2, 1] = yz

    exponent = mu * Q2 / (4 * xi**2)

    factor = np.exp(exponent)

    bad = (

        ~np.isfinite(factor)

        | ~np.all(np.isfinite(G), axis=(-1, -2))

    )

    G *= factor[..., None, None]

    G[bad] = 0.0

    return G


def ewald_pair_tensor(
        ri,
        rj,
        k,
        xi,
        lattice_vectors,
        real_cut,
        recip_cut,
    ):
        """
        Returns 3x3 Ewald coupling tensor
        between particle i and particle j.
        """

        G_real = np.zeros((3,3), dtype=complex)
        unit_cell_area = lattice_area(lattice_vectors)
        real_shifts = sorted_lattice_vectors(lattice_vectors, real_cut)

        # --------------------------------------
        # REAL SPACE
        # --------------------------------------

        for shift_2d in real_shifts:

            shift = np.array([
                shift_2d[0],
                shift_2d[1],
                0.0
            ])

            rvec = ri - rj + shift

            if (
                np.linalg.norm(rvec) < 1e-12
            ):
                continue

            Gtmp = real_space_tensor(
                rvec,
                k,
                xi,
            )

            if not np.all(np.isfinite(Gtmp)):
                print("Found NaN/Inf!")
                print("rvec =", rvec)
                print("r =", np.linalg.norm(rvec))
                print("k =", k)
                print("xi =", xi)
                print(Gtmp)

            G_real += Gtmp
                

        # --------------------------------------
        # RECIPROCAL SPACE
        # --------------------------------------

        Gk = np.zeros((3,3), dtype=complex)

        dr = ri - rj

        rho = dr[:2]
        z = dr[2]
        reciprocal_basis = reciprocal_lattice_basis(lattice_vectors)
        reciprocal_vectors = sorted_lattice_vectors(reciprocal_basis, recip_cut)

        for reciprocal_vector in reciprocal_vectors:

            Qx = reciprocal_vector[0]
            Qy = reciprocal_vector[1]
            Q = np.sqrt(Qx**2 + Qy**2)

            if np.isclose(Q, k):
                continue

            phase = np.exp(
                1j*(Qx*rho[0] + Qy*rho[1])
            )

            Gtmp = reciprocal_tensor_kernel(
                Qx,
                Qy,
                z,
                k,
                xi,
            )

            if not np.all(np.isfinite(Gtmp)):
                print("Bad reciprocal tensor!")
                print(f"Qx={Qx}, Qy={Qy}")
                print(f"Q={Q}")
                print(f"z={z}")
                print(f"k={k}")
                print(f"xi={xi}")
                print(Gtmp)

            Gk += Gtmp * phase

        Gk /= unit_cell_area

        # --------------------------------------
        # SELF TERM
        # --------------------------------------

        G = G_real + Gk

        if np.allclose(ri, rj):

            Gself_scalar = self_term_scalar(
                k,
                xi,
            )

            if not np.isfinite(Gself_scalar):
                print("Bad self term!")
                print(f"k = {k}")
                print(f"xi = {xi}")
                print(f"Gself = {Gself_scalar}")

            Gself = (
                Gself_scalar
                * np.eye(3)
            )

            G -= Gself
        return G
def build_ewald_matrix(
        num_particles,
        positions,
        k,
        xi,
        lattice_vectors,
        real_cut,
        reciprocal_cut
    ):
        

        M = np.zeros(
            (3*num_particles, 3*num_particles),
            dtype=complex,
        )

        for i in range(num_particles):
            for j in range(num_particles):

                Gij = ewald_pair_tensor(
                    positions[i],
                    positions[j],
                    k,
                    xi,
                    lattice_vectors,
                    real_cut,
                    reciprocal_cut
                )

                M[
                    3*i:3*i+3,
                    3*j:3*j+3
                ] = Gij

        return M