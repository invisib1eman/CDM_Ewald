import numpy as np
from scipy.special import erfc, erfcx, erf, erfi
from cdm_ewald.lattice_utils import check_lattice_basis, reciprocal_lattice_basis, lattice_area, nearest_lattice_length, sorted_lattice_vectors

SQRT_PI = np.sqrt(np.pi)

LIGHTLINE_TOL = 1e-12
DECAYING_PARA = 1e-12

def _stable_evanescent_terms(gamma, z_distance, xi):
    """Return the two bounded terms in the evanescent Ewald kernel."""
    scaled_gamma = gamma / (2 * xi)
    scaled_distance = xi * z_distance

    x1 = scaled_gamma - scaled_distance
    x2 = scaled_gamma + scaled_distance

    term1 = np.exp(-gamma * z_distance) * erfc(x1)
    term2 = np.exp(
        -scaled_gamma**2 - scaled_distance**2
    ) * erfcx(x2)
    return term1, term2


def f_ewald(Q: float, k: float, z: float, xi: float) -> complex:
    """Evaluate the scalar reciprocal-space Ewald kernel."""
    z_distance = abs(z)
    delta = k**2 - Q**2

    if abs(delta) < LIGHTLINE_TOL:
        # Limiting absorption prescription
        kc = complex(k, DECAYING_PARA)

        # Complex vertical wavevector
        beta = np.sqrt(kc**2 - Q**2 + 0j)

        # Enforce outgoing / decaying branch:
        # Im(beta) >= 0
        if beta.imag < 0:
            beta = -beta

        a = beta / (2 * xi)
        b = xi * z_distance
        return (
            1j
            / (4 * beta)
            * (
                np.exp(1j * beta * z_distance)
                * erfc(-1j * a - b)
                +
                np.exp(-1j * beta * z_distance)
                * erfc(-1j * a + b)
            )
        )
        

    if delta > 0:
        beta = np.sqrt(delta)
        return (
            1j
            * np.exp(-1j * beta * z_distance)
            * (
                np.exp(2j * beta * z_distance)
                * erfc(-1j * beta / (2 * xi) - xi * z_distance)
                + erfc(
                    -1j * beta / (2 * xi) + xi * z_distance
                )
            )
            / (4 * beta)
        )
    else:
        gamma = np.sqrt(-delta)
        term1, term2 = _stable_evanescent_terms(
            gamma, z_distance, xi
        )
        return (term1 + term2) / (4 * gamma)
# def f_ewald(
#     Q: float,
#     k: float,
#     z: float,
#     xi: float,
#     eta: float = 1e-12,
# ) -> complex:
#     """
#     Scalar reciprocal-space Ewald kernel with limiting absorption.

#     Parameters
#     ----------
#     Q : float
#         Magnitude of in-plane reciprocal wavevector.
#     k : float
#         Real background-medium wavenumber.
#     z : float
#         Out-of-plane separation.
#     xi : float
#         Ewald splitting parameter.
#     eta : float
#         Small positive damping used as k -> k + i eta.

#     Returns
#     -------
#     complex
#         Reciprocal-space scalar Ewald kernel.
#     """

#     z_distance = abs(z)

#     # Limiting absorption prescription
#     kc = complex(k, eta)

#     # Complex vertical wavevector
#     beta = np.sqrt(kc**2 - Q**2 + 0j)

#     # Enforce outgoing / decaying branch:
#     # Im(beta) >= 0
#     if beta.imag < 0:
#         beta = -beta

#     a = beta / (2 * xi)
#     b = xi * z_distance
#     if abs(b)>
#     return (
#         1j
#         / (4 * beta)
#         * (
#             np.exp(1j * beta * z_distance)
#             * erfc(-1j * a - b)
#             +
#             np.exp(-1j * beta * z_distance)
#             * erfc(-1j * a + b)
#         )
#     )

def d_fEwaldZ_dz(
    Q: float, k: float, z: float, xi: float
) -> complex:
    """Evaluate the first z derivative of the Ewald kernel."""
    z_distance = abs(z)
    z_sign = np.sign(z)
    delta = k**2 - Q**2

    if abs(delta) < LIGHTLINE_TOL:
        return -0.5 * z_sign * erf(xi * z_distance)

    if delta > 0:
        beta = np.sqrt(delta)
        derivative_at_distance = (
            0.25
            * np.exp(-1j * beta * z_distance)
            * (
                -np.exp(2j * beta * z_distance)
                * erfc(
                    -1j * beta / (2 * xi) - xi * z_distance
                )
                + erfc(
                    -1j * beta / (2 * xi) + xi * z_distance
                )
            )
        )
        return z_sign * derivative_at_distance
    else:
        gamma = np.sqrt(-delta)
        term1, term2 = _stable_evanescent_terms(
            gamma, z_distance, xi
        )
        return z_sign * 0.25 * (-term1 + term2)


def d2_fEwaldZ_dz2(
    Q: float, k: float, z: float, xi: float
) -> complex:
    """Evaluate the second z derivative of the Ewald kernel."""
    z_distance = abs(z)
    delta = k**2 - Q**2

    gaussian = (
        xi
        * np.exp(
            (delta - 4 * xi**4 * z_distance**2)
            / (4 * xi**2)
        )
        / SQRT_PI
    )

    if abs(delta) < LIGHTLINE_TOL:
        return (
            -xi
            * np.exp(-xi**2 * z_distance**2)
            / SQRT_PI
        )

    if delta > 0:
        beta = np.sqrt(delta)
        return (
            -gaussian
            - 0.25j
            * np.exp(1j * beta * z_distance)
            * beta
            * erfc(
                -1j * beta / (2 * xi)
                - xi * z_distance
            )
            - 0.25j
            * np.exp(-1j * beta * z_distance)
            * beta
            * erfc(
                -1j * beta / (2 * xi)
                + xi * z_distance
            )
        )
    else:
        gamma = np.sqrt(-delta)
        term1, term2 = _stable_evanescent_terms(
            gamma, z_distance, xi
        )
        return 0.25 * (
            -4 * gaussian + gamma * (term1 + term2)
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


def f_ewald_vec(
    Q: np.ndarray,
    k: float,
    z: float,
    xi: float,
) -> np.ndarray:
    """Evaluate f_ewald elementwise, preserving the input array shape."""
    Q = np.asarray(Q)
    z_distance = abs(z)
    delta = k**2 - Q**2
    out = np.zeros_like(Q, dtype=complex)

    # Match the scalar function's branch conditions exactly.
    light = np.abs(delta) < LIGHTLINE_TOL
    prop = (~light) & (delta > 0)
    evan = ~(light | prop)

    # Grazing modes: same limiting-absorption prescription as f_ewald.
    kc = complex(k, DECAYING_PARA)
    beta = np.sqrt(kc**2 - Q[light]**2 + 0j)
    beta = np.where(beta.imag < 0, -beta, beta)

    a = beta / (2 * xi)
    b = xi * z_distance

    out[light] = (
        1j
        / (4 * beta)
        * (
            np.exp(1j * beta * z_distance)
            * erfc(-1j * a - b)
            + np.exp(-1j * beta * z_distance)
            * erfc(-1j * a + b)
        )
    )

    # Propagating modes.
    beta = np.sqrt(delta[prop])
    out[prop] = (
        1j
        * np.exp(-1j * beta * z_distance)
        * (
            np.exp(2j * beta * z_distance)
            * erfc(-1j * beta / (2 * xi) - xi * z_distance)
            + erfc(-1j * beta / (2 * xi) + xi * z_distance)
        )
        / (4 * beta)
    )

    # Evanescent modes: retain the stable evaluation.
    gamma = np.sqrt(-delta[evan])
    term1, term2 = _stable_evanescent_terms(
        gamma, z_distance, xi
    )
    out[evan] = (term1 + term2) / (4 * gamma)

    return out


def d_fEwaldZ_dz_vec(Q, k, z, xi):
    """Vectorized first z derivative for an array of Q values."""
    Q = np.asarray(Q)
    z_distance = abs(z)
    z_sign = np.sign(z)
    delta = k**2 - Q**2
    out = np.zeros_like(Q, dtype=complex)
    light = np.abs(delta) < LIGHTLINE_TOL
    prop = delta > LIGHTLINE_TOL
    evan = delta < -LIGHTLINE_TOL

    out[light] = (
        -0.5 * z_sign * erf(xi * z_distance)
    )

    beta = np.sqrt(delta[prop])
    out[prop] = (
        z_sign
        * 0.25
        * np.exp(-1j * beta * z_distance)
        * (
            -np.exp(2j * beta * z_distance)
            * erfc(
                -1j * beta / (2 * xi) - xi * z_distance
            )
            + erfc(
                -1j * beta / (2 * xi) + xi * z_distance
            )
        )
    )

    gamma = np.sqrt(-delta[evan])
    term1, term2 = _stable_evanescent_terms(
        gamma, z_distance, xi
    )
    out[evan] = z_sign * 0.25 * (-term1 + term2)
    return out


def d2_fEwaldZ_dz2_vec(Q, k, z, xi):
    """Vectorized second z derivative for an array of Q values."""
    Q = np.asarray(Q)
    z_distance = abs(z)
    delta = k**2 - Q**2
    out = np.zeros_like(Q, dtype=complex)
    light = np.abs(delta) < LIGHTLINE_TOL
    prop = delta > LIGHTLINE_TOL
    evan = delta < -LIGHTLINE_TOL

    gaussian = (
        xi
        * np.exp(
            (delta - 4 * xi**4 * z_distance**2)
            / (4 * xi**2)
        )
        / SQRT_PI
    )

    out[light] = (
        -xi
        * np.exp(-xi**2 * z_distance**2)
        / SQRT_PI
    )

    beta = np.sqrt(delta[prop])
    out[prop] = (
        -gaussian[prop]
        - 0.25j
        * np.exp(1j * beta * z_distance)
        * beta
        * erfc(
            -1j * beta / (2 * xi) - xi * z_distance
        )
        - 0.25j
        * np.exp(-1j * beta * z_distance)
        * beta
        * erfc(
            -1j * beta / (2 * xi) + xi * z_distance
        )
    )

    gamma = np.sqrt(-delta[evan])
    term1, term2 = _stable_evanescent_terms(
        gamma, z_distance, xi
    )
    out[evan] = 0.25 * (
        -4 * gaussian[evan]
        + gamma * (term1 + term2)
    )
    return out


def tensor_kernel_vec(Qx, Qy, dz, k, xi, mu):

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
        k_parallel,
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
            phase_shift = np.exp(1j*np.dot(shift,k_parallel))
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

            G_real += Gtmp*phase_shift

        # --------------------------------------
        # RECIPROCAL SPACE
        # --------------------------------------

        Gk = np.zeros((3,3), dtype=complex)

        dr = ri - rj

        rho = dr[:2]
        z = dr[2]
        reciprocal_basis = reciprocal_lattice_basis(lattice_vectors)
        reciprocal_vectors = sorted_lattice_vectors(reciprocal_basis, recip_cut)
        grazing = False
        for reciprocal_vector in reciprocal_vectors:

            Qx = reciprocal_vector[0] + k_parallel[0]
            Qy = reciprocal_vector[1] + k_parallel[1]
            Q = np.sqrt(Qx**2 + Qy**2)

            if np.isclose(Q, k):
                print(f"Grazing")
                # grazing = True
                # break

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

            # if not np.all(np.isfinite(Gtmp)):
            #     print("Bad reciprocal tensor!")
            #     print(f"Qx={Qx}, Qy={Qy}")
            #     print(f"Q={Q}")
            #     print(f"z={z}")
            #     print(f"k={k}")
            #     print(f"xi={xi}")
            #     print(Gtmp)

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
        wavevector_direction,
        xi,
        lattice_vectors,
        real_cut,
        reciprocal_cut
    ):
        
        k_parallel = k*np.array([wavevector_direction[0],wavevector_direction[1],0])
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
                    k_parallel,
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
