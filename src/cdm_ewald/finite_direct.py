"""Finite free-space coupled dipoles; no periodic images or Ewald sums.

Lengths are in sphere radii and alpha includes 4*pi. The outgoing Green
function uses exp(ikr), corresponding to the exp(-i*omega*t) convention.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import spherical_jn, spherical_yn


@dataclass(frozen=True)
class DipoleSolution:
    """Complex dipoles and independently checked linear-solve diagnostics."""

    dipoles: NDArray[np.complex128]
    relative_residual: float
    iterations: int


def twisted_bilayer(side: int, spacing: float, separation: float) -> NDArray:
    """Build two centered side-by-side square index grids at a 36.87 deg twist.

    Each layer contains side**2 particles. Primitive vectors match the original
    silver_twisted_bilayer notebook; centering aligns the two patch centroids.
    """
    if not isinstance(side, (int, np.integer)) or side < 1:
        raise ValueError('side must be a positive integer')
    if not np.isfinite([spacing, separation]).all() or min(spacing, separation) <= 0:
        raise ValueError('spacing and separation must be finite and positive')
    indices = np.indices((side, side)).reshape(2, -1).T - (side - 1) / 2
    bottom = spacing / np.sqrt(5) * np.array([[1, 2], [-2, 1]])
    top = spacing / np.sqrt(5) * np.array([[2, 1], [-1, 2]])
    return np.vstack([
        np.column_stack((indices @ bottom.T, np.zeros(side**2))),
        np.column_stack((indices @ top.T, np.full(side**2, separation))),
    ])


def mie_polarizability(wavenumber: NDArray, permittivity: NDArray) -> NDArray:
    """Return electric-dipole Mie alpha for radius-one spheres in vacuum."""
    x = np.asarray(wavenumber, dtype=float)
    eps = np.asarray(permittivity, dtype=complex)
    if np.any(x <= 0) or not np.isfinite(x).all() or not np.isfinite(eps).all():
        raise ValueError('Wavenumbers must be positive and material inputs finite')
    m = np.sqrt(eps)

    def riccati(z: NDArray) -> tuple[NDArray, NDArray]:
        return z * spherical_jn(1, z), spherical_jn(1, z) + z * spherical_jn(1, z, True)

    psi_x, prime_x = riccati(x)
    psi_m, prime_m = riccati(m * x)
    xi_x = psi_x + 1j * x * spherical_yn(1, x)
    xi_prime = prime_x + 1j * (spherical_yn(1, x) + x * spherical_yn(1, x, True))
    a1 = (m * psi_m * prime_x - psi_x * prime_m) / (m * psi_m * xi_prime - xi_x * prime_m)
    return 6j * np.pi * a1 / x**3


class DirectInteraction:
    """Apply every off-diagonal free-space interaction in bounded-size blocks.

    Work is O(N**2) per application, memory O(block_size*N); there is no
    distance cutoff, image replication, mesh, or lattice summation.
    """

    def __init__(self, positions: NDArray, wavenumber: float, block_size: int = 64):
        self.positions = np.array(positions, dtype=float, copy=True)
        self.k = float(wavenumber)
        self.block_size = block_size
        if self.positions.ndim != 2 or self.positions.shape[1] != 3 or len(self.positions) == 0:
            raise ValueError('positions must have nonempty shape (N, 3)')
        if not np.isfinite(self.positions).all() or not np.isfinite(self.k) or self.k <= 0:
            raise ValueError('positions must be finite and wavenumber positive')
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        if len(np.unique(self.positions, axis=0)) != len(self.positions):
            raise ValueError('Distinct dipoles cannot occupy the same position')

    def apply(self, dipoles: NDArray) -> NDArray:
        """Return G p, omitting self terms already represented in Mie alpha."""
        p = np.asarray(dipoles, dtype=complex).reshape(self.positions.shape)
        field = np.zeros_like(p)
        for start in range(0, len(p), self.block_size):
            stop = min(start + self.block_size, len(p))
            displacement = self.positions[start:stop, None] - self.positions[None, :]
            distance = np.linalg.norm(displacement, axis=-1)
            rows = np.arange(stop - start)
            columns = np.arange(start, stop)
            distance[rows, columns] = 1.0
            direction = displacement / distance[..., None]
            kr = self.k * distance
            prefactor = np.exp(1j * kr) / (4 * np.pi * distance**3)
            transverse = prefactor * (kr**2 + 1j * kr - 1)
            longitudinal = prefactor * (3 - 3j * kr - kr**2)
            transverse[rows, columns] = 0
            longitudinal[rows, columns] = 0
            projection = np.einsum('ijc,jc->ij', direction, p)
            field[start:stop] = np.einsum('ij,jc->ic', transverse, p)
            field[start:stop] += np.einsum('ij,ij,ijc->ic', longitudinal, projection, direction)
        return field


def solve_direct(
    positions: NDArray, wavenumber: float, polarizability: complex,
    incident_field: NDArray, *, block_size: int = 64, rtol: float = 1e-6,
    restart: int = 60, maxiter: int = 100, initial_dipoles: NDArray | None = None,
) -> DipoleSolution:
    """Solve (I - alpha G)p = alpha E using GMRES, raising on failure.

    maxiter counts restart cycles. The residual is checked against the actual
    equation after iteration; unconverged dipoles are never returned as results.
    """
    interaction = DirectInteraction(positions, wavenumber, block_size)
    incident = np.asarray(incident_field, dtype=complex)
    if incident.shape != interaction.positions.shape or not np.isfinite(incident).all():
        raise ValueError('incident_field must be finite with shape (N, 3)')
    if not np.isfinite(polarizability) or polarizability == 0 or not 0 < rtol < 1:
        raise ValueError('alpha must be finite and nonzero; rtol must lie in (0, 1)')
    rhs = (polarizability * incident).ravel()
    if np.linalg.norm(rhs) == 0:
        raise ValueError('incident_field must be nonzero')
    operator = LinearOperator((rhs.size, rhs.size), dtype=complex,
                              matvec=lambda p: p - polarizability * interaction.apply(p).ravel())
    history = []
    guess = None if initial_dipoles is None else np.asarray(initial_dipoles).ravel()
    result, info = gmres(operator, rhs, x0=guess, rtol=rtol, atol=0,
                         restart=restart, maxiter=maxiter,
                         callback=history.append, callback_type='pr_norm')
    residual = float(np.linalg.norm(operator @ result - rhs) / np.linalg.norm(rhs))
    if info != 0 or not np.isfinite(residual) or residual > rtol:
        raise RuntimeError(f'Direct CDM failed: info={info}, relative residual={residual:.3e}')
    return DipoleSolution(result.reshape(incident.shape), residual, len(history))


def extinction_cross_section(wavenumber: float, incident_field: NDArray,
                             dipoles: NDArray, amplitude: float = 1.0) -> float:
    """Optical-theorem extinction in radius squared (alpha includes 4*pi)."""
    if amplitude <= 0 or not np.isfinite(amplitude):
        raise ValueError('amplitude must be finite and positive')
    return float(wavenumber * np.vdot(incident_field, dipoles).imag / amplitude**2)
