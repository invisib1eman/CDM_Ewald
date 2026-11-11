from __future__ import annotations

import ast
import logging
import time
from pathlib import Path
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cdm_ewald.ewald_kernels import reciprocal_tensor_kernel, real_space_tensor, self_term_scalar, tensor_kernel_vec
from cdm_ewald.neighbor_list import neighbor_pairs
from scipy.sparse.linalg import gmres,LinearOperator,cgs



def build_real_space_coupling_tensor(

    positions,

    pairs,

    box,

    k,

    xi,

    include_symmetric=True,

    include_self=False,

):

    """

    Build sparse real-space dipole tensor.

    Parameters

    ----------

    positions : ndarray, shape (N, 3)

        Particle positions.

    pairs : ndarray/list, shape (M, 2)

        Neighbor pairs (i, j).

    box : tuple/list/ndarray

        Periodic box size. Use (Lx, Ly) or (Lx, Ly, Lz).

        Only x/y periodic wrapping is applied here.

    k, alpha : float

        Parameters for real_space_tensor.

    include_symmetric : bool

        If True, also adds G_ji from G_ij.

    include_self : bool

        If True, include diagonal self blocks. Usually False for real-space Ewald.

    Returns

    -------

    G_sparse : scipy.sparse.csr_matrix

        Sparse tensor of shape (3*N, 3*N).

    """

    positions = np.asarray(positions, dtype=float)

    pairs = np.asarray(pairs, dtype=int)

    N = positions.shape[0]

    Lx, Ly = box[0], box[1]

    rows = []

    cols = []

    data = []

    def add_block(i, j, Gij):

        for a in range(3):

            for b in range(3):

                val = Gij[a, b]

                if val != 0:

                    rows.append(3 * i + a)

                    cols.append(3 * j + b)

                    data.append(val)

    for i, j in pairs:

        if i == j:

            if not include_self:

                continue

        rvec = positions[i] - positions[j]

        # 2D minimum image convention

        rvec[0] -= Lx * np.round(rvec[0] / Lx)

        rvec[1] -= Ly * np.round(rvec[1] / Ly)

        Gij = real_space_tensor(rvec, k, xi)

        add_block(i, j, Gij)

        if include_symmetric and i != j:

            # G_ji uses r_j - r_i = -rvec

            # For this tensor, real_space_tensor(-rvec) is usually equal to Gij,

            # because it depends on rhat rhat.

            add_block(j, i, Gij)

    G_sparse = coo_matrix(

        (data, (rows, cols)),

        shape=(3 * N, 3 * N),

        dtype=complex,

    ).tocsr()

    return G_sparse
def build_spread_matrix(

    positions,

    box,

    n_grids,

    xi,

    mu,

    error_tol=1e-3,

):

    """

    Build sparse interpolation matrix S.

    sigma_flat = S @ dipoles_flat

    where

        dipoles_flat.shape = (3*N,)

        sigma_flat.shape   = (3*Nx*Ny,)

    """

    Lx, Ly = box[:2]

    if np.isscalar(n_grids):

        Nx = Ny = int(n_grids)

    else:

        Nx, Ny = n_grids

    N = len(positions)

    Ngrid = Nx * Ny

    dx = Lx / Nx

    dy = Ly / Ny

    x = np.arange(Nx) * dx

    y = np.arange(Ny) * dy

    beta = 2 * xi**2 / mu

    prefactor = beta / np.pi

    r_cut = np.sqrt(-np.log(error_tol) / beta)

    nx_cut = int(np.ceil(r_cut / dx))

    ny_cut = int(np.ceil(r_cut / dy))

    dix = np.arange(-nx_cut, nx_cut + 1)

    diy = np.arange(-ny_cut, ny_cut + 1)

    DX, DY = np.meshgrid(dix, diy, indexing="ij")

    rows = []

    cols = []

    data = []

    for pid, pos in enumerate(positions):

        ix0 = int(np.floor(pos[0] / dx))

        iy0 = int(np.floor(pos[1] / dy))

        ix = (ix0 + DX) % Nx

        iy = (iy0 + DY) % Ny

        dxp = x[ix] - pos[0]

        dxp -= Lx * np.round(dxp / Lx)

        dyp = y[iy] - pos[1]

        dyp -= Ly * np.round(dyp / Ly)

        r2 = dxp**2 + dyp**2

        mask = r2 <= r_cut**2

        weights = prefactor * np.exp(-beta * r2[mask])

        grid_index = ix[mask] * Ny + iy[mask]

        for g, w in zip(grid_index, weights):

            for c in range(3):

                rows.append(3 * g + c)

                cols.append(3 * pid + c)

                data.append(w)

    S = coo_matrix(

        (data, (rows, cols)),

        shape=(3 * Ngrid, 3 * N),

        dtype=complex,

    ).tocsr()

    return S
def build_gather_matrix(

    spread_matrix,

    box,

    n_grids,

):

    """

    Construct the sparse gather matrix.

    Parameters

    ----------

    spread_matrix : scipy.sparse.csr_matrix

        Matrix returned by build_spread_matrix().

        Shape = (3*Ngrid, 3*N)

    box : (Lx, Ly)

    n_grids : int or (Nx, Ny)

    Returns

    -------

    gather_matrix : scipy.sparse.csr_matrix

        Shape = (3*N, 3*Ngrid)

    Satisfies

        field = G @ grid_field

    where

        grid_field.shape = (3*Ngrid,)

        field.shape      = (3*N,)

    """

    if np.isscalar(n_grids):

        Nx = Ny = int(n_grids)

    else:

        Nx, Ny = n_grids

    Lx, Ly = box[:2]

    dx = Lx / Nx

    dy = Ly / Ny

    # Match the normalization used in gather_from_grid()

    return (dx * dy) * spread_matrix.transpose().tocsr()

def reciprocal_space_field_function(dipoles, k, box, xi, mu, precomputed_spread_matrix, precomputed_contract_matrix, n_grids):
    Nx, Ny = n_grids
    dx = box[0] / Nx
    dy = box[1] / Ny
    qx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    qy = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    Qx, Qy = np.meshgrid(qx, qy, indexing="ij")
    dipoles = dipoles.reshape(-1) 
    dipoles_grid_spread = precomputed_spread_matrix @ dipoles
    dipoles_grid_spread = dipoles_grid_spread.reshape(Nx, Ny, 3)
    dipoles_fft = np.fft.fft2(dipoles_grid_spread, axes=(0, 1))
    dz = 0
    Gk = tensor_kernel_vec(Qx, Qy, dz, k, xi, mu)
    Fk = np.einsum("...ij,...j->...i", Gk, dipoles_fft)
    F = np.fft.ifft2(Fk, axes=(0, 1))
    F = F.reshape(-1)
    particle_field = precomputed_contract_matrix @ F
    return particle_field

def total_field(dipoles, k, polarizability, box, xi, mu, real_space_sparse_tensor, precomputed_spread_matrix, precomputed_contract_matrix, n_grids):
    dipoles = dipoles.reshape(-1)      # shape (3N,)
    real_space_field = real_space_sparse_tensor @ dipoles # shape (3N,) 
    reciprocal_space_field = reciprocal_space_field_function(dipoles, k, box, xi, mu, precomputed_spread_matrix, precomputed_contract_matrix, n_grids)
    self_field = self_term_scalar(k,xi) * dipoles
    E_total = dipoles / polarizability
    return E_total - (real_space_field + reciprocal_space_field - self_field)
    