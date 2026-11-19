"""Layer-resolved PME fields against an independent reciprocal Ewald sum."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from cdm_ewald.electric_field import (
    build_gather_matrix,
    build_spread_matrix,
    reciprocal_space_field_function,
    total_field,
)
from cdm_ewald.ewald_kernels import (
    reciprocal_tensor_kernel,
    self_term_scalar,
    tensor_kernel_vec,
)


class ParticleMeshLayerTests(unittest.TestCase):
    box = np.array([16.0, 18.0])
    k = 0.73
    xi = 0.6
    mu = 0.5
    grids = (64, 72)

    def matrices(self, positions, grids=None, num_layers=None):
        if grids is None:
            grids = self.grids
        if num_layers is None:
            num_layers = len(np.unique(positions[:, 2]))
        spread = build_spread_matrix(
            positions, self.box, grids, self.xi, self.mu,
            error_tol=1e-12, num_layers=num_layers,
        )
        return spread, build_gather_matrix(spread, self.box, grids)

    def mesh_field(self, positions, dipoles, grids=None):
        if grids is None:
            grids = self.grids
        heights = np.unique(positions[:, 2])
        spread, gather = self.matrices(positions, grids)
        return reciprocal_space_field_function(
            dipoles, self.k, self.box, self.xi, self.mu,
            spread, gather, grids,
            num_layers=len(heights), layer_heights=heights,
        ).reshape(-1, 3)

    def direct_reciprocal_field(self, positions, dipoles):
        # Independent particle-pair sum: no spreading, FFT, or deconvolution.
        # These reciprocal cutoffs resolve the unsmoothed Ewald kernel.
        qx = 2 * np.pi * np.arange(-16, 17) / self.box[0]
        qy = 2 * np.pi * np.arange(-18, 19) / self.box[1]
        q = np.stack(np.meshgrid(qx, qy, indexing="ij"), axis=-1)
        q = q.reshape(-1, 2)
        tensors = {}
        field = np.zeros_like(dipoles, dtype=complex)
        for i, target in enumerate(positions):
            for j, source in enumerate(positions):
                dz = target[2] - source[2]
                if dz not in tensors:
                    tensors[dz] = np.array([
                        reciprocal_tensor_kernel(a, b, dz, self.k, self.xi)
                        for a, b in q
                    ])
                phase = np.exp(1j * (q @ (target[:2] - source[:2])))
                field[i] += np.einsum(
                    "q,qab,b->a", phase, tensors[dz], dipoles[j]
                ) / np.prod(self.box)
        return field

    def test_default_single_layer_matches_previous_fft_expression(self):
        positions = np.array([[3.2, 4.1, 2.5], [6.7, 8.6, 2.5]])
        dipoles = np.array([[1.0, 0.2j, -0.3], [0.4j, 0.7, 0.5j]])
        spread = build_spread_matrix(
            positions, self.box, self.grids, self.xi, self.mu, 1e-12
        )
        gather = build_gather_matrix(spread, self.box, self.grids)
        nx, ny = self.grids
        qx = 2 * np.pi * np.fft.fftfreq(nx, self.box[0] / nx)
        qy = 2 * np.pi * np.fft.fftfreq(ny, self.box[1] / ny)
        qx, qy = np.meshgrid(qx, qy, indexing="ij")
        grid_dipoles = (spread @ dipoles.ravel()).reshape(nx, ny, 3)
        spectrum = np.fft.fft2(grid_dipoles, axes=(0, 1))
        kernel = tensor_kernel_vec(qx, qy, 0, self.k, self.xi, self.mu)
        grid_field = np.fft.ifft2(
            np.einsum("...ij,...j->...i", kernel, spectrum), axes=(0, 1)
        )
        expected = gather @ grid_field.ravel()
        actual = reciprocal_space_field_function(
            dipoles, self.k, self.box, self.xi, self.mu,
            spread, gather, self.grids,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)

    def test_three_unequally_populated_layers_converge_to_direct_sum(self):
        positions = np.array([
            [4.3, 5.6, 2.4], [7.1, 6.2, 0.0], [5.5, 9.3, 5.1],
            [8.8, 10.4, 0.0], [3.7, 8.1, 0.0],
        ])
        rng = np.random.default_rng(15)
        dipoles = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
        expected = self.direct_reciprocal_field(positions, dipoles)
        coarse = self.mesh_field(positions, dipoles, grids=(12, 14))
        fine = self.mesh_field(positions, dipoles)
        np.testing.assert_allclose(fine, expected, rtol=1e-6, atol=1e-9)
        self.assertLess(
            np.linalg.norm(fine - expected),
            0.05 * np.linalg.norm(coarse - expected),
        )

        permutation = np.array([4, 2, 0, 3, 1])
        shifted = positions[permutation].copy()
        shifted[:, 2] += 11.0
        reordered = self.mesh_field(shifted, dipoles[permutation])
        np.testing.assert_allclose(
            reordered, fine[permutation], rtol=1e-12, atol=1e-13
        )

    def test_z_dipole_generates_cross_layer_field_in_both_directions(self):
        positions = np.array([[4.2, 6.4, 0.0], [5.7, 7.1, 2.3]])
        for source in (0, 1):
            with self.subTest(source_layer=source):
                dipoles = np.zeros((2, 3), dtype=complex)
                dipoles[source, 2] = 1.0 + 0.4j
                expected = self.direct_reciprocal_field(positions, dipoles)
                actual = self.mesh_field(positions, dipoles)
                self.assertGreater(np.linalg.norm(expected[1 - source, :2]), 1e-5)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)

                spread, gather = self.matrices(positions)
                alpha = 0.9 + 0.2j
                operator_result = total_field(
                    dipoles, self.k, alpha, self.box, self.xi, self.mu,
                    csr_matrix((6, 6), dtype=complex),
                    spread, gather, self.grids,
                    num_layers=2, layer_heights=np.unique(positions[:, 2]),
                )
                expected_operator = (
                    dipoles / alpha - expected
                    + self_term_scalar(self.k, self.xi) * dipoles
                )
                np.testing.assert_allclose(
                    operator_result.reshape(-1, 3), expected_operator,
                    rtol=1e-6, atol=1e-9,
                )

    def test_layer_count_must_match_positions(self):
        positions = np.array([[4.2, 6.4, 0.0], [5.7, 7.1, 2.3]])
        for count in (1, 3):
            with self.subTest(num_layers=count):
                with self.assertRaises(ValueError):
                    self.matrices(positions, num_layers=count)

    def test_multilayer_field_requires_heights(self):
        positions = np.array([[4.2, 6.4, 0.0], [5.7, 7.1, 2.3]])
        spread, gather = self.matrices(positions)
        with self.assertRaises(ValueError):
            reciprocal_space_field_function(
                np.ones((2, 3)), self.k, self.box, self.xi, self.mu,
                spread, gather, self.grids, num_layers=2,
            )


if __name__ == "__main__":
    unittest.main()
