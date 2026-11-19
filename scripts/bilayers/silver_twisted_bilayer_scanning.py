import sys
import numpy as np
import pandas as pd

from scipy.special import erfc, erfcx, erf
from cdm_ewald.CDM_Ewald_direct import CDM_Ewald_direct


# ============================================================
# Input: layer separation in nm
# ============================================================

if len(sys.argv) != 2:
    raise ValueError(
        "Usage: python run_moire.py <separation_nm>"
    )

layer_separation = float(sys.argv[1])

print(f"Running layer separation = {layer_separation} nm")


# ============================================================
# Optical parameters
# ============================================================

optical_data = pd.read_csv("silver_optical_data.csv")

wavelength_orig = optical_data["lambda"].to_numpy()

epsP_orig = (
    optical_data["e1"].to_numpy()
    + 1j * optical_data["e2"].to_numpy()
)

# Additional wavelengths
add_wavelength_list = np.arange(
    470 / 50,
    490 / 50,
    0.1
)

wavelength_list = np.unique(
    np.concatenate(
        (wavelength_orig, add_wavelength_list)
    )
)

epsP = (
    np.interp(
        wavelength_list,
        wavelength_orig,
        epsP_orig.real
    )
    + 1j
    * np.interp(
        wavelength_list,
        wavelength_orig,
        epsP_orig.imag
    )
)

k_list = 2 * np.pi / wavelength_list

wavevector_direction = np.array([0, 0, 1])
polarization_direction = np.array([1, 0, 0])


# ============================================================
# Mie polarizability
# ============================================================

def phi_1(x):
    return np.sin(x) / x - np.cos(x)


def xi_1(x):
    return (
        np.sin(x) / x
        - np.cos(x)
        + 1j * (
            -np.cos(x)
            - np.sin(x) / x
        )
    )


def phi_1_prime(x):
    return (
        -np.sin(x) / x**2
        + np.cos(x) / x
        + np.sin(x)
    )


def xi_1_prime(x):
    return (
        -np.sin(x) / x**2
        + np.cos(x) / x
        + np.sin(x)
        + 1j * (
            np.cos(x) / x**2
            + np.sin(x) / x
            - np.cos(x)
        )
    )


x_array = k_list
m_array = np.sqrt(epsP)

a1_array_up = (
    m_array
    * phi_1(m_array * x_array)
    * phi_1_prime(x_array)
    - phi_1(x_array)
    * phi_1_prime(m_array * x_array)
)

a1_array_down = (
    m_array
    * phi_1(m_array * x_array)
    * xi_1_prime(x_array)
    - xi_1(x_array)
    * phi_1_prime(m_array * x_array)
)

a1_array = a1_array_up / a1_array_down

polarizability_list = (
    1j
    * 1.5
    * a1_array
    / x_array**3
    * 4
    * np.pi
)


# ============================================================
# Moire geometry
# ============================================================

a = 200.0  # nm

a_cell = np.sqrt(5) * a

C1 = np.array([
    a_cell,
    0.0
])

C2 = np.array([
    0.0,
    a_cell
])

C = np.column_stack([
    C1,
    C2
])


# Bottom primitive lattice
A = (
    a / np.sqrt(5)
) * np.array([
    [1.0, 2.0],
    [-2.0, 1.0],
])


# Top primitive lattice
B = (
    a / np.sqrt(5)
) * np.array([
    [2.0, 1.0],
    [-1.0, 2.0],
])


# ============================================================
# Generate positions inside the supercell
# ============================================================

def points_inside_supercell(
    primitive_vectors,
    supercell_vectors,
    search_limit=20,
    tolerance=1e-9,
):

    inverse_supercell = np.linalg.inv(
        supercell_vectors
    )

    positions = []
    lattice_indices = []
    fractional_coordinates = []

    for i in range(
        -search_limit,
        search_limit + 1
    ):
        for j in range(
            -search_limit,
            search_limit + 1
        ):

            position = (
                primitive_vectors
                @ np.array(
                    [i, j],
                    dtype=float
                )
            )

            fractional = (
                inverse_supercell
                @ position
            )

            inside = (
                np.all(
                    fractional >= -tolerance
                )
                and np.all(
                    fractional
                    < 1.0 - tolerance
                )
            )

            if inside:
                positions.append(position)
                lattice_indices.append(
                    [i, j]
                )
                fractional_coordinates.append(
                    fractional
                )

    positions = np.asarray(positions)

    lattice_indices = np.asarray(
        lattice_indices,
        dtype=int
    )

    fractional_coordinates = np.asarray(
        fractional_coordinates
    )

    order = np.lexsort((
        fractional_coordinates[:, 1],
        fractional_coordinates[:, 0],
    ))

    return (
        positions[order],
        lattice_indices[order],
        fractional_coordinates[order],
    )


bottom_xy, bottom_ij, bottom_uv = (
    points_inside_supercell(
        A,
        C
    )
)

top_xy, top_ij, top_uv = (
    points_inside_supercell(
        B,
        C
    )
)


# ============================================================
# Add z coordinates
# ============================================================

bottom_xyz = np.column_stack([
    bottom_xy,
    np.zeros(
        len(bottom_xy)
    )
])

top_xyz = np.column_stack([
    top_xy,
    np.full(
        len(top_xy),
        layer_separation
    )
])

positions = np.vstack([
    bottom_xyz,
    top_xyz
])


print(
    f"Particles per layer: "
    f"{len(bottom_xyz)}"
)

print(
    f"Total particles: "
    f"{len(positions)}"
)


# ============================================================
# CDM Ewald calculation
#
# All lengths divided by 50, following your original code.
# ============================================================

cdm = CDM_Ewald_direct(
    C / 50,
    positions / 50,
    polarizability_list,
    k_list,
    wavevector_direction,
    polarization_direction,
)

cdm.compute()


# ============================================================
# Output files
# ============================================================

# Avoid filename such as 100.0 if input was 100
if layer_separation.is_integer():
    separation_string = str(
        int(layer_separation)
    )
else:
    separation_string = str(
        layer_separation
    )


dipole_filename = (
    f"moire_separation"
    f"{separation_string}"
    f"_dipoles.txt"
)

extinction_filename = (
    f"moire_separation"
    f"{separation_string}"
    f"_extinction.txt"
)


# ============================================================
# Save dipoles
# ============================================================

np.savetxt(
    dipole_filename,
    cdm.dips
)


# ============================================================
# Save extinction efficiency
# ============================================================

np.savetxt(
    extinction_filename,
    np.asarray(
        cdm.extinction_efficiency
    )
)


print(
    f"Saved dipoles to: "
    f"{dipole_filename}"
)

print(
    f"Saved extinction to: "
    f"{extinction_filename}"
)