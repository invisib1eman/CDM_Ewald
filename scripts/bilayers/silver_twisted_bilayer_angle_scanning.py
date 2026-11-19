import numpy as np
import numpy as np
from scipy import special
from scipy.special import erfc, erfi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from scipy.special import erfc, erfcx, erf
from cdm_ewald.CDM_Ewald_direct import CDM_Ewald_direct
from cdm_ewald.CDM_Ewald_particle_mesh import CDM_Ewald_pme
# parameters
# import silver dielectric constant
optical_data = pd.read_csv(
    "silver_optical_data.csv"
)
# Original data
radius = 50 # nm
wavelength_orig = optical_data["lambda"].to_numpy()

epsP_orig = optical_data["e1"].to_numpy() + 1j * optical_data["e2"].to_numpy()

# Add new wavelengths

add_wavelength_list = np.arange(470/radius, 490/radius, 0.1)  # 470, 475, 480, 485

# Merge and sort uniquely

wavelength_list = np.unique(

    np.concatenate((wavelength_orig, add_wavelength_list))

)

# Interpolate the real and imaginary parts separately

epsP = (

    np.interp(wavelength_list, wavelength_orig, epsP_orig.real)

    + 1j * np.interp(wavelength_list, wavelength_orig, epsP_orig.imag)

)

# Other quantities

k_list = 2 * np.pi / wavelength_list

k_direction = np.array([0, 0, 1])
def phi_1(x):
    return np.sin(x)/x - np.cos(x)
def xi_1(x):
    return np.sin(x)/x - np.cos(x) + 1j*(-np.cos(x) - np.sin(x)/x)
def phi_1_prime(x):
    return -np.sin(x)/x**2 + np.cos(x)/x + np.sin(x)
def xi_1_prime(x):
    return -np.sin(x)/x**2 + np.cos(x)/x + np.sin(x) + 1j*(np.cos(x)/x**2 + np.sin(x)/x - np.cos(x))
x_array = k_list
m_array = np.sqrt(epsP)
a1_array_up = m_array*phi_1(m_array*x_array)*phi_1_prime(x_array)-phi_1(x_array)*phi_1_prime(m_array*x_array)
a1_array_down = m_array*phi_1(m_array*x_array)*xi_1_prime(x_array)-xi_1(x_array)*phi_1_prime(m_array*x_array)
a1_array = a1_array_up/a1_array_down
polarizability_list = 1j*1.5*a1_array/(x_array**3) * 4 * np.pi
wavelength_nm = wavelength_list * radius
wavevector_direction = np.array([0,0,1])
polarization_direction = np.array([1,0,0])
def points_inside_supercell(
        primitive_vectors,
        supercell_vectors,
        search_limit=20,
        tolerance=1e-9,
    ):

        inverse_supercell = np.linalg.inv(supercell_vectors)

        positions = []
        lattice_indices = []
        fractional_coordinates = []

        for i in range(-search_limit, search_limit + 1):
            for j in range(-search_limit, search_limit + 1):

                position = primitive_vectors @ np.array(
                    [i, j],
                    dtype=float,
                )

                fractional = inverse_supercell @ position

                inside = (
                    np.all(fractional >= -tolerance)
                    and np.all(fractional < 1.0 - tolerance)
                )

                if inside:
                    positions.append(position)
                    lattice_indices.append([i, j])
                    fractional_coordinates.append(fractional)

        positions = np.asarray(positions)
        lattice_indices = np.asarray(lattice_indices, dtype=int)
        fractional_coordinates = np.asarray(fractional_coordinates)

        order = np.lexsort((
            fractional_coordinates[:, 1],
            fractional_coordinates[:, 0],
        ))

        return (
            positions[order],
            lattice_indices[order],
            fractional_coordinates[order],
        )


def twisted_square_bilayer(a, m, n, layer_separation):

    theta = 2 * np.arctan2(n, m)

    def rotation_matrix(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)],
        ])

    # Rotate bottom and top symmetrically
    A = a * rotation_matrix(-theta / 2)
    B = a * rotation_matrix(+theta / 2)

    # Minimal axis-aligned common supercell
    L = a * np.sqrt(m**2 + n**2)
    C = np.diag([L, L])

    search_limit = 2 * (m + n + 1)

    bottom_xy, _, _ = points_inside_supercell(
        A, C, search_limit=search_limit
    )

    top_xy, _, _ = points_inside_supercell(
        B, C, search_limit=search_limit
    )

    bottom_xyz = np.column_stack([
        bottom_xy,
        np.zeros(len(bottom_xy))
    ])

    top_xyz = np.column_stack([
        top_xy,
        np.full(len(top_xy), layer_separation)
    ])

    positions = np.vstack([bottom_xyz, top_xyz])

    return L, positions, np.rad2deg(theta)
def multilayer_transmittance(
    dips,
    k_dimensionless_list,
    incident_field,
    area_unit_cell,
    theta,
):


    # Wavevector at every frequency
    k = np.asarray(k_dimensionless_list)


    # Project dipoles onto transmitted TM polarization
    # Shape: (n_freq, n_dipoles)

    p_proj = np.einsum("fij,fij->f", dips, np.conj(incident_field))






    # Complex transmission coefficient
    t = (
        1
        + 1j * k
        / (2 * area_unit_cell * np.cos(theta))
        * p_proj
    )

    return t, np.abs(t)**2
a = 300
layer_separation = 100
num_layer = 2

n = 1


extinction_results = {}

for m in range(3, 20):
    L, positions, theta = twisted_square_bilayer(
        a, m, n, layer_separation
    )

    box = np.array([L, L])

    cdm = CDM_Ewald_pme(
        box / radius,
        positions / radius,
        polarizability_list,
        k_list,
        wavevector_direction,
        polarization_direction,
        num_layers=num_layer
    )

    cdm.compute()

    t, T = multilayer_transmittance(
        cdm.dips,
        cdm.dimensionless_angular_wavenumber,
        cdm.incident_field_array,
        (L / radius)**2,
        0
    )

    extinction_results[m] = {
        "theta": theta,
        "extinction": 1 - T
    }

fig, ax = plt.subplots(figsize=(9, 8))

offset = 0.15  # Adjust depending on extinction amplitude

for i, m in enumerate(m_values):
    theta = extinction_results[m]["theta"]
    extinction = extinction_results[m]["extinction"]

    ax.plot(
        wavelength_list * radius,
        extinction + i * offset,
        color=cmap(norm(theta)),
        lw=1.2
    )

    ax.text(
        wavelength_list[-1] * radius,
        i * offset,
        f"{theta:.2f}°",
        fontsize=8,
        va="center",
        ha="left"
    )

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Extinction + offset")
ax.set_title("Twist-Angle-Dependent Extinction")

ax.set_yticks([])
ax.grid(axis="x", alpha=0.15)

ax.margins(x=0.12)
plt.tight_layout()
plt.savefig(f"../../figures/twisted_silver_bilayer/Twist_Angle_Dependent_Extinction_silver_radius{radius}nm_layer_separation{layer_separation}nm_original_lattice_constant{a}nm.png",
            dpi=600)