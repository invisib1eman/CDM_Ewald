from cProfile import label

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import special
from scipy.special import erfc, erfi, erfcx, erf
from scipy.optimize import curve_fit

from cdm_ewald.CDM_Ewald_direct import CDM_Ewald_direct


# ============================================================
# HEDA parameters
# ============================================================

df_heda = pd.read_excel("heda_parameters.xlsx")


def HEDA_conversion(
    nc_diameter,
    ne,
    eta,
    bulk_mean_free_path,
):
    """
    Convert HEDA parameters to plasma frequency
    and damping coefficient.

    Parameters
    ----------
    nc_diameter : float
        Nanocrystal diameter [m].
    ne : float
        Electron density [m^-3].
    eta : float
        Electron core volume fraction.
    bulk_mean_free_path : float
        Bulk electron mean free path [m].

    Returns
    -------
    plasma_frequency : float
        Plasma frequency [cm^-1].
    damping_coefficient : float
        Drude damping coefficient [cm^-1].
    """

    # Physical constants
    hbar = 1.054571817e-34
    electron_charge = 1.602176634e-19
    electron_mass = 9.1093837e-31
    vacuum_permittivity = 8.8541878128e-12
    speed_of_light_cm = 2.998e10

    # Effective nanocrystal radius
    nc_radius = nc_diameter / 2
    effective_radius = nc_radius * eta**(1 / 3)

    if np.any(effective_radius <= 0):
        raise ValueError(
            "Effective electron-core radius must be positive."
        )

    # Effective electron mass
    effective_electron_mass = 0.4 * electron_mass

    # Plasma frequency
    plasma_frequency = (
        np.sqrt(
            ne * electron_charge**2
            / (vacuum_permittivity * effective_electron_mass)
        )
        / (2 * np.pi * speed_of_light_cm)
    )

    # Damping coefficient
    damping_coefficient = (
        (hbar / effective_electron_mass)
        * (3 * np.pi**2 * ne)**(1 / 3)
        * (
            1 / bulk_mean_free_path
            + 3 / (4 * effective_radius)
        )
        / (2 * np.pi * speed_of_light_cm)
    )

    return plasma_frequency, damping_coefficient


def lorentzian(x, A, x0, gamma):
    """
    A     : peak amplitude
    x0    : resonance position
    gamma : HWHM
    """
    return A * gamma**2 / ((x - x0)**2 + gamma**2)


def two_lorentzian(
    x,
    Q0,
    slope,
    A1,
    x01,
    gamma1,
    A2,
    x02,
    gamma2,
):
    return (
        Q0
        + slope * x
        + lorentzian(x, A1, x01, gamma1)
        + lorentzian(x, A2, x02, gamma2)
    )


# ============================================================
# Input parameters
# ============================================================

idx = 0  # 9.5 nm NC

diameter = df_heda["diameter(nm)"][idx]
r_nc = diameter / 2

doping_ratio_sequence = np.arange(1,1.5,0.05)
ne_undoped = df_heda["ne(10^26m^-3)"][idx]
bulk_mean_free_path = df_heda["mean_free_path(nm)"][idx]


# eta scan
eta_undoped = 0.470

eta_doped_min = eta_undoped
eta_doped_max = 1.0
eta_doped_intev = 0.01

eta_doped_sequence = np.arange(
    eta_doped_min,
    eta_doped_max,
    eta_doped_intev,
)


# Material parameters
gap = 1.4  # nm

eps_vacuum = 1.0
eps_il = 2.02
eps_inf = 4.0
eps_substrate = 2.07


# Frequency scan
light_freq_min = 1000
light_freq_max = 10000
light_freq_intv = 1

light_freq_list = np.arange(
    light_freq_min,
    light_freq_max,
    light_freq_intv,
)

wavelength_nm = 1e7 / light_freq_list


# ============================================================
# Wavevector
# ============================================================

# Angular wavenumber in ionic liquid
# unit: cm^-1
k_real_list = (
    2
    * np.pi
    * light_freq_list
    * np.sqrt(eps_il)
)

# Dimensionless k in units of 1 / particle radius
k_dimensionless_list = (
    k_real_list
    * r_nc
    * 1e-7
)


# ============================================================
# Lattice
# ============================================================

lattice_type = "hexagonal"

positions = np.array([
    [0, 0, 0]
]) / r_nc

lattice_constant = (
    r_nc * 2 + gap
) / r_nc


# ============================================================
# Incident light
# ============================================================

incident_theta_air = np.pi / 3

incident_theta_il = np.arcsin(
    np.sin(incident_theta_air)
    * np.sqrt(eps_vacuum / eps_il)
)

print(
    "Incident angle in ionic liquid:",
    incident_theta_il * 180 / np.pi,
    "degree",
)

wavevector_direction = np.array([
    np.sin(incident_theta_il),
    0,
    -np.cos(incident_theta_il),
])

polarization_direction = np.array([
    np.cos(incident_theta_il),
    0,
    np.sin(incident_theta_il),
])


# ============================================================
# Lattice vectors
# ============================================================

if lattice_type == "square":

    v1 = np.array([
        lattice_constant,
        0.0,
    ])

    v2 = np.array([
        0.0,
        lattice_constant,
    ])

    lattice_vectors = np.stack(
        (v1, v2),
        axis=1,
    )

elif lattice_type == "hexagonal":

    v1 = np.array([
        lattice_constant,
        0.0,
    ])

    v2 = np.array([
        0.5 * lattice_constant,
        np.sqrt(3)
        * lattice_constant
        / 2,
    ])

    lattice_vectors = np.stack(
        (v1, v2),
        axis=1,
    )

else:

    raise ValueError(
        f"Unknown lattice type: {lattice_type}"
    )


# ============================================================
# Create output folder
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_dir = Path(
    "eta_scan_results"
) / f"run_{timestamp}"

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    f"Saving data to: {output_dir.resolve()}"
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "timestamp": timestamp,

    # HEDA
    "heda_index": int(idx),
    "diameter_nm": float(diameter),
    "radius_nm": float(r_nc),
    "ne_undoped_1e26_m-3": float(ne_undoped),
    "bulk_mean_free_path_nm":
        float(bulk_mean_free_path),

    # eta scan
    "eta_undoped": float(eta_undoped),
    "eta_doped_min":
        float(eta_doped_min),
    "eta_doped_max":
        float(eta_doped_max),
    "eta_doped_interval":
        float(eta_doped_intev),
    "eta_doped_sequence":
        eta_doped_sequence.tolist(),
    # doping ratio scan
    "doping_ratio_sequence":
        doping_ratio_sequence.tolist(), 
    # materials
    "eps_vacuum":
        float(eps_vacuum),
    "eps_il":
        float(eps_il),
    "eps_inf":
        float(eps_inf),
    "eps_substrate":
        float(eps_substrate),

    # geometry
    "gap_nm":
        float(gap),
    "lattice_type":
        lattice_type,
    "lattice_constant_rnc":
        float(lattice_constant),
    "positions":
        positions.tolist(),
    "lattice_vectors":
        lattice_vectors.tolist(),

    # frequency scan
    "light_freq_min_cm-1":
        float(light_freq_min),
    "light_freq_max_cm-1":
        float(light_freq_max),
    "light_freq_interval_cm-1":
        float(light_freq_intv),
    "light_freq_list_cm-1":
        light_freq_list.tolist(),
    "wavelength_nm":
        wavelength_nm.tolist(),
    "k_real_list":
        k_real_list.tolist(),
    "k_dimensionless_list":
        k_dimensionless_list.tolist(),

    # incident light
    "incident_theta_air_deg":
        float(
            np.degrees(
                incident_theta_air
            )
        ),
    "incident_theta_il_deg":
        float(
            np.degrees(
                incident_theta_il
            )
        ),
    "wavevector_direction":
        wavevector_direction.tolist(),
    "polarization_direction":
        polarization_direction.tolist(),

    # fitting
    "fit_model":
        "two Lorentzians + linear background",
    "fit_width":
        "FWHM = 2 * gamma",
}


with open(
    output_dir / "metadata.json",
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


# ============================================================
# Save script itself
# ============================================================

try:

    script_path = Path(
        __file__
    ).resolve()

    shutil.copy2(
        script_path,
        output_dir / script_path.name,
    )

except NameError:

    # Jupyter / IPython
    try:

        from IPython import get_ipython

        ip = get_ipython()

        if ip is not None:

            history = "\n\n".join(
                command
                for command
                in ip.user_ns.get(
                    "In",
                    [],
                )[1:]
                if command
            )

            with open(
                output_dir
                / "script.py",
                "w",
            ) as f:

                f.write(history)

    except Exception as e:

        print(
            "Could not save script:",
            e,
        )





# ============================================================
# Iterate over eta
# ============================================================
for doping_ratio in doping_ratio_sequence:
    ne = ne_undoped * doping_ratio
    # ============================================================
    # Output lists
    # ============================================================

    peak_first_list = []
    FWHM_first_list = []

    peak_second_list = []
    FWHM_second_list = []

    plasma_freq_list = []
    damping_coef_list = []
    for eta in eta_doped_sequence:

        plasma_freq, damping_coef = (
            HEDA_conversion(
                diameter * 1e-9,
                ne * 1e26,
                eta,
                bulk_mean_free_path
                * 1e-9,
            )
        )

        plasma_freq_list.append(
            plasma_freq
        )

        damping_coef_list.append(
            damping_coef
        )

        print(
            f"eta = {eta:.3f}, "
            f"plasma frequency = "
            f"{plasma_freq:.3f}, "
            f"damping = "
            f"{damping_coef:.3f}"
        )


        # ========================================================
        # Core-shell permittivity
        # ========================================================

        eps_core = (
            eps_inf
            - plasma_freq**2
            / (
                light_freq_list**2
                + 1j
                * damping_coef
                * light_freq_list
            )
        )

        eps_shell = eps_inf

        eps_p = (
            eps_shell
            * (
                eps_core
                + 2 * eps_shell
                + 2
                * eta
                * (
                    eps_core
                    - eps_shell
                )
            )
            / (
                eps_core
                + 2 * eps_shell
                - eta
                * (
                    eps_core
                    - eps_shell
                )
            )
        )


        # ========================================================
        # Polarizability
        # ========================================================

        polarizability_list = (
            4
            * np.pi
            * (
                eps_p
                - eps_il
            )
            / (
                eps_p
                + 2 * eps_il
            )
        )


        # ========================================================
        # CDM calculation
        # ========================================================

        cdm = CDM_Ewald_direct(
            lattice_vectors,
            positions,
            polarizability_list,
            k_dimensionless_list,
            wavevector_direction,
            polarization_direction,
        )

        cdm.compute()


        # ========================================================
        # Save dipole data
        # ========================================================

        dips = np.asarray(
            cdm.dips
        )

        dips_2d = dips.reshape(
            len(light_freq_list),
            -1,
        )

        dips_with_freq = np.column_stack([
            light_freq_list,
            dips_2d,
        ])

        dip_filename = (
            output_dir
            / f"doping_ratio_{doping_ratio:.3f}_eta_{eta:.3f}_dips.dat"
        )

        np.savetxt(
            dip_filename,
            dips_with_freq,
            header=(
                "First column: "
                "light frequency [cm^-1]. "
                "Remaining columns: "
                "flattened complex "
                "dipole components."
            ),
        )


        # ========================================================
        # Lorentzian fit
        # ========================================================

        p0 = [
            0.0,
            0.0,

            1.0,
            2500,
            300,

            0.5,
            4000,
            100,
        ]

        bounds = (
            [
                -np.inf,
                -np.inf,

                0,
                100,
                1,

                0,
                100,
                1,
            ],

            [
                np.inf,
                np.inf,

                np.inf,
                8000,
                5000,

                np.inf,
                8000,
                5000,
            ],
        )

        popt, pcov = curve_fit(
            two_lorentzian,
            light_freq_list,
            cdm.extinction_efficiency,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
        )

        (
            Q0,
            slope,
            A1,
            x01,
            gamma1,
            A2,
            x02,
            gamma2,
        ) = popt


        peak_first_list.append(
            x01
        )

        peak_second_list.append(
            x02
        )

        FWHM_first_list.append(
            2 * gamma1
        )

        FWHM_second_list.append(
            2 * gamma2
        )


    # ============================================================
    # Save summary
    # ============================================================

    summary_df = pd.DataFrame({
        "eta":
            eta_doped_sequence,

        "eta":
            eta_doped_sequence,

        "plasma_frequency_cm^-1":
            plasma_freq_list,

        "damping_coefficient_cm^-1":
            damping_coef_list,

        "peak1_position_cm^-1":
            peak_first_list,

        "peak1_FWHM_cm^-1":
            FWHM_first_list,

        "peak2_position_cm^-1":
            peak_second_list,

        "peak2_FWHM_cm^-1":
            FWHM_second_list,
    })


    summary_df.to_csv(
        output_dir
        / f"doping_ratio_{doping_ratio}_eta_scan_summary.csv",
        index=False,
    )


    # ============================================================
    # Plot
    # ============================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        eta_doped_sequence,
        peak_first_list,
        label="first peak (in plane)",
    )

    plt.plot(
        eta_doped_sequence,
        peak_second_list,
        label="second peak (out of plane)",
    )

    plt.plot(
        eta_doped_sequence,
        FWHM_first_list,
        label="first peak width",
    )

    plt.plot(
        eta_doped_sequence,
        FWHM_second_list,
        label="second peak width",
    )

    plt.xlabel(
        "electron core volume fraction"
    )

    plt.ylabel(
        "Lorentzian parameters (cm$^{-1}$)"
    )

    plt.legend()

    plt.tight_layout()


    # Save PNG
    plt.savefig(
        output_dir
        / f"doping_ratio_{doping_ratio}_eta_peak_and_width.png",
        dpi=300,
        bbox_inches="tight",
    )


    # Save PDF too
    plt.savefig(
        output_dir
        / "eta_peak_and_width.pdf",
        bbox_inches="tight",
    )


    plt.close()

    print(
        "\nSimulation finished."
    )

    print(
        f"Results saved in:\n"
        f"{output_dir.resolve()}"
    )

    plt.plot(
        eta_doped_sequence,
        np.array(peak_second_list)/np.array(peak_first_list),
    )
    plt.xlabel("electron core volume fraction")
    plt.ylabel("peak ratio")
    # Save PNG
    plt.savefig(
        output_dir
        / f"doping_ratio_{doping_ratio}_peak_ratio.png",
        dpi=300,
        bbox_inches="tight",
    )