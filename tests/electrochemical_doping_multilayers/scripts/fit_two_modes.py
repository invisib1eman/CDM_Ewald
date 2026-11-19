import numpy as np
from scipy import special
from pathlib import Path
import csv
from scipy.special import erfc, erfi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, least_squares
from scipy.stats import t as student_t
from functools import partial
import json
from scipy.special import erfc, erfcx, erf
def convert_core_volume_fraction_to_depletion_layer_thickness(R,eta_core):
    r_eff = R*(eta_core**(1/3))
    return R-r_eff
def convert_depletion_layer_thickness_to_core_volume_fraction(R,d_depletion):
    r_eff = R-d_depletion
    return (r_eff/R)**3
def HEDA_conversion(
    nc_diameter,
    ne,
   depletion_layer_thickness,
    bulk_mean_free_path,
):
    """
    Convert HEDA parameters to effective electron mass,
    plasma frequency, and damping coefficient.

    Parameters
    ----------
    nc_diameter : float
        Nanocrystal diameter [m].
    ne : float
        Electron density [m^-3].
    depletion_layer_thickness : float
        Depletion layer thickness [m].
    bulk_mean_free_path : float
        Bulk electron mean free path [m].

    Returns
    -------
    electron_mass_ratio : float
        Effective electron mass divided by free electron mass.
    plasma_frequency : float
        Plasma angular frequency [rad/s].
    damping_coefficient : float
        Drude damping coefficient [s^-1].
    """

    # Physical constants (SI)
    hbar = 1.054571817e-34          # J s
    electron_charge = 1.602176634e-19  # C
    electron_mass = 9.1093837e-31  # kg
    vacuum_permittivity = 8.8541878128e-12  # F/m
    speed_of_light_cm = 2.998e10 # cm/s
    # Effective nanocrystal radius
    nc_radius = nc_diameter / 2
    effective_radius = nc_radius - depletion_layer_thickness

    if np.any(effective_radius <= 0):
        raise ValueError(
        "Depletion layer thickness must be smaller than the nanocrystal radius."
    )

    # Effective electron mass
    effective_electron_mass = 0.4 * electron_mass

    # Plasma frequency
    plasma_frequency = np.sqrt(
        ne * electron_charge**2
        / (vacuum_permittivity * effective_electron_mass)
    ) / (2*np.pi*speed_of_light_cm)

    # Damping coefficient
    damping_coefficient = (hbar/effective_electron_mass) * (3*np.pi**2*ne)**(1/3) * (
        1 / bulk_mean_free_path
        + 3 / (4 * effective_radius)
    ) / (2*np.pi*speed_of_light_cm)


    return plasma_frequency, damping_coefficient
def extinction_convertion(extinction):
    T = 10**(-extinction)
    return 1-T
# get parameters from HEDA
df_heda = pd.read_excel("../raw_data/heda_parameters.xlsx")
df_heda["plasma_frequency(cm^-1)"], df_heda["damping_coefficient(cm^-1)"] = HEDA_conversion(
    df_heda["diameter(nm)"]*1e-9,
    df_heda["ne(10^26m^-3)"]*1e26,
    df_heda["depletion_layer_thickness(nm)"]*1e-9,
    df_heda["mean_free_path(nm)"]*1e-9,
)
# load extinction spectra for 13.4 nm 
df_extinction = pd.read_csv("../raw_data/13p4nm_8L_CaF2_60deg_ppol_all_voltages.csv")
# freq list
freq_light = df_extinction["wavenumber_cm-1"].to_numpy()
# freq filtered from 1700 to 5500
lower_bond_freq = 1700
higher_bond_freq = 5500
MOL_MASK = (2900, 3250) 
mask = (freq_light>lower_bond_freq) & (freq_light<higher_bond_freq)

freq_filtered = freq_light[mask] # shared by all curves
# extinction of different voltage
voltage_series = pd.Series({
    float(col.split("V")[0]): df_extinction[col].to_numpy()
    for col in df_extinction.columns
    if "V (" in col
})
# filtered extinction
# Each curve has its own experimental extinction spectrum
freq_list = []
extinction_list = []
voltage_list = []
fit_parameter_list = []
for idx, extinction in enumerate(voltage_series):
    voltage_list.append(voltage_series.index[idx]) 
    freq_filtered = freq_light[mask]
    extinction_filtered = extinction[mask]
    extinction_filtered_converted = extinction_convertion(extinction_filtered)
    extinction_list.append(extinction_filtered_converted)
    freq_list.append(freq_filtered)
    
def lorentzian(x, A, x0, gamma):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)
def two_lorentzian(x,
                   Q0, slope,
                   A1, x01, gamma1,
                   A2, x02, gamma2):

    return (
        Q0 + slope*x
        + lorentzian(x, A1, x01, gamma1)
        + lorentzian(x, A2, x02, gamma2)
    )
p0 = [
    0.0, 0.0,       # background
    1.0, 2500, 300, # peak 1: amplitude, center, HWHM
    0.5, 4000, 100  # peak 2
]

bounds = (
    [
        -np.inf, -np.inf,
        0, 100, 1,
        0, 100, 1
    ],
    [
        np.inf, np.inf,
        np.inf, 8000, 5000,
        np.inf, 8000, 5000
    ]
)
figures_dir = Path("../figures/two_modes_fit")
figures_dir.mkdir(parents=True, exist_ok=True)

# Open the CSV before the loop.
with open("fitting_parameters.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "voltage_V", "Q0", "slope",
        "A1", "x01", "gamma1",
        "A2", "x02", "gamma2",
    ])

    for curve_index in range(len(voltage_list)):
        voltage = voltage_list[curve_index]
        freq_filtered = freq_list[curve_index]
        extinction_converted = extinction_list[curve_index]

        # First fit: use the whole range.
        popt_full, _ = curve_fit(
            two_lorentzian,
            freq_filtered,
            extinction_converted,
            p0=p0,
            bounds=bounds,
            method="trf",
            x_scale="jac",
            maxfev=50000,
        )

        # Second fit: exclude MOL_MASK and start from the first fit.
        fit_mask = (
            (freq_filtered < MOL_MASK[0])
            | (freq_filtered > MOL_MASK[1])
        )

        popt, pcov = curve_fit(
            two_lorentzian,
            freq_filtered[fit_mask],
            extinction_converted[fit_mask],
            p0=popt_full,
            bounds=bounds,
            method="trf",
            x_scale="jac",
            maxfev=50000,
        )

        # Use the second fit as the initial guess for the next voltage.
        p0 = popt.copy()

        # Evaluate the second fit over the whole range for plotting.
        y_fit = two_lorentzian(freq_filtered, *popt)

        # Save this voltage and its fitting parameters.
        writer.writerow([voltage, *popt])
        f.flush()

        # Plot the experimental data and fitted curve.
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(
            freq_filtered, extinction_converted,
            label="Experimental",
        )
        ax.plot(
            freq_filtered, y_fit,
            "--", label="Fit",
        )

        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax.set_ylabel(r"$1-T$")
        ax.set_title(f"Voltage: {voltage:+g} V")
        ax.legend()
        fig.tight_layout()

        # Save one figure for this voltage.
        fig.savefig(
            figures_dir / f"fit_{curve_index:02d}_{voltage:+g}V.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)