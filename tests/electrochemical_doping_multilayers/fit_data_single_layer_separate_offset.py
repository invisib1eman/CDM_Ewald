"""Fit each voltage extinction curve independently with ne_ratio, eta, and offset.

Use the same environment as the original script: numpy, scipy, pandas,
matplotlib, openpyxl, and your local cdm_ewald package. Run beside
heda_parameters.xlsx and 13p4nm_8L_CaF2_60deg_ppol_all_voltages.csv,
or supply --input-dir. Outputs default to fit_results/.

The physical model, HEDA conversion, geometry, frequency window, voltage
exclusions (+1 V and +2 V), layer count (11), and plot are retained.
Bounds are the original ACTIVE bounds: ne_ratio [0.7, 1.5], eta [0.6, 1.0],
offset [0.01, 0.10]. Each curve starts from (1.3, 0.8, 0.05).

Uncertainties use each curve's full three-parameter Jacobian, its own residual
variance, and N_i - 3 degrees of freedom. Blank CSV uncertainties indicate a
boundary, nonconverged, or rank-deficient fit; uncertainty_status explains why.
The 95% intervals are local linear approximations, not constrained intervals.

Example: python independent_curve_fitting.py --no-show
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import t as student_t


LOGGER = logging.getLogger(__name__)
PARAMETER_NAMES = ("ne_ratio", "eta", "offset")
FIT_INITIAL_GUESS = (1.3, 0.8, 0.05)
FIT_LOWER_BOUNDS = (0.7, 0.6, 0.01)
FIT_UPPER_BOUNDS = (1.5, 1.0, 0.10)
MAX_NFEV = 100
COVARIANCE_RCOND = 1e-8
BOUNDARY_RELATIVE_TOLERANCE = 1e-7


# HEDA conversion
def effective_mass_kane(ne, m0_ratio=0.3, alpha_eV=0.5):
    """
    ne: electron density [m^-3]
    m0_ratio: band-edge effective mass / electron mass
    alpha_eV: Kane nonparabolicity parameter [eV^-1]
    """
    hbar = 1.054571817e-34
    m_e = 9.1093837e-31
    e = 1.602176634e-19

    m0 = m0_ratio * m_e

    kF = (3 * np.pi**2 * ne)**(1/3)

    # Parabolic reference energy in eV
    E_parabolic = hbar**2 * kF**2 / (2*m0*e)

    # Solve alpha*EF^2 + EF = E_parabolic
    EF = 2*E_parabolic / (
        1 + np.sqrt(1 + 4*alpha_eV*E_parabolic)
    )

    m_eff = m0 * (1 + 2*alpha_eV*EF)

    return m_eff, EF
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
    effective_electron_mass = effective_mass_kane(ne)[0]

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

# transmittance of multilayers including single layer
def multilayer_transmittance(
    dips,
    k_dimensionless_list,
    incident_field,
    eps_il,
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
        / (2 * eps_il * area_unit_cell * np.cos(theta))
        * p_proj
    )

    return t, np.abs(t)**2

# single layer fitting
def extinction_cdm_single_layer(
    freq, ne_ratio, eta,
    offset,
    ne_dispersion,
    incident_field_array,
    nc_diameter,
    bulk_mean_free_path,
    eps_m,
    eps_inf,
    G_array,
    area_unit_cell,
    incident_theta_il,
    n_layers
):

    nc_radius = nc_diameter / 2


    effective_radius = nc_radius * eta**(1/3)
    depletion_layer_thickness = nc_radius - effective_radius
    ne = ne_dispersion*ne_ratio
    plasma_freq, damping_coef = HEDA_conversion(
        nc_diameter*1e-9,
        ne*1e26,
        depletion_layer_thickness*1e-9,
        bulk_mean_free_path*1e-9,
    )
    LOGGER.debug("Plasma frequency %s, damping %s", plasma_freq, damping_coef)


    eps_core = eps_inf - plasma_freq**2 / (
        freq**2 + 1j * damping_coef * freq
    )

    eps_shell = eps_inf

    eps_p = eps_shell * (
        eps_core + 2*eps_shell + 2*eta*(eps_core-eps_shell)
    ) / (
        eps_core + 2*eps_shell - eta*(eps_core-eps_shell)
    )

    polarizability_list = (
        4*np.pi*(eps_p-eps_m)/(eps_p+2*eps_m)
    )

    k_real_list = 2*np.pi*freq*np.sqrt(eps_m)
    k_dimensionless_list = k_real_list*nc_radius*1e-7
    dips = np.zeros((len(freq), 1, 3), dtype=complex)

    # Shape: (n_freq, 3, 3)
    A = (
        np.eye(3)[None, :, :]
        / polarizability_list[:, None, None]
        - G_array
    )

    # Shape: (n_freq, 3)
    E0 = incident_field_array.reshape(len(freq), 3)

    # Solve all frequencies
    # Explicit right-hand-side axis works with both NumPy 1.x and 2.x.
    P = np.linalg.solve(A, E0[..., None])[..., 0]

    # Shape: (n_freq, 1, 3)
    dips = P.reshape(len(freq), 1, 3)

    t, T = multilayer_transmittance(
        dips,
        k_dimensionless_list,
        incident_field_array,
        eps_m,
        area_unit_cell,
        incident_theta_il
    )

    return -np.log10(T) * n_layers + offset

def prepare_model_and_data(
    input_dir: Path,
) -> tuple[Callable[..., np.ndarray], list[float], list[np.ndarray], list[np.ndarray]]:
    """Load the original data and build its unchanged single-layer CDM model."""
    from cdm_ewald.CDM_Ewald_direct import CDM_Ewald_direct
    from cdm_ewald.lattice_utils import lattice_area

    input_dir = Path(input_dir)
    # get parameters from HEDA
    df_heda = pd.read_excel(input_dir / "heda_parameters.xlsx")
    df_heda["plasma_frequency(cm^-1)"], df_heda["damping_coefficient(cm^-1)"] = HEDA_conversion(
        df_heda["diameter(nm)"]*1e-9,
        df_heda["ne(10^26m^-3)"]*1e26,
        df_heda["depletion_layer_thickness(nm)"]*1e-9,
        df_heda["mean_free_path(nm)"]*1e-9,
    )
    # load extinction spectra for 13.4 nm
    df_extinction = pd.read_csv(input_dir / "13p4nm_8L_CaF2_60deg_ppol_all_voltages.csv")
    # freq list
    freq_light = df_extinction["wavenumber_cm-1"].to_numpy()
    # freq filtered from 1700 to 4500
    lower_bond_freq = 1700
    higher_bond_freq = 4500
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
    for idx, extinction in enumerate(voltage_series):
        if voltage_series.index[idx] == 2 or voltage_series.index[idx] == 1:
            continue
        voltage_list.append(voltage_series.index[idx])
        freq_filtered = freq_light[mask]
        extinction_filtered = extinction[mask]
        extinction_list.append(extinction_filtered)
        freq_list.append(freq_filtered)
    # build geometry so that the G matrix can be built, example polarizability is also here, eta is not important here
    idx = 1
    r_nc = df_heda["diameter(nm)"][idx] / 2
    depletion_layer_thickness = df_heda["depletion_layer_thickness(nm)"][idx]
    plasma_freq = df_heda["plasma_frequency(cm^-1)"][idx]
    damping_coef = df_heda["damping_coefficient(cm^-1)"][idx]
    eta = 1
    gap = 1.4 # nm
    eps_vacuum = 1 # vacuum/air
    eps_il = 2.02 # ionic liquid
    eps_inf = 4 # indium oxide
    eps_substrate = 2.07 # CaF2
    light_freq_list = freq_filtered
    # convert to wavelength in vacuum/air and angular wavenumber in the medium
    wavelength_nm = 10**7/light_freq_list
    # angular wavenumber list in the unit of cm^-1 (angular wavenumber in the medium, which is ionic liquid)
    k_real_list = 2*np.pi*light_freq_list*np.sqrt(eps_il)
    # wavenumber list in the unit of 1/particle radius first convert to m^-1 then convert to 1/particle radius
    k_dimensionless_list = k_real_list*r_nc*1e-7 # 1e-7 is to convert nm to cm
    # permittivity core-shell
    eps_core = eps_inf - plasma_freq**2 / (light_freq_list**2 + 1j*damping_coef*light_freq_list)
    eps_shell = eps_inf
    eps_p = eps_shell*(eps_core+2*eps_shell+2*eta*(eps_core-eps_shell))/(eps_core+2*eps_shell-eta*(eps_core-eps_shell))
    # polarizability in unit of r_nc^3, add 4pi for consistency
    polarizability_list = 4*np.pi*(eps_p-eps_il)/(eps_p+2*eps_il)
    # define lattice
    lattice_type = "hexagonal"

    lattice_constant = (r_nc*2+gap)/r_nc

    incident_theta_air = np.pi/3 # 60 degree from air/vacuum
    incident_theta_il = np.arcsin(np.sin(incident_theta_air)*np.sqrt(eps_vacuum/eps_il)) # refraction angle in the ionic liquid, which is the true incident angle to the metasurface
    LOGGER.info("Incident angle in ionic liquid: %.6g degrees", incident_theta_il*180/np.pi)
    incident_phi = 0
    wavevector_direction = np.array([np.sin(incident_theta_il)*np.cos(incident_phi),np.sin(incident_theta_il)*np.sin(incident_phi),-np.cos(incident_theta_il)]) # z
    polarization_direction = np.array([np.cos(incident_theta_il)*np.cos(incident_phi),np.cos(incident_theta_il)*np.sin(incident_phi),np.sin(incident_theta_il)]) # x
    if lattice_type == "square":
        v1 = np.array([lattice_constant, 0.0])
        v2 = np.array([0.0, lattice_constant])
        lattice_vectors = np.stack((v1,v2),axis=0)
    elif lattice_type == "hexagonal":
        v1 = np.array([lattice_constant, 0.0])
        v2 = np.array([-0.5 * lattice_constant, np.sqrt(3) * lattice_constant / 2])
        lattice_vectors = np.stack((v1,v2),axis=0)
    # single layer
    single_layer_position = np.array([[0,0,0]])
    cdm_single_layer = CDM_Ewald_direct(lattice_vectors,single_layer_position,polarizability_list,k_dimensionless_list, wavevector_direction,polarization_direction)
    cdm_single_layer.compute()
    # get garray and incident field from the cdm computation
    G_array_single_layer = np.array(cdm_single_layer.G_set)
    incident_field_array_single_layer = cdm_single_layer.incident_field_array
    area_unit_cell = lattice_area(lattice_vectors)
    num_layers = 11


    # Fix all non-fitting parameters
    single_layer_model = partial(
        extinction_cdm_single_layer,
        ne_dispersion=df_heda["ne(10^26m^-3)"].iloc[1],
        incident_field_array=incident_field_array_single_layer,
        nc_diameter=13.4,
        bulk_mean_free_path=df_heda["mean_free_path(nm)"].iloc[1],
        eps_m=eps_il,
        eps_inf=eps_inf,
        G_array=G_array_single_layer,
        area_unit_cell=area_unit_cell,
        incident_theta_il=incident_theta_il,
        n_layers=num_layers
    )
    return single_layer_model, voltage_list, freq_list, extinction_list


@dataclass
class CurveFit:
    """One curve's independent optimizer, prediction, and local uncertainty."""

    optimizer: OptimizeResult
    prediction: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    statistics: dict[str, object]
    uncertainty_status: str


def goodness_of_fit(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_parameters: int,
) -> dict[str, object]:
    """Calculate the original goodness-of-fit measures for the supplied data."""
    residuals = predicted - observed
    n_points = observed.size
    dof = n_points - n_parameters
    sse = float(residuals @ residuals)
    sst = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "R2": float(1 - sse / sst) if sst > 0 else None,
        "RMSE": float(np.sqrt(sse / n_points)),
        "MAE": float(np.mean(np.abs(residuals))),
        "SSE": sse,
        "residual_std": float(np.sqrt(sse / dof)) if dof > 0 else None,
        "n_data_points": int(n_points),
        "n_parameters": int(n_parameters),
        "degrees_of_freedom": int(dof),
    }


def parameter_uncertainty(
    result: OptimizeResult,
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Estimate local OLS covariance using this curve's full raw Jacobian.

    Each curve has three fitted parameters, so its residual variance and
    Student-t intervals use N_i - 3 degrees of freedom. Ordinary symmetric
    intervals are withheld for failed, rank-deficient, or boundary fits.
    """
    n_parameters = result.x.size
    dof = result.fun.size - n_parameters
    covariance = np.full((n_parameters, n_parameters), np.nan)
    standard_errors = np.full(n_parameters, np.nan)
    ci_lower = np.full(n_parameters, np.nan)
    ci_upper = np.full(n_parameters, np.nan)
    lower, upper = bounds
    boundary_tolerance = BOUNDARY_RELATIVE_TOLERANCE * (upper - lower)

    if not result.success:
        status = "unavailable: optimization did not converge"
    elif dof <= 0:
        status = "unavailable: no residual degrees of freedom"
    elif (
        np.any(result.active_mask != 0)
        or np.any(result.x - lower <= boundary_tolerance)
        or np.any(upper - result.x <= boundary_tolerance)
    ):
        status = "unavailable: fitted parameter at or near a bound"
    else:
        # All three parameters are explicit in the residual, so this Jacobian
        # includes the offset column and its correlations with ne_ratio/eta.
        _, singular_values, right_vectors = np.linalg.svd(
            result.jac, full_matrices=False
        )
        if (
            singular_values.size != n_parameters
            or not np.all(np.isfinite(singular_values))
            or singular_values[0] == 0
            or singular_values[-1] <= COVARIANCE_RCOND * singular_values[0]
        ):
            status = "unavailable: numerically rank-deficient Jacobian"
        else:
            residual_variance = float(result.fun @ result.fun) / dof
            covariance = residual_variance * (
                (right_vectors.T / singular_values**2) @ right_vectors
            )
            standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            t_critical = student_t.ppf(0.975, dof)
            ci_lower = result.x - t_critical * standard_errors
            ci_upper = result.x + t_critical * standard_errors
            status = "available: local linear 95% Student-t intervals"
    return covariance, standard_errors, ci_lower, ci_upper, status


def fit_curve(
    physical_model: Callable[..., np.ndarray],
    frequency: np.ndarray,
    observed: np.ndarray,
    *,
    initial_guess: Sequence[float] = FIT_INITIAL_GUESS,
    lower_bounds: Sequence[float] = FIT_LOWER_BOUNDS,
    upper_bounds: Sequence[float] = FIT_UPPER_BOUNDS,
    max_nfev: int = MAX_NFEV,
) -> CurveFit:
    """Fit this curve's (ne_ratio, eta, offset) with no other curve's data."""
    frequency = np.asarray(frequency, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if frequency.ndim != 1 or observed.shape != frequency.shape:
        raise ValueError("Frequency and extinction must be matching 1D arrays.")
    if observed.size <= len(PARAMETER_NAMES):
        raise ValueError("Each curve needs more than three points for uncertainties.")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(observed)):
        raise ValueError("Frequency and extinction must contain only finite values.")

    initial = np.asarray(initial_guess, dtype=float).copy()
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if any(values.shape != (3,) for values in (initial, lower, upper)):
        raise ValueError("Supply three initial values and bounds: ne_ratio, eta, offset.")
    if not all(np.all(np.isfinite(values)) for values in (initial, lower, upper)):
        raise ValueError("Initial values and bounds must be finite.")
    if np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Bounds must be ordered and contain the initial guess.")
    if not isinstance(max_nfev, int) or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer.")

    def residual(parameters: np.ndarray) -> np.ndarray:
        prediction = np.asarray(physical_model(frequency, *parameters), dtype=float)
        if prediction.shape != observed.shape:
            raise ValueError("The model must return one extinction value per frequency.")
        if not np.all(np.isfinite(prediction)):
            raise ValueError("The physical model returned non-finite extinction values.")
        return prediction - observed

    # Offset is an explicit third parameter in each separate optimization.
    # SciPy selects bounded three-point finite differences for all parameters.
    result = least_squares(
        residual,
        x0=initial,
        jac="3-point",
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=max_nfev,
    )
    prediction = observed + result.fun
    covariance, errors, ci_lower, ci_upper, status = parameter_uncertainty(
        result, (lower, upper)
    )
    statistics = goodness_of_fit(observed, prediction, len(PARAMETER_NAMES))
    statistics.update({
        "optimization_success": bool(result.success),
        "optimization_message": str(result.message),
        "n_function_evaluations": int(result.nfev),
        "uncertainty_status": status,
    })
    return CurveFit(
        result, prediction, covariance, errors, ci_lower, ci_upper,
        statistics, status,
    )


def fit_independent_curves(
    physical_model: Callable[..., np.ndarray],
    frequencies: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
    *,
    max_nfev: int = MAX_NFEV,
) -> list[CurveFit]:
    """Start a fresh bounded fit for every curve, using its actual frequency grid."""
    if len(frequencies) == 0 or len(frequencies) != len(observations):
        raise ValueError("Supply matching, nonempty lists of frequencies and curves.")
    fits = []
    for index, (frequency, observed) in enumerate(zip(frequencies, observations)):
        LOGGER.info("Fitting curve %d of %d", index + 1, len(observations))
        fit = fit_curve(physical_model, frequency, observed, max_nfev=max_nfev)
        if not fit.optimizer.success:
            LOGGER.warning("Curve %d: %s", index + 1, fit.optimizer.message)
        LOGGER.info(
            "Curve %d: ne_ratio=%.6g, eta=%.6g, offset=%.6g; %s",
            index + 1, *fit.optimizer.x, fit.uncertainty_status,
        )
        fits.append(fit)
    return fits


def save_fit_results(
    output_dir: Path,
    voltages: Sequence[float],
    frequencies: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
    fits: Sequence[CurveFit],
) -> pd.DataFrame:
    """Save the original result files plus a per-voltage statistics table."""
    if not len(fits) == len(voltages) == len(frequencies) == len(observations) > 0:
        raise ValueError("Voltages, frequencies, observations, and fits must align.")
    if not all(np.array_equal(frequency, frequencies[0]) for frequency in frequencies):
        raise ValueError("The wide fitted_curves.csv output needs matching frequency grids.")
    parameter_rows = []
    curve_statistics = []
    curves = pd.DataFrame({"wavenumber_cm-1": frequencies[0]})
    for index, (voltage, observed, fit) in enumerate(zip(voltages, observations, fits)):
        for column, name in enumerate(PARAMETER_NAMES):
            parameter_rows.append({
                "parameter": f"{name}_{index + 1}",
                "estimate": float(fit.optimizer.x[column]),
                "std_error": float(fit.standard_errors[column]),
                "CI_95_lower": float(fit.ci_lower[column]),
                "CI_95_upper": float(fit.ci_upper[column]),
                "curve_index": index + 1,
                "voltage_V": float(voltage),
                "uncertainty_status": fit.uncertainty_status,
            })
        curve_statistics.append({
            "curve_index": index + 1,
            "voltage_V": float(voltage),
            **dict(zip(PARAMETER_NAMES, map(float, fit.optimizer.x))),
            **fit.statistics,
        })
        label = f"{voltage:+g}V"
        curves[f"{label}_experimental"] = observed
        curves[f"{label}_fitted"] = fit.prediction
        curves[f"{label}_residual"] = fit.prediction - observed

    # Aggregate statistics are descriptive; every covariance above uses only
    # that curve's residual variance, never a pooled variance across voltages.
    statistics = goodness_of_fit(
        np.concatenate(observations),
        np.concatenate([fit.prediction for fit in fits]),
        len(PARAMETER_NAMES) * len(fits),
    )
    statistics.update({
        "fit_mode": "independent",
        "n_curves": len(fits),
        "offsets": [float(fit.optimizer.x[2]) for fit in fits],
        "optimization_success": all(fit.optimizer.success for fit in fits),
        "optimization_message": [str(fit.optimizer.message) for fit in fits],
        "per_curve": curve_statistics,
    })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = pd.DataFrame(parameter_rows)
    parameters.to_csv(output_dir / "fit_parameters.csv", index=False)
    curves.to_csv(output_dir / "fitted_curves.csv", index=False)
    pd.DataFrame(curve_statistics).to_csv(
        output_dir / "fit_statistics_per_curve.csv", index=False
    )
    with (output_dir / "fit_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=4, allow_nan=False)
    LOGGER.info("Fitting parameters and uncertainties:\n%s", parameters.to_string(index=False))
    LOGGER.info(
        "Aggregate R2=%s, RMSE=%.6g, MAE=%.6g, SSE=%.6g",
        statistics["R2"], statistics["RMSE"], statistics["MAE"], statistics["SSE"],
    )
    LOGGER.info("Results saved to %s", output_dir.resolve())
    return curves


def plot_fit_comparison(
    curves: pd.DataFrame,
    voltages: Sequence[float],
    output_dir: Path,
    *,
    show_plot: bool = True,
) -> None:
    """Preserve the solid-data/dashed-fit plot and reversed FTIR frequency axis."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.rainbow_r(np.linspace(0.05, 0.95, len(voltages)))
    for voltage, color in zip(voltages, colors):
        label = f"{voltage:+g}V"
        for kind, linestyle, suffix in (
            ("experimental", "-", "Exp"), ("fitted", "--", "Fit")
        ):
            ax.plot(
                curves["wavenumber_cm-1"], curves[f"{label}_{kind}"],
                color=color, linestyle=linestyle, linewidth=1.5,
                label=f"{label} {suffix}",
            )
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Extinction")
    ax.set_title("Experimental vs CDM Fitted Extinction")
    ax.set_xlim(4500, 1700)
    ax.legend(fontsize=9, ncol=2, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "fit_comparison.png", dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def run_fit(
    input_dir: Path = Path("."),
    output_dir: Path = Path("fit_results"),
    *,
    max_nfev: int = MAX_NFEV,
    show_plot: bool = True,
) -> list[CurveFit]:
    """Load the original data/model, fit each voltage, and export results."""
    physical_model, voltages, frequencies, observations = prepare_model_and_data(input_dir)
    fits = fit_independent_curves(
        physical_model, frequencies, observations, max_nfev=max_nfev
    )
    curves = save_fit_results(output_dir, voltages, frequencies, observations, fits)
    plot_fit_comparison(curves, voltages, output_dir, show_plot=show_plot)
    return fits


def main() -> None:
    """Run the standalone script with the original filenames and defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("fit_results"))
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening it.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fits = run_fit(
        args.input_dir, args.output_dir,
        max_nfev=args.max_nfev, show_plot=not args.no_show,
    )
    if not all(fit.optimizer.success for fit in fits):
        raise SystemExit("At least one curve did not converge; inspect fit_statistics.json.")


if __name__ == "__main__":
    main()
