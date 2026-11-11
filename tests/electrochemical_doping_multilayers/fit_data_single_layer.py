import numpy as np
import numpy as np
from scipy import special
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
from cdm_ewald.CDM_Ewald_direct import CDM_Ewald_direct
from cdm_ewald.lattice_utils import lattice_area
from cdm_ewald.incident_field_builder import build_incident_field



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
    effective_electron_mass = 0.4*electron_mass 

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
    # Your existing calculations ...
    
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
    print(f"plasma freq {plasma_freq} damping {damping_coef}")


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
    k_dimensionless_list = k_real_list*nc_radius*1e-7 # quasistatic multiply by another 1e-2
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
    P = np.linalg.solve(A, E0)

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

# get parameters from HEDA
df_heda = pd.read_excel("heda_parameters.xlsx")
df_heda["plasma_frequency(cm^-1)"], df_heda["damping_coefficient(cm^-1)"] = HEDA_conversion(
    df_heda["diameter(nm)"]*1e-9,
    df_heda["ne(10^26m^-3)"]*1e26,
    df_heda["depletion_layer_thickness(nm)"]*1e-9,
    df_heda["mean_free_path(nm)"]*1e-9,
)
# load extinction spectra for 13.4 nm 
df_extinction = pd.read_csv("13p4nm_8L_CaF2_60deg_ppol_all_voltages.csv")
# freq list
freq_light = df_extinction["wavenumber_cm-1"].to_numpy()
# freq filtered from 1700 to 5500
lower_bond_freq = 1700
higher_bond_freq = 5500
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
# light_freq_min = 1800
# light_freq_max = 5500
# light_freq_intv = 1
# light_freq_list = np.arange(light_freq_min,light_freq_max,light_freq_intv)
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
print(f"incident angle from il is {incident_theta_il*180/np.pi} degree")
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
n_curves = len(extinction_list)
num_layers = 8
# Concatenate experimental data
y_exp = np.concatenate(extinction_list)


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



# Initial guesses: ne1, eta1, ne2, eta2, ..., offset
p0 = [1.0, 0.8] * n_curves + [0.05]

lower = [0.7, 0.6] * n_curves + [0.01]
upper = [1.3, 1.0] * n_curves + [0.1]







N_total = len(y_exp)

# Choose your physical model
# It must accept (freq, ne_ratio, eta, offset)
# physical_model = multi_curve_model
physical_model = single_layer_model


# --------------------------------------------------
# 2. Cache model evaluations
# --------------------------------------------------

cache = {}

def evaluate_curve(i, ne_ratio, eta):
    key = (i, float(ne_ratio), float(eta))

    if key not in cache:
        cache[key] = np.asarray(
            physical_model(
                freq_filtered,
                ne_ratio,
                eta,
                0.0
            ),
            dtype=float
        )

    return cache[key]


def evaluate_all(x):
    predictions = []

    for i in range(n_curves):
        ne = x[2*i]
        eta = x[2*i + 1]

        predictions.append(
            evaluate_curve(i, ne, eta)
        )

    return np.concatenate(predictions)


# --------------------------------------------------
# 3. Eliminate shared offset analytically
# --------------------------------------------------

OFFSET_MIN = 0.01
OFFSET_MAX = 0.10

def optimal_offset(y_sim):
    return np.clip(
        np.mean(y_exp - y_sim),
        OFFSET_MIN,
        OFFSET_MAX
    )


def residual(x):
    y_sim = evaluate_all(x)

    offset = optimal_offset(y_sim)

    return y_sim + offset - y_exp


# --------------------------------------------------
# 4. Custom Jacobian
# --------------------------------------------------

def jacobian(x):
    y_sim = evaluate_all(x)
    offset = optimal_offset(y_sim)

    n_params = len(x)

    J = np.zeros((N_total, n_params))

    # Finite-difference steps in physical parameter units
    steps = [0.005, 0.005]

    for i in range(n_curves):
        start = i * len(freq_filtered)
        end = (i + 1) * len(freq_filtered)

        ne = x[2*i]
        eta = x[2*i + 1]

        E0 = evaluate_curve(i, ne, eta)

        for j in range(2):
            x_perturbed = x.copy()

            param_idx = 2*i + j
            step = steps[j]

            # Stay within parameter bounds
            if x[param_idx] + step <= upper[param_idx]:
                x_perturbed[param_idx] += step
                actual_step = step
            else:
                x_perturbed[param_idx] -= step
                actual_step = -step

            E1 = evaluate_curve(
                i,
                x_perturbed[2*i],
                x_perturbed[2*i + 1]
            )

            derivative = (E1 - E0) / actual_step

            # Derivative of raw predictions
            J[start:end, param_idx] = derivative

            # Shared offset derivative:
            # dc/dparam = -mean(dE/dparam)
            #
            # Only when offset is inside its bounds.
            if OFFSET_MIN < offset < OFFSET_MAX:
                J[:, param_idx] -= (
                    np.sum(derivative) / N_total
                )

    return J


# --------------------------------------------------
# 5. Optimization
# --------------------------------------------------

p0 = np.array(
    [1.0, 0.8] * n_curves,
    dtype=float
)

lower = np.array(
    [0.7, 0.6] * n_curves
)

upper = np.array(
    [1.3, 1.0] * n_curves
)

result = least_squares(
    residual,
    x0=p0,
    jac=jacobian,
    bounds=(lower, upper),
    method="trf",
    x_scale="jac",
    verbose=2,
    max_nfev=100
)

# --------------------------------------------------
# 6. Extract results
# --------------------------------------------------

x_fit = result.x

y_sim_fit = evaluate_all(x_fit)
offset_fit = optimal_offset(y_sim_fit)
# Fitted values
x_fit = result.x
y_sim_fit = evaluate_all(x_fit)
offset_fit = optimal_offset(y_sim_fit)

r = y_sim_fit + offset_fit - y_exp

N = len(r)
p = len(x_fit) + 1  # Include shared offset
dof = N - p

if dof <= 0:
    raise ValueError("Not enough data points for uncertainty estimation.")

# Goodness of fit
SSE = np.sum(r**2)
MSE = SSE / N
RMSE = np.sqrt(MSE)
MAE = np.mean(np.abs(r))

SST = np.sum((y_exp - np.mean(y_exp))**2)
R2 = 1 - SSE / SST if SST > 0 else np.nan

residual_std = np.sqrt(SSE / dof)

# Full Jacobian: physical parameters + shared offset
# result.jac is the profiled Jacobian.
J_full = np.column_stack([
    result.jac,
    np.ones(N)
])
def raw_jacobian(x):
    n_freq = len(freq_filtered)
    n_total = len(y_exp)
    J = np.zeros((n_total, len(x)))

    steps = [0.005, 0.005]

    for i in range(n_curves):
        ne = x[2*i]
        eta = x[2*i + 1]

        E0 = evaluate_curve(i, ne, eta)

        for j in range(2):
            col = 2*i + j
            h = steps[j]

            x1 = x.copy()

            if x[col] + h <= upper[col]:
                x1[col] += h
            else:
                x1[col] -= h

            actual_h = x1[col] - x[col]

            E1 = evaluate_curve(
                i, x1[2*i], x1[2*i + 1]
            )

            start = i * n_freq
            end = (i + 1) * n_freq

            J[start:end, col] = (E1 - E0) / actual_h

    return J


J_raw = raw_jacobian(x_fit)
J_full = np.column_stack([J_raw, np.ones(N)])

# Use SVD-based pseudoinverse for numerical stability
cov = residual_std**2 * np.linalg.pinv(J_full.T @ J_full)

se = np.sqrt(np.maximum(np.diag(cov), 0))

# 95% confidence intervals
t_crit = student_t.ppf(0.975, dof)
ci_lower = np.r_[x_fit, offset_fit] - t_crit * se
ci_upper = np.r_[x_fit, offset_fit] + t_crit * se

parameter_names = [
    name
    for i in range(n_curves)
    for name in (f"ne_ratio_{i+1}", f"eta_{i+1}")
] + ["shared_offset"]

uncertainty_table = pd.DataFrame({
    "parameter": parameter_names,
    "estimate": np.r_[x_fit, offset_fit],
    "SE": se,
    "CI_95_lower": ci_lower,
    "CI_95_upper": ci_upper
})

print(uncertainty_table)

print(f"R²:   {R2:.6f}")
print(f"RMSE: {RMSE:.6f}")
print(f"MAE:  {MAE:.6f}")
print(f"SSE:  {SSE:.6f}")
print(f"Residual standard deviation: {residual_std:.6f}")
# IMPORTANT:
# For uncertainty calculation, use the raw physical
# Jacobian, not the offset-corrected one.
# Recover it using the projection relation only if
# its column means are available separately.

# Output directory
output_dir = Path("fit_results")
output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Fitting parameters and uncertainties
# ==========================================

parameter_names = [
    name
    for i in range(n_curves)
    for name in (f"ne_ratio_{i+1}", f"eta_{i+1}")
] + ["shared_offset"]

parameter_df = pd.DataFrame({
    "parameter": parameter_names,
    "estimate": np.r_[x_fit, offset_fit],
    "std_error": se,
    "CI_95_lower": ci_lower,
    "CI_95_upper": ci_upper
})

parameter_df.to_csv(
    output_dir / "fit_parameters.csv",
    index=False
)

# ==========================================
# 2. Experimental and fitted curves
# ==========================================

curves_df = pd.DataFrame({
    "wavenumber_cm-1": freq_filtered
})

for i, voltage in enumerate(voltage_list):

    E_exp = np.asarray(extinction_list[i])

    E_fit = (
        evaluate_curve(
            i,
            x_fit[2*i],
            x_fit[2*i + 1]
        )
        + offset_fit
    )

    label = f"{voltage:+g}V"

    curves_df[f"{label}_experimental"] = E_exp
    curves_df[f"{label}_fitted"] = E_fit
    curves_df[f"{label}_residual"] = E_fit - E_exp

curves_df.to_csv(
    output_dir / "fitted_curves.csv",
    index=False
)

# ==========================================
# 3. Goodness-of-fit statistics
# ==========================================

r = y_sim_fit + offset_fit - y_exp

N = len(r)
p = len(x_fit) + 1
dof = N - p

SSE = float(np.sum(r**2))
RMSE = float(np.sqrt(np.mean(r**2)))
MAE = float(np.mean(np.abs(r)))

SST = np.sum((y_exp - np.mean(y_exp))**2)
R2 = float(1 - SSE / SST) if SST > 0 else None

residual_std = (
    float(np.sqrt(SSE / dof))
    if dof > 0 else None
)

statistics = {
    "R2": R2,
    "RMSE": RMSE,
    "MAE": MAE,
    "SSE": SSE,
    "residual_std": residual_std,
    "n_data_points": int(N),
    "n_parameters": int(p),
    "degrees_of_freedom": int(dof),
    "shared_offset": float(offset_fit),
    "optimization_success": bool(result.success),
    "optimization_message": result.message
}

with open(output_dir / "fit_statistics.json", "w") as f:
    json.dump(statistics, f, indent=4)

print("Results saved to:", output_dir.resolve())
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.rainbow_r(np.linspace(0.05, 0.95, len(voltage_list)))

for i, voltage in enumerate(voltage_list):
    label = f"{voltage:+g}V"
    color = colors[i]

    # Experimental: solid line
    ax.plot(
        curves_df["wavenumber_cm-1"],
        curves_df[f"{label}_experimental"],
        color=color,
        linestyle="-",
        linewidth=1.5,
        label=f"{label} Exp"
    )

    # Fitted: dashed line, same color
    ax.plot(
        curves_df["wavenumber_cm-1"],
        curves_df[f"{label}_fitted"],
        color=color,
        linestyle="--",
        linewidth=1.5,
        label=f"{label} Fit"
    )

ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Extinction")
ax.set_title("Experimental vs CDM Fitted Extinction")

ax.set_xlim(5500, 1700)  # Reverse FTIR x-axis
ax.legend(
    fontsize=9,
    ncol=2,
    loc="best"
)

ax.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    output_dir / "fit_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()