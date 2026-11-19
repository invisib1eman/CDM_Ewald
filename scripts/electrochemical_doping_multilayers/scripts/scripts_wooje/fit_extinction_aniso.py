"""Fit 60-deg p-pol extinction spectra with the anisotropic-TMM model.

Strategy
--------
1. For the most-modulated voltage (the last entry of SEQS, -2 V), fit all 7 parameters:
       A_ip, nu0_ip, gamma_ip, A_oop, nu0_oop, gamma_oop, offset
   using differential_evolution (global, no initial guess required).
2. For the remaining voltages, FIX gamma_ip and gamma_oop to their -2V values
   (linewidths are microscopic damping rates and should not vary strongly with
   applied voltage for the same sample).  Fit the other 5 parameters, seeding
   the optimizer with the neighboring voltage's solution.

Why two stages?
    Widths and amplitudes are correlated; fixing gamma at the highest-SNR
    voltage prevents drift and makes per-voltage A and nu0 trends interpretable.

The molecular-vibration window (2900-3250 cm^-1) is masked during the fit so
that the broad Drude-Lorentz fit is not distorted by the EMIM C-H stretch.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from tests.electrochemical_doping_multilayers.scripts.scripts_wooje.tmm_aniso import calc_extinction   # engine

# -------------------- USER INPUTS --------------------
# Example: 13.4 nm, 8-layer film on CaF2, April 28 data (60 deg p-pol).
DATA_DIR = ("/Users/wooje92/Library/CloudStorage/OneDrive-TheUniversityofTexasatAustin/"
            "raw data/FTIR/20260428 ITO on CaF2 again")
PATTERN  = ("230123-1 8layers_260414sample on CaF2_0_8cmdiamter_CaF2_"
            "60degree_ppol_IL_seq{}.0.dpt")
# Sequence numbers listed in VOLTAGE ORDER (+2 V first, -2 V last).
# The measurement order does not follow voltage order, so map explicitly.
SEQS           = [10, 11, 9, 8, 12, 13, 14, 15]
VOLTAGE_LABELS = ["+2 V", "+1 V", "0 V", "-1 V", "-1.2 V", "-1.5 V", "-1.8 V", "-2 V"]

# sample-specific physical constants
EPS_INF     = 2.2        # high-frequency permittivity of the ITO film
D_ITO_NM    = 104        # total ITO thickness (nm) ~ N_layers * (d_NC + gap)
THETA_DEG   = 60.0       # angle of incidence in air
N_CAF2      = 1.44
N_IL        = 1.42
D_IL_UM     = 1.0        # IL gel thickness in microns

FIT_RANGE     = (1700, 5500)     # wavenumber range used for the fit [cm^-1]
PLOT_RANGE    = (1700, 4500)
MOL_MASK      = (2900, 3250)     # exclude EMIM C-H stretch during fit

# bounds for stage-1 (full 7-param) fit
BOUNDS_FULL = [
    (1e5, 5e8),     # A_ip
    (1500, 2800),   # nu0_ip      (in-plane LSPR)
    (500, 3000),    # gamma_ip
    (1e5, 5e8),     # A_oop
    (2000, 4500),   # nu0_oop     (out-of-plane Berreman mode)
    (500, 3000),    # gamma_oop
    (-0.15, 0.15),  # baseline offset
]
# -----------------------------------------------------


def load_data():
    """Load all voltages in the order given by SEQS (+2 V first, -2 V last)."""
    data = {}
    for s in SEQS:
        data[s] = np.loadtxt(os.path.join(DATA_DIR, PATTERN.format(s)))
    return data, list(SEQS)


def prepare(d):
    wn, ext = d[:, 0], d[:, 1]
    m = (wn >= FIT_RANGE[0]) & (wn <= FIT_RANGE[1])
    wn_r = wn[m][::-1]
    ext_r = ext[m][::-1]
    mol = ~((wn_r >= MOL_MASK[0]) & (wn_r <= MOL_MASK[1]))
    return wn_r, ext_r, mol


def fit_full(wn_fit, ext_fit):
    """Stage-1: 7 free parameters, global optimization."""
    def cost(p):
        A_ip, nu0_ip, g_ip, A_oop, nu0_oop, g_oop, off = p
        m = calc_extinction(wn_fit, THETA_DEG, EPS_INF,
                            A_ip, nu0_ip, g_ip,
                            A_oop, nu0_oop, g_oop,
                            D_ITO_NM, N_IL, D_IL_UM, N_CAF2, off)
        return np.sum((ext_fit - m) ** 2)
    res = differential_evolution(cost, BOUNDS_FULL, seed=42,
                                  maxiter=5000, tol=1e-12, polish=True,
                                  popsize=30, mutation=(0.5, 1.5),
                                  recombination=0.9)
    return res.x, res.fun


def fit_fixed_gamma(wn_fit, ext_fit, g_ip, g_oop, x0_full):
    """Stage-2: 5 free parameters (A_ip, nu0_ip, A_oop, nu0_oop, offset),
    gammas fixed to stage-1 values, seeded with the neighboring voltage."""
    bounds = [BOUNDS_FULL[0], BOUNDS_FULL[1], BOUNDS_FULL[3],
              BOUNDS_FULL[4], BOUNDS_FULL[6]]
    def cost(p):
        A_ip, nu0_ip, A_oop, nu0_oop, off = p
        m = calc_extinction(wn_fit, THETA_DEG, EPS_INF,
                            A_ip, nu0_ip, g_ip,
                            A_oop, nu0_oop, g_oop,
                            D_ITO_NM, N_IL, D_IL_UM, N_CAF2, off)
        return np.sum((ext_fit - m) ** 2)

    # seed the initial population around x0 (neighbor voltage) for speed
    ndim = 5
    popsize = 25
    rng = np.random.RandomState(42)
    pop = rng.uniform(size=(popsize, ndim))
    for i in range(ndim):
        lo, hi = bounds[i]
        pop[:, i] = lo + pop[:, i] * (hi - lo)
    x0 = np.clip(np.array([x0_full[0], x0_full[1], x0_full[3],
                           x0_full[4], x0_full[6]]),
                 [b[0] for b in bounds], [b[1] for b in bounds])
    pop[0] = x0
    for j in range(1, 5):
        pop[j] = np.clip(x0 * (1 + 0.05 * rng.randn(ndim)),
                         [b[0] for b in bounds], [b[1] for b in bounds])

    res = differential_evolution(cost, bounds, seed=42, init=pop,
                                  maxiter=3000, tol=1e-12, polish=True)
    v = res.x
    return np.array([v[0], v[1], g_ip, v[2], v[3], g_oop, v[4]]), res.fun


# =========================== MAIN ===========================
if __name__ == "__main__":
    data, sorted_seqs = load_data()
    NV = len(sorted_seqs)

    # 1) -2V full fit
    wn_r, ext_r, mol = prepare(data[sorted_seqs[-1]])
    popt_m2, cost_m2 = fit_full(wn_r[mol], ext_r[mol])
    gamma_ip_fix, gamma_oop_fix = popt_m2[2], popt_m2[5]

    fits = {NV - 1: (popt_m2, wn_r, ext_r)}

    # 2) other voltages, propagating from -2V towards +2V
    for idx in range(NV - 2, -1, -1):
        wn_r, ext_r, mol = prepare(data[sorted_seqs[idx]])
        prev = fits[idx + 1][0]
        popt, cost_v = fit_fixed_gamma(wn_r[mol], ext_r[mol],
                                        gamma_ip_fix, gamma_oop_fix, prev)
        fits[idx] = (popt, wn_r, ext_r)

    # -------- results table --------
    print(f"\nANISOTROPIC TMM FIT  eps_inf={EPS_INF}  d={D_ITO_NM} nm  theta={THETA_DEG} deg")
    print(f"{'V':>6s} {'A_ip':>10s} {'nu0_ip':>7s} {'g_ip':>6s}  "
          f"{'A_oop':>10s} {'nu0_oop':>8s} {'g_oop':>6s}  {'off':>7s}  {'nu_LO':>6s}")
    for idx in range(NV):
        p = fits[idx][0]
        nu_LO = np.sqrt(p[4]**2 + p[3]/EPS_INF)        # formal LO frequency
        print(f"{VOLTAGE_LABELS[idx]:>6s} "
              f"{p[0]:>10.0f} {p[1]:>7.0f} {p[2]:>6.0f}  "
              f"{p[3]:>10.0f} {p[4]:>8.0f} {p[5]:>6.0f}  "
              f"{p[6]:>7.4f}  {nu_LO:>6.0f}")

    # -------- plot data + fit --------
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.get_cmap("rainbow_r")(np.linspace(0.05, 0.95, NV))
    for idx in range(NV):
        p, wn_r, ext_r = fits[idx]
        pm = (wn_r >= PLOT_RANGE[0]) & (wn_r <= PLOT_RANGE[1])
        model = calc_extinction(wn_r, THETA_DEG, EPS_INF, *p[:6],
                                 D_ITO_NM, N_IL, D_IL_UM, N_CAF2, p[6])
        ax.plot(wn_r[pm], ext_r[pm], color=colors[idx],
                lw=1.2, label=f"{VOLTAGE_LABELS[idx]} data")
        ax.plot(wn_r[pm], model[pm], "--", color=colors[idx], lw=0.9, alpha=0.75)
        ax.plot(p[1], model[np.argmin(np.abs(wn_r - p[1]))],
                 "o", color=colors[idx], ms=5)         # IP marker
        ax.plot(p[4], model[np.argmin(np.abs(wn_r - p[4]))],
                 "s", color=colors[idx], ms=5)         # OOP marker
    ax.set_xlim(*PLOT_RANGE[::-1])
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Extinction")
    ax.set_title("Anisotropic-TMM fit (circles = IP, squares = OOP)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("tmm_aniso_fit.png", dpi=200, bbox_inches="tight")
    print("\nSaved tmm_aniso_fit.png")
