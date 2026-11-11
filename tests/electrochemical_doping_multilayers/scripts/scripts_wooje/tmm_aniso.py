"""Transfer-matrix method for an anisotropic uniaxial layer at oblique incidence
with p-polarization. Optic axis is normal to the film (z-direction).

Stack (top -> bottom, incidence from air):
    Air   (n = 1)
      |----- Layer 1 : IL gel      (isotropic, n_IL, thickness d_IL)
      |----- Layer 2 : ITO NC film (uniaxial, eps_ip, eps_oop, thickness d_ITO)
    CaF2 substrate  (n = n_CaF2)   -- exit half-space

Reference: Air -> CaF2 directly (no film, no IL).  The CaF2 back-surface loss is
common to both measurements and cancels in the extinction ratio.

Extinction:
    E(nu) = -log10( T_sample(nu) / T_ref(nu) ) + offset

Dielectric model (anisotropic Lorentz, one oscillator each direction):
    eps_{||}(nu) = eps_inf + A_ip  / (nu0_ip^2  - nu^2 - i*gamma_ip*nu)
    eps_{perp}(nu) = eps_inf + A_oop / (nu0_oop^2 - nu^2 - i*gamma_oop*nu)

Physics notes
--------------
For p-polarized light in a uniaxial layer with optic axis along z, the
extraordinary wave sees an effective permittivity

    eps_eff(theta) = eps_|| * ( 1 - sin^2(theta)/eps_perp )

so the z-component of the wavevector inside the layer is
    q_z = sqrt( eps_|| * (1 - sin^2 theta / eps_perp) )
Snell's law conserves sin(theta) = sin(theta_0) across all interfaces because
it's the in-plane component of the wavevector that is continuous.

The p-pol admittance is
    eta_p = eps / q_z
(The s-pol admittance would be eta_s = q_z.)  This is the convention where the
characteristic matrix is
    M = [[ cos(delta),          -i sin(delta) / eta ],
         [-i eta sin(delta),      cos(delta)         ]],   delta = (2*pi*nu) * q_z * d

The total stack matrix is the ORDERED product from the side closest to the
incoming wave:  M_total = M_IL @ M_ITO.

Amplitude transmission:
    [B; C] = M_total @ [1; eta_exit]
    t = 2 * eta_in / (eta_in * B + C)
    T = Re(eta_exit)/Re(eta_in) * |t|^2

All arrays are vectorized over wavenumber nu (cm^-1).
"""
from __future__ import annotations
import numpy as np


def eps_lorentz(nu, eps_inf, A, nu0, gamma):
    """Single-oscillator Lorentzian dielectric function."""
    return eps_inf + A / (nu0**2 - nu**2 - 1j * gamma * nu)


def calc_extinction(
    nu,                 # wavenumber array [cm^-1]
    theta_deg,          # angle of incidence in AIR
    eps_inf,
    A_ip, nu0_ip, gamma_ip,          # in-plane oscillator
    A_oop, nu0_oop, gamma_oop,       # out-of-plane oscillator
    d_ITO_nm,           # ITO film total thickness [nm]
    n_IL, d_IL_um,      # IL gel index and thickness [um]
    n_CaF2,             # substrate index
    offset=0.0,         # additive baseline in extinction
):
    """
    p-pol extinction of (Air | IL | ITO_aniso | CaF2) referenced to (Air | CaF2).
    Returns real-valued extinction array.
    """
    nu = np.asarray(nu, dtype=float)
    sin2 = np.sin(np.deg2rad(theta_deg)) ** 2
    k0 = 2.0 * np.pi * nu           # vacuum wavenumber in 2*pi/cm

    # ---- dielectric functions ----
    eps_IL   = n_IL ** 2
    eps_CaF2 = n_CaF2 ** 2
    eps_ip   = eps_lorentz(nu, eps_inf, A_ip,  nu0_ip,  gamma_ip)
    eps_oop  = eps_lorentz(nu, eps_inf, A_oop, nu0_oop, gamma_oop)

    # ---- thicknesses in cm (same unit system as 1/nu for delta = k0*qz*d) ----
    d_ITO = d_ITO_nm * 1e-7
    d_IL  = d_IL_um  * 1e-4

    # ---- admittances (p-pol : eta = eps / q_z) and phase thicknesses ----
    # Air half-space
    qz_air  = np.sqrt(1.0 - sin2 + 0j)
    eta_air = 1.0 / qz_air

    # IL (isotropic)
    qz_IL   = np.sqrt(eps_IL - sin2 + 0j)
    eta_IL  = eps_IL / qz_IL
    delta_IL = k0 * qz_IL * d_IL

    # ITO (uniaxial, extraordinary wave for p-pol)
    qz_ITO   = np.sqrt(eps_ip * (1.0 - sin2 / eps_oop) + 0j)
    eta_ITO  = eps_ip / qz_ITO
    delta_ITO = k0 * qz_ITO * d_ITO

    # CaF2 exit
    qz_CaF2  = np.sqrt(eps_CaF2 - sin2 + 0j)
    eta_CaF2 = eps_CaF2 / qz_CaF2

    # ---- characteristic matrices ----
    def char_matrix(delta, eta):
        c, s = np.cos(delta), np.sin(delta)
        return c, -1j * s / eta, -1j * eta * s, c      # M11, M12, M21, M22

    a11, a12, a21, a22 = char_matrix(delta_IL,  eta_IL)
    b11, b12, b21, b22 = char_matrix(delta_ITO, eta_ITO)

    # total = IL * ITO (wave enters IL first)
    M11 = a11 * b11 + a12 * b21
    M12 = a11 * b12 + a12 * b22
    M21 = a21 * b11 + a22 * b21
    M22 = a21 * b12 + a22 * b22

    # transmission through sample stack
    B = M11 + M12 * eta_CaF2
    C = M21 + M22 * eta_CaF2
    t_sample = 2.0 * eta_air / (eta_air * B + C)
    T_sample = np.real(eta_CaF2) / np.real(eta_air) * np.abs(t_sample) ** 2

    # reference : air -> CaF2 directly
    t_ref = 2.0 * eta_air / (eta_air + eta_CaF2)
    T_ref = np.real(eta_CaF2) / np.real(eta_air) * np.abs(t_ref) ** 2

    extinction = -np.log10(np.clip(T_sample / T_ref, 1e-10, None)) + offset
    return np.real(extinction)
