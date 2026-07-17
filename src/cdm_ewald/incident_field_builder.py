import numpy as np

def build_incident_field(

    wavevector_direction,

    dimensionless_angular_wavenumber,

    positions,

    polarization_direction,

):

    """

    Construct the incident electric field for multiple wavenumbers.

    Parameters

    ----------

    wavevector_direction : (3,) array_like

        Unit propagation direction.

    dimensionless_angular_wavenumber : (num_wavenumbers,) array_like

        Array of dimensionless angular wavenumbers.

    positions : (num_particles, 3) array_like

        Particle positions.

    polarization_direction : (3,) array_like

        Polarization vector.

    Returns

    -------

    E : ndarray, shape (num_wavenumbers, num_particles, 3)

        Incident electric field.

    """

    k = np.asarray(dimensionless_angular_wavenumber, dtype=float)

    direction = np.asarray(wavevector_direction, dtype=float)

    positions = np.asarray(positions, dtype=float)

    polarization = np.asarray(polarization_direction, dtype=complex)

    # (num_wavenumbers, 3)

    wave_vectors = k[:, None] * direction

    # (num_wavenumbers, num_particles)

    phase = np.exp(1j * wave_vectors @ positions.T)

    # (num_wavenumbers, num_particles, 3)

    E = phase[:, :, None] * polarization

    return E