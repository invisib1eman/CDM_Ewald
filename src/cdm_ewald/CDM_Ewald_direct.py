import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import gmres,LinearOperator,cgs
import time
import itertools
import warnings
from cdm_ewald.ewald_kernels import reciprocal_tensor_kernel, real_space_tensor, self_term_scalar, ewald_pair_tensor, build_ewald_matrix
from cdm_ewald.neighbor_list import neighbor_pairs
from cdm_ewald.lattice_utils import check_lattice_basis, reciprocal_lattice_basis, lattice_area, nearest_lattice_length, sorted_lattice_vectors
from cdm_ewald.incident_field_builder import build_incident_field
class CDM_Ewald_direct():
    '''
    Calculate the dipoles of particles for an incident light wave with angular wavenumber k and polarization E_0. Works for lattices with few particles, and the real space cutoff could be comparable to box size.
    
    **Parameters**
        *lattice vectors*
            (3,2) or (2,2) arrays of lattice basis vectors in 2d. If (3,2), z component must be 0.
        *positions*
            positions of the particles
        *polarizability*
            Either a 1D or 2D array specifying the frequency dependent polarizability function of the particles. Here polarizability is defined as 4*pi*(eps_i-eps_m)/(eps_i+2*eps_m)
            If *polarizability* is a 1D array, it is assumed to be of length *num_wavelengths* and to specify the frequency dependent polarizability function for all particles. 
            If *polarizability* is 2D, it is assumed to be of shape *num_wavelengths* x *num_particles* and to specify the frequency dependent polarizability function of each particle individually.
        *dimensionless_angular_wavenumber*
            The angular wavenumber in the medium (k=k_0sqrt(eps_m), where k_0 is the vacuum wavenumber) times the unit length of the system (typically radius of the particles). 
        *wavevector_direction
            The unit vector of the wavevector direction.
        *polarization_direction
            The polarization direction of the incident light. Needs to be perpendicular to the wavevector direction
        *radius*
            The radius of the particles. Defaults to 1 (assumed nondimentionalized).
        
        
        *tol*
            The tolerance for solving the dipoles. Defaults to 0.001.
        
        
    '''
    def __init__(self,lattice_vectors,positions,polarizability,dimensionless_angular_wavenumber, wavevector_direction=np.array([0,0,1]),polarization_direction=np.array([1,0,0]),radius=1,tol=1e-3):# {{{
        self.indent_level = 0
        



        self.errortol = tol
        self.num_particles = None

      

        self.lattice_vectors = lattice_vectors
        self.positions = positions
        self.radius = radius
        self.polarizability = polarizability
        self.dimensionless_angular_wavenumber = dimensionless_angular_wavenumber
        self.wavevector_direction = wavevector_direction
        self.polarization_direction = polarization_direction
        self.num_particles = self.positions.shape[0] 
        self.num_wavenumbers = self.dimensionless_angular_wavenumber.shape[0]
        self.xi_ewald = 0.5
        #np.sqrt(np.pi) / 9.4
        #nearest_lattice_length(self.lattice_vectors)
        self.dips = np.zeros((self.num_wavenumbers,self.num_particles,3), dtype=np.complex128)
        self.extinction_efficiency = np.zeros(self.num_wavenumbers,dtype=np.float64)
        self.incident_field_array = build_incident_field(self.wavevector_direction,
                self.dimensionless_angular_wavenumber,
                self.positions,
                self.polarization_direction)
        self.G_set = []
        self.A_set = []
    def set_positions(self, positions):

        self.positions = np.asarray(positions, dtype=float)

        self.num_particles = self.positions.shape[0] 
    def compute(self):
   
        """
        Solve

            (alpha^-1 I - G) P = E0

        Returns
        -------
        p : (N,3)
            dipole moments
        """



        # ----------------------------------------
        # Build Ewald matrix
        # ----------------------------------------
        
        for wavenumber_idx in range(self.num_wavenumbers):
            print(f"run for {wavenumber_idx}")
            dimensionless_angular_wavenumber = self.dimensionless_angular_wavenumber[wavenumber_idx]
            polarizability = self.polarizability[wavenumber_idx]
            s = np.log(10/self.errortol)
            real_cut = np.sqrt(s/self.xi_ewald**2 + dimensionless_angular_wavenumber**2/(4*self.xi_ewald**2))
            reciprocal_cut = np.sqrt(dimensionless_angular_wavenumber**2 + 4*s*self.xi_ewald**2)
            G = build_ewald_matrix(
                self.num_particles,
                self.positions,
                dimensionless_angular_wavenumber,
                self.wavevector_direction,
                self.xi_ewald,
                self.lattice_vectors,
                real_cut,
                reciprocal_cut
            )
            print(f"G {G}")
             # Polarizability
            # ----------------------------------------
            self.G_set.append(G)

            print(G)
            A = (
                np.eye(3 * self.num_particles, dtype=complex)
                / polarizability
                - G
            )
            self.A_set.append(A)
            # ----------------------------------------
            # RHS
            # ----------------------------------------

            E0 = self.incident_field_array[wavenumber_idx].flatten()

            # ----------------------------------------
            # Solve
            # ----------------------------------------

            P = np.linalg.solve(A, E0)

            self.dips[wavenumber_idx] = P.reshape(self.num_particles, 3)
        self.extinction_efficiency = (
            (1.0 / np.pi)
            * self.dimensionless_angular_wavenumber
            * np.imag(
                np.sum(
                    np.conjugate(self.incident_field_array) * self.dips,
                    axis=(1, 2),
                )
            )
)

    
            
     
            
        