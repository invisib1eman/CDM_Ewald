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
from cdm_ewald.electric_field import build_real_space_coupling_tensor, build_spread_matrix, build_gather_matrix, reciprocal_space_field_function, total_field

class CDM_Ewald_pme():
    '''
    Calculate the dipoles of particles for an incident light wave with angular wavenumber k and polarization E_0. Works for lattices with few particles, and the real space cutoff could be comparable to box size.
    
    **Parameters**
        *box size*
            L1,L2
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
    def __init__(self,box,positions,polarizability,dimensionless_angular_wavenumber, wavevector_direction=np.array([0,0,1]),polarization_direction=np.array([1,0,0]),radius=1,tol=1e-6,xi=0.5,mu=0.5):# {{{
        self.indent_level = 0
        



        self.errortol = tol
        self.num_particles = None

        self.xi = xi
        self.mu = mu

        self.box = box
        self.positions = positions
        self.radius = radius
        self.r_hardcore = self.radius
        self.polarizability = polarizability
        self.dimensionless_angular_wavenumber = dimensionless_angular_wavenumber
        self.wavevector_direction = wavevector_direction
        self.polarization_direction = polarization_direction
        self.num_particles = self.positions.shape[0] 
        self.num_wavenumbers = self.dimensionless_angular_wavenumber.shape[0]
        self.dips = np.zeros((self.num_wavenumbers,self.num_particles,3), dtype=np.complex128)
        self.extinction_efficiency = np.zeros(self.num_wavenumbers,dtype=np.float64)
        self.incident_field_array = build_incident_field(self.wavevector_direction,
                self.dimensionless_angular_wavenumber,
                self.positions,
                self.polarization_direction)
        self._precalculation()
    def _precalculation(self):
        self.r_cut = np.sqrt(-np.log(self.errortol)) / self.xi
        self.k_cut = 2*self.xi*np.sqrt(-np.log(self.errortol))
        self.num_grid = np.ceil(1+self.box*self.k_cut/np.pi).astype(int)
        self.grid_spacing = self.box/self.num_grid
        
        self.pairs, self.overlap = neighbor_pairs(
            self.positions,
            self.box,
            self.r_cut,
            self.r_hardcore,
        )
        self.spread_matrix = build_spread_matrix(self.positions, self.box, self.num_grid, self.xi, self.mu, self.errortol)
        self.gather_matrix = build_gather_matrix(self.spread_matrix, self.box, self.num_grid)
    def set_positions(self, positions):

        self.positions = np.asarray(positions, dtype=float)

        self.num_particles = self.positions.shape[0] 
    def compute(self):
   
        """
        Solve

            alpha^-1 P - E_dipole = E0

        Returns
        -------
        p : (N,3)
            dipole moments
        """
        for kindex, k in enumerate(self.dimensionless_angular_wavenumber):
            print(f"running {k}")
            polarizability = self.polarizability[kindex]
            incident_field = self.incident_field_array[kindex].flatten()
            dip_guess = incident_field / polarizability
            real_space_sparse_tensor = build_real_space_coupling_tensor(self.positions, self.pairs, self.box, k, self.xi)
            def solve_kernel(dipoles):
                H = total_field(dipoles, k, polarizability, self.box, self.xi, self.mu, real_space_sparse_tensor, self.spread_matrix, self.gather_matrix, self.num_grid)
                ret = H.flatten()
                return ret
            solve = LinearOperator(2*[3*self.num_particles], matvec = solve_kernel,dtype = np.complex128)
            restart = min([self.num_particles*3,10])
            maxiter = min([self.num_particles*3,100])
            #start = time.time()
            dip,info = gmres(solve,incident_field,x0 = dip_guess,rtol=self.errortol,
                                    restart = restart, maxiter = maxiter)
            dip = dip.reshape(self.num_particles,3)


            self.dips[kindex] = dip