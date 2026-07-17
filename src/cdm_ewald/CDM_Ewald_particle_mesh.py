import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import gmres,LinearOperator,cgs
import time
import itertools
import warnings
from cdm_ewald.ewald_kernels import tensor_kernel, real_space_tensor, self_term_scalar
from cdm_ewald.neighbor_list import neighbor_pairs


class CDM_Ewald
