## Get config:
from _config_reader import DEBUG_MODE

from collections.abc import Iterable
from containers import TNDimensions, ITEConfig, Config, MatrixMetrics
from containers.imaginary_time_evolution import HamiltonianFuncAndInputs
from tensor_networks import TensorNetwork, TensorNode, KagomeTN, EdgeTN, UnitCell, create_kagome_tn
from utils import lists, logs, assertions, prints

# Used types:
from _error_types import ITEError

# needed libs and algos:
from libs.ITE import g_from_exp_h, apply_2local_gate, rho_ij
from algo.density_matrices import rho_ij_to_rho, calc_metrics

# Other ite stuff:
from algo.imaginary_time_evolution._constants import ENV_HERMICITY_THRESHOLD

# For quick and smart function caching:
import functools

import numpy as np


@functools.cache
def get_imaginary_time_evolution_operator(hamiltonian_func:HamiltonianFuncAndInputs, delta_t:float) -> tuple[np.ndarray, np.ndarray]: 
    hamiltonian_func = HamiltonianFuncAndInputs.standard(hamiltonian_func)
    h = hamiltonian_func.call()
    if delta_t is None:
        g = None
    else:
        g = g_from_exp_h(h, delta_t)
    return h, g


def _check_rdms_metrics(rdm:np.ndarray)->MatrixMetrics:
    env_metrics = calc_metrics(rho_ij_to_rho(rdm))    
    ## Check state:
    if env_metrics.hermicity>ENV_HERMICITY_THRESHOLD:
        raise ITEError(f"env_hermicity={env_metrics.hermicity}")
    sum_eigenvalues = assertions.real(env_metrics.sum_eigenvalues)
    if abs(sum_eigenvalues-1)>ENV_HERMICITY_THRESHOLD:
        raise ITEError(f"env is not psd. sum-eigenvalues={sum_eigenvalues}")
    if env_metrics.negativity>0.1:
        # raise ITEError(f"env is not psd. negativity={env_metrics.negativity}") 
        prints.print_warning(f"env is not psd. negativity={env_metrics.negativity}")
    return env_metrics
       

def _calc_environment_equivalent_matrix(environment_tensors:list[np.ndarray]) -> np.ndarray:

    ## Parse inputs:
    num_tensors = len(environment_tensors)
    d_physical = environment_tensors[0].shape[2]
    d_virtual  = environment_tensors[0].shape[0]

    ## Prepare ncon inputs:
    tensors_ids = [i for i, _ in enumerate(environment_tensors)]
    edge_list = []

    for prev, crnt, next in lists.iterate_with_periodic_prev_next_items(tensors_ids):
        tensor_edges = []
        tensor_edges.append(crnt)
        tensor_edges.append(-100-crnt)
        tensor_edges.append(-200-crnt)
        tensor_edges.append(next)

        edge_list.append(tensor_edges)

    ## Get matrix from using ncon:
    env_tensor = ncon(environment_tensors, edge_list)
    assert isinstance(env_tensor, np.ndarray)
    assert len(env_tensor.shape) == num_tensors*2
    assert env_tensor.shape[0] == d_physical
    mat_d = d_physical**num_tensors

    perm_list = []
    half_size = len(env_tensor.shape)//2
    for i in range(half_size):
        perm_list.append(i)
        perm_list.append(i+half_size)

    m = env_tensor.copy()
    # m = m.transpose(perm_list)
    m = m.reshape([mat_d, mat_d])
    # m /= norm(m)

    return m


def update_unit_cell(
    edge_tn:EdgeTN,
    unit_cell:UnitCell,
    ite_config:ITEConfig,
    delta_t:float,
    logger:logs.Logger
)->tuple[
    UnitCell,     # unit_cell
    float,        # energy
    MatrixMetrics # env_data
]:

    ## rdm and metrics before update:
    t1, t2, mps_env = edge_tn.edge_and_environment()
    rdm_before = rho_ij(t1, t2, mps_env=mps_env)
    _check_rdms_metrics(rdm_before)

    ## Prepare inputs for time evolution update:
    # Get Time Evolution Operator
    h, g = get_imaginary_time_evolution_operator(ite_config.interaction_hamiltonian, delta_t)
    # Derive dimensions:
    assert (d_virtual := t1.shape[1]) == t1.shape[2] == t1.shape[3] == t1.shape[4]

    ## Perform ITE step on edge:
    t1_new, t2_new = apply_2local_gate( g=g, Dmax=d_virtual, Ti=t1, Tj=t2, mps_env=mps_env )    

    ## Calc energy and updated env metrics:
    rdm_after = rho_ij(t1_new, t2_new, mps_env=mps_env)
    energy_after = np.dot(rdm_after.flatten(),  h.flatten())
    env_metrics = _check_rdms_metrics(rdm_after)

    ## normalize
    t1_new = t1_new / np.linalg.norm(t1_new)
    t2_new = t2_new / np.linalg.norm(t2_new)

    ## Update tensors:
    f1, f2 = edge_tn.unit_cell_flavors
    unit_cell[f1] = t1_new
    unit_cell[f2] = t2_new

    ## Keep copy 
    unit_cell.save("last_unit_cell")

    return unit_cell, energy_after, env_metrics


