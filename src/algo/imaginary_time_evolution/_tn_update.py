## Get config:
from _config_reader import DEBUG_MODE

from containers import TNDimensions, ITEConfig, Config, MatrixMetrics
from containers.imaginary_time_evolution import HamiltonianFuncAndInputs
from tensor_networks import TensorNetwork, TensorNode, KagomeTNRepeatedUntiCell, EdgeTN, UnitCell, create_kagome_tn
from utils import lists, logs, assertions, prints
from enums import UnitCellFlavor
from _types import PermutationOrdersType


# Used types:
from _error_types import ITEError

# needed libs and algos:
from libs.ITE import g_from_exp_h, apply_2local_gate, rho_ij
from algo.density_matrices import rho_ij_to_rho, calc_metrics

# Other ite stuff:
from algo.imaginary_time_evolution._constants import ENV_HERMICITY_THRESHOLD

# For quick and smart function caching:
import functools  #TODO  use cache

# For math and tensors:
import numpy as np


def get_imaginary_time_evolution_operator(hamiltonian_func:HamiltonianFuncAndInputs, delta_t:float|None) -> tuple[np.ndarray, np.ndarray]: 
    hamiltonian_func = HamiltonianFuncAndInputs.standard(hamiltonian_func)
    if delta_t is None:
        h = hamiltonian_func.call(delta_t=0.0)
        g = None
    else:
        h = hamiltonian_func.call(delta_t=delta_t)
        g = g_from_exp_h(h, delta_t)
    return h, g


def _raise_ite_error_or_print_warning(message:str) -> None:
    if DEBUG_MODE:
        raise ITEError(message) 
    else:
        prints.print_warning(message)


def _check_rdms_metrics(rdm:np.ndarray)->MatrixMetrics:
    env_metrics = calc_metrics(rho_ij_to_rho(rdm))    
    ## Check state:
    if env_metrics.hermicity>ENV_HERMICITY_THRESHOLD:
        _raise_ite_error_or_print_warning(f"env_hermicity={env_metrics.hermicity}")
    sum_eigenvalues = assertions.real(env_metrics.sum_eigenvalues)
    if abs(sum_eigenvalues-1)>ENV_HERMICITY_THRESHOLD:
        _raise_ite_error_or_print_warning(f"env is not psd. sum-eigenvalues={sum_eigenvalues}")
    if env_metrics.negativity>0.1:
        _raise_ite_error_or_print_warning(f"env is not psd. negativity={env_metrics.negativity}")
    return env_metrics


def _calc_original_negativity_ratio(origin_eigen_vals:np.ndarray|None)->float:
    if origin_eigen_vals is None:
        return 0
    
    positive = sum(origin_eigen_vals[origin_eigen_vals>0])
    negative = sum(origin_eigen_vals[origin_eigen_vals<0])
    negative = np.abs(negative)
    ratio = negative / (negative + positive)
    return ratio


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


def _measures_on_edge(t1_new, t2_new, mps_env, h:np.ndarray=None, eigen_values=None) -> tuple[complex, MatrixMetrics]:
    rdm = rho_ij(t1_new, t2_new, mps_env=mps_env)

    if h is None:
        energy = None
    else:
        energy = np.dot(rdm.flatten(),  h.flatten())
        energy /= 2  # Divide by 2 to get effective energy per site

    metrics = _check_rdms_metrics(rdm)
    if eigen_values is not None:
        metrics.other["original_negativity_ratio"] = _calc_original_negativity_ratio(eigen_values)

    return energy, metrics


def invert_permutation(permutation:list[int]) -> list[int]:
	"""
	Given a permutation in the form of a list of integers, find its
	inverse permutation. 
	"""
	inv = np.zeros_like(np.array(permutation))
	inv[permutation] = np.arange(len(inv), dtype=inv.dtype)

	return list(inv)


def _derive_permutation(flavour:UnitCellFlavor, edge_tn:EdgeTN, permutation_orders:PermutationOrdersType)->list[int]:
    old_node = edge_tn.get_nodes_by_cell_flavor(flavour)[0]
    original_permutation_order = permutation_orders[old_node.name]
    reversed_permutation_order = invert_permutation(original_permutation_order)
    assert old_node.is_ket
    full_ket_tensor_permutation = [0]+[i+1 for i in reversed_permutation_order]  # ket tensors have additional physical leg
    return full_ket_tensor_permutation


def _update_unit_cell_tensors_in_canonical_leg_order(
    unit_cell:UnitCell, edge_tn:EdgeTN, permutation_orders:PermutationOrdersType, t1_new:np.ndarray, t2_new:np.ndarray
) -> UnitCell:

    f1, f2 = edge_tn.unit_cell_flavors
    for flavour, tensor in zip([f1, f2], [t1_new, t2_new], strict=True):
        permutation = _derive_permutation(flavour, edge_tn, permutation_orders)
        tensor = tensor.transpose(permutation)
        unit_cell[flavour] = tensor
    
    return unit_cell


def ite_update_unit_cell(
    edge_tn:EdgeTN,
    unit_cell:UnitCell,
    permutation_orders:PermutationOrdersType,
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
    _measures_on_edge(t1, t2, mps_env, h=None, eigen_values=None)

    ## Prepare inputs for time evolution update:
    # Get Time Evolution Operator
    h, g = get_imaginary_time_evolution_operator(ite_config.interaction_hamiltonian, delta_t)
    # Derive dimensions:
    assert (d_virtual := t1.shape[1]) == t1.shape[2] == t1.shape[3] == t1.shape[4]

    ## Perform ITE step on edge:
    t1_new, t2_new, origin_eigen_vals = apply_2local_gate( g=g, Dmax=d_virtual, Ti=t1, Tj=t2, mps_env=mps_env )    

    # Calc energy and calc env metrics:
    energy_after, metrics = _measures_on_edge(t1_new, t2_new, mps_env, h=h, eigen_values=origin_eigen_vals)

    ## normalize
    if ite_config.normalize_tensors_after_update:
        t1_new = t1_new / np.linalg.norm(t1_new)
        t2_new = t2_new / np.linalg.norm(t2_new)

    ## Update tensors:
    unit_cell = _update_unit_cell_tensors_in_canonical_leg_order(unit_cell, edge_tn, permutation_orders, t1_new, t2_new) 
    # Keep copy:
    unit_cell.save()

    return unit_cell, energy_after, metrics


