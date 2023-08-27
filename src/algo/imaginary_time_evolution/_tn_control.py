from containers import TNDimensions, ITEConfig, Config
from tensor_networks import TensorNetwork, TensorNode, KagomeTN, UnitCell
from utils import lists, logs

import numpy as np


def _calc_environment_equivalent_matrix(environment_tensors:list[np.ndarray]) -> np.ndarray:

    ## Parse inputs:
    num_tensors = len(environment_tensors)
    d_physical = environment_tensors[0].shape[2]
    d_virutal  = environment_tensors[0].shape[0]

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


def kagome_tn_from_unit_cell(unit_cell:UnitCell, dims:TNDimensions) -> KagomeTN:
    assert dims.core_size == core.original_lattice_dims[0] == core.original_lattice_dims[1]
    repeats = assertions.odd(dims.big_lattice_size/dims.core_size)
    return repeat_core(core, repeats=repeats)



def update_unit_cell_tensors(
    mode_tn:KagomeTN,
    core1:TensorNode,
    core2:TensorNode,
    environment_tensors:list[np.ndarray],
    ite_config:ITEConfig,
    delta_t:float,
    logger:logs.Logger
)->tuple[
    TensorNode,  # core1
    TensorNode,  # core2 
    float, # energy
    MatrixMetrics # env_data
]:

    # environment_mat = _calc_environment_equivalent_matrix(environment_tensors)
    # env_metrics = calc_metrics(environment_mat)

    ## Collect data:
    #  Get Time Evolution Operator
    h, g = get_imaginary_time_evolution_operator(ite_config.interaction_hamiltonain, delta_t)
    # dimensions:
    d_virtual = mode_tn.tensor_dims.virtual
    # Get physical core tensors:
    ti = core1.physical_tensor
    tj = core2.physical_tensor
    assert isinstance(ti, np.ndarray)
    assert isinstance(tj, np.ndarray)

    ## Get metrics
    rdm_before = ite.rho_ij(ti, tj, mps_env=environment_tensors)
    env_metrics = calc_metrics(rho_ij_to_rho(rdm_before))    

    ## Check state:
    if env_metrics.hermicity>ENV_HERMICITY_THRESHOLD:
        raise ITEError(f"env_hermicity={env_metrics.hermicity}")
    sum_eigenvalues = assertions.real(env_metrics.sum_eigenvalues)
    if abs(sum_eigenvalues-1)>ENV_HERMICITY_THRESHOLD:
        raise ITEError(f"env is not psd. sum-eigenvalues={sum_eigenvalues}")
    if env_metrics.negativity>0.1:
        raise ITEError(f"env is not psd. negativity={env_metrics.negativity}")    

    ## Update tensors:
    new_ti, new_tj = ite.apply_2local_gate( g=g, Dmax=d_virtual, Ti=ti, Tj=tj, mps_env=environment_tensors )    

    ## Calc energy and updated env metrics:
    rdm_after = ite.rho_ij(new_ti, new_tj, mps_env=environment_tensors)
    energy_after  = np.dot(rdm_after.flatten(),  h.flatten())
    env_metrics = calc_metrics(rho_ij_to_rho(rdm_after))    

    ## Check change
    if DEBUG_MODE and VERBOSE_MODE:
        #
        rdm_diff = norm(rdm_after-rdm_before)/norm(rdm_before)
        s_ = strings.add_color(f"rdm_diff={rdm_diff}", strings.PrintColors.GREEN)
        print(f"        "+s_)
        #
        ti_diff = norm(new_ti-ti)/norm(ti)
        s_ = strings.add_color(f"ti_diff ={ti_diff}", strings.PrintColors.GREEN)
        print(f"        "+s_)
        #
        tj_diff = norm(new_tj-tj)/norm(tj)
        s_ = strings.add_color(f"tj_diff ={tj_diff}", strings.PrintColors.GREEN)
        print(f"        "+s_)

    ## normalize
    new_ti = new_ti / norm(new_ti)
    new_tj = new_tj / norm(new_tj)

    ## keep new data in nodes:
    core1.tensor = new_ti
    core2.tensor = new_tj
    assert core1.is_ket == True
    assert core2.is_ket == True

    return core1, core2, energy_after, env_metrics

