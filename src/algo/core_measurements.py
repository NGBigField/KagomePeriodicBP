if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )


## Get config:
from _config_reader import DEBUG_MODE

import numpy as np

from tensor_networks import KagomeTensorNetwork, TensorNode
from tensor_networks.construction import repeat_core
from algo.tensor_network import reduce_tn_to_core_and_environment, calc_edge_environment, calc_unit_cell_expectation_values
from libs.ITE import rho_ij
from physics import pauli
from lattices.directions import BlockSide

from containers import Config
from containers.results import Measurements, Expectations
from copy import deepcopy

from _error_types import BPNotConvergedError

from algo.belief_propagation import belief_propagation, belief_propagation_pashtida

from utils import assertions, logs, errors

from typing import TypeVar


_T = TypeVar('_T')


all_paulis = [pauli.x, pauli.y, pauli.z]
all_puali_names = ['x', 'y', 'z'] 

H = np.matrix(
    [[1,  1],
     [1, -1]]
) / np.sqrt(2)   # Hadamard


def _mean(list_:list[_T])->_T:
    return sum(list_)/len(list_)  #type: ignore


def _get_z_projection(rho:np.ndarray)->complex:
    return rho[0,0] - rho[1,1] 


def measure_xyz_expectation_values_with_tn(tn_stable_around_core:KagomeTensorNetwork, reduce:bool=True, bubblecon_trunc_dim:int=18)->dict[str, float|complex]:
    res = calc_unit_cell_expectation_values(tn_stable_around_core, operators=all_paulis, bubblecon_trunc_dim=bubblecon_trunc_dim, force_real=True, reduce=reduce)
    return dict(x=res[0], y=res[1], z=res[2])


def measure_xyz_expectation_values_with_rdms(rdms:list[np.ndarray])->dict[str, float|complex]:

    ## Collect all projections:
    projections_per_edge : list[dict[str, complex]] = []
    for rdm_ij in rdms:
        rho_i = np.trace(rdm_ij, axis1=2, axis2=3)
        rho_j = np.trace(rdm_ij, axis1=0, axis2=1)

        projections_per_axis = dict()
        for pauli_name, pauli_op in zip(all_puali_names, all_paulis, strict=True):

            match pauli_name:       
                case 'z':           
                    rotated_rho_i = rho_i                        
                    rotated_rho_j = rho_j                           
                case 'y':           
                    rotated_rho_i = -1j * H @ rho_i @ pauli.z @ H                        
                    rotated_rho_j = -1j * H @ rho_j @ pauli.z @ H                            
                case 'x':           
                    rotated_rho_i = H @ rho_i @ H                        
                    rotated_rho_j = H @ rho_j @ H    
                case _:
                    raise ValueError("Not an option")
                
            assert isinstance(rotated_rho_i, np.ndarray)
            assert isinstance(rotated_rho_j, np.ndarray)
            projection_i = _get_z_projection(rotated_rho_i)
            projection_j = _get_z_projection(rotated_rho_j)

            projections_per_axis[pauli_name] = 0.5*(projection_i + projection_j)

        projections_per_edge.append(projections_per_axis)

    ## sum results by axis:
    per_projection_dict : dict[str, list[complex]] = dict()
    for pauli_name in all_puali_names:
        per_projection_dict[pauli_name] = []

    for edge_dict in projections_per_edge:
        for pauli_name, projection in edge_dict.items():
            per_projection_dict[pauli_name].append(projection)

    res : dict[str, complex] = dict()
    for key, list_ in per_projection_dict.items():
        res[key] = _mean(list_)

    return res
        

def measure_core_energies(
    tn_stable_around_core:KagomeTensorNetwork, hamiltonian:np.ndarray, bubblecon_trunc_dim:int
)->tuple[
    list[complex],
    list[np.ndarray]
]:
    energies_per_site = []
    # Reduce to small square of core:    
    if DEBUG_MODE: tn_stable_around_core.validate()
    ## For each edge, reduce a bit more and calc energy:
    rdms = []
    for side in BlockSide.all_in_counter_clockwise_order():
        # do the final needed contraction for this specific edge:
        core1, core2, environment_tensors, tn_env = calc_edge_environment(tn_stable_around_core, side, bubblecon_trunc_dim, already_reduced_to_core=True)
        # Get physical core tensors:
        ti = core1.physical_tensor
        tj = core2.physical_tensor
        ## Get metrics
        rdm = rho_ij(ti, tj, mps_env=environment_tensors)
        edge_energy  = np.dot(rdm.flatten(),  hamiltonian.flatten())
        if DEBUG_MODE:
            edge_energy = assertions.real(edge_energy)
        energies_per_site.append(edge_energy/2)
        # keep rdm:
        rdms.append(rdm)
    return energies_per_site, rdms


def _measurements_everything_on_duplicated_core_specific_size(
    core:KagomeTensorNetwork, 
    repeats:int,
    config:Config,
    logger:logs.Logger,
    reduce_for_xyz:bool=True,
    use_rdms_for_xyz:bool=True
)->Measurements:
    ## Parse commonly used inputs:
    chi = config.bubblecon_trunc_dim

    ## Get small stable network:
    tn_open = repeat_core(core, repeats)
    # Beliefe Propagation:
    tn_stable, _, bp_stats = belief_propagation(tn_open, messages=None, bp_config=config.bp)
    if bp_stats.final_error>config.bp.target_msg_diff:
        raise BPNotConvergedError("")
    tn_stable_around_core = reduce_tn_to_core_and_environment(tn_stable, chi, method=config.reduce2core_method)

    ## Calc 
    # Energies:
    energies_per_site, rdms = measure_core_energies(tn_stable_around_core, config.ite.interaction_hamiltonain, chi)    
    # XYZ
    if use_rdms_for_xyz:
        expectation_values = measure_xyz_expectation_values_with_rdms(rdms)
    else:
        if reduce_for_xyz:
            expectation_values = measure_xyz_expectation_values_with_tn(tn_stable_around_core, reduce=False)
        else:
            expectation_values = measure_xyz_expectation_values_with_tn(tn_stable, reduce=False)

    ## Pack results:
    measurements = Measurements(
        expectations=Expectations(
            x=np.abs(np.real(expectation_values['x'])),
            y=np.abs(np.real(expectation_values['y'])),
            z=np.abs(np.real(expectation_values['z']))
        ),
        energy=sum([np.real(e) for e in energies_per_site])/len(energies_per_site)
    )

    logger.info("")
    logger.info(f"Measurements on TN of edge size {tn_open.original_lattice_dims[0]}")
    logger.info(f"{measurements}")

    return measurements


def measurements_everything_on_duplicated_core(
    core:KagomeTensorNetwork, 
    repeats:int,
    config:Config,
    logger:logs.Logger|None = None
)->Measurements:
    ## Check:
    repeats = assertions.odd(repeats, reason="Must be an odd number >= 3")
    assert repeats>=3, "Must be an odd number >= 3"
    if logger is None:
        logger = logs.get_logger()

    ## Some changes to config:
    crnt_config = deepcopy(config)
    crnt_config.bp.allowed_retries = 1 # Force Only a single bp attempt per tn size
    crnt_repeats = repeats

    ## Multiple tries
    measurements = None
    while crnt_repeats >= 3:
        try:
            measurements = _measurements_everything_on_duplicated_core_specific_size(core, crnt_repeats, crnt_config, logger)
        except Exception as e:
            logger.warn(errors.get_traceback(e))
            crnt_repeats -= 2
            continue
        else:
            break
    ## Return
    assert measurements is not None, f"Not even a single attempt succeeded"
    assert isinstance(measurements, Measurements) , f"Not even a single attempt succeeded"
    return measurements

    

def main_test():
    pass


if __name__ == "__main__":
    main_test()
    print("Done")
