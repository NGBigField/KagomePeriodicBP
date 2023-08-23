if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )


## Get config:
from _config_reader import DEBUG_MODE

import numpy as np

# Other of our modules we need here:
from libs.ITE import rho_ij

# Common types in the code:
from containers import Config, BubbleConConfig
from tensor_networks import KagomeTN, BaseTensorNetwork, ModeTN, EdgeTN, TensorNode, MPS
from lattices.directions import BlockSide, check
from _error_types import TensorNetworkError, BPNotConvergedError
from enums import ContractionDepth, NodeFunctionality, UnitCellFlavor
from physics import pauli
from tensor_networks.unit_cell import UnitCell
from containers.results import Measurements, Expectations

## Algos we need:
from tensor_networks.construction import repeat_core
from algo.tn_reduction import reduce_full_kagome_to_core
from algo.belief_propagation import belief_propagation
from physics import pauli
from lattices.directions import BlockSide


# Utils:
from utils import assertions, logs, errors

# A bit of OOP:
from copy import deepcopy
from typing import TypeVar

_T = TypeVar('_T')


all_paulis = [pauli.x, pauli.y, pauli.z]
all_pauli_names = ['x', 'y', 'z'] 

H = np.matrix(
    [[1,  1],
     [1, -1]]
) / np.sqrt(2)   # Hadamard



if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)
	sys.path.append(
		pathlib.Path(__file__).parent.parent.parent.__str__()
	)


# Control flags:
from _config_reader import DEBUG_MODE

# Everyone needs numpy:
import numpy as np

# For type annotation:
from typing import TypeVar



# Our utilities:
from utils import lists, assertions, parallel_exec, prints

# Our needed algos:
from tensor_networks.tensor_network import get_common_edge_legs
from tensor_networks.mps import physical_tensor_with_split_mid_leg
from algo.contract_tensor_network import contract_tensor_network
from algo.tn_reduction import reduce_full_kagome_to_core

# For energy estimation:
from libs.ITE import rho_ij


MULTIPROCESSING = False

TensorNetworkType = TypeVar("TensorNetworkType", bound=BaseTensorNetwork)



def _mean(list_:list[_T])->_T:
    return sum(list_)/len(list_)  #type: ignore


def _get_z_projection(rho:np.ndarray)->complex:
    return rho[0,0] - rho[1,1] 


def measure_xyz_expectation_values_with_rdms(rdms:list[np.ndarray])->dict[str, float|complex]:

    ## Collect all projections:
    projections_per_edge : list[dict[str, complex]] = []
    for rdm_ij in rdms:
        rho_i = np.trace(rdm_ij, axis1=2, axis2=3)
        rho_j = np.trace(rdm_ij, axis1=0, axis2=1)

        projections_per_axis = dict()
        for pauli_name in all_pauli_names:

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
    for pauli_name in all_pauli_names:
        per_projection_dict[pauli_name] = []

    for edge_dict in projections_per_edge:
        for pauli_name, projection in edge_dict.items():
            per_projection_dict[pauli_name].append(projection)

    res : dict[str, complex] = dict()
    for key, list_ in per_projection_dict.items():
        res[key] = _mean(list_)

    return res
        

def measure_core_energies(
    tn_stable_around_core:KagomeTN, hamiltonian:np.ndarray, bubblecon_trunc_dim:int
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
    core:KagomeTN, 
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
    tn_stable_around_core = reduce_full_kagome_to_core(tn_stable, chi, method=config.reduce2core_method)

    ## Calc 
    # Energies:
    energies_per_site, rdms = measure_core_energies(tn_stable_around_core, config.ite.interaction_hamiltonain, chi)    
    # XYZ
    if use_rdms_for_xyz:
        expectation_values = measure_xyz_expectation_values_with_rdms(rdms)
    else:
        if reduce_for_xyz:
            expectation_values = derive_xyz_expectation_values_with_tn(tn_stable_around_core, reduce=False)
        else:
            expectation_values = derive_xyz_expectation_values_with_tn(tn_stable, reduce=False)

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
    core:KagomeTN, 
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


def _sandwich_fused_tensors_with_expectation_values(tn_in:TensorNetworkType, mat:np.matrix, ind:int)->TensorNetworkType:

    ## Get peps tensor and node data
    node = tn_in.nodes[ind]
    assert node.is_ket
    ket = node.physical_tensor
    bra = np.conj(ket)

    D = ket.shape[1]
    D2 = D*D
    
    ket_op = np.tensordot(ket, mat, axes=([0],[0]))
    res = np.tensordot(ket_op, bra, axes=([4],[0]))

    res2 = np.transpose(res, axes=[0, 4, 1, 5, 2, 6, 3, 7])
    fused_data = res2.reshape([D2, D2, D2, D2])

    ## Replace node in original tensor_network:
    tn_out = tn_in.copy()
    tn_out.nodes[ind] = TensorNode(
        is_ket          = False,
        tensor          = fused_data,
        edges           = node.edges,
        directions      = node.directions,
        pos             = node.pos,
        index           = node.index,
        name            = node.name,
        boundaries      = node.boundaries,
        functionality   = node.functionality,
        unit_cell_flavor= node.unit_cell_flavor
    )
    if DEBUG_MODE: tn_out.validate()

    return tn_out


def _calc_and_check_expectation_value(numerator, denominator, force_real:bool) -> float:
    ## Control:
    separate_exp = BubbleConConfig.separate_exp

    ## Check inputs:
    if DEBUG_MODE:
        err_msg = f"Bracket results should be scalar values. Got numerator={numerator}, denominator={denominator}"
        for val in [numerator, denominator]:
            if separate_exp:
                assert isinstance(val, tuple), "BubbleCon should return tuple[complex, int]"
                assert len(val)==2, "BubbleCon should return tuple[complex, int]"
                assert isinstance(val[0], complex|float), err_msg  # mantissa
                assert isinstance(val[1], int), f"Second return value of BubbleCon should be the exponent. Instead got {val[1]}"  # exponent
            else:
                assert isinstance(val, complex), err_msg

    ## Assign values in common format man*10**exp
    if separate_exp:
        numerator_mantissa   = numerator[0]
        numerator_exponent   = numerator[1]
        denominator_mantissa = denominator[0]
        denominator_exponent = denominator[1]
    else:
        numerator_mantissa   = numerator
        numerator_exponent   = 0
        denominator_mantissa = denominator
        denominator_exponent = 0

    if numerator_mantissa==denominator_mantissa==0:
        raise FloatingPointError(f"Both numerator and denominator are zero.")
            
    ## Compute Result:
    mantissa = numerator_mantissa/denominator_mantissa
    exponent = numerator_exponent-denominator_exponent

    ## Check result:
    if DEBUG_MODE and force_real:
        mantissa = assertions.real(mantissa, reason = f"Solution should be a real value. Instead got {mantissa}")
    elif force_real:
        mantissa = float(np.real(mantissa))
    else:
        pass

    return mantissa*10**exponent

def _sandwich_with_operator_and_contract_fully(
    node_ind:int,
    tn:BaseTensorNetwork, 
    operator:np.matrix,
    max_con_dim:int, 
    direction:BlockSide,
    print_progress:bool=False
) -> complex|tuple:
    # Replace fused-tensor <psi|psi> in `node_ind` with  <psi|Z|psi>:
    tn_with_observable = _sandwich_fused_tensors_with_expectation_values(tn, operator, node_ind)
    ## Calculate Expectation Value:
    numerator, _, _ = contract_tensor_network(
        tn_with_observable, 
        direction=direction, 
        depth=ContractionDepth.Full, 
        bubblecon_trunc_dim=max_con_dim, 
        print_progress=print_progress 
    )
    # complete contraction so must be a number:
    assert isinstance(numerator, complex|tuple)
    return numerator



def _expectation_values_with_rdm(
    rdm:np.ndarray,
    force_real:bool=True
) -> dict[str, tuple[complex, complex]]:
    rho_i = np.trace(rdm, axis1=2, axis2=3)
    rho_j = np.trace(rdm, axis1=0, axis2=1)
    return {
        pauli_name : _per_op_expectation_values_with_rdm(rho_i, rho_j, pauli_name, force_real=force_real) 
        for pauli_name in all_pauli_names
    }


def _per_op_expectation_values_with_rdm(
    rho_i:np.ndarray, 
    rho_j:np.ndarray, 
    pauli_name:str,
    force_real:bool=True
) -> tuple[complex, complex]:
    """ Compute expectation values of ab edge using its RDM
    """
    match pauli_name:       
        case 'z'|'Z':           
            rotated_rho_i = rho_i                        
            rotated_rho_j = rho_j                           
        case 'y'|'Y':           
            rotated_rho_i = -1j * H @ rho_i @ pauli.z @ H                        
            rotated_rho_j = -1j * H @ rho_j @ pauli.z @ H                            
        case 'x'|'X':           
            rotated_rho_i = H @ rho_i @ H                        
            rotated_rho_j = H @ rho_j @ H    
        case _:
            raise ValueError("Not an option")
            
    assert isinstance(rotated_rho_i, np.ndarray)
    assert isinstance(rotated_rho_j, np.ndarray)
    projection_i = _get_z_projection(rotated_rho_i)
    projection_j = _get_z_projection(rotated_rho_j)

    if force_real:
        r_i = assertions.real(projection_i)
        r_j = assertions.real(projection_j)
    else:
        r_i = complex(projection_i)
        r_j = complex(projection_j)

    return r_i, r_j


#TODO assert used
def calc_interaction_energies_in_core(tn:KagomeTN, interaction_hamiltonain:np.ndarray, bubblecon_trunc_dim:int) -> list[float]:
    energies = []
    reduced_tn = reduce_tn_to_core_and_environment(tn, bubblecon_trunc_dim, swallow_corners_=False)
    for side in Direction:
        core1, core2, environment_tensors, tn_env = calc_edge_environment(reduced_tn, side, bubblecon_trunc_dim)
        rdm = rho_ij(core1.physical_tensor, core2.physical_tensor, mps_env=environment_tensors)
        energy  = np.dot(rdm.flatten(),  interaction_hamiltonain.flatten())
        energies.append(energy)
    return energies


def calc_unit_cell_expectation_values_from_tn(
    tn:KagomeTN, 
    operators:list[np.matrix], 
    bubblecon_trunc_dim:int, 
    direction:BlockSide|None=None, 
    force_real:bool=False, 
    reduce:bool=True,
    print_progress:bool=True,
    parallel:bool=False
) -> list[UnitCell]:
    """ Compute expectation values of unit cell nodes

    For each operator (np.matrix) in `operators`, returns a UnitCell object (with A,B,C attributes) with the 
    corresponding expectation values of this specific tensor in the unit cell for the operator.

    """
    ## Prepare output:
    results = []

    ## Check or choose direction:
    if direction is None:
        if isinstance(tn, ModeTN):
            direction = lists.random_item(tn.major_directions)
        else:
            direction = BlockSide.random()
    else:
        assert isinstance(direction, BlockSide)

    ## Perform all common actions:
    if reduce:
        tn_reduced = reduce_full_kagome_to_core(tn, bubblecon_trunc_dim, parallel)
    else:
        tn_reduced = tn
    denominator, _, _ = contract_tensor_network(tn_reduced, direction=direction, depth=ContractionDepth.Full, bubblecon_trunc_dim=bubblecon_trunc_dim)
    assert not isinstance(denominator, MPS), "Full contraction should result in a number, not an MPS"
    center_nodes = tn_reduced.get_center_core_nodes()
    unit_cell_indices = UnitCell(
        A = next(n.index for n in center_nodes if n.unit_cell_flavor is UnitCellFlavor.A),
        B = next(n.index for n in center_nodes if n.unit_cell_flavor is UnitCellFlavor.B),
        C = next(n.index for n in center_nodes if n.unit_cell_flavor is UnitCellFlavor.C)
    )

    ## Prepare progress-bar:
    if print_progress:
        prog_bar = prints.ProgressBar(len(operators)*unit_cell_indices.size(), f"Calculating expectation-values: ", print_length=51)
    else:
        prog_bar = prints.ProgressBar.inactive()

    ## Perform calculation per operator:
    for operator in operators:     
        res_unit_cell = dict(A=0.0, B=0.0, C=0.0)   
        for key in UnitCell.all_keys():
            prog_bar.next()
            index = unit_cell_indices[key]
            value = calc_mean_value(
                tn_reduced, [index], operator, bubblecon_trunc_dim=bubblecon_trunc_dim, 
                direction=direction, denominator=denominator, force_real=force_real,
                print_progress=print_progress
            )
            res_unit_cell[key] = value
        results.append(res_unit_cell)
    prog_bar.clear()
    return results


def calc_mean_value(
    tn:KagomeTN, 
    node_indices:list[int], 
    operator:np.matrix,
    bubblecon_trunc_dim:int, 
    direction:BlockSide = BlockSide.random(), 
    print_progress:bool = False,
    print_results:bool = False,
    force_real:bool = False,
    denominator:complex|tuple|None=None
) -> float:
    ## Common denominator <psi|psi>:
    if isinstance(denominator, complex|tuple):
        pass
    elif denominator is None:
        res, _, _ = contract_tensor_network(tn, direction=direction, depth=ContractionDepth.Full, bubblecon_trunc_dim=bubblecon_trunc_dim)        
        assert not isinstance(res, MPS), "Full contraction should result in a number, not an MPS"
        denominator = res
    else:
        raise TypeError(f"Not an expected type {type(denominator)!r} for denominator")

    ## Decide parallel or not:    
    fixed_arguments = dict(tn=tn, max_con_dim=bubblecon_trunc_dim, direction=direction, operator=operator)
    in_parallel = MULTIPROCESSING and len(node_indices)>1 and tn.tensor_dims.virtual>2
    if in_parallel:
        fixed_arguments["print_progress"] = False
    else:
        fixed_arguments["print_progress"] = print_progress

    ## Calculate Numerators:
    numerators = parallel_exec.concurrent_or_parallel(
        func=_sandwich_with_operator_and_contract_fully, 
        values=node_indices, value_name="node_ind", 
        fixed_arguments=fixed_arguments,
        in_parallel=in_parallel
    ) 

    ## Calc numerator/denominator for each value
    expectation_values = [
        _calc_and_check_expectation_value(numerator, denominator, force_real)
        for numerator in numerators.values()
    ]    

    if print_results:
        for expectation_value, numerator in zip(expectation_values, numerators.values(), strict=True):
            print(f"\texpectation_value = {expectation_value}  =  {numerator} / {denominator}  ")

    return lists.average(expectation_values)


def derive_xyz_expectation_values_with_tn(tn_stable_around_core:KagomeTN, reduce:bool=True, bubblecon_trunc_dim:int=18)->dict[str, float|complex]:
    res = calc_unit_cell_expectation_values_from_tn(tn_stable_around_core, operators=all_paulis, bubblecon_trunc_dim=bubblecon_trunc_dim, force_real=True, reduce=reduce)
    return dict(x=res[0], y=res[1], z=res[2])


def derive_xyz_expectation_values_using_rdm(
    edge_tn:EdgeTN
) -> dict[str, complex]:
    ## Get RDM:
    t1, t2, mps_env = edge_tn.edge_and_environment()
    rdm = rho_ij(t1, t2, mps_env=mps_env)

    ## Compute Expectation values:
    per_ij_results = _expectation_values_with_rdm(rdm, force_real=True)

    ## Rearrange:
    type1 = edge_tn.core1.unit_cell_flavor
    type2 = edge_tn.core2.unit_cell_flavor
    res = {}
    for key, values in per_ij_results.items():
        crnt_res = dict(A=None, B=None, C=None)
        for value, type_ in zip(values, [type1, type2], strict=True):
            crnt_res[type_.name] = value
        res[key] = crnt_res
    return res



if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import bp_test
    bp_test.main_test()


