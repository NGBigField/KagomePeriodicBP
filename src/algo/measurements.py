
if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)
	sys.path.append(
		pathlib.Path(__file__).parent.parent.parent.__str__()
	)



## Get config:
from _config_reader import DEBUG_MODE

# Everyone needs numpy:
import numpy as np


# Other of our modules we need here:
from libs.ITE import rho_ij

# Common types in the code:
from containers import Config, BubbleConConfig, UpdateEdge
from containers.imaginary_time_evolution import HamiltonianFuncAndInputs
from tensor_networks import KagomeTN, TensorNetwork, ModeTN, EdgeTN, TensorNode, MPS
from lattices.directions import BlockSide
from _error_types import TensorNetworkError, BPNotConvergedError
from enums import ContractionDepth, NodeFunctionality, UnitCellFlavor, UpdateMode
from physics import pauli
from unit_cell import UnitCell
from containers.results import Measurements, Expectations

## Algos we need:
from algo.tn_reduction import reduce_full_kagome_to_core, reduce_tn
from algo.belief_propagation import belief_propagation
from algo.contract_tensor_network import contract_tensor_network
from algo.imaginary_time_evolution._tn_update import get_imaginary_time_evolution_operator

# For energy estimation:
from libs.ITE import rho_ij

# Utils:
from utils import assertions, logs, errors, lists, parallel_exec, prints, strings

# A bit of OOP:
from copy import deepcopy
from typing import TypeVar, TypeAlias

_T = TypeVar('_T')

UnitCellExpectationValuesDict : TypeAlias = dict[str, dict[str, float]]


all_paulis = list(pauli.all_paulis(with_names=False))
all_pauli_names = ['x', 'y', 'z'] 

H : np.matrix = np.matrix(
    [[1,  1],
     [1, -1]]
) / np.sqrt(2)   # Hadamard
Z : np.matrix = pauli.z


MULTIPROCESSING = False

TensorNetworkType = TypeVar("TensorNetworkType", bound=TensorNetwork)





def _find_not_none_item_in_double_dict( d :dict[str, dict[str, _T]], keys1, keys2) -> _T:
    for k1 in keys1:
        for k2 in keys2:
            item = d[k1][k2]
            if item is not None:
                return item


def print_results_table(results:dict[str, dict[str, float]])->None:
    node_keys = ["A", "B", "C"]
    proj_keys = ["x", "y", "z"]
    space_per_node = 25
    space_between_numbers = 3
    space_per_number = space_per_node - space_between_numbers

    _dummy = _find_not_none_item_in_double_dict(results, proj_keys, node_keys)
    if isinstance(_dummy, complex):
        space_per_node *= 2


    row_str = "   "+" "*space_between_numbers
    for node_key in node_keys:
        s = " "*(space_per_node//2)+f"{node_key}"+" "*(space_per_node//2-1)
        row_str += s
    print(row_str)

    for proj_key in proj_keys:
        proj_res = results[proj_key]
        row_str = f" {proj_key}:"+" "*space_between_numbers
        for node_key in node_keys:
            node_proj_res = proj_res[node_key]
            if node_proj_res is None:
                s = " "*(space_per_node-2)
            else:
                s = strings.formatted(node_proj_res, fill=' ', alignment="<", width=space_per_number, precision=space_per_number-3, signed=True)
                s += " "*space_between_numbers
            row_str += s
        print(row_str)



def _mean(list_:list[_T])->_T:
    return sum(list_)/len(list_)  #type: ignore
        
def mean_expectation_values(expectation:UnitCellExpectationValuesDict)->dict[str, float]:
    mean_per_direction : dict[str, float] = dict(x=0, y=0, z=0)
    # Add
    for abc, xyz_dict in expectation.items():
        for xyz, value in xyz_dict.items():
            mean_per_direction[xyz] += value/3
    return mean_per_direction



def measure_energies_and_observables_together(
    tn:TensorNetwork, 
    hamiltonian:HamiltonianFuncAndInputs|np.ndarray, 
    trunc_dim:int,
    mode:UpdateMode|None=None,
    force_real:bool=True
)->tuple[
    dict[tuple[str, str], float],
    UnitCellExpectationValuesDict,
    float
]:
    ## Prepare outputs and check inputs:
    # inputs:
    if DEBUG_MODE: tn.validate()

    if mode is None:
        mode = UpdateMode.random()

    if isinstance(hamiltonian, np.ndarray):
        h = hamiltonian
    elif callable(hamiltonian):
        h = hamiltonian()
    elif isinstance(hamiltonian, tuple|HamiltonianFuncAndInputs):
        h, _ = get_imaginary_time_evolution_operator(hamiltonian, None)
    else:
        raise TypeError(f"Not a valid type for input 'hamiltonian' of type {hamiltonian!r}")

    # outputs:
    energies = dict()
    expectations = {
        abc : { xyz : [0.0, 0] for xyz in ['x', 'y', 'z'] } 
        for abc in ['A', 'B', 'C']
    }
    energies4mean = []
    
    ## Contract to mode:
    mode_tn = reduce_tn(tn, ModeTN, trunc_dim, copy=True, mode=mode)

    ## For each edge, reduce a bit more and calc energy:
    for edge_tuple in UpdateEdge.all_options():
        # do the final needed contraction for this specific edge:
        edge_tn = reduce_tn(mode_tn, target_type=EdgeTN, trunc_dim=trunc_dim, copy=True, edge_tuple=edge_tuple)

        # Compute Reduce-Density-Matrix (RDM)
        rdm = edge_tn.rdm

        # Calc energy:
        edge_energy  = np.dot(rdm.flatten(),  h.flatten()) 
        edge_energy  /= 2  # Divide by 2 to get energy per site instead of per edge
        if DEBUG_MODE and force_real:
            edge_energy = assertions.real(edge_energy)
        elif force_real:
            edge_energy = float(np.real(edge_energy))

        # keep energies:
        energies4mean.append(edge_energy)
        energies[edge_tuple.as_strings] = edge_energy

        # Calc expectations:
        per_edge_results = expectation_values_with_rdm(rdm, force_real=force_real)

        # Sort expectation values:
        f1, f2 = edge_tn.unit_cell_flavors
        for xyz, tuple_ in per_edge_results.items():
            for value, abc_flavor in zip(tuple_, [f1, f2]):
                abc = abc_flavor.name
                expectations[abc][xyz][0] += value # accumulate values
                expectations[abc][xyz][1] += 1     # count appearances

    # mean expectation values from all appearances :
    for abc in ['A', 'B', 'C']:
        for xyz in ['x', 'y', 'z']:
            sum_, count_ = expectations[abc][xyz]
            expectations[abc][xyz] = sum_/count_
    
    # mean energy from all energies prt site:
    mean_energy = sum(energies4mean)/len(energies4mean)

    return energies, expectations, mean_energy


def _measurements_everything_on_duplicated_core_specific_size(
    core:KagomeTN, 
    repeats:int,
    config:Config,
    logger:logs.Logger,
    reduce_for_xyz:bool=True,
    use_rdms_for_xyz:bool=True
)->Measurements:
    ## Parse commonly used inputs:
    chi = config.trunc_dim

    ## Get small stable network:
    tn_open = repeat_core(core, repeats)
    # Beliefe Propagation:
    tn_stable, _, bp_stats = belief_propagation(tn_open, messages=None, bp_config=config.bp)
    if bp_stats.final_error>config.bp.target_msg_diff:
        raise BPNotConvergedError("")
    tn_stable_around_core = reduce_full_kagome_to_core(tn_stable, chi, method=config.reduce2core_method)

    ## Calc 
    # Energies:
    energies_per_site, rdms = measure_energies_and_observables_together(tn_stable_around_core, config.ite.interaction_hamiltonian, chi)    
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
    ket_op = np.tensordot(ket, mat, axes=([0], [1]))
    ket_op_bra = np.tensordot(ket_op, bra, axes=([4],[0]))

    res2 = np.transpose(ket_op_bra, axes=[0, 4, 1, 5, 2, 6, 3, 7])
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
        cell_flavor= node.cell_flavor
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
    tn:TensorNetwork, 
    operator:np.matrix,
    max_con_dim:int, 
    direction:BlockSide,
    print_progress:bool=False
) -> complex|tuple:
    # Replace ket tensor |psi> in `node_ind` with the bracket <psi|Z|psi>:
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


def expectation_values_with_rdm(
    rdm:np.ndarray,
    force_real:bool=True
) -> dict[str, tuple[complex, complex]]:
    rho_i = np.trace(rdm, axis1=2, axis2=3)
    rho_j = np.trace(rdm, axis1=0, axis2=1)
    return {
        pauli_name : _per_pauli_expectation_values_with_two_rdm(rho_i, rho_j, pauli_name, force_real=force_real) 
        for pauli_name in all_pauli_names
    }


def _per_pauli_expectation_values_with_two_rdm(
    rho_i:np.matrix, 
    rho_j:np.matrix, 
    pauli_name:str,
    force_real:bool=True
) -> tuple[complex, complex]:
    """ Compute expectation values of ab edge using its RDM
    """
    projection_i = _calc_rdm_projection_in_axis(rho_i, pauli_name, force_real=force_real)
    projection_j = _calc_rdm_projection_in_axis(rho_j, pauli_name, force_real=force_real)
    return projection_i, projection_j


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
    tn:TensorNetwork, 
    operators:list[np.matrix], 
    bubblecon_trunc_dim:int, 
    direction:BlockSide|None=None, 
    force_real:bool=False, 
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
            direction = lists.random_item(tn.major_sides)
        else:
            direction = BlockSide.random()
    else:
        assert isinstance(direction, BlockSide)

    ## Find nodes to sandwich with observables:
    denominator, _, _ = contract_tensor_network(tn, direction=direction, depth=ContractionDepth.Full, bubblecon_trunc_dim=bubblecon_trunc_dim)
    assert not isinstance(denominator, MPS), "Full contraction should result in a number, not an MPS"
    center_nodes = tn.get_center_core_nodes()
    unit_cell_indices = UnitCell(
        A = next(n.index for n in center_nodes if n.cell_flavor is UnitCellFlavor.A),
        B = next(n.index for n in center_nodes if n.cell_flavor is UnitCellFlavor.B),
        C = next(n.index for n in center_nodes if n.cell_flavor is UnitCellFlavor.C)
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
            value = _calc_mean_value_by_bracket_tn(
                tn, [index], operator, bubblecon_trunc_dim=bubblecon_trunc_dim, 
                direction=direction, denominator=denominator, force_real=force_real,
                print_progress=print_progress
            )
            res_unit_cell[key] = value
        results.append(res_unit_cell)
    prog_bar.clear()
    return results


def _calc_mean_value_by_bracket_tn(
    tn:TensorNetwork, 
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


def _get_z_projection(rho:np.ndarray)->complex:
    return rho[0,0] - rho[1,1] 


def _rotate_rdm(rho:np.matrix, pauli_name:str)->np.matrix:
    match pauli_name:       
        case 'x'|'X':           
            return H @ rho @ H                        
        case 'y'|'Y':        
            return 1j * H @ Z @ rho @ H  
        case 'z'|'Z':           
            return rho
        case _:
            raise ValueError("Not an option")


def _calc_rdm_projection_in_axis(rho:np.matrix, pauli_name:str, force_real:bool=False, using_rotation_method:bool=False)->complex|float:

    ## Calculate:
    if using_rotation_method:
        rotated_rho = _rotate_rdm(rho, pauli_name)
        projection =  _get_z_projection(rotated_rho)     
    else:
        obs = pauli.by_name(pauli_name)
        projection = np.trace( obs @ rho )

    ## Real/Complex:
    if force_real:
        if DEBUG_MODE:
            return assertions.real(projection)
        else:
            return float(np.real(projection))
    else:
        return complex(projection)


def derive_xyz_expectation_values_with_tn(
    tn:TensorNetwork, 
    bubblecon_trunc_dim:int=18,
    force_real:bool=True
)->dict[str, float|complex]:
    res = calc_unit_cell_expectation_values_from_tn(tn, operators=all_paulis, bubblecon_trunc_dim=bubblecon_trunc_dim, force_real=force_real)
    return dict(x=res[0], y=res[1], z=res[2])


def derive_xyz_expectation_values_using_rdm(
    edge_tn:EdgeTN,
    force_real:bool=True
) -> dict[str, complex]:
    ## Get RDM:
    assert isinstance(edge_tn, EdgeTN)
    t1, t2, mps_env = edge_tn.edge_and_environment()
    rdm = rho_ij(t1, t2, mps_env=mps_env)

    ## Compute Expectation values:
    per_ij_results = expectation_values_with_rdm(rdm, force_real=force_real)

    ## Rearrange:
    type1 = edge_tn.core1.cell_flavor
    type2 = edge_tn.core2.cell_flavor
    res = {}
    for key, values in per_ij_results.items():
        crnt_res = dict(A=None, B=None, C=None)
        for value, type_ in zip(values, [type1, type2], strict=True):
            crnt_res[type_.name] = value
        res[key] = crnt_res
    return res


if __name__ == "__main__":
    from project_paths import add_scripts
    add_scripts()
    from scripts import test_ite
    test_ite.main_test()


