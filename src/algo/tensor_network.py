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

# For type anotation:
from typing import TypeVar

# Common types in the code:
from tensor_networks import KagomeTN, BaseTensorNetwork, ArbitraryTN, ModeTN, TensorNode, MPS
from lattices.directions import LatticeDirection, BlockSide, check
from lattices.edges import edges_dict_from_edges_list
from _error_types import TensorNetworkError
from enums import ContractionDepth, NodeFunctionality, UnitCellFlavor
from containers import BubbleConConfig
from physics import pauli
from _types import EdgeIndicatorType
from tensor_networks.unit_cell import UnitCell

# Our utilities:
from utils import tuples, lists, assertions, parallel_exec, prints

# Our needed algos:
from tensor_networks.tensor_network import get_common_edge_legs
from algo.mps import physical_tensor_with_split_mid_leg
from algo.contract_tensor_network import contract_tensor_network
from algo.tn_reduction import reduce_tn_to_core

# For energy estimation:
from libs.ITE import rho_ij


MULTIPROCESSING = False

TensorNetworkType = TypeVar("TensorNetworkType", bound=BaseTensorNetwork)


def _get_corner_tensors(tn:KagomeTN) -> list[TensorNode]:
    min_x, max_x, min_y, max_y = tn.positions_min_max()
    corner_tesnors = [] 
    for x in [min_x, max_x]:
        for y in  [min_y, max_y]:
            t = tn.get_node_in_pos((x, y))
            corner_tesnors.append(t)    
    return corner_tesnors

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
        err_msg = f"Braket results should be scalar values. Got numerator={numerator}, denominator={denominator}"
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


def calc_unit_cell_expectation_values(
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
    corrosponding expectation values of this specific tensor in the unit cell for the operator.

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
        tn_reduced = reduce_tn_to_core(tn, bubblecon_trunc_dim, parallel)
    else:
        tn_reduced = tn
    denominator, _, _ = contract_tensor_network(tn_reduced, direction=direction, depth=ContractionDepth.Full, bubblecon_trunc_dim=bubblecon_trunc_dim)
    assert not isinstance(denominator, MPS), "Full contraction should result in a number, not an MPS"
    center_nodes = tn_reduced.get_center_unit_cell_nodes()
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
        res_unit_cell = UnitCell(A=0.0, B=0.0, C=0.0)   
        for key in UnitCell.all_keys():
            prog_bar.next()
            index = unit_cell_indices[key]
            value = calc_mean_value(
                tn_reduced, [index], operator, bubblecon_trunc_dim=bubblecon_trunc_dim, 
                direction=direction, denominator=denominator, force_real=force_real,
                print_progress=print_progress
            )
            res_unit_cell.__setattr__(key, value)
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



if __name__ == "__main__":
    from project_paths import add_scripts; 
    add_scripts()
    from scripts import bp_test
    bp_test.main_test()

