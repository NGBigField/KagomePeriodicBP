if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
    from project_paths import add_src, add_base, add_scripts
    add_src()
    add_base()
    add_scripts()

from typing import TypeAlias, Callable, Any
import functools, itertools
from unit_cell import UnitCell
import numpy as np
from utils import saveload, numpys, tuples

from lattices.kagome import UpperTriangle, KagomeLattice, Node, BlockSide

from libs.tnsu._utils import plot_convergence_curve
from libs.tnsu.tensor_network import TensorNetwork
import libs.tnsu.simple_update as su
import libs.tnsu.structure_matrix_constructor as smg


TnsuReturnType : TypeAlias = TensorNetwork
DATA_SUBFOLDER = "simple_update_results"


# Pauli matrices
pauli_x = np.array([[0, 1],
                    [1, 0]])
pauli_y = np.array([[0, -1j],
                    [1j, 0]])
pauli_z = np.array([[1, 0],
                    [0, -1]])


_FIXED_KAGOME_STRUCTURE_MATRIX  =  np.array(
    [[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T1
     [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T2
     [0, 0, 0, 0, 1, 0, 0, 0, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T3
     [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T4
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T5
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],  # T6
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0],  # T7
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0],  # T8
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 4, 0, 0],  # T9
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 4, 0],  # T10
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 4],  # T11
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 4]]  # T12
)   #E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14 E15 E16 E17 E18 E19 E20 E21 E22 E23 E24
# Taken from Jahromi, Saeed S., and Román Orús. "Universal tensor-network algorithm for any infinite lattice." Physical Review B 99.19 (2019): 195105.


_FIXED_KAGOME_UPPER_TRIANGLES : list[UpperTriangle] = [
    UpperTriangle(up=0, left= 3, right= 4),
    UpperTriangle(up=6, left= 9, right=10),
    UpperTriangle(up=7, left=11, right= 8),
    UpperTriangle(up=1, left=5 , right= 2),
]

_FIXED_KAGOME_FIRST_TRIANGLE_OUR_LEG_ORDER  = UpperTriangle(up=[1,2,3,4], left=[1,2,3,4], right=[1,2,3,4])
_FIXED_KAGOME_FIRST_TRIANGLE_TNSU_LEG_ORDER = UpperTriangle(up=[3,1,2,4], left=[2,4,3,1], right=[1,2,4,3])



_SINGLE_PERIODIC_CELL_STRUCTURE_MATRIX = np.array([
    [1, 2, 3, 4, 0, 0],   # A
    [0, 4, 0, 2, 1, 3],   # B
    [3, 0, 1, 0, 4, 2],   # C
]) #E0 E1 E2 E3 E4 E5
# Taken from Nir Gutman's thesis. 


def _kagome_sm_value(
    node:Node, edge_name:str, edge_tuple:tuple[int, int]
) -> int:
    if edge_name not in node.edges:
        return 0
    
    ## basic info:
    node_index = node.index
    leg_index = node.edges.index(edge_name) + 1

    ## Check data:
    assert node_index in edge_tuple

    return leg_index



def _kagome_disconnect_boundary_edges_to_open_nodes(kagome_lattice:KagomeLattice, edge1:str, edge2:str) -> None:
    ## assert an edge at boundary:
    assert kagome_lattice.edges[edge1][0] == kagome_lattice.edges[edge1][1]
    assert kagome_lattice.edges[edge2][0] == kagome_lattice.edges[edge2][1]

    ## Define node creation function:
    new_node_index = kagome_lattice.size
    def _kagome_disconnect_boundary_edges_to_open_nodes_create_node(edge:str)->Node:
        nonlocal new_node_index

        # Find neighbor and its orientation to the open leg:
        neigbor_index = kagome_lattice.edges[edge][0]
        neighbor = kagome_lattice.nodes[neigbor_index]
        neighbor_leg_index = neighbor.edges.index(edge)
        neighbor_leg_direction = neighbor.directions[neighbor_leg_index]

        # Create node:
        open_node = Node(
            index=new_node_index,
            pos=tuples.add(neighbor.pos, neighbor_leg_direction.unit_vector), 
            edges=[edge], 
            directions=[neighbor_leg_direction.opposite()]
        )
        new_node_index += 1

        return open_node

    ## Create two new tensors:
    n1 = _kagome_disconnect_boundary_edges_to_open_nodes_create_node(edge1)
    n2 = _kagome_disconnect_boundary_edges_to_open_nodes_create_node(edge2)
    
    ## connect new nodes to lattice:
    kagome_lattice.nodes.append(n1)
    kagome_lattice.nodes.append(n2)
    kagome_lattice.edges[edge1]  = tuples.copy_with_replaced_val_at_index(kagome_lattice.edges[edge1], 1, n1.index)
    kagome_lattice.edges[edge2]  = tuples.copy_with_replaced_val_at_index(kagome_lattice.edges[edge2], 1, n2.index)


def _kagome_connect_boundary_edges_periodically(kagome_lattice:KagomeLattice, edge1:str, edge2:str) -> None:
    ## assert an edge at boundary:
    assert kagome_lattice.edges[edge1][0] == kagome_lattice.edges[edge1][1]
    assert kagome_lattice.edges[edge2][0] == kagome_lattice.edges[edge2][1]
    
    node1_index = kagome_lattice.edges[edge1][0]
    node2_index = kagome_lattice.edges[edge2][0]         

    ## replace two dict entries with a single entry for two tensors
    new_periodic_edge = _new_periodic_edge_name(edge1, edge2)
    del kagome_lattice.edges[edge1]
    del kagome_lattice.edges[edge2]
    kagome_lattice.edges[new_periodic_edge] = (node1_index, node2_index)

    ## Replace in both nodes, as-well:
    # get nodes:
    node1 = kagome_lattice.nodes[node1_index]
    node2 = kagome_lattice.nodes[node2_index]
    # change edges:
    node1.edges[node1.edges.index(edge1)] = new_periodic_edge
    node2.edges[node2.edges.index(edge2)] = new_periodic_edge


def _kagome_structure_matrix(size:int, periodic:bool) -> np.ndarray:
    
    ## Lattice:
    kagome_lattice : KagomeLattice = KagomeLattice(N=size)
    kagome_lattice.change_boundary_conditions(periodic)

    # Plot lattice:
    # fig = plt.figure()
    # kagome_lattice.plot()
    # fig.suptitle(f"periodic={periodic}")

    ## init structure matrix:
    m = len(kagome_lattice.nodes)  # num rows
    n = len(kagome_lattice.edges)  # num columns
    sm = np.zeros(shape=(m, n), dtype=int)

    ## Complete structure matrix
    for node in kagome_lattice.nodes:
        i = node.index
        for j, (edge_name, edge_tuple) in enumerate(kagome_lattice.edges.items()):            
            value = _kagome_sm_value(node, edge_name, edge_tuple)
            sm[i, j] = value
        
        ## check row:
        row = sm[i, :]
        non_zero_values = [x for x in row if x!=0]
        assert len(non_zero_values)==len(node.edges), f"Node is not completely connected! Must have all edges!"

    ## Check columns:
    for j in range(n):
        col = sm[:, j]
        non_zero_values = [x for x in col if x!=0]
        if periodic:
            assert len(non_zero_values)==2, f"Each edge must have only two connected nodes!"
        else:
            assert len(non_zero_values)<=2, f"Each edge must have at most two connected nodes!"

    return sm


def _periodicity_str(periodic:bool)->str:
    if periodic:
        return "PBC"
    else:
        return "OBC"


def _common_filename(D:int, size:int, periodic:bool)->str:
    name = f"tnsu_AFH_D={D}_size={size}_{_periodicity_str(periodic)}.dat"
    return name


def load_or_compute_tnsu_unit_cell(D:int=2, size:int=1, periodic:bool=True) -> tuple[UnitCell, float]:
    tnsu_network = _load_or_compute_tnsu_network(D=D, size=size, periodic=periodic)
    unit_cell = _parse_tnsu_network_to_unit_cell(D=D, size=size, periodic=periodic, tnsu_network=tnsu_network)
    try:
        tnsu_energy = tnsu_network.final_energy  #type: ignore
    except AttributeError:
        tnsu_energy = None
    return unit_cell, tnsu_energy  #type: ignore


def _load_or_compute_tnsu_network(D:int=2, size:int=1, periodic:bool=True)->TnsuReturnType:
    filename = _common_filename(D=D, size=size, periodic=periodic)
    if saveload.exist(filename, sub_folder=DATA_SUBFOLDER):
        tnsu_network = saveload.load(filename, sub_folder=DATA_SUBFOLDER)
    else:
        tnsu_network = _kagome_afh_peps_ground_state_search(D=D, size=size, periodic=periodic)
        saveload.save(tnsu_network, name=filename, sub_folder=DATA_SUBFOLDER)
    return tnsu_network


def _mean_unit_cell(unit_cells:list[UnitCell]) -> UnitCell:
    A = numpys.tensor_mean( [unit_cell.A for unit_cell in unit_cells] )
    B = numpys.tensor_mean( [unit_cell.B for unit_cell in unit_cells] )
    C = numpys.tensor_mean( [unit_cell.C for unit_cell in unit_cells] )

    return UnitCell(A=A, B=B, C=C)



def _parse_tnsu_network_to_unit_cell_get_triangles(size:int)->list[UpperTriangle]:
    kagome_lattice : KagomeLattice = KagomeLattice(N=size)
    # if PBC:
    #     return kagome_lattice.triangles
    # else:
    #     return [kagome_lattice.get_center_triangle()]
    return [kagome_lattice.get_center_triangle()]


def _parse_tnsu_network_to_unit_cell_canonical_form(D:int, size:int, periodic:bool, tnsu_network:TnsuReturnType)->UnitCell:
    triangles_to_include = _parse_tnsu_network_to_unit_cell_get_triangles(size=size)
    unit_cells : list[UnitCell] = []

    for triangle in triangles_to_include:
        triangle_tensors = UpperTriangle()

        for corner_name in UpperTriangle.field_names():
            #$ Get tensor:
            node = triangle[corner_name] 
            tnsu_tensor : np.ndarray = tnsu_network.tensors[node.index]

            # # Permutation from their leg order to our:
            # our_leg_order   = FIRST_TRIANGLE_OUR_LEG_ORDER[corner_name]
            # their_leg_order = FIRST_TRIANGLE_TNSU_LEG_ORDER[corner_name]
            # permutation = [0]+[their_leg_order.index(i)+1 for i in our_leg_order]
            # our_tensor = tnsu_tensor.transpose(permutation)
            
            ## save in triangle:
            triangle_tensors[corner_name] = tnsu_tensor

        # unit_cell = UnitCell(A=triangle_nodes.up, B=triangle_nodes.left, C=triangle_nodes.right)
        unit_cell = UnitCell.from_upper_triangle(triangle_tensors)
        unit_cells.append(unit_cell)

    unit_cell = _mean_unit_cell(unit_cells)
    return unit_cell


def _parse_tnsu_network_to_unit_cell_single_periodic_single_cell(D:int, tnsu_network:TnsuReturnType)->UnitCell:
    return UnitCell(
        A=tnsu_network.tensors[0],
        B=tnsu_network.tensors[1],
        C=tnsu_network.tensors[2],
    )


def _parse_tnsu_network_to_unit_cell(D:int, size:int, periodic:bool, tnsu_network:TnsuReturnType)->UnitCell:

    if size>=2:
        return _parse_tnsu_network_to_unit_cell_canonical_form(D=D, size=size, periodic=periodic, tnsu_network=tnsu_network)
    
    elif size==1 and periodic:
        return _parse_tnsu_network_to_unit_cell_single_periodic_single_cell(D=D, tnsu_network=tnsu_network)
    
    else:
        raise NotImplementedError(f"Not yet implemented. size={size!r}.")


def _kagome_afh_peps_ground_state_search(
    D: int = 2, 
    error: float = 1e-7,
    size: int = 1,
    max_iterations: int = 500, 
    dts: list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
    plot_results: bool = False, 
    periodic:bool = True,
    print_process: bool = False
) -> TnsuReturnType:
    """
    1. construct a random Tensor Network state with spin dimension 2 and virtual dimensions d_max.
    2. Run Simple Update with an Antiferromagnetic Heisenberg Hamiltonian to find the Tensor Network ground state.
    3. Plot results: optional.
    4. Return network 

    d_max: maximal virtual bond dimensions
    error: The maximally allowed L2 norm convergence error (between two consecutive weight vectors)
    max_iterations: Maximal number of Simple update iterations per dt
    dts: List of dt values for ITE
    h_k: The Hamiltonian's magnetic field constant (for single spin energy)
    dir_path: Path for saving the networks in
    plot_results: Bool for plot resulst, True = plot, False = don't plot
    save_network: Bool for saving the resulting networks

    :return: A network
    """

    ## Basic params and inputs:
    dir_path: str = saveload.DATA_FOLDER + saveload.PATH_SEP + DATA_SUBFOLDER
    h_k: float = 0.

    np.random.seed()

    # Construct the spin operators for the Hamiltonian
    s_i = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
    s_j = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
    s_k = [pauli_x / 2.]

    # structure matrix:
    if size == 0:
        structure_matrix = _FIXED_KAGOME_STRUCTURE_MATRIX
    elif size == 1 and periodic:
        structure_matrix = _SINGLE_PERIODIC_CELL_STRUCTURE_MATRIX
    elif size > 1:
        structure_matrix = _kagome_structure_matrix(size=size, periodic=periodic)
    else:
        raise ValueError("Not an option")

    print(f'There are {structure_matrix.shape[1]} edges, and {structure_matrix.shape[0]} tensors.')

    # AFH Hamiltonian interaction parameters
    j_ij = [1.] * structure_matrix.shape[1]


    ## Run Simple Update:
    
    # create Tensor Network name for saving
    network_name = _common_filename(D=D, size=size, periodic=periodic)

    # create the Tensor Network object
    afh_tn = TensorNetwork(structure_matrix=structure_matrix,
                            virtual_dim=2,
                            network_name=network_name,
                            dir_path=dir_path)

    # create the Simple Update environment
    afh_tn_su = su.SimpleUpdate(tensor_network=afh_tn,
                                dts=dts,
                                j_ij=j_ij,
                                h_k=h_k,
                                s_i=s_i,
                                s_j=s_j,
                                s_k=s_k,
                                d_max=D,
                                max_iterations=max_iterations,
                                convergence_error=error,
                                log_energy=True,
                                print_process=print_process)

    # run Simple Update algorithm over the Tensor Network state
    afh_tn_su.run()

    # compute the energy per-site observable
    energy = afh_tn_su.energy_per_site()

    # Swallow matrices into tensors:
    afh_tn_su.absorb_all_weights()

    print(f'| D max: {D} | Energy: {energy}\n')
    afh_tn = afh_tn_su.tensor_network
    # Save energy. We might use it later
    afh_tn.__setattr__("final_energy", energy)

    # plot su convergence / energy curve
    if plot_results:
        plot_convergence_curve(afh_tn_su)

    return afh_tn




if __name__=="__main__":
    from scripts.test_unit_cell_from_tnsu import main_test
    main_test()
