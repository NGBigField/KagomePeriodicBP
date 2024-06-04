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
import functools
from unit_cell import UnitCell
import numpy as np
from utils import saveload

from lattices.kagome import UpperTriangle, KagomeLattice, Node, BlockSide

from libs.tnsu.utils import plot_convergence_curve
from libs.tnsu.tensor_network import TensorNetwork
import libs.tnsu.simple_update as su
import libs.tnsu.structure_matrix_constructor as smg


TnsuReturnType : TypeAlias = TensorNetwork

DATA_SUBFOLDER = "tnsu_results"


# Pauli matrices
pauli_x = np.array([[0, 1],
                    [1, 0]])
pauli_y = np.array([[0, -1j],
                    [1j, 0]])
pauli_z = np.array([[1, 0],
                    [0, -1]])


_KAGOME_STRUCTURE_MATRIX  =  np.array(
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


def _kagome_sm_value(
    kagome_lattice:KagomeLattice, node:Node, triangle:UpperTriangle, edge_name:str, edge_tuple:tuple[int, int]
) -> int:
    if edge_name not in node.edges:
        return 0
    
    ## basic info:
    node_index = node.index
    leg_index = node.edges.index(edge_name) + 1

    ## Check data:
    assert node_index in edge_tuple

    return leg_index
    
    

def _kagome_connect_boundary_edges_periodically(kagome_lattice:KagomeLattice, edge1:str, edge2:str) -> None:
    ## assert an edge at boundary:
    assert kagome_lattice.edges[edge1][0] == kagome_lattice.edges[edge1][1]
    assert kagome_lattice.edges[edge2][0] == kagome_lattice.edges[edge2][1]
    
    node1_index = kagome_lattice.edges[edge1][0]
    node2_index = kagome_lattice.edges[edge2][0]         

    ## replace two dict entries with a single entry for two tensors
    new_periodic_edge = edge1+"+"+edge2
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


def _periodic_kagome_structure_matrix(size:int) -> np.ndarray:
    
    ## Lattice:
    kagome_lattice : KagomeLattice = KagomeLattice(N=size)
    # kagome_lattice.plot_triangles_lattice()

    ## connect periodically:
    for block_side in [BlockSide.U, BlockSide.UL, BlockSide.DL]:
        boundary_edges          = kagome_lattice.sorted_boundary_edges(boundary=block_side)
        opposite_boundary_edges = kagome_lattice.sorted_boundary_edges(boundary=block_side.opposite())

        for edge1, edge2 in zip(boundary_edges, opposite_boundary_edges, strict=True):
            _kagome_connect_boundary_edges_periodically(kagome_lattice, edge1, edge2)

    ## init structure matrix:
    m = len(kagome_lattice.nodes)  # num rows
    n = len(kagome_lattice.edges)  # num columns
    sm = np.zeros(shape=(m, n), dtype=int)

    ## Complete structure matrix
    for node, triangle in kagome_lattice.nodes_and_triangles():
        i = node.index
        for j, (edge_name, edge_tuple) in enumerate(kagome_lattice.edges.items()):            
            value = _kagome_sm_value(kagome_lattice, node, triangle, edge_name, edge_tuple)
            sm[i, j] = value
        
        ## check row:
        row = sm[i, :]
        non_zero_values = [x for x in row if x!=0]
        assert len(non_zero_values)==4, f"Node is not completely connected! Must have 4 edges!"

    ## Check columns:
    for j in range(n):
        col = sm[:, j]
        non_zero_values = [x for x in col if x!=0]
        assert len(non_zero_values)==2, f"Each edge must have only two connected nodes!"

    return sm



UPPER_TRIANGLES : list[UpperTriangle] = [
    UpperTriangle(up=0, left= 3, right= 4),
    UpperTriangle(up=6, left= 9, right=10),
    UpperTriangle(up=7, left=11, right= 8),
    UpperTriangle(up=1, left=5 , right= 2),
]

FIRST_TRIANGLE_OUR_LEG_ORDER  = UpperTriangle(up=[1,2,3,4], left=[1,2,3,4], right=[1,2,3,4])
FIRST_TRIANGLE_TNSU_LEG_ORDER = UpperTriangle(up=[3,1,2,4], left=[2,4,3,1], right=[1,2,4,3])


def _common_filename(D:int)->str:
    return f"tnsu_AFH_D={D}.dat"


def load_or_compute_tnsu_unit_cell(D:int=2)->UnitCell:
    tnsu_network = _load_or_compute_tnsu_network(D=D)
    unit_cell = _parse_tnsu_network_to_unit_cell(D=D, tnsu_network=tnsu_network)
    return unit_cell


def _load_or_compute_tnsu_network(D:int=2)->TnsuReturnType:
    filename = _common_filename(D=D)
    if saveload.exist(filename, sub_folder=DATA_SUBFOLDER):
        tnsu_network = saveload.load(filename, sub_folder=DATA_SUBFOLDER)
    else:
        tnsu_network = _kagome_afh_peps_ground_state_search(d_max=D)
        saveload.save(tnsu_network, name=filename, sub_folder=DATA_SUBFOLDER)
    return tnsu_network


def _parse_tnsu_network_to_unit_cell(D:int, tnsu_network:TnsuReturnType)->UnitCell:
    triangle_indices = UPPER_TRIANGLES[0]
    triangle_nodes = UpperTriangle()

    for corner_name in UpperTriangle.field_names():
        #$ Get tensor:
        tensor_index = triangle_indices[corner_name] 
        tnsu_tensor : np.ndarray = tnsu_network.tensors[tensor_index]

        # Permutation from their leg order to our:
        our_leg_order   = FIRST_TRIANGLE_OUR_LEG_ORDER[corner_name]
        their_leg_order = FIRST_TRIANGLE_TNSU_LEG_ORDER[corner_name]
        permutation = [0]+[their_leg_order.index(i)+1 for i in our_leg_order]
        our_tensor = tnsu_tensor.transpose(permutation)
        
        ## save in triangle:
        triangle_nodes[corner_name] = our_tensor

    # unit_cell = UnitCell(A=triangle_nodes.up, B=triangle_nodes.left, C=triangle_nodes.right)
    unit_cell = UnitCell.from_upper_triangle(triangle_nodes)
    return unit_cell


def _kagome_afh_peps_ground_state_search(
    d_max: list = 2, 
    error: float = 1e-6,
    size: int = 4,
    max_iterations: int = 500, 
    dts: list = [0.1, 0.01, 0.001, 0.0001, 0.00001],
    plot_results: bool = True, 
    save_network: bool = False,
    print_process: bool = True
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
    bc: str = 'obc'  #  bc should be in ["obc", "pbc"]
    dir_path: str = saveload.DATA_FOLDER + saveload.PATH_SEP + DATA_SUBFOLDER
    h_k: float = 0.

    np.random.seed()

    # Construct the spin operators for the Hamiltonian
    s_i = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
    s_j = [pauli_x / 2., pauli_y / 2., pauli_z / 2.]
    s_k = [pauli_x / 2.]

    # structure matrix:
    structure_matrix = _periodic_kagome_structure_matrix(size=size)

    print(f'There are {structure_matrix.shape[1]} edges, and {structure_matrix.shape[0]} tensors.')

    # AFH Hamiltonian interaction parameters
    j_ij = [1.] * structure_matrix.shape[1]


    ## Run Simple Update:
    
    # create Tensor Network name for saving
    network_name = _common_filename(D=d_max)

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
                                d_max=d_max,
                                max_iterations=max_iterations,
                                convergence_error=error,
                                log_energy=True,
                                print_process=print_process)

    # run Simple Update algorithm over the Tensor Network state
    afh_tn_su.run()

    # compute the energy per-site observable
    energy = afh_tn_su.energy_per_site()
    print(f'| D max: {d_max} | Energy: {energy}\n')
    afh_tn = afh_tn_su.tensor_network

    # plot su convergence / energy curve
    if plot_results:
        plot_convergence_curve(afh_tn_su)

    # save the tensor network
    if save_network:
        afh_tn.save_network()

    # add to networks list

    return afh_tn




if __name__=="__main__":
    from scripts.test_unit_cell_from_tnsu import main_test
    main_test()