
if __name__ == "__main__":

    from pathlib import Path
    import sys

    ## Import itai and roy packages:
    this_folder = Path(__file__).parent.__str__()
    if this_folder not in sys.path:
        sys.path.append(this_folder)


d = 2


from itai.kagome_lattice import main as itai_create_kagome_lattice
from itai.Kagome_BPSU_evolution import main as itai_find_ground_state
# from itai.get_saved_results import find_results_by_dims


def get_itai_tn(N:int, D:int):
    kagome_block_data = itai_create_kagome_lattice(n=N, _print=False, periodic=True)
    output_dict = itai_find_ground_state(d=d, D=D, N_Kagome=N, lattice_fname=kagome_block_data['lattice']['path'])
    tensor_network = output_dict['data']
    return tensor_network
    

def main(
    N:int=2,
    D:int=2
):
    itai_tn = get_itai_tn(N=N, D=D)
    # get_roy_tn()

    itai_energy = 0


if __name__ == "__main__":
    main()