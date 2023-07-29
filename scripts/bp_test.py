import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn

# BP:
from algo.belief_propagation import belief_propagation

# Config and containers:
from containers import BPConfig

# Directions defining the lattice and the block:
from lattices.directions import block, lattice, BlockSide


def bp_test(
    d = 2,
    D = 3,
    N = 2
):

    ## Config:
    bp_config = BPConfig(
        max_swallowing_dim=18        
    )
    
    ## first network:
    tn = create_kagome_tn(d=d, D=D, N=N)

    ## BP:
    tn_with_messages, messages, stats = belief_propagation(tn, messages=None, bp_config=bp_config)

    print(stats)


def main_test():
    # draw_lattice()
    bp_test()


if __name__ == "__main__":
    main_test()