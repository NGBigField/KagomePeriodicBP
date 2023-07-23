import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Tensor-Networks creation:
from tensor_networks.construction import create_kagome_tn

# BP:
from algo.belief_propagation import belief_propagation

# Config and containers:
from containers import BPConfig


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


    ## With messages:
    from containers import Message
    from algo.belief_propagation import initial_message
    from lattices.directions import BlockSide
    D = tn.dimensions.virtual_dim
    message_length = tn.num_message_connections
    messages = { 
        edge_side : Message(
            mps=initial_message(D=D, N=message_length), 
            order_direction=edge_side.orthogonal_counterclockwise_lattice_direction() 
        ) 
        for edge_side in BlockSide.all_in_counter_clockwise_order()  \
    }
    tn.connect_messages(messages)
    tn.nodes

    ## Apply BP:
    tn_with_messages, messages, stats = belief_propagation(tn, messages=None, bp_config=bp_config)

    print(stats)


def main_test():
    # draw_lattice()
    bp_test()


if __name__ == "__main__":
    main_test()