# To call src files from scripts:
import _import_src 

# Basics:
from matplotlib import pyplot as plt
from utils import visuals

# Tensor-Networks creation:
from src.lattices.kagome import create_kagome_lattice
from src.tensor_networks.construction import create_kagome_tn




def draw_lattice():
    lattice = create_kagome_lattice(N=2)
    create_kagome_tn()

    node_color = 'red'

    for node in lattice:
        x, y = node.pos
        plt.scatter(x, y, c=node_color, zorder=3)

    edges_list = [node.edges for node in lattice]
    for edges in edges_list:
        for edge in edges:
            nodes = [node for node in lattice if edge in node.edges]
            if len(nodes)==2:
                n1, n2 = nodes
                x1, y1 = n1.pos
                x2, y2 = n2.pos
                plt.plot([x1, x2], [y1, y2], color="blue", zorder=2)
            if len(nodes)>2:
                raise ValueError(".")

    visuals.draw_now()
    print("Done")
    

def add_messages():
    d = 2
    D = 3
    N = 2
    tn = create_kagome_tn(d=d, D=D, N=N)
    
    nodes = tn.nodes
    tn.validate()
    tn.plot()
    
    print(tn)
    





def main_test():
    # draw_lattice()
    add_messages()


if __name__ == "__main__":
    main_test()