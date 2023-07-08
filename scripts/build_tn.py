import _import_src
from matplotlib import pyplot as plt
from lattices.triangle import create_triangle_lattice
from utils import visuals




def main_test():
    triangle_lattice = create_triangle_lattice(N=3)

    node_color = 'red'

    for node in triangle_lattice:
        x, y = node.pos
        plt.scatter(x, y, c=node_color, zorder=3)

    edges_list = [node.edges for node in triangle_lattice]
    for edges in edges_list:
        for edge in edges:
            nodes = [node for node in triangle_lattice if edge in node.edges]
            if len(nodes)==2:
                n1, n2 = nodes
                x1, y1 = n1.pos
                x2, y2 = n2.pos
                plt.plot([x1, x2], [y1, y2], color="blue", zorder=2)
            if len(nodes)>2:
                raise ValueError(".")

    visuals.draw_now()
    print("Done")
    



if __name__ == "__main__":
    main_test()