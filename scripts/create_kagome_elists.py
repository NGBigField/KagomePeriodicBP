import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Import our kagome structure functions:
from lattices import kagome, edges


def main(size:int) -> dict:
    nodes, triangles = kagome.create_kagome_lattice(size)
    edges_list = [node.edges for node in nodes]
    edges_dict = edges.edges_dict_from_edges_list(edges_list)

if __name__ == "__main__":
    main(size=2)