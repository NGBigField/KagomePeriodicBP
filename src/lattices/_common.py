from _types import EdgeIndicatorType, PosScalarType
from lattices.directions import Direction, BlockSide, DirectionError
from utils import tuples
from dataclasses import dataclass, field


@dataclass
class Node():
    index : int
    pos : tuple[PosScalarType, ...]
    edges : list[EdgeIndicatorType]
    directions : list[Direction]
    boundaries : set[BlockSide] = field(default_factory=set)

    def get_edge_in_direction(self, direction:Direction) -> EdgeIndicatorType:
        edge_index = self.directions.index(direction)
        return self.edges[edge_index]
    
    def set_edge_in_direction(self, direction:Direction, value:EdgeIndicatorType) -> None:
        edge_index = self.directions.index(direction)
        self.edges[edge_index] = value

    @property
    def angles(self)->list[float]:
        return [dir.angle for dir in self.directions]


def plot_lattice(
    lattice:list[Node], edges_dict:dict[str, tuple[int, int]]|None=None, 
    node_color:str="red", node_size=40, edge_style:str="b-", 
    periodic:bool=False, with_edge_names:bool=True
)->None:
    from matplotlib import pyplot as plt

    ## Plot node positions:
    for node in lattice:
        x, y = node.pos
        plt.scatter(x, y, c=node_color, s=node_size)
        plt.text(x,y, s=node.index, zorder=3)

    ## Plot edges:
    edges_list = [node.edges for node in lattice]
    for edges in edges_list:
        for edge in edges:
            nodes = [node for node in lattice if edge in node.edges]

            ## Assert with edges_dict, if given:
            if edges_dict is not None:
                indices = edges_dict[edge]
                assert set(indices) == set((node.index for node in nodes))
            
            if isinstance(edge, str) and edge.count("-")==2 and not edge[0]=='M':
                assert periodic==True
                assert len(nodes)==2
                for node in nodes:
                    direction = node.directions[ node.edges.index(edge) ]
                    x1, y1 = node.pos
                    x2, y2 = tuples.add((x1, y1), direction.unit_vector)
                    x_text = x2
                    y_text = y2
                    plt.plot([x1, x2], [y1, y2], edge_style)
                    if with_edge_names:
                        plt.text(x_text, y_text, edge, color="green")
                continue

            elif len(nodes)==2:
                n1, n2 = nodes
                x1, y1 = n1.pos
                x2, y2 = n2.pos
                x_text = (x1+x2)/2
                y_text = (y1+y2)/2

            elif len(nodes)>2:
                raise ValueError(f"len(nodes) = {len(nodes)}")
            
            elif len(nodes)==1:
                node = nodes[0]
                direction = node.directions[ node.edges.index(edge) ] 
                x1, y1 = node.pos
                x2, y2 = tuples.add((x1, y1), direction.unit_vector)
                x_text = x2
                y_text = y2

            plt.plot([x1, x2], [y1, y2], edge_style)
            if with_edge_names:
                plt.text(x_text, y_text, edge, color="green")
            


def sorted_boundary_nodes(nodes:list[Node], boundary:BlockSide)->list[Node]:
    # Get relevant nodes:
    boundary_nodes = [node for node in nodes if boundary in node.boundaries]

    # Choose sorting key:
    if   boundary is BlockSide.U:     sorting_key = lambda node: -node.pos[0]
    elif boundary is BlockSide.UR:    sorting_key = lambda node: +node.pos[1] 
    elif boundary is BlockSide.DR:    sorting_key = lambda node: +node.pos[1]
    elif boundary is BlockSide.UL:    sorting_key = lambda node: -node.pos[1]
    elif boundary is BlockSide.DL:    sorting_key = lambda node: -node.pos[1]
    elif boundary is BlockSide.D:     sorting_key = lambda node: +node.pos[0]
    else:
        raise DirectionError(f"Impossible direction {boundary!r}")

    # Sort:
    return sorted(boundary_nodes, key=sorting_key)