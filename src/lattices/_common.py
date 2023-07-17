from tensor_networks.node import _EdgeIndicator, _PosScalarType, Direction
from utils import tuples
from dataclasses import dataclass, field


class LatticeError(ValueError): ...
class DirectionError(ValueError): ...
class OutsideLatticeError(LatticeError): ...

@dataclass
class NodePlaceHolder():
    index : int
    pos : tuple[_PosScalarType, ...]
    edges : list[_EdgeIndicator]
    directions : list[Direction]
    boundaries : set[Direction] = field(default_factory=set)


    def get_edge_in_direction(self, direction:Direction) -> _EdgeIndicator:
        edge_index = self.directions.index(direction)
        return self.edges[edge_index]
    
    def set_edge_in_direction(self, direction:Direction, value:_EdgeIndicator) -> None:
        edge_index = self.directions.index(direction)
        self.edges[edge_index] = value


def plot(lattice:list[NodePlaceHolder], node_color:str="red", node_size=40, edge_style:str="b-")->None:
    from matplotlib import pyplot as plt
    from utils import visuals


    edges_list = [node.edges for node in lattice]
    for edges in edges_list:
        for edge in edges:
            nodes = [node for node in lattice if edge in node.edges]
            if len(nodes)==2:
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
            plt.text(x_text, y_text, edge, color="green")
            
    for node in lattice:
        x, y = node.pos
        plt.scatter(x, y, c=node_color, s=node_size)
        plt.text(x,y, s=node.index)

    visuals.draw_now()
    print("Done")