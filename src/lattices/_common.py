from tensor_networks.node import _EdgeIndicator, _PosScalarType, Direction
from dataclasses import dataclass, field


class LatticeError(ValueError): ...
class OutsideLatticeError(LatticeError): ...

@dataclass
class NodePlaceHolder():
    index : int
    pos : tuple[_PosScalarType, ...]
    edges : list[_EdgeIndicator]
    directions : list[Direction]


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
                plt.plot([x1, x2], [y1, y2], edge_style, zorder=2)

                x_mid = (x1+x2)/2
                y_mid = (y1+y2)/2
                plt.text(x_mid, y_mid, edge, color="green")
            if len(nodes)>2:
                raise ValueError(".")
            
    for node in lattice:
        x, y = node.pos
        plt.scatter(x, y, c=node_color, s=node_size, zorder=3)
        plt.text(x,y, s=node.index)

    visuals.draw_now()
    print("Done")