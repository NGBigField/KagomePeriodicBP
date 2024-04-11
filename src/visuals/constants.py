from typing import NamedTuple



class ScatterStyle(NamedTuple):
    color : str
    marker : str
    size : int 
    alpha : float 


class ScatterStyleFoeITE(NamedTuple):
    default : ScatterStyle
    energies_at_update : ScatterStyle 
    energies_after_segment : ScatterStyle


SCATTER_STYLE_FOR_ITE = ScatterStyleFoeITE(
    default = ScatterStyle(
        color="blue",
        marker="o",
        size=4,
        alpha=1.0
    )
    ,
    energies_at_update = ScatterStyle(
        color="black",
        marker="d",
        size=20,
        alpha=0.6
    )
    ,
    energies_after_segment = ScatterStyle(
        color="blue",
        marker="o",
        size=20,
        alpha=0.6
    )
)



EDGE_TUPLE_TO_MARKER = {
    "(A, B)" : "1",
    "(A, C)" : "2",
    "(B, A)" : "3",
    "(B, C)" : "4",
    "(C, A)" : "+",
    "(C, B)" : "x",
}