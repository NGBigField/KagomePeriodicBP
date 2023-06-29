from typing import List, Dict, Tuple


def edges_dict_from_edges_list(edges_list:List[List[str]])->Dict[str, Tuple[int, int]]:
    vertices = {}
    for i, i_edges in enumerate(edges_list):
        for e in i_edges:
            if e in vertices:
                (j1,j2) = vertices[e]
                vertices[e] = (i,j1)
            else:
                vertices[e] = (i,i)
    return vertices