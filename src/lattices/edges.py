from copy import deepcopy
from utils import tuples
from _types import EdgesDictType


def edges_dict_from_edges_list(edges_list:list[list[str]])->EdgesDictType:
    vertices = {}
    for i, i_edges in enumerate(edges_list):
        for e in i_edges:
            if e in vertices:
                (j1,j2) = vertices[e]
                vertices[e] = (i,j1)
            else:
                vertices[e] = (i,i)
    return vertices


def same_dicts(d1:EdgesDictType, d2:EdgesDictType)->bool:

    if len(d1) != len(d2):
        return False
    
    d2_copy = deepcopy(d2)
    for key, tuple1 in d1.items():
        if key not in d2_copy:
            return False
        tuple2 = d2_copy[key]
        if not tuples.equal(tuple1, tuple2, allow_permutation=True):
            return False
        d2_copy.pop(key)

    if len(d2_copy)>0:
        return False
    
    return True
        