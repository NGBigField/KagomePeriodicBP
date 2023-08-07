

def edges_dict_from_edges_list(edges_list:list[list[str]])->dict[str, tuple[int, int]]:
    vertices = {}
    for i, i_edges in enumerate(edges_list):
        for e in i_edges:
            if e in vertices:
                (j1,j2) = vertices[e]
                vertices[e] = (i,j1)
            else:
                vertices[e] = (i,i)
    return vertices