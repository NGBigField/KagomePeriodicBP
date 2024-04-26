if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
    from project_paths import add_src, add_base, add_scripts
    add_src()
    add_base()
    add_scripts()


from unit_cell import UnitCell
import numpy as np
from utils import saveload

from lattices.kagome import UpperTriangle

DATA_SUBFOLDER = "tnsu_results"


KAGOME_STRUCTURE_MATRIX  =  np.array(
    [[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T1
     [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T2
     [0, 0, 0, 0, 1, 0, 0, 0, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T3
     [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T4
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T5
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],  # T6
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0],  # T7
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0],  # T8
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 4, 0, 0],  # T9
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 4, 0],  # T10
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 4],  # T11
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 4]]  # T12
)   #E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14 E15 E16 E17 E18 E19 E20 E21 E22 E23 E24

UPPER_TRIAMGLES : list[UpperTriangle] = [
    UpperTriangle(up=0, left= 3, right= 4),
    UpperTriangle(up=6, left= 9, right=10),
    UpperTriangle(up=7, left=11, right= 8),
    UpperTriangle(up=1, left=5 , right= 2),
]



def get_3_sites_unit_cell(D:int=2)->UnitCell:
    ## Get data:
    
    filename = f"tnsu_AFH_D={D}.dat"
    data = saveload.load(filename, sub_folder=DATA_SUBFOLDER)

    for triangle_indices in UPPER_TRIAMGLES:
        triangle_nodes = UpperTriangle()

        for corner_name in UpperTriangle.field_names():
            index = triangle_indices[corner_name] 

            row_of_neighbors = KAGOME_STRUCTURE_MATRIX[index]
            edges = [i for i, val in enumerate(row_of_neighbors) if val!=0 ]
            assert len(edges)==4



    print("Done.")



if __name__=="__main__":
    from scripts.test_unit_cell_from_tnsu import main_test
    main_test()
