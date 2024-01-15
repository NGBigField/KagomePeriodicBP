import json
from enum import Enum, auto
import numpy as np

from utils import lists
import project_paths
from unit_cell import UnitCell


_data_folder = project_paths.src / "unit_cell" / "given_by" / "_JurajHasik"


class DataType(Enum):
    triangle_3_tensors = auto()  # 
    iPESS_5_TENSORS    = auto()  # iPESS ansatz for Kagome lattice composes five tensors

# DEFAULT_DATA_TYPE = DataType.triangle_3_tensors
DEFAULT_DATA_TYPE = DataType.iPESS_5_TENSORS



def _get_file_name(D:int, data_type:DataType)->str:
    if data_type == DataType.triangle_3_tensors:
        if D in [2, 3]: 
            return f"IPEPS_J1.0_JD0.0_D{D}_optchi40_state"
        elif D==4:
            return f"IPEPS_J1.0_JD0.0_D{D}_optchi20_state"
        else: 
            raise ValueError(f"No existing result for D={D}")
    elif data_type == DataType.iPESS_5_TENSORS:
        return f"IPESS_D_{D}_chi_40"


def _get_json_data(D:int, data_type:DataType=DEFAULT_DATA_TYPE)->dict:
    file_name = _get_file_name(D, data_type)
    full_path = str(_data_folder/file_name)+".json"
    with open(full_path, 'r') as f:
        data = json.load(f)
    return data
        
    ALLOW_VISUALS : bool = data["allow_visuals"]
    DEBUG_MODE : bool = data["debug_mode"]
    KEEP_LOGS : bool = data["keep_logs"]


def _parse_json_tensor_entry(entry:str)->tuple[np.array, float]:
    *indices, value = entry.split(" ")
    indices = [int(s) for s in indices]
    indices = tuple(indices)
    value = float(value)
    return indices, value

def _json_tensor_to_np_tensor(json_tensor:dict)->np.ndarray:
    ## Collect needed data:
    dims = json_tensor["dims"]
    entries = json_tensor["entries"]
    num_entries = json_tensor["numEntries"]

    ## Fill numpy tensor using entries:
    np_tensor = np.zeros(dims)
    for entry in entries:
        indices, value = _parse_json_tensor_entry(entry)
        np_tensor[indices] = value

    assert np_tensor.size == num_entries == lists.product(dims), f"Resulted tensor has dimensions that are not consistent with header"
    
    return np_tensor


def get_3_sites_unit_cell(D:int=2, data_type:DataType=DEFAULT_DATA_TYPE)->UnitCell:
    ## Get data:
    data = _get_json_data(D, data_type)

    ## type dependent:
    if data_type==DataType.triangle_3_tensors:
        A = _json_tensor_to_np_tensor(data['sites'][0])
        

    elif data_type==DataType.iPESS_5_TENSORS:
        tensors : dict = data["elem_tensors"]        
        T_u = _json_tensor_to_np_tensor(tensors["UP_T"])
        T_d = _json_tensor_to_np_tensor(tensors["DOWN_T"])
        B_1 = _json_tensor_to_np_tensor(tensors["BOND_S1"])
        B_2 = _json_tensor_to_np_tensor(tensors["BOND_S2"])
        B_3 = _json_tensor_to_np_tensor(tensors["BOND_S3"])


    print("Done.")