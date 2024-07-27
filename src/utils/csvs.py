import csv
from typing import Any, TypeVar, overload
import os, sys
from pathlib import Path
from utils import strings, lists
from project_paths import data as DATA_FOLDER
from utils.saveload import force_folder_exists
_T = TypeVar("_T")


PATH_SEP = os.sep
DEFAULT_RESULTS_CSV_FILE_NAME = "temp_results.csv" 
DEFAULT_RESULTS_CSV_FOLDER = (DATA_FOLDER/"results").__str__()


def _standard_filename(file_name:str)->str:
    *parts, extension = file_name.split('.') 
    if extension=="csv":
        return file_name
    else:
        return file_name+".csv"


def _write_or_append_to_csv(row:list, file_name:str, mode:str)->None:
    file_name = _standard_filename(file_name)
    with open(file_name, mode, newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)


def append_row_to_csv(row:list, file_name:str=DEFAULT_RESULTS_CSV_FILE_NAME):
    _write_or_append_to_csv(row=row, file_name=file_name, mode='a')      

    
def create_or_override_csv(row:list, file_name:str=DEFAULT_RESULTS_CSV_FILE_NAME):
    _write_or_append_to_csv(row=row, file_name=file_name, mode='w')      


def read_csv_table(fullpath:str|Path)->dict[str, list[str]]:
    # Parse inputs:
    fullpath_str : str
    if isinstance(fullpath, str):
        fullpath_str = fullpath
    elif isinstance(fullpath, Path):
        fullpath_str = fullpath.absolute()
    else:
        raise TypeError()
    # prepare output:
    results = dict()
    keys : list[str] = []
    is_numeric : list[bool] = []
    # main:
    mode = 'r'
    first = True
    second = False
    with open(fullpath_str, mode, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            row = [s.replace(' ', '') for s in row]  # remove white spaces
            row = [s for s in row if s!='']
            
            if first:
                keys = row
                for key in keys:
                    results[key] = []
                first = False
                second = True
                continue
            
            if len(keys)>len(row):
                continue

            if second:                
                for key, val in zip(keys, row, strict=True):    
                    try:
                        _ = float(val)
                        is_numeric.append(True)
                    except:
                        is_numeric.append(False)
                    second = False
                
            for key, val, numeric in zip(keys, row, is_numeric, strict=True):
                if numeric:
                    val = float(val)
                results[key].append(val)
    #
    return results
    
def get_matching_table_element(table:dict[str, _T], **kwargs)->list[dict[str, _T]]:
    # info:
    dummy_key_ = list(kwargs.keys())[0]
    dummy_val_ = table[dummy_key_]
    n = len(dummy_val_)

    # search matching indices of rows::
    matching_indices = [True for _ in range(n)]
    for key, val in kwargs.items():
        values = table[key]
        crnt_matching_indices = [val==val_ for val_ in values]
        matching_indices = [t1 and t2 for t1, t2 in zip(matching_indices, crnt_matching_indices, strict=True)]

    # Retrieve rows by using matching indices        
    matching_rows = []
    for i in range(n):
        if not matching_indices[i]:
            continue
        row = dict()
        for key, list_ in table.items():
            val = list_[i]
            row[key] = val
        matching_rows.append(row)
    return matching_rows

class CSVManager():
    __slots__ = {"fullpath", "columns"}

    def __init__(
        self, 
        columns:list,
        name:str=strings.time_stamp()+" "+strings.random(3),
        folder:str|None=DEFAULT_RESULTS_CSV_FOLDER
    ) -> None:
        if folder is not None:
            force_folder_exists(folder)
        self.fullpath = folder +PATH_SEP+_standard_filename(name)
        self.columns = columns
        create_or_override_csv(columns, file_name=self.fullpath)

    def append(self, row:list)->None:
        append_row_to_csv(row, file_name=self.fullpath)


class _TableByOrder():
    __slots__ = "_list"

    def __init__(self, input_list:list[dict[str, str]]) -> None:
        self._list : list[dict[str, str]] = input_list

    def __repr__(self) -> str:
        return _table_by_order_str(self._list)
    
    def __str__(self) -> str:
        return self.__repr__()

    @overload
    def __getitem__(self, key:str) -> list[Any]:...
    @overload
    def __getitem__(self, key:int) -> dict[Any]:...
    def __getitem__(self, key:str|int) -> list[Any]|dict[Any]:
        if isinstance(key, str):
            return [d[key] for d in self._list]
        elif isinstance(key, int):
            return self._list[key]
        else:
            raise TypeError("Not an expected type")

    def __len__(self) -> int:
        return len(self._list)


class TableByKey():
    __slots__ = "_table"

    def __init__(self, csv_fullpath:str) -> None:
        self._table : dict[str, list[str]] = read_csv_table(csv_fullpath)

    def __getitem__(self, key) -> list[str]:
        return self._table[key]
    
    def unique_values(self, key) -> set[str]:
        return lists.deep_unique(self[key])

    def get_matching_table_elements(self, **kwargs) -> _TableByOrder:
        matching_elements = get_matching_table_element(self._table, **kwargs)
        return _TableByOrder(matching_elements)
    

def _table_by_order_str(list_:list[dict[str, Any]]) -> str:
    ## Collect length data
    order_str_length = len(str(len(list_)-1)) + 2
    keys = list(list_[0].keys())
    str_lengths = {key:0 for key in keys}
    for dict_ in list_:
        for key, value in dict_.items():
            s = str(value)
            str_lengths[key] = max(str_lengths[key], len(s))

    ## Print title row:
    res = " "*order_str_length
    for key in keys:
        res += strings.formatted(key, width=str_lengths[key], alignment='^') + " "
    res += "\n" 
    res += " "*order_str_length
    for key in keys:
        res += strings.formatted("-"*len(key), width=str_lengths[key], alignment='^') + " "
    res += "\n" 

    ## Print values:
    for i, dict_ in enumerate(list_):
        res += str(i)+": "
        for key in keys:
            value = dict_[key]
            s = str(value)
            _remaining_len = str_lengths[key]-len(s)
            res += s + " " + " "*_remaining_len
        res += "\n" 
    
    return res