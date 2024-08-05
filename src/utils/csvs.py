import csv
from typing import Any, TypeVar, overload, Generic
import os, sys
from pathlib import Path
from utils import strings, lists, dicts
from project_paths import data as DATA_FOLDER
from utils.saveload import force_folder_exists
from copy import deepcopy

from dataclasses import dataclass, is_dataclass

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


def read_csv_table(fullpath:str|Path)->dict[str, list[_T]]:
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
    
def get_matching_table_element(table:dict[str, list[_T]], **kwargs)->list[dict[str, _T]]:
    # info:
    dummy_key_ = list(kwargs.keys())[0]
    dummy_val_ : list[_T] = table[dummy_key_]
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
        columns:list[str],
        name:str=strings.time_stamp()+" "+strings.random(3),
        folder:str=DEFAULT_RESULTS_CSV_FOLDER
    ) -> None:
        # Path
        force_folder_exists(folder)
        self.fullpath = folder +PATH_SEP+_standard_filename(name)
        # Columns
        self.columns = columns
        create_or_override_csv(columns, file_name=self.fullpath)

    def append(self, row:list)->None:
        append_row_to_csv(row, file_name=self.fullpath)


class _TableByOrder(Generic[_T]):
    __slots__ = "_list"

    def __init__(self, input_list:list[dict[str, _T]]) -> None:
        self._list : list[dict[str, _T]] = input_list

    def __repr__(self) -> str:
        return _table_by_order_str(self._list)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def unique_values(self, key:str) -> set[_T]:
        return lists.deep_unique(self[key])

    @overload
    def __getitem__(self, key:str) -> list[_T]:...
    @overload
    def __getitem__(self, key:int) -> dict[str, _T]:...
    def __getitem__(self, key:str|int) -> list|dict:
        if isinstance(key, str):
            return [d[key] for d in self._list]
        elif isinstance(key, int):
            return self._list[key]
        else:
            raise TypeError("Not an expected type")

    def __len__(self) -> int:
        return len(self._list)


class TableByKey(Generic[_T]):
    __slots__ = "_table"

    def __init__(self, csv_fullpath:str|None=None) -> None:
        if csv_fullpath is None:
            table = {}
        else:
            table = read_csv_table(csv_fullpath)
        self._table : dict[str, list[_T]] = table

    def __repr__(self) -> str:
        return _table_by_key_str(self._table)
    
    def __str__(self) -> str:
        return self.__repr__()

    def __iadd__(self, other:"TableByKey") -> "TableByKey":
        self.extend(other)
        return self

    def __getitem__(self, key) -> list[_T]:
        return self._table[key]

    def extend(self, other:"TableByKey") -> None:
        if self.is_empty:
            self._table = other._table
        else:
            self._table = _extend_matching_tables(self._table, other._table)

    def unique_values(self, key) -> set[str]:
        return lists.deep_unique(self[key])

    def get_matching_table_elements(self, **kwargs) -> _TableByOrder[_T]:
        matching_elements = get_matching_table_element(self._table, **kwargs)
        return _TableByOrder(matching_elements)
    
    @property
    def is_empty(self) -> bool:
        return len(self._table)==0
    
    def verify(self) -> None:
        d = self._table
        dummy_val = next(iter(d.values()))
        length = len(dummy_val)
        for key, value in d.items():
            assert len(value)==length, f"key {key!r} not matching length {length!r}. The length of this key is {len(value)!r}"
    

def  _extend_matching_tables(target:dict[str, list[_T]], source:dict[str, list[_T]]) -> dict[str, list[_T]]:
    trgt_copy = deepcopy(target)
    for key in target.keys():
        if key not in source:
            raise KeyError("Tables don't match")
        other_vals = source[key]
        trgt_copy[key].extend( other_vals )
    return trgt_copy


def _table_common_title_row_str(order_str_length:int, str_lengths:dict[str, int], keys:list[str]) -> str:
    ## Print title row:
    res = " "*order_str_length
    for key in keys:
        res += strings.formatted(key, width=str_lengths[key], alignment='^') + " "
    res += "\n" 
    res += " "*order_str_length
    for key in keys:
        res += strings.formatted("-"*len(key), width=str_lengths[key], alignment='^') + " "
    res += "\n" 
    return res


def _common_value_formatting(value:Any) -> str:
    if isinstance(value, int|float):
        if int(value)==value:
            return f"{int(value)}"
    return f"{value}"


def _table_common_value_str(value:Any, key:str, str_lengths:dict[str, int]) -> str:
    s = _common_value_formatting(value)
    _remaining_len = str_lengths[key]-len(s)
    res = s + " "*(_remaining_len+1)
    return res


def _table_by_key_str(table:dict[str, list[_T]]) -> str:
    ## Collect length data
    keys = list(table.keys())
    str_lengths = {key:0 for key in keys}
    dummy_list_ = table[keys[0]]
    num_items = len(dummy_list_)
    order_str_length = len(str(num_items-1)) + 2
    for key, _list in table.items():
        str_lengths[key] = max([len(_common_value_formatting(val)) for val in _list])

    ## Print title row:
    res = _table_common_title_row_str(order_str_length=order_str_length, str_lengths=str_lengths, keys=keys)

    ## Print values:
    for i in range(num_items):
        res += str(i)+": "
        for j, (key, list_ )in enumerate(table.items()):
            value = list_[i]
            res += _table_common_value_str(value, key, str_lengths)
        res += "\n" 

    return res


def _table_by_order_str(list_:list[dict[str, Any]]) -> str:
    ## Collect length data
    order_str_length = len(str(len(list_)-1)) + 2
    keys = list(list_[0].keys())
    str_lengths = {key:0 for key in keys}
    for dict_ in list_:
        for key, value in dict_.items():
            s = _common_value_formatting(value)
            str_lengths[key] = max(str_lengths[key], len(s))

    ## Print title row:
    res = _table_common_title_row_str(order_str_length=order_str_length, str_lengths=str_lengths, keys=keys)

    ## Print values:
    for i, dict_ in enumerate(list_):
        res += str(i)+": "
        for key in keys:
            value = dict_[key]
            res += _table_common_value_str(value, key, str_lengths)
        res += "\n" 
    
    return res