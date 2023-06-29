import csv
DEFAULT_RESULTS_CSV_FILE_NAME = "Results.csv" 
from typing import Any
import inspect
import os
from pathlib import Path
from utils import strings


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


def _get_caller_path(i:int=1):
    _stack = inspect.stack()[0]
    _place = (_stack)[1]
    abs_path = os.path.abspath(_place)
    directory_of_1py = os.path.dirname(abs_path)
    return directory_of_1py


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
    

class CSVManager():
    def __init__(
        self, 
        columns:list,
        name:str="Results "+strings.time_stamp()+" "+strings.random(3)
    ) -> None:
        self.file_name = _standard_filename(name)
        create_or_override_csv(columns, file_name=self.file_name)

    def append(self, row:list)->None:
        append_row_to_csv(row, file_name=self.file_name)