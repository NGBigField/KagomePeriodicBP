# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

# configuratoin:
from _config_reader import SAVE_FILES_WITH

# Typing hints:
from typing import Tuple, Optional, Any, List, Generator, Literal, TypeAlias, Type, overload

from abc import ABC, abstractmethod

from numpy import isin

# Other utilities:
from utils import arguments, strings, errors
import project_paths

# Operating System and files:
from pathlib import Path
import os
FileDescriptor : TypeAlias = int

# For saving stuff:
if SAVE_FILES_WITH == "pickle":
    import pickle
elif SAVE_FILES_WITH == "dill":
    import dill as pickle   # A common alias to simplify things
        



# ==================================================================================== #
#|                                  Constants                                         |#
# ==================================================================================== #
PATH_SEP = os.sep
DEFAULT_DATA_FOLDER = project_paths.data.__str__()
DATA_EXTENSION = "dat"
LOG_EXTENSION = "log"

# ==================================================================================== #
#|                                   Classes                                          |#
# ==================================================================================== #

class _Mode():
    @classmethod
    @abstractmethod
    def str(cls) -> str: ...

class Modes():  
    _ModeType : TypeAlias = _Mode   

    class Read(_Mode):
        @classmethod
        def str(cls) -> str:
            return 'rb'
    class Write(_Mode):
        @classmethod
        def str(cls) -> str:
            return 'wb'

    class Append(_Mode):
        @classmethod
        def str(cls) -> str:
            return 'a'
# ==================================================================================== #
#|                               Inner Functions                                      |#
# ==================================================================================== #
def derive_fullpath(name:str, sub_folder:Optional[str]=None, typ:Literal['data', 'log']='data', data_path:str=DEFAULT_DATA_FOLDER) -> str:
    # Complete missing inputs:
    sub_folder = arguments.default_value(sub_folder, "")
    name = _common_name(name, typ)
    folder = data_path+PATH_SEP+sub_folder
    force_folder_exists(folder)
    fullpath = folder+PATH_SEP+name
    return fullpath
    
def _open(fullpath:str, mode:str):
    return open(fullpath, mode)

def _common_name(name:str, typ:Literal['data', 'log']='data') -> str:
    assert isinstance(name, str)

    given_extension = name[-4:]
    if typ=="data":
        target_extension = DATA_EXTENSION
    elif typ=="log":
        target_extension = LOG_EXTENSION
    else:
        raise ValueError(f"Not a valid `typ` input. Given '{typ}'")
        
    if given_extension == "."+target_extension:
        return name
    elif given_extension == ".pkl" or given_extension == ".csv" :
        return name
    else:
        return name+"."+target_extension


def _default_load_catch_other_load(file) -> Any:
    try:
        data = pickle.load(file)
    except Exception as e:
        # Try the other type:
        if SAVE_FILES_WITH == "pickle":
            import dill as other_pickle   # A common alias to simplify things
        elif SAVE_FILES_WITH == "dill":
            import pickle as other_pickle
        data = other_pickle.load(file)
        
    return data


# ==================================================================================== #
#|                              Declared Functions                                    |#
# ==================================================================================== #
def append_text(text:str, name:str, sub_folder:Optional[str]=None, in_new_line:bool=True) -> None:
    assert isinstance(text, str), f"`text` Must be of type str"
    if in_new_line:
        text = "\n"+text
    fullpath = derive_fullpath(name, sub_folder, typ='log')
    mode = Modes.Append.str()
    flog = open(fullpath, mode)
    flog.write(text)
    flog.close()


def exist(name:str, sub_folder:Optional[str]=None) -> bool:
    fullpath = derive_fullpath(name, sub_folder)
    return os.path.exists(fullpath)


def all_saved_data(data_path:str=DEFAULT_DATA_FOLDER) -> Generator[Tuple[str, Any], None, None]:
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            fullpath = path + PATH_SEP + name
            file = _open(fullpath, Modes.Read.str())
            data = pickle.load(file)
            yield name, data



def save_or_load_with_fullpath(fullpath:str|Path, mode:Literal['save', 'load'], var:Any|None=None) -> Any:
    ## Derive basic path relations:
    if isinstance(fullpath, str):
        parent = Path(fullpath).parent
    elif isinstance(fullpath, Path):
        parent = fullpath.parent
        fullpath = fullpath.__str__()
    else: 
        raise TypeError("Not an expected type")

    ## Return appropriate value with common function:
    match mode:
        case 'load':
            return _common_save_or_load_with_fullpath(fullpath, Modes.Read)
        case 'save':
            force_folder_exists(parent.__str__())
            return _common_save_or_load_with_fullpath(fullpath, Modes.Write, var)



def _common_save_or_load_with_fullpath(fullpath:str, mode:Type[_Mode], var:Any|None=None) -> Any:
    file = _open(fullpath=fullpath, mode=mode.str())    
    match mode:
        case Modes.Write:
            return pickle.dump(var, file)
        case Modes.Read: 
            assert var is None
            return pickle.load(file)
        case _:
            raise ValueError("Not matching an existing case")

def save(var:Any, name:Optional[str]=None, sub_folder:Optional[str]=None, if_not_exist:bool=False, print_:bool=False) -> str:
    # Complete missing inputs:
    name = arguments.default_value(name, strings.time_stamp())
    if if_not_exist and exist(name=name, sub_folder=sub_folder):
        return ""
    # fullpath:
    fullpath = derive_fullpath(name, sub_folder)
    # Common save or load:
    _common_save_or_load_with_fullpath(fullpath, mode=Modes.Write, var=var)
    # print:
    if print_:
        print(f"Saved file of type {type(var)!r} in path {fullpath!r}")
    return fullpath


def load(name:str, sub_folder:Optional[str]=None, none_if_not_exist:bool=False) -> Any:
    if none_if_not_exist and not exist(name=name, sub_folder=sub_folder):
        return None
    # fullpath:
    fullpath = derive_fullpath(name, sub_folder)
    # Common save or load:
    data = _common_save_or_load_with_fullpath(fullpath, mode=Modes.Read)
    # Load:
    return data


def delete(name:str, sub_folder:Optional[str]=None, if_exist:bool=False)->None:
    if if_exist and not exist(name=name, sub_folder=sub_folder):
        return None 
    # fullpath:
    fullpath = derive_fullpath(name, sub_folder)   
    os.remove(fullpath)


def get_size(name:str, sub_folder:Optional[str]=None, if_exist:bool=False) -> int:
    if if_exist and not exist(name=name, sub_folder=sub_folder):
        return None  #type: ignore
    # fullpath:
    fullpath = derive_fullpath(name, sub_folder)
    # get size
    return os.path.getsize(fullpath)

def force_subfolder_exists(folder_name:str, data_path:str=DEFAULT_DATA_FOLDER) -> None:
    folder_path = data_path + PATH_SEP + folder_name
    force_folder_exists(folder_path)


def force_folder_exists(folderpath:str) -> None:
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)




# ==================================================================================== #
#|                                      Tests                                         |#
# ==================================================================================== #

def _test():
    d = dict(a="A", b=3.05)
    save(d, "test_file")
    del d
    e = load("test_file.dat")
    print(e)


if __name__ == "__main__":
    _test()
    print("Done.")