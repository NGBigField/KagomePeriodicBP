# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

# configuratoin:
from _config_reader import SAVE_FILES_WITH

# Typing hints:
from typing import (
    Tuple,
    Optional,
    Any,
    List,
	Generator,
    Literal,
)

from abc import ABC, abstractmethod

from numpy import isin

# Other utilities:
from utils import arguments, strings, errors
import project_paths

# Operating System and files:
from pathlib import Path
import os

# For saving stuff:
if SAVE_FILES_WITH == "pickle":
    import pickle
elif SAVE_FILES_WITH == "dill":
    import dill as pickle   # A common alias to simplify things
        



# ==================================================================================== #
#|                                  Constants                                         |#
# ==================================================================================== #
PATH_SEP = os.sep
DATA_FOLDER = project_paths.data.__str__()
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
def _fullpath(name:str, sub_folder:Optional[str]=None, typ:Literal['data', 'log']='data') -> str:
    # Complete missing inputs:
    sub_folder = arguments.default_value(sub_folder, "")
    name = _common_name(name, typ)
    folder = DATA_FOLDER+PATH_SEP+sub_folder
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
    else:
        return name+"."+target_extension


def _default_load_catch_other_load(file:str) -> Any:
    try:
        data = pickle.load(file)
    except Exception as e:
        # Try the other type:
        if SAVE_FILES_WITH == "pickle":
            import dill as crnt_pickle   # A common alias to simplify things
        elif SAVE_FILES_WITH == "dill":
            import pickle as crnt_pickle
        data = crnt_pickle.load(file)
        
    return data


# ==================================================================================== #
#|                              Declared Functions                                    |#
# ==================================================================================== #
def append_text(text:str, name:str, sub_folder:Optional[str]=None, in_new_line:bool=True) -> None:
    assert isinstance(text, str), f"`text` Must be of type str"
    if in_new_line:
        text = "\n"+text
    fullpath = _fullpath(name, sub_folder, typ='log')
    mode = Modes.Append.str()
    flog = open(fullpath, mode)
    flog.write(text)
    flog.close()


def exist(name:str, sub_folder:Optional[str]=None) -> bool:
    fullpath = _fullpath(name, sub_folder)
    return os.path.exists(fullpath)


def all_saved_data() -> Generator[Tuple[str, Any], None, None]:
    for path, subdirs, files in os.walk(DATA_FOLDER):
        for name in files:
            fullpath = path + PATH_SEP + name
            file = _open(fullpath, Modes.Read.str())
            data = pickle.load(file)
            yield name, data


def save_or_load_with_fullpath(fullpath:str, mode:_Mode, var:Any|None=None) -> Any:
    file = _open(fullpath=fullpath, mode=mode.str())    
    match mode:
        case Modes.Write:
            return pickle.dump(var, file)
        case Modes.Read: 
            return _default_load_catch_other_load(file)
        case _:
            raise ValueError("Not matching an existing case")


def save(var:Any, name:Optional[str]=None, sub_folder:Optional[str]=None, if_not_exist:bool=False, print_:bool=False) -> str:
    if if_not_exist and exist(name=name, sub_folder=sub_folder):
        return None
    # Complete missing inputs:
    name = arguments.default_value(name, strings.time_stamp())
    # fullpath:
    fullpath = _fullpath(name, sub_folder)
    # Common save or load:
    save_or_load_with_fullpath(fullpath, mode=Modes.Write, var=var)
    # print:
    if print_:
        print(f"Saved file of type {type(var)!r} in path {fullpath!r}")
    return fullpath


def load(name:str, sub_folder:Optional[str]=None, none_if_not_exist:bool=False) -> Any:
    if none_if_not_exist and not exist(name=name, sub_folder=sub_folder):
        return None
    # fullpath:
    fullpath = _fullpath(name, sub_folder)
    # Common save or load:
    data = save_or_load_with_fullpath(fullpath, mode=Modes.Read)
    # Load:
    return data


def delete(name:str, sub_folder:Optional[str]=None, if_exist:bool=False)->None:
    if if_exist and not exist(name=name, sub_folder=sub_folder):
        return None 
    # fullpath:
    fullpath = _fullpath(name, sub_folder)   
    os.remove(fullpath)


def get_size(name:str, sub_folder:Optional[str]=None, if_exist:bool=False) -> int:
    if if_exist and not exist(name=name, sub_folder=sub_folder):
        return None
    # fullpath:
    fullpath = _fullpath(name, sub_folder)
    # get size
    return os.path.getsize(fullpath)

def force_subfolder_exists(folder_name:str) -> None:
    folderpath = DATA_FOLDER + PATH_SEP + folder_name
    force_folder_exists(folderpath)


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