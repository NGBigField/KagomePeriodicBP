from dataclasses import dataclass, field
from pathlib import Path

from typing import TypeAlias, Union

from containers._meta import _ConfigClass
import project_paths



def _parse_fullpath(fullpath:Union[str,Path]) -> Path:
    if isinstance(fullpath, str):
        return Path(fullpath)
    elif isinstance(fullpath, Path):
        return fullpath
    else:
        raise TypeError("Not an expected type")


_FolderInterfaceType : TypeAlias = str

class _FolderDescriptor:
    """ Folder Descriptor for IOConfig.
    Here we assume that the instance is a dataclass
    """
    __slots__ = '_fullpath', '_name'

    def __init__(self, name:str, fullpath:str|Path) -> None:
        self._name : str = name
        self._fullpath : Path = _parse_fullpath(fullpath)

    ## =-= property for user access =-= ##
    @property
    def fullpath(self) -> _FolderInterfaceType:
        return self._fullpath.__str__()
    
    @fullpath.setter
    def fullpath(self, value:str|Path) -> None:
        self._fullpath = _parse_fullpath(value)

    @fullpath.deleter
    def fullpath(self) -> None:
        del self._fullpath

    ## =-= Other methods =-= ##
    def subfolder(self, name:str) -> str:
        return str(self._fullpath / name)

    def __str__(self) -> str:
        return self._fullpath.__str__()

    def __repr__(self) -> str:
        return self.__str__()



@dataclass
class IOConfig(_ConfigClass):
    logs : _FolderDescriptor = _FolderDescriptor('logs', project_paths.logs)
    data : _FolderDescriptor = _FolderDescriptor('data', project_paths.data)

    def __setattr__(self, k, v):
        assert k in ("logs", "data")
        if isinstance(v, str):
            v = _FolderDescriptor(k, v)
        return super().__setattr__(k, v)

    def __repr__(self) -> str:
        return super().__repr__()
    

    