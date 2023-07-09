# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

from typing import Any, Literal, Optional, Generator

from utils import arguments, lists

import time
import string 

# for basic OOP:
from enum import Enum


# ==================================================================================== #
#|                                  Constants                                         |#
# ==================================================================================== #

class SpecialChars:
    NewLine = "\n"
    CarriageReturn = "\r"
    Tab = "\t"
    BackSpace = "\b"
    LineUp = '\033[1A'
    LineClear = '\x1b[2K'

ASCII_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ==================================================================================== #
#|                               declared classes                                     |#
# ==================================================================================== #

class StrEnum(Enum): 

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self._str_value() == other
        return super().__eq__(other)
    
    def __add__(self, other:str) -> str:
        if not isinstance(other, str):
            raise TypeError(f"other is {type(other)}")
        return self._str_value()+other
    
    def __radd__(self, other:str) -> str:
        if not isinstance(other, str):
            raise TypeError(f"other is {type(other)}")
        return other+self._str_value()
    
    def __hash__(self):
        return hash(self._str_value())
    
    def __str__(self) -> str:
        return self._str_value()
    
    def _str_value(self) -> str:
        s = self.value
        if not isinstance(s, str):
            s = self.name.lower()
        return s
        

# ==================================================================================== #
#|                              declared functions                                    |#
# ==================================================================================== #

def to_list(s:str)->list[str]:
    return [c for c in s]


def random(len:int=1)->str:
    ascii_list = to_list(ASCII_UPPERCASE)
    s = ""
    for _ in range(len):
        s += lists.random_item(ascii_list)
    return s


def formatted(
    val:Any, 
    fill:str=' ', 
    alignment:Literal['<','^','>']='>', 
    width:Optional[int]=None, 
    precision:Optional[int]=None,
    signed:bool=False
) -> str:
    
    # Check info:
    try:
        if round(val)==val and precision is None:
            force_int = True
        else:
            force_int = False
    except:
        force_int = False
           
    # Simple formats:
    format = f"{fill}{alignment}"
    if signed:
        format += "+"
    
    # Width:
    width = arguments.default_value(width, len(f"{val}"))
    format += f"{width}"            
    
    precision = arguments.default_value(precision, 0)
    format += f".{precision}f"    
        
    return f"{val:{format}}"  

def num_out_of_num(num1, num2):
    width = len(str(num2))
    format = lambda num: formatted(num, fill=' ', alignment='>', width=width )
    return format(num1)+"/"+format(num2)

def time_stamp():
    t = time.localtime()
    return f"{t.tm_year}.{t.tm_mon:02}.{t.tm_mday:02}_{t.tm_hour:02}.{t.tm_min:02}.{t.tm_sec:02}"


def insert_spaces_in_newlines(s:str, num_spaces:int) -> str:
    spaces = ' '*num_spaces
    s2 = s.replace('\n','\n'+spaces)
    return s2

def str_width(s:str, last_line_only:bool=False) -> int:
    lines = s.split('\n')
    widths = [len(line) for line in lines]
    if last_line_only:
        return widths[-1]
    else:
        return max(widths)
        
def num_lines(s:str)->int:
    n = s.count(SpecialChars.NewLine)
    return n + 1
def alphabet(upper_case:bool=False)->Generator[str, None, None]:
    if upper_case is True:
        l = list( string.ascii_uppercase )
    else:
        l = list( string.ascii_lowercase )
    for s in l:
        yield s
