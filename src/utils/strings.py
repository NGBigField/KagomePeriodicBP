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


ASCII_UPPERCASE_LIST = to_list(ASCII_UPPERCASE)
DIGITS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def random(len:int=1)->str:
    s = ""
    for _ in range(len):
        s += lists.random_item(ASCII_UPPERCASE_LIST)
    return s


def random_digits(len:int=1)->str:
    s = ""
    for _ in range(len):
        s += f"{lists.random_item(DIGITS_LIST)}"
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


def search_pattern_in_text(pattern:str, text:str)->int:
    return _kmp_search(text, pattern)


def _compute_lps_array_for_kmp(pattern):
	"""
	This function computes the Longest Proper Prefix which is also a Suffix (LPS) array for a given pattern.

	Args:
		pattern: The pattern string to compute LPS for.

	Returns:
		A list representing the LPS array for the given pattern.
	"""
	n = len(pattern)
	lps = [0] * n

	i = 1
	j = 0
	while i < n:
		if pattern[i] == pattern[j]:
			lps[i] = j + 1
			i += 1
			j += 1
		else:
			if j != 0:
				j = lps[j - 1]
			else:
				lps[i] = 0
				i += 1
	return lps

def _kmp_search(text, pattern):
	"""
	This function searches for the pattern in the text using the KMP algorithm.

	Args:
		text: The text string to search in.
		pattern: The pattern string to search for.

	Returns:
		The starting index of the first occurrence of the pattern in the text, 
		or -1 if the pattern is not found.
	"""
	n = len(text)
	m = len(pattern)
	lps = _compute_lps_array_for_kmp(pattern)

	i = 0
	j = 0
	while i < n:
		if text[i] == pattern[j]:
			i += 1
			j += 1
		if j == m:
			return i - j  # pattern found at index i - j
		elif i < n and text[i] != pattern[j]:
			if j != 0:
				j = lps[j - 1]
			else:
				i += 1
	return -1  # pattern not found
