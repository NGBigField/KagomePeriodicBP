if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)
	sys.path.append(
		pathlib.Path(__file__).parent.parent.parent.__str__()
	)



# ============================================================================ #
#|                                  Imports                                   |#
# ============================================================================ #
from collections.abc import Mapping
import sys, os

import logging
from logging import Logger
from typing import Any, Final, Mapping
from utils import strings, saveload, prints
from enum import Enum, auto
import types, project_paths

from typing import Iterable


# ============================================================================ #
#|                                  Helpers                                   |#
# ============================================================================ #   

# Logger config:
DEFAULT_PRINT_FORMAT : Final = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_DATE_TIME_FORMAT : Final = "%H:%M:%S" # "%Y-%m-%d %H:%M:%S"

# File system:
PATH_SEP = os.sep
LOGS_FOLDER = os.getcwd()+PATH_SEP+"logs"


class LoggerLevels(Enum):
    LOWEST    = logging.DEBUG//2
    DEBUG     = logging.DEBUG
    INFO      = logging.INFO
    WARNING   = logging.WARNING
    ERROR     = logging.ERROR
    CRITICAL  = logging.CRITICAL
    

# ============================================================================ #
#|                                 Constants                                  |#
# ============================================================================ #   



# ============================================================================ #
#|                               Inner Functions                              |#
# ============================================================================ #   

def _force_log_file_extension(filename:str)->str:
    file_parts = filename.split(".")
    extension = file_parts[-1]
    if extension != "log":
        filename += ".log"
    return filename

def _get_fullpath(filename:str)->str:
    saveload.force_folder_exists(LOGS_FOLDER)
    fullpath = LOGS_FOLDER+PATH_SEP+filename
    return fullpath

# ============================================================================ #
#|                               Inner Classes                                |#
# ============================================================================ #           

class _MyStdoutFormatter(logging.Formatter):

    def format(self, record)->str:
        s = super().format(record)
        level_value = record.levelno

        if level_value in [LoggerLevels.CRITICAL.value, LoggerLevels.ERROR.value]:
            color = prints.PrintColors.RED
            s = prints.add_color(s, color)
        elif level_value == LoggerLevels.WARNING.value:
            warn1color = prints.PrintColors.HIGHLIGHTED_YELLOW
            warn2color = prints.PrintColors.YELLOW_DARK
            s = prints.add_color("Warning:", warn1color) + prints.add_color(s, warn2color)

        return s
    
class _MyFileFormatter(logging.Formatter):

    def format(self, record)->str:
        s = super().format(record)
        
        # Remove coloring strings from message:
        for color in prints.PrintColors:
            color_s = f"{color}"
            s = s.replace(color_s, '')

        return s


class FakeLogger(Logger):
    def to_file( self, msg: object, *args: object, **kwargs) -> None: ...
    def debug(   self, msg: object, *args: object, **kwargs) -> None: ...
    def info(    self, msg: object, *args: object, **kwargs) -> None: ...
    def warning( self, msg: object, *args: object, **kwargs) -> None: ...
    def error(   self, msg: object, *args: object, **kwargs) -> None: ...
    def critical(self, msg: object, *args: object, **kwargs) -> None: ...
        
# ============================================================================ #
#|                             Declared Functions                             |#
# ============================================================================ #           

def inactive_logger()->FakeLogger:
    """inactive_logger Fake logger object.

    Returns:
        Logger: Fake Logger-like object
    """
    return FakeLogger(name="Fake Logger", level=100)



def get_logger(
    verbose:bool=False,
    write_to_file:bool=True,
    filename:str|None=None
    )->Logger:

    # Define default arguments:
    if filename is None:
        filename = strings.time_stamp()+" "+strings.random(6)

    # Define logger level:
    if verbose:
        level = LoggerLevels.DEBUG
    else:
        level = LoggerLevels.INFO

    # Get logger obj:
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(LoggerLevels.LOWEST.value)
    
    ## Stdout (usually print to screen) handler:
    c_formatter = _MyStdoutFormatter(fmt="%(message)s")    
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(c_formatter)
    c_handler.setLevel(level.value)    # Print only logs above this level
    logger.addHandler(c_handler)   

    ## Write to file handler:
    if write_to_file:
        # Path:
        filename = _force_log_file_extension(filename)
        fullpath = _get_fullpath(filename)
        # Formatting:
        f_formatter = _MyFileFormatter(fmt=DEFAULT_PRINT_FORMAT, datefmt=DEFAULT_DATE_TIME_FORMAT)
        f_handler = logging.FileHandler(fullpath)
        f_handler.setFormatter(f_formatter)
        f_handler.setLevel(logging.DEBUG)   # Write all logs to file 
        logger.addHandler(f_handler)      
    

    return logger


def _look_for_matching_sentence(
    parts:list[str], get_values_of:Iterable[str]
)->tuple[int, list[str]]:
    for search_index, search_sentence in enumerate(get_values_of):
        for part_index, part in enumerate(parts):
            search_words = search_sentence.split(" ")
            first_search_word = search_words[0]

            ## If we found match at first word:
            if part==first_search_word:
                for i in range(part_index+1, len(parts)):
                    found_sentence = " ".join(parts[part_index:i])
                    if search_sentence == found_sentence:
                        values = " ".join(parts[i:])
                        return search_index, values
    return -1, None



def search_words_in_log(
    filename:str,
    get_values_of:Iterable[str]
)->tuple[list[str], ...]:
    ## Read file:
    folder = project_paths.logs
    name_with_extension = saveload._common_name(filename, typ='log')
    full_path = str(folder)+PATH_SEP+name_with_extension

    ## Init outputs:
    res = [list() for value in get_values_of]

    ## Iterate:
    with open(full_path, "r") as file:
        for line in file:

            parts = line.split(" ")
            search_index, values = _look_for_matching_sentence(parts, get_values_of)
            if values is not None:
                res[search_index].append(values)


    return res

    



def test():

    from matplotlib import pyplot as plt

    log_name = "AFM-stable-log"
    found_results = search_words_in_log(log_name, ("Edge-Energies", "Mean energy after sequence") )
    energies = []

    for word in found_results[1]:
        assert isinstance(word, str)
        word = word.removeprefix("= ")
        word = word.removesuffix("\n")
        energies.append(float(word))
    
    plt.plot(energies)
    plt.show()
    plt.grid()
    plt.title("Mean Energy")
    plt.xlabel("Iteration")

    print("Done.")

if __name__ == "__main__":
    test()
    print("Done")