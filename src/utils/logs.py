# ============================================================================ #
#|                                  Imports                                   |#
# ============================================================================ #
from collections.abc import Mapping
import sys, os

import logging
from logging import Logger
from typing import Any, Final, Mapping
from utils import strings, saveload, prints, tuples, lists
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



def search_words_in_log(
    filename:str,
    words:Iterable[str]
)->tuple[list[str], ...]:
    ## Read file:
    folder = project_paths.logs
    name_with_extension = saveload._common_name(filename, typ='log')
    full_path = str(folder)+PATH_SEP+name_with_extension

    ## Init outputs:
    res = [list() for _ in words]

    ## Iterate:
    with open(full_path, "r") as file:
        for line in file:    
            for word_index, word in enumerate(words):

                location_in_line = strings.search_pattern_in_text(word, line)
                if location_in_line != -1:  # if found
                    index_after_word = location_in_line + len(word)
                    proceeding_str = line[index_after_word:]
                    res[word_index].append(proceeding_str)

    return tuple(res)

    