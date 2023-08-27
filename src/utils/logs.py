# ============================================================================ #
#|                                  Imports                                   |#
# ============================================================================ #
import sys, os

if __name__ == "__main__":
    import pathlib
    sys.path.append(pathlib.Path(__file__).parent.parent.__str__())
       
import logging
from logging import Logger
from typing import Final, Mapping
from utils import strings, saveload, prints
from enum import Enum, auto
import types


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

class _MyCFormatter(logging.Formatter):

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
    
class _MyFFormatter(logging.Formatter):

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



def _to_file(logger:Logger, msg: object, *args: object, **kwargs) -> None:
    raise NotImplementedError("Not yet")
    for hdlr in c.handlers:
        if record.levelno >= hdlr.level:
            hdlr.handle(record)
    self._log(LoggerLevels.ONLYFILE.value, msg, args, **kwargs)

def get_logger(
    verbose:bool=False,
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
    
    # Derive fullpath:
    filename = _force_log_file_extension(filename)
    fullpath = _get_fullpath(filename)
    
    ## Configuration:
    f_formatter = _MyFFormatter(fmt=DEFAULT_PRINT_FORMAT, datefmt=DEFAULT_DATE_TIME_FORMAT)
    c_formatter = _MyCFormatter(fmt="%(message)s")
    #
    f_handler = logging.FileHandler(fullpath)
    c_handler = logging.StreamHandler(sys.stdout)
    #
    f_handler.setFormatter(f_formatter)
    c_handler.setFormatter(c_formatter)
    #
    f_handler.setLevel(logging.DEBUG)  # Write all logs to file
    c_handler.setLevel(level.value)    # Print only logs above level
    
    ## set handlers:        
    logger.addHandler(f_handler)      
    logger.addHandler(c_handler)   

    ## add to_file function to object:
    logger.to_file = types.MethodType( _to_file, logger )  #type: ignore

    return logger


# ============================================================================ #
#|                       Scripts to analyze saved logs:                       |#
# ============================================================================ #


        
        
        

     

# ============================================================================ #
#|                                    Test                                    |#
# ============================================================================ #     



if __name__ == "__main__":
    # _main_test()
    pass