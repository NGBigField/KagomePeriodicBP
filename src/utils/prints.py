from utils.strings import StrEnum, SpecialChars, num_out_of_num
from utils import decorators, lists, arguments
from typing import Any, Literal, Optional, TextIO, List

# For defining print std_out or other:
import sys

from numpy import inf


class StaticPrinter():

    def __init__(self, print_out:TextIO=sys.stdout, in_place:bool=False) -> None:
        self.printed_lines_lengths : List[int] = []
        self.print_out : TextIO = print_out
        self.in_place : bool = in_place

    @property
    def end_char(self)->str:
        if self.in_place:
            return ''
        else:
            return '\n'

    def _print(self, s:str, end:Optional[str]=None)->None:
        end = arguments.default_value(end, default=self.end_char)
        file = self.print_out
        print(s, end=end, file=file)
    
    @decorators.ignore_first_method_call
    def clear(self) -> None:
        # Get info about what was printed until now:
        reversed_prined_lengths = self.printed_lines_lengths.copy()
        reversed_prined_lengths.reverse()

        # Act according to `in_place`:
        for is_first, is_last, line_width in lists.iterate_with_edge_indicators(reversed_prined_lengths):
            if self.in_place:
                if not is_first:
                    pass   #TODO: Here we have a small bug that causes stacked static printers to override one-another                    
                self._print(SpecialChars.BackSpace*line_width)
                self._print(" "*line_width)
                self._print(SpecialChars.BackSpace*line_width)
                if not is_last:
                    self._print(SpecialChars.LineUp)
            else:
                self._print(SpecialChars.LineUp, end=SpecialChars.LineClear)

        # Reset printed lengths:
        self.printed_lines_lengths = []
                
    def print(self, s:str) -> None:
        if self.print_out is None:
            return
        self.clear()
        print_lines = s.split(SpecialChars.NewLine)
        self.printed_lines_lengths = [len(line) for line in print_lines]
        self._print(s)
    


class StaticNumOutOfNum():
    def __init__(self, expected_end:int, print_prefix:str="", print_suffix:str="", print_out:TextIO=sys.stdout, in_place:bool=False) -> None:
        self.static_printer : StaticPrinter = StaticPrinter(print_out=print_out, in_place=in_place)
        self.expected_end :int = expected_end    
        self.print_prefix :str = print_prefix
        self.print_suffix :str = print_suffix
        self.counter : int = -1
        self._sparse_show_counter : int = 0
        self._as_iterator : bool = False
        # First print:
        if expected_end>0:
            self._show()

    def __next__(self) -> int:
        try:
            val = self.next()
        except StopIteration:
            self.clear()
            raise StopIteration
        return val

    def __iter__(self) -> "StaticNumOutOfNum":
        self._as_iterator = True
        return self

    def _check_end_iterations(self)->bool:
        return self._as_iterator and self.iteration_num > self.expected_end

    def next(self, increment:int=1, extra_str:Optional[str]=None, every:int=1) -> int:
        self.counter += increment
        self._sparse_show_counter += 1
        if self._sparse_show_counter < every:
            return self.counter
        self._sparse_show_counter = 0
        self._show(extra_str)
        if self._check_end_iterations():
            raise StopIteration
        return self.counter

    def append_extra_str(self, extra_str:str)->None:
        self._show(extra_str)

    def clear(self):
        self.static_printer.clear()

    def _print(self, s:str):
        self.static_printer.print( s + self.print_suffix )

    @property
    def iteration_num(self) -> int:
        """Iteration-Number starting from 1"""
        return self.counter+1

    def _show(self, extra_str:Optional[str]=None):
        i = self.iteration_num
        expected_end = int( self.expected_end )
        s = num_out_of_num(i, expected_end)
        self._print( s )


class ProgressBar(StaticNumOutOfNum):
    def __init__(self, expected_end:int, print_prefix:str="", print_suffix:str="", print_length:int=60, print_out:TextIO=sys.stdout, in_place:bool=False): 
        # Save basic data:        
        self.print_length :int = print_length
        super().__init__(expected_end, print_prefix, print_suffix, print_out, in_place)

    @staticmethod
    def inactive()->"InactiveProgressBar":
        return InactiveProgressBar()

    @staticmethod
    def unlimited(print_prefix:str="", print_suffix:str="", print_length:int=60, print_out:TextIO=sys.stdout, in_place:bool=False)->"ProgressBar":
        return UnlimitedProgressBar(print_prefix=print_prefix, print_suffix=print_suffix, print_length=print_length, print_out=print_out, in_place=in_place)

    def _show(self, extra_str:Optional[str]=None):
        # Unpack properties:
        i = self.iteration_num
        prefix = self.print_prefix
        expected_end = int( self.expected_end )
        print_length = int( self.print_length )

        # Derive print:
        if i>expected_end:
            crnt_bar_length = print_length
        else:
            crnt_bar_length = int(print_length*i/expected_end)
        s = f"{prefix}[{u'█'*crnt_bar_length}{('.'*(print_length-crnt_bar_length))}] {i:d}/{expected_end:d}"

        if extra_str is not None:
            s += " "+extra_str

        # Print:
        self._print(s)


class InactiveProgressBar(ProgressBar):
    def __init__(self):
        print_prefix=""
        print_suffix=""
        print_length=60 
        print_out=sys.stdout
        in_place=False
        expected_end=-1
        super().__init__(expected_end, print_prefix, print_suffix, print_length, print_out, in_place)

    def next(self, *args, **kwargs)->None:
        return None

    def __next__(self)->None:
        return None

    def __iter__(self):
        i = 0
        while True:
            yield i
            i += 1

    def _show(self, *args, **kwargs)->None:
        return None


class UnlimitedProgressBar(ProgressBar):
    def __init__(self, print_prefix: str = "", print_suffix: str = "", print_length: int = 60, print_out: TextIO = sys.stdout, in_place: bool = False):
        expected_end = -1
        super().__init__(expected_end, print_prefix, print_suffix, print_length, print_out, in_place)

    def _check_end_iterations(self)->Literal[False]:
        return False
        
    def _show(self, extra_str:Optional[str]=None):
        # Unpack properties:
        prefix = self.print_prefix
        print_length = int( self.print_length )
        marker_loc =self.counter % print_length
        if marker_loc == 0:
            marker_loc = print_length

        # Derive print:        
        s =  f"{prefix}["
        s += f"{('.'*(marker_loc-1))}"
        s += f"{u'█'}"
        s += f"{('.'*(print_length-marker_loc))}"
        s += f"] {self.counter:d}"

        if extra_str is not None:
            s += " "+extra_str

        # Print:
        self._print(s)

class PrintColors(StrEnum):
    DEFAULT = '\033[0m'
    # Styles
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    UNDERLINE_THICK = '\033[21m'
    HIGHLIGHTED = '\033[7m'
    HIGHLIGHTED_BLACK = '\033[40m'
    HIGHLIGHTED_RED = '\033[41m'
    HIGHLIGHTED_GREEN = '\033[42m'
    HIGHLIGHTED_YELLOW = '\033[43m'
    HIGHLIGHTED_BLUE = '\033[44m'
    HIGHLIGHTED_PURPLE = '\033[45m'
    HIGHLIGHTED_CYAN = '\033[46m'
    HIGHLIGHTED_GREY = '\033[47m'

    HIGHLIGHTED_GREY_LIGHT = '\033[100m'
    HIGHLIGHTED_RED_LIGHT = '\033[101m'
    HIGHLIGHTED_GREEN_LIGHT = '\033[102m'
    HIGHLIGHTED_YELLOW_LIGHT = '\033[103m'
    HIGHLIGHTED_BLUE_LIGHT = '\033[104m'
    HIGHLIGHTED_PURPLE_LIGHT = '\033[105m'
    HIGHLIGHTED_CYAN_LIGHT = '\033[106m'
    HIGHLIGHTED_WHITE_LIGHT = '\033[107m'

    STRIKE_THROUGH = '\033[9m'
    MARGIN_1 = '\033[51m'
    MARGIN_2 = '\033[52m' # seems equal to MARGIN_1
    # colors
    BLACK = '\033[30m'
    RED_DARK = '\033[31m'
    GREEN_DARK = '\033[32m'
    YELLOW_DARK = '\033[33m'
    BLUE_DARK = '\033[34m'
    PURPLE_DARK = '\033[35m'
    CYAN_DARK = '\033[36m'
    GREY_DARK = '\033[37m'

    BLACK_LIGHT = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[96m'



def add_color(s:str, color:PrintColors)->str:
    return color+s+PrintColors.DEFAULT

def print_warning(s:str)->None:
    warn1color = PrintColors.HIGHLIGHTED_YELLOW
    warn2color = PrintColors.YELLOW_DARK
    s = add_color("Warning: ", warn1color)+add_color(s, warn2color)
    print(s)


