# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

# For error handling:
import traceback

# for type hints:
from typing import List, TypeVar, ClassVar

from utils import strings

# ============================================================================ #
#|                               Helper Types                                 |#
# ============================================================================ #
Self = TypeVar('Self')


# ============================================================================ #
#|                           Declared Functions                               |#
# ============================================================================ #


def print_traceback(e: Exception) -> None:
    s = get_traceback(e)
    s = strings.add_color(s, strings.PrintColors.RED)
    print(s)


def get_traceback(e: Exception) -> str:
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)


