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


# ============================================================================ #
#|                                Classes                                     |#
# ============================================================================ #

class _CumulativeErrorMeta(type):  # metaclass
    def __new__(mcs, name, bases, namespace, **kwargs):
        msg = namespace['_msg']
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if  hasattr(cls, '__cumulative_msg__'):
            cls.__cumulative_msg__.append(msg)
        else:
            cls.__cumulative_msg__ : List[str] = [msg]
        return cls
    

class CumulativeError(Exception, metaclass=_CumulativeErrorMeta):
    _msg = ''
    def __init__(self, *args: object) -> None:
        messages : List[str] = self.__cumulative_msg__
        messages.extend(args)
        msg = " ".join(messages)

        type_ = type(self.__class__.__name__, (Exception, ), {})
        self = type_(msg)

    def __new__(cls: type[Self], *args) -> Self:
        messages : List[str] = cls.__cumulative_msg__
        messages.extend(args)
        msg = " ".join(messages)

        new_constructor = type(cls.__name__, (Exception, ), {})

        return new_constructor(msg)
