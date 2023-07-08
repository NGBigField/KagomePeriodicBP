from typing import Callable, Any, ParamSpec, TypeVar, List, Tuple
from utils import tuples, strings, errors
from utils.arguments import Stats
import time
from numpy import ndarray as np_ndarray
from functools import wraps

# ============================================================================ #
#|                             Helper Types                                   |#
# ============================================================================ #
_In = ParamSpec("_In")
_Out = TypeVar("_Out")
_T = TypeVar("_T")


def when_fails_do(func_secondary:Callable[_In, _Out])->Callable[[Callable[_In, _Out]],Callable[_In, _Out]]:
    def decorator(func_primary:Callable[_In, _Out])->Callable[_In, _Out]:
        def wrapper(*args:_In.args, **kwargs:_In.kwargs)->_Out:
            try:
                results = func_primary(*args, **kwargs)
            except Exception as e:
                strings.print_warning(
                    f"Function {func_primary.__name__!r} failed because of error:"+
                    f"\n{errors.get_traceback(e)}"+
                    f"\nRunning {func_secondary.__name__!r} in its stead."
                )
                results = func_secondary(*args, **kwargs)
            return results
        return wrapper
    return decorator


def add_stats(
    execution_time:bool=True
) -> Callable[[Callable[_In, _Out]], Callable[_In, _Out]]:  # A function that return a decorator which depends on inputs

    def _add_stats_to_object(stats:Stats, t1:float, t2:float)->None:        
        if execution_time and stats.execution_time is None:
            stats.execution_time = t2-t1

    def decorator(func:Callable[_In, _Out]) -> Callable[_In, _Out]:  # decorator that return a wrapper to `func`
        def wrapper(*args:_In.args, **kwargs:_In.kwargs) -> _Out:  # wrapper that calls `func`

            ## Call function:
            t1 = time.perf_counter()
            results = func(*args, **kwargs)
            t2 = time.perf_counter()

            ## Add stats:
            data = (t1, t2)
            if isinstance(results, Stats):
                _add_stats_to_object(results, *data)
            elif isinstance(results, tuple):
                for i, element in enumerate(results):
                    if isinstance(element, Stats):
                        _add_stats_to_object(element, *data)
                        results = tuples.copy_with_replaced_val_at_index(results, i, element)
                        break
                else:
                    # No available `Stats` object in results:
                    raise TypeError(f"Couldn't find an output of type 'Stats'.")

            ## Return
            return results

        return wrapper
    return decorator


def copy_tensors(func:Callable[_In, _Out]) -> Callable[_In, _Out]:  # decorator that return a wrapper to `func`
    def _copy_only_tensors(val:_T)->_T:
        if isinstance(val, np_ndarray):
            return val.copy()
        elif isinstance(val, list):
            new_list = list()
            for item in val:
                new_list.append(_copy_only_tensors(item))
            return new_list
        return val
    def wrapper(*args:_In.args, **kwargs:_In.kwargs) -> _Out:  # wrapper that calls `func`
        ## Copy tensors
        new_args = list()
        new_kwargs = dict()
        for value in args:
            new_args.append(_copy_only_tensors(value))
        for key, value in kwargs.items():
            new_kwargs[key] = _copy_only_tensors(value)
        ## Call function:
        results = func(*new_args, **new_kwargs)  #type: ignore
        ## Return
        return results
    return wrapper


def ignore_first_method_call(func:Callable)->Callable: # decorator that returns a wrapper:
    objects_that_already_called : List[Tuple[object, Callable]] = []

    def wrapper(self, *args, **kwargs)->Any: # wrapeer that cals the function            
        nonlocal objects_that_already_called
        if (self, func) in objects_that_already_called:
            results = func(self, *args, **kwargs)
        else:
            objects_that_already_called.append((self, func))
            results = None
        return results
    return wrapper

def multiple_tries(num:int)->Callable[[Callable[_In, _Out]], Callable[_In, _Out]]: # function that returns a decorator
    # Return decorator:
    def decorator(func:Callable[_In, _Out])->Callable[_In, _Out]: # decorator that returns a wrapper:
        # Return a wrapper to func:
        @wraps(func)
        def wrapper(*args:_In.args, **kwargs:_In.kwargs)->_Out: # wrapeer that cals the function            
            last_error = Exception("Temp Exception")
            for i in range(num):
                try:
                    results = func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                else:
                    return results
            raise last_error
        return wrapper
    return decorator


def list_tries(list_:list)->Callable[[Callable], Callable]: # function that returns a decorator
    # Return decorator:
    def decorator(func:Callable)->Callable: # decorator that returns a wrapper:
        def wrapper(*args, **kwargs)->Any: # wrapeer that cals the function            
            last_error = Exception("Temp Exception")
            for val in list_:
                try:
                    results = func(val)
                    return results
                except Exception as e:
                    last_error = e
            raise last_error
        return wrapper
    return decorator


