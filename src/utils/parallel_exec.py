if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)

# For multiprocessing:
from multiprocessing import Pool

# Use our utilities:
from utils import decorators

# typing hints
from typing import Any, Callable, ParamSpec, Iterable, TypeVar, Generic
_InType      = TypeVar("_InType")
_OtherInputs = TypeVar("_OtherInputs")
_OutType     = TypeVar("_OutType")


class _FunctionObject(Generic[_InType, _OtherInputs, _OutType]):
    def __init__(self, func:Callable[[_InType, _OtherInputs], _OutType], value_name:str, fixed_arguments:dict[str,Any]|None) -> None:
        self.func : Callable[[_InType, _OtherInputs], _OutType] = func
        self.single_arg_name = value_name
        self.fixed_arguments : dict[str,Any]
        if fixed_arguments is None:
            self.fixed_arguments = dict()
        elif isinstance(fixed_arguments, dict):
            self.fixed_arguments = fixed_arguments
        else: 
            raise TypeError(fixed_arguments, "of type ", type(fixed_arguments))
    
    def __call__(self, single_arg) -> _OutType:
        kwargs = self.fixed_arguments
        kwargs[self.single_arg_name] = single_arg 
        res : _OutType = self.func(**kwargs) #type: ignore  
        return res
    


def concurrent(
    func:Callable[[_InType, _OtherInputs], _OutType], 
    values:Iterable[_InType], 
    value_name:str,
    fixed_arguments:dict[str, _OtherInputs]|None=None
)->dict[_InType, _OutType]:
    f = _FunctionObject[_InType, _OtherInputs, _OutType](func=func, value_name=value_name, fixed_arguments=fixed_arguments)
    res = dict()    
    for input in values:
        res[input] = f(input)
    return res

@decorators.when_fails_do(concurrent)
def parallel(
    func:Callable[[_InType, _OtherInputs], _OutType], 
    values:Iterable[_InType], 
    value_name:str,
    fixed_arguments:dict[str, _OtherInputs]|None=None
)->dict[_InType, _OutType]:
    if not isinstance(values, list):
        values = list(values)
    f = _FunctionObject[_InType, _OtherInputs, _OutType](func=func, value_name=value_name, fixed_arguments=fixed_arguments)
    with Pool() as pool:
        results = pool.map(f, values)  
    assert len(results)==len(values)      
    res : dict[_InType, _OutType] = {input:output for input, output in zip(values, results, strict=True) }   # type: ignore 
    return res


def concurrent_or_parallel(
    func:Callable[[_InType, _OtherInputs], _OutType], 
    values:Iterable[_InType], 
    value_name:str,
    in_parallel:bool, 
    fixed_arguments:dict[str, _OtherInputs]|None=None
)->dict[_InType, _OutType]:

    if in_parallel:
        return parallel(func=func, values=values, value_name=value_name, fixed_arguments=fixed_arguments)
    else:
        return concurrent(func=func, values=values, value_name=value_name, fixed_arguments=fixed_arguments)