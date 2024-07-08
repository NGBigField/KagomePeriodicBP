if __name__ == "__main__":
    # add src to available imports:
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()   
	)

# For multiprocessing:
from _config_reader import PARALLEL_METHOD  #TODO Implement
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
  

def _parallel_with_pool(f:_FunctionObject[_InType, _OtherInputs, _OutType], values:list[_InType]) -> dict[_InType, _OutType]:
    with Pool() as pool:
        results = pool.map(f, values)  
    assert len(results)==len(values)      
    return {input:output for input, output in zip(values, results, strict=True) }  


def _parallel_with_multithreading(f:_FunctionObject[_InType, _OtherInputs, _OutType], values:list[_InType]) -> dict[_InType, _OutType]:
    pass


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
    ## Arrange inputs in lists:
    if not isinstance(values, list):
        values = list(values)
    ## _FunctionObject helps with a simplified call with multiple inputs were only 1 is different for every worker:
    f = _FunctionObject[_InType, _OtherInputs, _OutType](func=func, value_name=value_name, fixed_arguments=fixed_arguments)
    ## Decide how to execute in parallel:
    match PARALLEL_METHOD:
        case "mpi":
            raise NotImplementedError("Not yet implemented")
        case "multiprocessing_pool":
            return _parallel_with_pool(f, values)
        case "multithreading":
            return _parallel_with_multithreading(f, values)
        

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
    


if __name__ == "__main__":
    # add scripts to available imports:
    import pathlib, sys
    base = pathlib.Path(__file__).parent.parent.parent
    sys.path.append(base.__str__())

    from scripts.tests import parallel as parallel_test
    parallel_test.test_parallel_execution_time_single_input()