from typing import Literal, TypeVar, Generator

BitType = Literal[0, 1]
BitsType = list[BitType]



T = TypeVar('T')
def _in_reverse(L: list[T]) -> Generator[T, None, None]:
    N = len(L)
    for i in range(N-1, -1, -1):
        yield L[i]

def binary2dec(bits: BitsType ) -> int:
    num = 0
    # Iterate over reversed order of bits
    for n, bit in enumerate(_in_reverse(bits)):
        num += bit*(2**n)
    return num        

def dec2binary(n: int, length: int|None = None ) -> BitsType:  
    # Transform integer to binary string:
    if length is None:
        binary_str = f"{n:0b}" 
    else:
        binary_str = f"{n:0{length}b}" 
    # Transform binary string to list:
    binary_list = [int(c) for c in binary_str]  
    return binary_list
