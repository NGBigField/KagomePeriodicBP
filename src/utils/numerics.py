EPSILON = 0.000001

def force_zero_for_small_numbers(x:float|int)->float|int:
    if abs(x)<EPSILON:
        return 0
    else:
        return x
    
def force_integers_on_close_to_round(x:float|int)->float|int:
    if abs(int(x)-x)<EPSILON:
        return int(x)
    return x
    