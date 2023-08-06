import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent)
    )
    import _import_src
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )


from scripts.bp_test import bp_single_call, _standard_row_from_results


def main(
    D : int = 2,
    N : int = 2,
    method : int = 0   
) -> dict:
    
    ## Parse:
    if method==0:
        with_bp = False
    else:
        with_bp = True


    ## Run:
    results = bp_single_call(
        d=2,
        D=D,
        N=N,
        with_bp=with_bp
    )

    print("Results:")
    print(results)

    row = [with_bp]+_standard_row_from_results(D, N, results)

    return row


if __name__ == "__main__":    
    main(h=2.2, method=2)

