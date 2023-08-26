import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent)
    )
    import _import_src
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )


from scripts.test_bp import bp_single_call, _standard_row_from_results


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
    func_results = bp_single_call(
        d=2,
        D=D,
        N=N,
        with_bp=with_bp
    )

    ## Collect as a row:
    row = [method]+_standard_row_from_results(D, N, func_results)
    
    ## Collect as a dict:
    # row: ["with_bp", 'D', 'N', 'A_X', 'A_Y', 'A_Z', 'B_X', 'B_Y', 'B_Z', 'C_X', 'C_Y', 'C_Z']
    results = dict(
        with_bp=row[0],
        D=row[1],
        N=row[2],
        A_X=row[3],
        A_Y=row[4],
        A_Z=row[5],
        B_X=row[6],
        B_Y=row[7],
        B_Z=row[8],
        C_X=row[9],
        C_Y=row[10],
        C_Z=row[11],
    )

    return results


if __name__ == "__main__":    
    main(h=2.2, method=2)

