if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from copy import deepcopy
from sender import main as main_sender



BASE_VALS = {}
BASE_VALS['N'] = range(2, 11, 1)
BASE_VALS['D'] = [2, 3, 4, 5, 6, 7]
BASE_VALS['method'] = [0, 1]
BASE_VALS['seed'] = range(5)


LOW  = 0
HIGH = 1


def main():


    for D in [LOW, HIGH]:
        for N in [LOW, HIGH]:
            for par in [False, True]:

                method = int(par)

                vals = deepcopy(BASE_VALS)
                vals['method'] = [method]

                if D==LOW:  vals['D'] = [2, 3, 4]
                else:       vals['D'] = [5, 6, 7]
                
                if N==LOW:  vals['N'] = [2, 3, 4]
                else:       vals['N'] = [5, 6, 7]

                if par:     cpu=14
                else:       cpu=4

                if D==HIGH and N==HIGH:     mem=64
                elif D==HIGH or N==HIGH:    mem=16
                else:                       mem=8

                filename = f"Timing-Test-D{D}-N{N}-P{int(par)}"

                main_sender(
                    job_type="parallel_timings",
                    request_cpus=cpu,
                    request_memory_gb=mem,
                    vals=vals,
                    result_file_name=filename
                )


if __name__ == "__main__":
    main()