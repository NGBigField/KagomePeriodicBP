import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config
from tensor_networks import UnitCell

from utils import strings


# Algos we test here:
from algo.imaginary_time_evolution import full_ite
from physics import hamiltonians


d = 2


def main(
    D = 2,
    N = 6,
    live_plots:bool = 1,
    parallel:bool = 0,
    chi_factor : int = 2,
    results_filename:str = strings.time_stamp()+"_"+strings.random(4),
    afm_or_fm:str = "AFM"  # Anti-Ferro-Magnetic or Ferro-Magnetic
)->tuple[float, str]:
    
    unit_cell_file_name = f"crnt_heisenberg_{afm_or_fm}_D{D}_chi{chi_factor}_"+strings.random(3)
    # unit_cell = UnitCell.load(unit_cell_file_name)
    # if unit_cell is None:
    unit_cell = UnitCell.random(d=d, D=D)
    unit_cell._file_name = results_filename

    ## Config:
    config = Config.derive_from_physical_dim(D)
    config.dims.big_lattice_size = N
    config.visuals.live_plots = live_plots
    if afm_or_fm=="AFM":
        config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_afm, None)
    elif afm_or_fm=="FM":
        config.ite.interaction_hamiltonian = (hamiltonians.heisenberg_fm, None)
    else:
        raise ValueError("not matching any option.")
    #
    config.trunc_dim *= chi_factor
    config.bp.max_swallowing_dim *= chi_factor
    # Parallel:
    config.bp.parallel_msgs = parallel
    if N>10:
        config.bp.max_iterations *= 2 
    config.visuals.progress_bars = True
    # delta-t's:
    t = 3
    config.ite.time_steps = [1e-1]*20 + [1e-2]*10 + [1e-3]*t + [1e-4]*t + [1e-5]*t + [1e-6]*t + [1e-7]*t + [1e-8]*t + [1e-9]*t + [1e-10]*t + [1e-11]*t + [1e-12]*t + [1e-16]*t + [0]*t    
    # config.ite.time_steps = [0.2]*5 + [0.1]*20 + [0.01]*50 + [0.001]*100 + [0.05]*20 + [0.001]*100 \
    #     + [1e-4]*100 + [1e-5]*100 + [1e-6]*100 + [1e-7]*100 + [1e-8]*100 + [1e-9]*100 
    config.ite.bp_every_edge = False
    # BP:
    # config.bp.target_msg_diff = 1e-6


    ## Run:
    energy, unit_cell_out, ite_tracker, logger = full_ite(unit_cell, config=config)
    fullpath = unit_cell_out.save(results_filename+"_final")
    print("Done")

    return energy, fullpath



if __name__ == "__main__":
    main()