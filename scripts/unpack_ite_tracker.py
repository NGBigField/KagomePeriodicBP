import _import_src  ## Needed to import src folders when scripts are called from an outside directory

# Types in the code:
from containers import Config, ITEProgressTracker
from tensor_networks import UnitCell




def main(
    filename:str = "2024.04.18_15.49.57_006_f=0.8999999999999995"
)->tuple[UnitCell, Config]:
    
    ite_tracker = ITEProgressTracker.load(file_name=filename)
    config = ite_tracker.config
    unit_cell = ite_tracker.last_unit_cell

    return config, unit_cell 




if __name__ == "__main__":
    main()