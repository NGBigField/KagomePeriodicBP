import _import_src  ## Needed to import src folders when scripts are called from an outside directory

from unit_cell.get_from._simple_update import load_or_compute_tnsu_unit_cell





def main_test(D=2):
    unit_cell = load_or_compute_tnsu_unit_cell(D=D)
    print(unit_cell)

if __name__ == "__main__":
    main_test()




