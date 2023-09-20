import _import_src  ## Needed to import src folders when scripts are called from an outside directory

from unit_cell.given_by.juraj_hasik import get_3_sites_unit_cell







def main_test(D=3):
    
    unit_cell = get_3_sites_unit_cell(D=D)
    print(unit_cell)

if __name__ == "__main__":
    main_test()




