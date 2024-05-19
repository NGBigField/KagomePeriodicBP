import os

from .saveload import force_folder_exists

def get_all_file_names_in_folder(folder_full_path:str) -> list[str]:
    return [file for file in os.listdir(folder_full_path)]

def get_last_file_in_folder(folder_full_path:str, none_if_empty:bool=True)->str:
    file_names = get_all_file_names_in_folder(folder_full_path=folder_full_path)
    if none_if_empty and len(file_names)==0:
        return None
    return file_names[-1]