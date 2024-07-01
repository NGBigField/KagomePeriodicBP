import os
from .saveload import force_folder_exists
from os.path import isfile
from os.path import isdir as isfolder
from os.path import sep as foldersep

def get_all_files_fullpath_in_folder(folder_full_path:str) -> list[str]:
    return [folder_full_path+foldersep+filename for filename in get_all_file_names_in_folder(folder_full_path=folder_full_path)]


def get_all_file_names_in_folder(folder_full_path:str) -> list[str]:
    return [file for file in os.listdir(folder_full_path) if not isfolder(folder_full_path+foldersep+file)]


def get_last_file_in_folder(folder_full_path:str, none_if_empty:bool=True)->str:
    file_names = get_all_file_names_in_folder(folder_full_path=folder_full_path)
    if none_if_empty and len(file_names)==0:
        return None
    return file_names[-1]