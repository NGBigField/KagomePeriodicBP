import os

def get_last_file_in_folder(folder_full_path:str)->str:
    file_names = [file for file in os.listdir(folder_full_path)]
    return file_names[-1]