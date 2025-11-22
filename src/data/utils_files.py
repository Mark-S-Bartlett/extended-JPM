import pathlib as pl
import os, glob

def check_files_exist(files: list) -> bool: 
    '''
    Checks if files exist

    Parameters:
    -----------
    files: list
        list of paths to files

    Returns:
    --------
    bool
        True if all files exist, False otherwise
    '''
    return all(pl.Path(file).exists() for file in files)

def create_directories(directory_list):
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")
    return None


def get_file_paths(directory, extension=''):
    """
    Get a list of file paths with the specified extension in the given directory.

    Parameters:
    - directory (str): Path to the directory containing files.
    - extension (str, optional): File extension to filter files. Default is an empty string.

    Returns:
    - list: List of file paths.
    """
    file_paths = []

    if os.path.exists(directory):
        file_paths = [file for file in os.listdir(directory) if file.endswith(extension)]
        file_paths = glob.glob(f"{directory}/*{extension}")
        return file_paths
    else:
        print("Directory does not exist.")
        return file_paths