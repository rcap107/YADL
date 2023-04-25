#%%
"""
Extremely simple script for downloading datasets from kaggle. 

Paste the full url of the dataset when prompted and the file will be downloaded in 
`/storage/store/work/rcappuzz/kaggle`

Note that this assumes that you have a Kaggle account and that you created an API key:
https://github.com/Kaggle/kaggle-api#api-credentials

Following instructions in 
https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python
https://technowhisp.com/kaggle-api-python-documentation/


"""
#%% 
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import os
from urllib.parse import urlparse
import zipfile

#%%
def extract_dataset_name(dataset_url):
    """Read a Kaggle url and extract username and dataset_name.
    
    The slug `username/dataset_name` is used to identify a single dataset. 

    Args:
        dataset_url (str): Full URL to the dataset.

    Returns:
        tuple(str, str): Username and dataset_name. 
    """
    out =  urlparse(dataset_url)
    username, dataset_name = out[2].split("/")[-2:]
    return username, dataset_name

def extract_file(dataset_name, data_dir):
    """Extract the given dataset, found in the provided `data_dir`. 
    If successful, return `downloaded_file_path`. 

    Args:
        dataset_name (str): Dataset name.
        data_dir (str): Path to the dir.

    Returns:
        str: Path to the destination folder. 
    """
    downloaded_file_path = Path(data_dir, dataset_name)
    with zipfile.ZipFile(str(downloaded_file_path) + ".zip", "r") as fp:
        fp.extractall(path=downloaded_file_path)
    
    return downloaded_file_path

#%%
os.chdir("/home/soda/rcappuzz/work/prepare-data-lakes")
#%%
data_dir = ('/storage/store/work/rcappuzz/kaggle')
# %%
# The Kaggle API assumes there is a kaggle.json file in ~/.kaggle with the credentials. 
api = KaggleApi()
api.authenticate()

#%%
dataset_url = input("Dataset url: ")
username, dataset_name = extract_dataset_name(dataset_url)

#%%
print(f"Downloading url {dataset_url}.")
api.dataset_download_files(str(Path(username, dataset_name)), data_dir)
print("Downloaded.")
        
# %%
print(f"Extracting {dataset_name}.")
extract_file(dataset_name, data_dir)
print("Extracted.")
