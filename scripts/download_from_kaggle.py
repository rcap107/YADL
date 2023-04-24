#%%
"""
Extremely simple script for downloading datasets from kaggle. 

Paste the full url of the dataset when prompted and the file will be downloaded in 
`/storage/store/work/rcappuzz/kaggle`

Note that this assumes that you have a Kaggle account and that you created an API key:
https://github.com/Kaggle/kaggle-api#api-credentials

"""
#%% 
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import os
from urllib.parse import urlparse
# %%
os.chdir("/home/soda/rcappuzz/work/prepare-data-lakes")
#%%
data_dir = ('/storage/store/work/rcappuzz/kaggle')
# %%
# Following instructions in https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python
# https://technowhisp.com/kaggle-api-python-documentation/
api = KaggleApi()
api.authenticate()

#%%
def extract_dataset_name(dataset_url):
    out =  urlparse(dataset_url)
    dataset_name = "/".join(out[2].split("/")[-2:])
    return dataset_name

#%%
dataset_url = input("Dataset url: ")
dataset_id = extract_dataset_name(dataset_url)
print(f"Downloading url {dataset_url}")

#%%
api.dataset_download_files(dataset_id, data_dir)

# %%
