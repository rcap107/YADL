#%% 
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import os
# %%
os.chdir("/home/soda/rcappuzz/work/prepare-data-lakes")
# %%
# Following instructions in https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python
# https://technowhisp.com/kaggle-api-python-documentation/
api = KaggleApi()
api.authenticate()

# %%
dataset_id = "sobhanmoosavi/us-accidents"
output_file_name = Path(dataset_id).name
destination_path = Path("data/kaggle")

#%%
api.dataset_download_files(dataset_id, destination_path)

# %%
