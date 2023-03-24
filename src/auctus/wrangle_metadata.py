"""This script is needed to parse the metadata files (in format .json) and save
the relevant information in a pandas df for further analysis. At the same time,
it produces a list of "viable datasets" which contain exclusively tabular data. 

"""
import pandas as pd
import json
import os.path as osp
import os
from copy import deepcopy
import glob
    
def extract_from_json(json_dict):
    this_info = deepcopy(info_fields)

    key = json_dict["about"]["datasetID"]
    this_info["datasetName"] = json_dict["about"]["datasetName"]

    for resource in json_dict["dataResources"]:
        res_rtype = resource["resType"]
        this_info[res_rtype] += 1
        if resource["resType"] != "table":
            this_info["tabular_only"] = False

    this_info["tot_res"] = len(json_dict["dataResources"])
    
    return key, {key: this_info}

DATA_PATH = "datasets-master"

possible_dtypes = ["image","video","audio","speech","text","graph","edgeList","table","timeseries","raw"]

info_fields = {
    "datasetName": None,
    "tabular_only": True,
}
info_fields.update(zip(possible_dtypes,[0 for _ in possible_dtypes])) 

overall_stats_dict = {}
list_viable_datasets = []
list_viable_with_splits = []

folders = glob.glob(f"{DATA_PATH}/training_datasets/*/*") + \
    glob.glob(f"{DATA_PATH}/seed_datasets_current/*")
    
for folder in folders:
    dataset_name = osp.basename(folder)
    dataset_folder = dataset_name + "_dataset"
    problem_folder = dataset_name + "_problem"
    json_dataset = json.load(open(
        osp.join(folder, dataset_folder, "datasetDoc.json")))
    dsetID, info_ = extract_from_json(json_dataset)
    if info_[dsetID]["tabular_only"]:
        list_viable_datasets.append(folder) 
        if osp.exists(osp.join(folder,problem_folder, "dataSplits.csv")):
            list_viable_with_splits.append(folder)
    overall_stats_dict.update(info_)


df_stats = pd.DataFrame().from_dict(overall_stats_dict, orient="index")
print(df_stats.drop(["datasetName", "tot_res"], axis=1).sum())
print(f'Dataset with the largest number of tables: {df_stats["table"].max()}')
print(f"Number of viable dataset: {len(list_viable_datasets)}")
print(f"Number of viable dataset with splits: {len(list_viable_with_splits)}")

with open("viable_datasets.txt", "w") as fp:
    # Saving only viable with splits to maintain consistency with previous behavior
    fp.write(f"{len(list_viable_with_splits)}\n")
    for ds in list_viable_with_splits:
        fp.write(f"{ds}\n")
