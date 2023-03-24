import pandas as pd
import os.path as osp

data_file = osp.expanduser("~/store/d3m/list_files.txt")

# cleaned_file = open("~/store/d3m/list_files_cleaned.txt")

ext_tally = {}

for idx, row in enumerate(open(data_file, "r")):
    name, ext = osp.splitext(row.strip())
    if len(ext) > 0:
        if ext in ext_tally:
            ext_tally[ext] += 1
        else:
            ext_tally[ext] = 1
print(ext_tally)