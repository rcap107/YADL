import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
import pyarrow.parquet as pq
import shutil


hashes = json.load(open(Path("data/digests/benchmark-datasets-full/benchmark-datasets-full_tally.json"), "r"))

src_dir = Path("data/benchmark-datasets-full")
dest_dir = Path("data/benchmark-datasets-dedup/")


for hash, path_list in tqdm(hashes.items(), total=len(hashes)):
    # Loading only the first dataset metadata
    path_to_data = Path(*(Path(path_list[0]).parts[:-2]))
    pth_mdata = Path(path_to_data, "datasetDoc.json")
    dataset_mdata = json.load(open(pth_mdata, "r"))
    
    # Copying the dataset
    shutil.copy(path_list[0], dest_dir/Path("datasets", hash+".csv"))

    # Copying dataset metadata
    shutil.copy(pth_mdata, dest_dir/Path("metadata", hash+".metadata.json"))
        

    # Parquet is having issues, I'll just save everything as csv for now 
    continue
    
    # Loading the actual dataset
    df = pd.read_csv(path_list[0], low_memory=False)
    
    # Converting 
    tab = pa.Table.from_pandas(df)
    
    # Expanding metadata
    existing_metadata = tab.schema.metadata
    new_metadata = json.dumps(dataset_mdata).encode("utf8")
    merged_metadata = {**{"Record Metadata": new_metadata}, **existing_metadata}
    tab = tab.replace_schema_metadata(merged_metadata)
    
    