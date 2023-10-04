"""I am writing this script to put together the objects I need to build the new
directory tree and all the files involved."""

import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq


def dump_table_to_csv(idx, table_path, dest_dir):
    try:
        tgt_dir, path = osp.split(osp.relpath(table_path, src_dir))
        path = osp.join(dest_dir, tgt_dir, osp.basename(path) + ".txt")
        if osp.exists(path):
            return (idx, 0)
        # Reading single table from parquet file
        tab = pq.read_table(table_path)
        # Converting table to csv passing from pandas, removing separators
        # and escape characters and forcing no quoting with QUOTE_NONE
        tab.to_pandas().to_csv(
            path, index=False, sep=" ", escapechar=" ", quoting=QUOTE_NONE
        )
        return (idx, 0)
    except Exception:
        # Avoid stopping the conversion procedure, count thefailures.
        return (idx, 1)


def prepare_hash(fp, block_size=2**20):
    md5 = hashlib.md5()
    while True:
        data = fp.read(block_size)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()


class Dataset:
    def __init__(self, dataset_name, src_path, dest_path):
        self.name = dataset_name
        self.src_path = src_path
        self.dest_path = dest_path

        self.data, self.metadata, self.auxiliary_data = self.load_data()
        self.hash = self.get_hash()

        self.path_dir = Path(dest_path, self.hash)
        self.path_tables = Path(self.path_dir, "tables")
        self.path_metadata = Path(self.path_dir, "metadata")

        self.make_dirtree()

    def load_data(self):
        raise NotImplementedError("Please implement this method.")

    def get_hash(self):
        raise NotImplementedError("Please implement this method.")

    def make_dirtree(self):
        os.makedirs(self.path_dir, exist_ok=True)
        os.makedirs(self.path_tables, exist_ok=True)
        os.makedirs(self.path_metadata, exist_ok=True)

        open(Path(self.path_metadata, f"{self.hash}.metadata.json"), "w")
        open(Path(self.path_metadata, f"{self.hash}.problem.json"), "w")
        open(Path(self.path_metadata, f"{self.hash}.candidates.json"), "w")


class SeedDataset(Dataset):
    """Assumes D3M folder structure"""

    def __init__(self, dataset_name, src_path, dest_path):
        super().__init__(dataset_name, src_path, dest_path)
        self.copy_data()

    def load_data(self):
        src_tables_path = Path(self.src_path, f"{self.name}_dataset/tables")
        src_metadata_path = Path(self.src_path, f"{self.name}_dataset/datasetDoc.json")

        data = None
        auxiliary_data = []
        for ff in os.listdir(src_tables_path):
            if ff == "learningData.csv":
                data = pd.read_csv(ff)
            else:
                aux_d = pd.read_csv(ff)
                auxiliary_data.append(aux_d)

        assert data is not None

        metadata = json.load(open(src_metadata_path, "r"))

        return data, metadata, auxiliary_data

    def get_hash(self):
        dummy = io.BytesIO(self.data.to_csv(index=False).encode())
        hash = prepare_hash(dummy)
        dummy.close()

        return hash

    def copy_data(self):
        src_tables_path = Path(self.src_path, f"{self.name}_dataset/tables")
        for ff in os.listdir(src_tables_path):
            shutil.copy(Path(src_tables_path, ff), Path(self.path_tables, ff))

        src_metadata_path = Path(self.src_path, f"{self.name}_dataset/datasetDoc.json")
        shutil.copy(
            Path(src_metadata_path),
            Path(self.path_metadata, f"{self.hash}.metadata.json"),
        )

        src_problem_path = Path(self.src_path, f"{self.name}_problem/problemDoc.json")
        shutil.copy(
            Path(src_problem_path),
            Path(self.path_metadata, f"{self.hash}.problem.json"),
        )


class GitTablesDataset(Dataset):
    def __init__(self, dataset_name, dataset_hash, src_path):
        super().__init__(dataset_name, dataset_hash, src_path)

    def load_data(self):
        # TODO Implement this
        pass

    def get_hash(self):
        # TODO Implement this
        pass

    def unpack_parquet(self):
        # TODO
        # table = parquet read
        # metadata = table.metadata
        # data = table.data

        # hash data
        # set hash for path
        # copy metadata
        # copy data

        try:

            # Reading single table from parquet file
            tab = pq.read_table(self.src_path)

            # Preparing hash
            dummy = io.BytesIO(tab.to_pandas().to_csv(index=False).encode())
            self.hash = prepare_hash(dummy)
            dummy.close()

            # Saving csv
            path_table = Path(self.dest_path, self.hash, "tables", self.name + ".csv")
            if path_table.exists():
                # Table already exists, skipping
                return (idx, 0)
            pcsv.write_csv(tab, path_table)

            path_metadata = Path(
                self.dest_path, self.hash, "metadata", self.name + "metadata.json"
            )
            decoded = {k.decode(): v.decode() for k, v in tab.schema.metadata.items()}
            json.dump(decoded, open(path_metadata))
            return (idx, 0)
        except Exception:
            # Avoid stopping the conversion procedure, count thefailures.
            return (idx, 1)


def build_dirtree(root="."):
    os.makedirs(Path(root, "seed_datasets"), exist_ok=True)

    os.makedirs(Path(root, "datalake", exist_ok=True))


with open("small_set.txt", "r") as fp:
    for idx, row in enumerate(fp):
        m = hashlib.md5()
        if idx == 0:
            continue
        m.update(bytes(row, encoding="utf8"))
        digest = m.hexdigest()
        print(digest)
        path = Path(row.strip())
        ds_name = path.stem
        dataset = SeedDataset(ds_name, digest, path)
        dataset.make_dirtree(Path("debug_dirtree/seed_datasets"))
        dataset.copy_data(path)
        if idx >= 1:
            break
