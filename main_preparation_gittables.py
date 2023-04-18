"""GITTABLES
This script gathers the functions used to prepare all the files in the gittables archive, in such a way that it 
becomes possible to run experiments on them. 

More specifically:
    - it extracts all tables in their respective path
    - it converts the content of the tables into text-like files for evaluation with tokenizers
    
Author: Riccardo Cappuzzo
"""
import argparse
import glob
import json
import os
import os.path as osp
from collections import Counter
from csv import QUOTE_NONE
from zipfile import ZipFile

import pyarrow.csv as pv
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tokenizers import Encoding, Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Select the command to execute.", required=True
    )

    parser_extract = subparsers.add_parser(
        "extract", help="Extract archives to their subfolders."
    )
    parser_extract.add_argument(
        "-s",
        "--src_dir",
        action="store",
        required=True,
        help="Path to data folder " "that contains the archives. ",
    )
    parser_extract.add_argument(
        "-d",
        "--dest_dir",
        action="store",
        required=True,
        help="Path to data folder " "that will contain the extracted folders.",
    )
    parser_extract.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print archive size for all archives.",
    )
    parser_extract.set_defaults(func=extract_tables)

    parser_convert = subparsers.add_parser(
        "convert", help="Convert tables to pseudo-text format."
    )
    parser_convert.add_argument(
        "-p",
        "--data_folder",
        action="store",
        required=True,
        help="Path to the folder where tables are stored. ASSUMES PARQUET FORMAT.",
    )
    parser_convert.add_argument(
        "--n_jobs",
        action="store",
        type=int,
        default=1,
        help="Number of jobs used by joblib.",
    )
    parser_convert.set_defaults(func=convert_tables_to_text)

    parser_tokenize = subparsers.add_parser(
        "tokenize", help="Tokenize pseudo-text with a huggingface tokenizer."
    )

    parser_tokenize.add_argument(
        "--src_dir",
        action="store",
        required=True,
        help="Path to the folder that stores the corpus files.",
    )
    parser_tokenize.add_argument(
        "--dest_dir",
        action="store",
        required=True,
        help="Path to the folder where the tokenizer will be saved.",
    )
    parser_tokenize.add_argument(
        "--tokenizer_name",
        action="store",
        default="tokenizer",
        help="Name to use for the tokenizer file.",
    )

    parser_tally = subparsers.add_parser("tally", help="Measure the frequceny of occurence of tokens in the full repo.")

    parser_tally.add_argument(
        ""
    )




    args = parser.parse_args()
    return args


def extract_tables(args):
    src_dir = args.src_dir
    dest_dir = args.dest_dir
    verbose = args.verbose

    # src_dir: data/zenodo/archives
    assert osp.exists(src_dir)
    # dest_dir: data/zenodo/tables
    assert osp.exists(dest_dir)

    for filename in tqdm(sorted(os.listdir(src_dir))):
        fpath = osp.join(src_dir, filename)
        if osp.isdir(fpath):
            continue

        basename, ext = osp.splitext(filename)
        tgt_folder = osp.join(dest_dir, basename)
        # The file has already been extracted
        if osp.exists(tgt_folder):
            continue

        # If the verbose flag is set, print the size of the archive.
        if verbose:
            file_size = osp.getsize(fpath)
            print(f"{filename:.<80}{file_size/1024:.>10.0f}KB")

        os.makedirs(tgt_folder, exist_ok=True)
        fpath = osp.join(src_dir, filename)
        with ZipFile(fpath) as myzip:
            myzip.extractall(tgt_folder)
            print(fpath)

    return


def convert_tables_to_text(args):
    root_data_dir = args.data_folder
    n_jobs = args.n_jobs

    dest_dir = osp.join(root_data_dir, "txt_tables")
    src_dir = osp.join(root_data_dir, "tables")

    if not osp.exists(src_dir):
        raise OSError(f"Directory {src_dir} does not exist. Aborting.")
    if len(os.listdir(src_dir)) == 0:
        raise OSError(f"Directory {src_dir} is empty. Aborting.")

    # Creating a new dir for the dumped data
    print("Creating dirs")
    os.makedirs(dest_dir, exist_ok=True)

    for tab in glob.glob(osp.join(src_dir, "*")):
        os.makedirs(osp.join(dest_dir, osp.basename(tab)), exist_ok=True)

    # Creating a list with the paths for all tables in the collection
    print("Creating tables glob")
    list_paths = glob.glob(osp.join(src_dir, "*", "*"))
    print(f"Found {len(list_paths)} files.")
    assert len(list_paths) > 0

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

    print("Converting.")

    # Using joblib to parallelize.
    r = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(dump_table_to_csv)(idx, table_path, dest_dir)
        for idx, table_path in tqdm(
            enumerate(list_paths), position=0, leave=False, total=len(list_paths)
        )
    )

    return


def tokenize_corpus(args):
    src_dir = args.src_dir
    dest_dir = args.dest_dir
    assert osp.exists(src_dir)
    assert osp.exists(dest_dir)

    # Creating list of files for the corpus
    glob_path = osp.join(src_dir, "*/*.txt")
    corpus_files = glob.glob(glob_path)

    # Setting default values for the tokenizer
    normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits()])

    # Creating the tokenizer with the assigned normalizer and pre-tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    trainer = BpeTrainer()

    # Training tokenizer
    tokenizer.train(corpus_files, trainer)

    # Saving tokenizer to file
    tokenizer.save(osp.join(args.dest_dir, f"{args.tokenizer_name}.json"))

    return



if __name__ == "__main__":
    args = parse_args()
    args.func(args)
