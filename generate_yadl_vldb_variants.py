# %%
import argparse
import os
import random
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from src import utils

random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_path", action="store", help="Path with base tables to replicate. "
    )
    parser.add_argument(
        "--col_resample",
        action="store",
        type=int,
        default=5,
        help="Number of subtables to generate.",
    )
    parser.add_argument(
        "--row_resample",
        action="store",
        type=int,
        default=2,
        help="Number of resamplings for each subtable.",
    )
    parser.add_argument(
        "--sample_fraction",
        action="store",
        type=float,
        default=0.7,
        help="Fraction of lines to keep for row resampling.",
    )
    parser.add_argument(
        "--minimum_arity",
        action="store",
        type=int,
        default=8,
        help="Minimum number of columns of columns.",
    )
    parser.add_argument(
        "--minimum_rows",
        action="store",
        type=int,
        default=100,
        help="Minimum number of rows of columns.",
    )

    return parser.parse_args()


# %%
if __name__ == "__main__":
    args = parse_args()
    base_path = Path(args.base_path)
    dest_path = base_path.with_stem(base_path.stem + "_" + str(args.col_resample))

    os.makedirs(dest_path, exist_ok=True)

    col_resample = args.col_resample
    row_resample = args.row_resample
    sample_fraction = args.sample_fraction

    min_occurrences = args.minimum_rows
    min_width = args.minimum_arity

    print("Generating subtables.")
    print(f"Base path: {base_path}")
    print(f"Destination path: {dest_path}")
    print(f"Number of subtables: {col_resample}")
    print(f"Number of subtable resamples: {row_resample}")
    print(f"Fraction of rows in row resamples: {sample_fraction}")
    print(f"Min. number of rows: {min_occurrences}")
    print(f"Min. number of columns: {min_width}")

    total_paths = sum(1 for _ in base_path.glob("*.parquet"))
    viable_paths = []

    arity_list = []
    for pth in tqdm(
        base_path.glob("*.parquet"), total=total_paths, desc="Scanning schemas"
    ):
        schema = pl.read_parquet_schema(pth)
        arity_list.append(len(schema))
        if len(schema) > min_width:
            viable_paths.append(pth)
    arr = np.array(arity_list)

    total_generated = 0

    for pth in tqdm(viable_paths, total=len(viable_paths), position=1):
        table_name = pth.stem

        tgt_table = pl.read_parquet(pth)
        n_rows, n_cols = tgt_table.shape
        n_cols -= 1  # Not counting column `subject`

        subject = [tgt_table.columns[0]]
        target_columns = tgt_table.columns[1:]

        num_columns = tgt_table.select(cs.numeric()).columns
        cat_columns = [
            _ for _ in tgt_table.select(~cs.numeric()).columns if _ != "subject"
        ]

        if n_rows == 0:
            continue

        new_dir = Path(dest_path, table_name)
        os.makedirs(new_dir, exist_ok=True)

        # # All columns
        total_generated += utils.generate_batch(
            dest_path,
            target_columns,
            tgt_table,
            table_name,
            subject,
            "",
            row_sample_fraction=sample_fraction,
            min_occurrences=min_occurrences,
            col_resample=col_resample,
            row_resample=row_resample,
        )

        # Numerical columns
        if len(num_columns) >= 2:
            # print(table_name)
            total_generated += utils.generate_batch(
                dest_path,
                num_columns,
                tgt_table,
                table_name,
                subject,
                "num",
                row_sample_fraction=sample_fraction,
                min_occurrences=min_occurrences,
                col_resample=col_resample,
                row_resample=row_resample,
            )

    print(f"Total generated: {total_generated}")
