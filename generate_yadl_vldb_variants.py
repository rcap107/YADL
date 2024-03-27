# %%
import os
import random
from pathlib import Path

import polars as pl
import polars.selectors as cs
from sklearn.utils import murmurhash3_32
from tqdm import tqdm


# %%
def generate_table(table, col_comb, subject):
    selected_ = subject + list(col_comb)
    new_df = table.select(selected_).filter(
        pl.any_horizontal(pl.col(col_comb).is_not_null())
    )
    return new_df


# %%
def generate_batch(
    dest_dir,
    target_columns,
    base_table,
    table_name,
    subject,
    case,
    resample=False,
    sample_fraction=0.7,
    min_occurrences=100,
    limit_break=50,
    gen_break=10,
):
    new_dir = Path(dest_dir, table_name)
    break_counter = 0
    gen_counter = 0
    good_comb = set()
    bad_comb = set()

    min_sample_size = max(2, len(target_columns) - 2)
    max_sample_size = len(target_columns)

    while break_counter < limit_break and gen_counter < gen_break:
        comb = random.sample(
            target_columns, k=random.randint(min_sample_size, max_sample_size)
        )
        comb = tuple(comb)
        if (comb not in good_comb) and (comb not in bad_comb):
            table = generate_table(base_table, comb, subject)
            if len(table) > min_occurrences:
                good_comb.add(comb)
                gen_counter += 1
                fname = (
                    table_name
                    + "-"
                    + str(murmurhash3_32("-".join(comb[:])))
                    + case
                    + ".parquet"
                )
                destination_path = Path(new_dir, fname)
                table.write_parquet(destination_path)
                if resample:
                    for sample_counter in range(n_resample):
                        resampled_table = table.sample(fraction=sample_fraction)
                        fname = (
                            table_name
                            + "-"
                            + str(murmurhash3_32("-".join(comb[:])))
                            + "-"
                            + str(sample_counter)
                            + "-"
                            + case
                            + ".parquet"
                        )
                        destination_path = Path(new_dir, fname)
                        resampled_table.write_parquet(destination_path)
            else:
                bad_comb.add(comb)
                break_counter += 1
        else:
            break_counter += 1


# %%
base_path = Path("data/yadl/wordnet_vldb")
dest_path = Path("data/yadl/wordnet_vldb_wide")

os.makedirs(dest_path, exist_ok=True)

comb_size = 2
min_occurrences = 50
limit_break = 100
gen_break = 50
n_resample = 2
sample_fraction = 0.7
resample = False

minimum_width = 8


total_paths = sum(1 for _ in base_path.glob("*.parquet"))
viable_paths = []
for pth in tqdm(base_path.glob("*.parquet"), total=total_paths):
    schema = pl.read_parquet_schema(pth)
    if len(schema) > minimum_width:
        viable_paths.append(pth)

# %%
for pth in tqdm(viable_paths, total=len(viable_paths)):
    table_name = pth.stem

    tgt_table = pl.read_parquet(pth)
    n_rows, n_cols = tgt_table.shape
    n_cols -= 1  # Not counting column `subject`

    subject = [tgt_table.columns[0]]
    target_columns = tgt_table.columns[1:]

    num_columns = tgt_table.select(cs.numeric()).columns
    cat_columns = [_ for _ in tgt_table.select(~cs.numeric()).columns if _ != "subject"]

    if n_rows == 0:
        continue

    new_dir = Path(dest_path, table_name)
    os.makedirs(new_dir, exist_ok=True)

    # # All columns
    generate_batch(
        dest_path,
        target_columns,
        tgt_table,
        table_name,
        subject,
        "",
        min_occurrences,
        limit_break=limit_break,
        gen_break=gen_break,
    )

    # Numerical columns
    if len(num_columns) >= 2:
        # print(table_name)
        generate_batch(
            dest_path,
            num_columns,
            tgt_table,
            table_name,
            subject,
            "num",
            min_occurrences,
            limit_break=limit_break,
            gen_break=gen_break,
        )

# %%
