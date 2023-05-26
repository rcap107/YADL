import polars as pl
import src.yago.utils as utils
from pathlib import Path
import os

os.chdir(Path("~/work/prepare-data-lakes/").expanduser())

print("Reading files")
yagofacts, yagotypes = utils.read_yago_files()

dest_dir = Path("data/yago3-dl/binary")

n_type_tabs = 20
print(f"Extracting the top {n_type_tabs} subjects.")
subjects_in_selected_types, _ = utils.get_subjects_in_wordnet_categories(
    yagofacts, yagotypes, top_k=n_type_tabs
)

for gname, group in subjects_in_selected_types.groupby("predicate"):
    print(f"Working on group {gname}")
    dff = group.clone()
    dff = dff[[s.name for s in dff if not (s.null_count() == dff.height)]]
    col_name = gname.replace("<", "").replace(">", "")
    new_df = None

    # If num_object is present, select the numeric version of the parameter. 
    if "num_object" in dff.columns:
        new_df = dff.with_columns(
            pl.col("subject"),
            pl.col("num_object").alias(col_name)
        ).select(
            pl.col("subject"),
            pl.col(col_name)
        )
    # Else, use the categorical value.
    else:
        new_df = dff.with_columns(
            pl.col("subject"),
            pl.col("cat_object").alias(col_name)
        ).select(
            pl.col("subject"),
            pl.col(col_name)
        )
    if new_df is not None:
        df_name = f"yago_binary_{col_name}.parquet"
        new_df.write_parquet(Path(dest_dir, df_name))
    else:
        print(f"Something wrong with group {gname}")
