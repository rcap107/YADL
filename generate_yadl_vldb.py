#%%
import os
import re
from pathlib import Path

import polars as pl
from tqdm import tqdm

import src.yago.utils as utils

yago_path = Path("/storage/store3/work/jstojano/yago3/")


#%%
def clean_string(string_to_clean):
    pattern = re.compile(r"<{1}([a-zA-Z0-9_]+)>{1}")
    m = re.sub(pattern, "\\1", string_to_clean)
    return m


def prepare_facts_df():
    facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")
    yagofacts_path = Path(facts2_path, "yagoFacts.tsv.parquet")
    yagofacts = utils.import_from_yago(yagofacts_path)
    yagofacts = yagofacts.drop("num_object")

    yagoliteralfacts_path = Path(facts2_path, f"yagoLiteralFacts.tsv.parquet")
    yagoliteralfacts = utils.import_from_yago(yagoliteralfacts_path)

    yagodatefacts_path = Path(facts2_path, f"yagoDateFacts.tsv.parquet")
    yagodatefacts = utils.import_from_yago(yagodatefacts_path)
    yagodatefacts = (
        yagodatefacts.with_columns(
            pl.col("cat_object")
            .str.split("^^")
            .list.first()
            .str.to_datetime(strict=False)
            .dt.date()
            .cast(pl.Utf8)
            .alias("cat_object")
        )
        .drop_nulls("cat_object")
        .drop("num_object")
    )
    yagoliteralfacts = yagoliteralfacts.with_columns(
        pl.when(pl.col("num_object").is_not_null())
        .then(pl.col("num_object"))
        .otherwise(pl.col("cat_object"))
        .alias("cat_object")
    ).drop("num_object")
    return pl.concat([yagofacts, yagoliteralfacts, yagodatefacts]).drop("id")


def get_subjects():
    facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
    fname = "yagoTypes"
    yagotypes_path = Path(facts1_path, f"{fname}.tsv.parquet")
    df_types = utils.import_from_yago(yagotypes_path)

    subjects_with_wordnet = (
        df_types.filter(pl.col("cat_object").str.starts_with("<wordnet_"))
        .select(pl.col("subject"), pl.col("cat_object"))
        .rename({"cat_object": "type"})
    )
    return subjects_with_wordnet


#%%
df_facts = prepare_facts_df()

# %%
subjects_with_wordnet = get_subjects()
n_groups = len(subjects_with_wordnet.unique("type"))

#%%

max_fields = 2
dest_path = Path("data/yago3-dl/wordnet_vldb")
os.makedirs(dest_path, exist_ok=True)
#%%
for this_type, this_df in tqdm(subjects_with_wordnet.group_by("type"), total=n_groups):
    clean_type = clean_string(this_type)
    joined_df = df_facts.join(this_df.select(pl.col("subject", "type")), on="subject")
    if len(joined_df) == 0:
        continue
    base_df = joined_df.select(pl.col("subject").unique()).lazy()
    for idx, grp in joined_df.group_by(by=["predicate"]):
        this_predicate = clean_string(idx[0])
        grp = (
            grp.group_by("subject")
            .head(max_fields)
            .select(pl.col("subject", "cat_object"))
            .rename({"cat_object": this_predicate})
        )
        try:
            grp = grp.with_columns(pl.col(this_predicate).cast(pl.Float64))
        except pl.ComputeError:
            pass
        grp = grp.lazy()
        base_df = base_df.join(grp, on="subject", how="left")
    base_df = base_df.rename({"subject": clean_type})
    df_name = f"wordnet_vldb-{clean_type}.parquet"
    base_df = base_df.collect()
    if len(base_df) > 0:
        base_df.write_parquet(Path(dest_path, df_name))
# %%
