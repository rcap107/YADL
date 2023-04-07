#%%
import src.yago.utils as utils
import polars as pl
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re


#%%
yago_path = Path("/storage/store3/work/jstojano/yago3/")
facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")
#%%
fname = "yagoTypes"
yagotypes_path = Path(facts1_path, f"{fname}.tsv.parquet")
yagotypes = utils.import_from_yago(yagotypes_path, engine="polars")

fname = "yagoFacts"
yagofacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
yagofacts = utils.import_from_yago(yagofacts_path, engine="polars")

fname = "yagoLiteralFacts"
yagoliteralfacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
yagoliteralfacts = utils.import_from_yago(yagoliteralfacts_path, engine="polars")

fname = "yagoDateFacts"
yagodatefacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
yagodatefacts = utils.import_from_yago(yagodatefacts_path, engine="polars")

# %%
yagofacts_overall = pl.concat(
    [
        yagofacts,
        yagoliteralfacts,
        yagodatefacts
    ]
)
# %%
subject_count_sorted=(yagofacts_overall.lazy().groupby(
    "subject"
).agg(
    pl.count()
).sort(
    "count", descending=True
    ).collect())
# %%
selected_types=subject_count_sorted.lazy().limit(
    10000 # Taking the top 10000 subjects
    ).join(
    yagotypes.lazy(), on="subject" # Joining on types table
    ).select( # Selecting only subject and type
    [
        pl.col("subject"),
        pl.col("cat_object")
    ]
).join( # Joining again on yagotypes to get the count of subjects by type
    yagotypes.groupby( # Grouping to get the count of subjects by type
        "cat_object"
    ).agg(
        pl.count() # Count of subjects by type
    ).lazy(), on="cat_object"
).sort( # Sorting to have the most frequent types at the start
    ["subject", "count"], descending=True
).groupby(["subject"]).agg( # Grouping by subject to have all types close to each other
    pl.first("cat_object").suffix("_first") # Selecting only the first entry from each group (i.e. the most frequent)
    ).groupby( # At this point the selection is done
        "cat_object_first" # Grouping by type to remove duplicates
        ).agg(
            pl.count() # counting the number of occurrences of each type
        ).sort( # Sorting by group (for convenience)
            "count", descending=True
            ).select(
                [
                    pl.col("cat_object_first").alias("type"),
                    pl.col("count")
                ],
            ).collect()

# %%

top_selected = selected_types.filter(
    pl.col("count") > 10
)

subjects_in_selected_types = yagofacts_overall.lazy().join(top_selected.lazy().join(
    yagotypes.lazy(), left_on=["type"], right_on=["cat_object"]
).select(
    [
        pl.col("subject"),
        pl.col("type")
    ]
), left_on="subject", right_on="subject"
).collect()
# %%
types_predicates_selected = utils.join_types_predicates(yagotypes, yagofacts_overall, types_subset=top_selected)

# %%
adj_dict = {}
for row in types_predicates_selected.iter_rows():
    left, right, _ = row
    if left not in adj_dict:
        adj_dict[left] = [right]
    else:
        adj_dict[left].append(right)

for k, v in adj_dict.items():
    adj_dict[k] = set(v)

# %%
def convert_df(df: pl.DataFrame, predicate):
    return df.select(
        pl.col("subject"),
        pl.col("cat_object").alias(predicate)
    ).lazy()
# %%
type_groups = (yagotypes.lazy().join(
    selected_types.head(10).lazy(), 
    left_on="cat_object", right_on="type"
    ).select(
        [
            pl.col("subject"),
            pl.col("cat_object")
        ]
    ).groupby("cat_object").all().select(
        [
            pl.col("cat_object"),
            pl.col("subject"),
        ]
        ).collect())

# %%
tabs_by_type = {}
for tup in type_groups.iter_rows():
    type_str, values = tup
    tab = pl.DataFrame(
        {
        "type": [type_str]*len(values),
        "subject": values
        },
    )
    tabs_by_type[type_str]=tab

# %%
dest_path = Path("data/yago3-dl/seltab")

# %%
def clean_keys(type_name):
    pattern = r"(<)([a-zA-Z_]+)[_0-9]*>"
    replacement = r"\g<2>"
    return re.sub(pattern, replacement, type_name).rstrip("_")


# %%

output_format = "parquet"

for type_str, tab in tqdm(tabs_by_type.items(), total=len(tabs_by_type)):
    new_tab = tab.clone()
    tqdm.write(type_str)
    for pred_name, pred_group in yagofacts.groupby("predicate"):
        if pred_name in adj_dict[type_str] and len(new_tab.columns) < 15:
        # if pred_name in G.neighbors(type_str) and pred_name in count_facts[:15]["predicate"]:
            transformed_tab = convert_df(pred_group, pred_name)
            new_tab= new_tab.lazy().join(
                transformed_tab.lazy(),
                on="subject",
                how="left"
            )
    
    clean_type_str = clean_keys(type_str)
    if output_format == "csv":
        fname = f"yago_seltab_{clean_type_str}.csv"
        new_tab.collect().write_csv(Path(dest_path, fname))
    elif output_format == "parquet":
        fname = f"yago_seltab_{clean_type_str}.parquet"
        new_tab.collect().write_parquet(Path(dest_path, fname))
    else:
        raise ValueError(f"Unkown output format {output_format}")

# %%
