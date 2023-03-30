# %%
%load_ext autoreload
%autoreload 2
import src.yago.utils as utils
import src.yago.queries as queries
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import polars as pl
from pathlib import Path
import numpy as np
from tqdm import tqdm
# %%
yago_path = Path("/storage/store3/work/jstojano/yago3/")
facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")

# %% [markdown]
# ## Reading yago triplets

# %%
fname = "yagoFacts"
yagofacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
yagofacts = utils.import_from_yago(yagofacts_path, engine="polars")

# %%
fname = "yagoTypes"
yagotypes_path = Path(facts1_path, f"{fname}.tsv.parquet")
yagotypes = utils.import_from_yago(yagotypes_path, engine="polars")

# %% 
# Converting to pandas
yagofacts_pd = yagofacts.to_pandas()
yagotypes_pd = yagotypes.to_pandas()

# %% [markdown]
# ## Some profiling

# %% [markdown]
# ### Predicates

# %%
unique_facts = queries.query_find_unique_predicates(yagofacts)
print(unique_facts)

# %% [markdown]
# These predicates will be used as `attributes` in the wide-form version of YAGO. 

# %%
count_facts=queries.query_count_occurrences_by_columns(yagofacts, "predicate")
print(count_facts)

# %% [markdown]
# Selecting only the top 10 facts to work with. 

# %%
top10facts = count_facts.head(10)

# %% [markdown]
# ### Types

# %%
unique_types = queries.query_count_occurrences_by_columns(yagotypes, "cat_object")

# %%
top10types= unique_types.head(10)

# %% [markdown]
# While looking at entity types, count the number of types each entity has and select the first for each of them. 

# %%
(yagotypes.lazy().groupby(
    "subject"
).agg(
    [
        pl.first("cat_object"),
        pl.count()
    ]
).sort("count", descending=True).collect())

# %%
most_frequent_types=queries.query_most_frequent_types(yagotypes)

# %%
yagotypes.lazy().filter(
    pl.col("cat_object").is_in(top10types["cat_object"])
).select(
    pl.col("subject").unique()
).collect()

# %% [markdown]
# ## Filter facts to include only frequent types

# %% [markdown]
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# Here I am filtering the subjects to keep only those whose type is in `most_frequent_types`. 

# %%
yagofacts_frequenttypes=(yagofacts.lazy().filter(
    pl.col("subject").is_in(most_frequent_types["subject"])
).collect())

# %%
pair_tab_list = []
groups = yagofacts_frequenttypes.groupby("predicate")
for group_name, group in groups:
    print(group_name)
    pair_tab=group.select(
    [    pl.col("subject").alias("subject"),
        pl.col("cat_object").alias(group_name),
    ])
    pair_tab_list.append(pair_tab)

# %% [markdown]
# Here I am executing a self-join on `yagofacts_frequenttypes` to generate pairs of `predicate`s: these are the cases in 
# which one `subject` has multiple `predicate`s, and these predicates co-occur. 
# 
# The reason why I am extracting these pairs is because this ensures that I can build tables by joining on the `subject`. 

# %%
cooccurring_predicates=(yagofacts_frequenttypes.lazy().join(
    yagofacts_frequenttypes.lazy(), left_on="subject",
    right_on="subject", how="left"
).select(
    [
        pl.col("predicate"),
        pl.col("predicate_right")
    ]
).groupby("predicate_right").agg(
    pl.first("predicate")
).collect())

# %%
cooccurring_predicates.head(10)

# %% [markdown]
# ## Plotting co-occurring pairs

# %% [markdown]
# Here I am trying to build a histogram that shows the pairs of columns that appear together the most, and those that never
# co-occur. First off, I am looking for "at least one co-occurrence". As I'll show later, this is not very insightful 
# because there is a huge variance in the co-occurrence frequency. 

# %%
sample_cooccurring=(yagofacts_frequenttypes.lazy().join(
    yagofacts_frequenttypes.lazy(), left_on="subject",
    right_on="subject", how="left"
).select(
    [
        pl.col("predicate"),
        pl.col("predicate_right")
    ]
).groupby(
    [
        "predicate",
        "predicate_right"
    ]
).all().collect())

# %%
sample_cooccurring

# %%
cooccurring_predicates=(yagofacts_frequenttypes.lazy().join(
    yagofacts_frequenttypes.lazy(), left_on="subject",
    right_on="subject", how="left"
).select(
    [
        pl.col("predicate"),
        pl.col("predicate_right")
    ]
).collect())

# %% [markdown]
# The size of this self-join is pretty large.

# %%
cooccurring_predicates.shape

# %% [markdown]
# Then, I am counting the number of occurrences of each pair of predicates. 

# %%
count_cooccurring_predicates=(cooccurring_predicates.lazy().groupby(
    ["predicate","predicate_right"]
).agg(
    pl.count()
).sort("count", descending=True).collect())

# %% [markdown]
# As the plot above attests, there is a huge difference in the number of occurrences of each pair: the most frequent pairs
# are found millions of times, the least frequent pairs appear as few as once. 
# 
# To account for this, the heatmap has a log-normalized color bar.

# %% [markdown]
# ## Join predicates with types

# %%
types_predicates=(yagotypes.lazy().filter(
        pl.col("cat_object").is_in(top10types["cat_object"])
    ).join(
        yagofacts.lazy(),
        left_on="subject",
        right_on="subject",
        how="left"
    ).select(
        pl.col("subject"),
        pl.col("cat_object").alias("type"),
        pl.col("predicate_right").alias("predicate")
    ).unique(
        ).drop_nulls(
            ).select(
                [
                    pl.col("type"),
                    pl.col("predicate")
                ]
            ).groupby(
                [
                    pl.col("type"),
                    pl.col("predicate")
                ]
            ).agg(
                [
                    pl.count()
                ]
            ).sort("count", descending=True).collect())

# %% [markdown]
# ## Building tables

# %%
groups = yagotypes.lazy().join(
    top10types.lazy().select(pl.col("cat_object")),
    on="cat_object",
    how="inner"
).groupby(
    pl.col("cat_object")
).all().select(
    [
        pl.col("cat_object"), 
        pl.col("subject")
    ]
    ).collect()

# %%
tabs_by_type = {}
for tup in groups.iter_rows():
    type_str, values = tup
    tab = pl.DataFrame(
        {
        "type": [type_str]*len(values),
        "subject": values
        },
    )
    tabs_by_type[type_str]=tab
    print(type_str)

# %%
tabs_by_type[type_str]

# %%
groups_predicates = yagofacts.groupby("predicate")

# %%
def convert_df(df: pl.DataFrame, predicate):
    return df.select(
        pl.col("subject"),
        pl.col("cat_object").alias(predicate)
    ).lazy()

# %%
full_tables_by_type = {}
for type_str, tab in tqdm(tabs_by_type.items(), total=len(tabs_by_type)):
    full_tables_by_type[type_str] = tab.clone()
    tqdm.write(type_str)
    for pred_name, pred_group in yagofacts.groupby("predicate"):
        if pred_name in G.neighbors(type_str) and pred_name in count_facts["predicate"]:
        # if pred_name in G.neighbors(type_str) and pred_name in count_facts[:15]["predicate"]:
            transformed_tab = convert_df(pred_group, pred_name)
            full_tables_by_type[type_str]= full_tables_by_type[type_str].lazy().join(
                transformed_tab.lazy(),
                on="subject",
                how="left"
            )
    full_tables_by_type[type_str].collect()

# %% [markdown]
# ### Saving tables

# %%
dest_path = Path("data/yago3-dl")

# %%
dest_path.exists()

# %%
for type_str, tab in full_tables_by_type.items():
    fname = f"yago_typetab_{type_str}.parquet"
    tab.collect().write_parquet(Path(dest_path, fname))

# %%



