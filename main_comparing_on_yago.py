# %%
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
from timerit import Timerit

# %%
yago_path = Path("/storage/store3/work/jstojano/yago3/")
facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")

# %%
rows = []
# %%
ti = Timerit(num=10)

print("reading")

for engine in ["pandas", "polars"]:
    fname = "yagoFacts"
    yagofacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
    fname = "yagoTypes"
    yagotypes_path = Path(facts1_path, f"{fname}.tsv.parquet")

    for timer in ti.reset(engine):
        with timer:
            yagofacts = utils.import_from_yago(yagofacts_path, engine=engine)
            yagotypes = utils.import_from_yago(yagotypes_path, engine=engine)
    row = {
        "experiment": "reading_dataset",
        "mean": ti.mean(),
        "min": ti.min(),
        "key": engine,
    }
    rows.append(row)

with open("runs.txt", "a") as fp:
    for row in rows:
        fp.write(",".join([str(_) for _ in row.values()]) + "\n")


# %%
# Converting to pandas
print("converting")
yagofacts_pd = yagofacts.to_pandas()
yagotypes_pd = yagotypes.to_pandas()

# %% [markdown]
# ## Some profiling

# %% [markdown]
# ### Predicates

# %%
print("find_unique_predicates")
for engine, df in {"polars": yagofacts, "pandas": yagofacts_pd}.items():
    for timer in ti.reset(engine):
        unique_facts = queries.query_find_unique_predicates(df)
    row = {
        "experiment": "find_unique_predicates",
        "mean": ti.mean(),
        "min": ti.min(),
        "key": engine,
    }
    rows.append(row)

with open("runs.txt", "a") as fp:
    for row in rows:
        fp.write(",".join([str(_) for _ in row.values()]) + "\n")


# %%
print("query_count_occurrences_by_columns")
for engine, df in {"polars": yagofacts, "pandas": yagofacts_pd}.items():
    for timer in ti.reset(engine):
        count_facts = queries.query_count_occurrences_by_columns(df, "predicate")
    row = {
        "experiment": "query_count_occurrences_by_columns",
        "mean": ti.mean(),
        "min": ti.min(),
        "key": engine,
    }
    rows.append(row)

with open("runs.txt", "a") as fp:
    for row in rows:
        fp.write(",".join([str(_) for _ in row.values()]) + "\n")


print("query_count_occurrences_by_columns")
for engine, df in {"polars": yagotypes, "pandas": yagotypes_pd}.items():
    for timer in ti.reset(engine):
        unique_types = queries.query_count_occurrences_by_columns(df, "cat_object")
    row = {
        "experiment": "query_count_occurrences_by_columns",
        "mean": ti.mean(),
        "min": ti.min(),
        "key": engine,
    }
    rows.append(row)

with open("runs.txt", "a") as fp:
    for row in rows:
        fp.write(",".join([str(_) for _ in row.values()]) + "\n")


# %%
print("query_most_frequent_types")
for engine, df in {"polars": yagotypes, "pandas": yagotypes_pd}.items():
    for timer in ti.reset(engine):
        most_frequent_types = queries.query_most_frequent_types(yagotypes)
    row = {
        "experiment": "query_most_frequent_types",
        "mean": ti.mean(),
        "min": ti.min(),
        "key": engine,
    }
    rows.append(row)

# %%
with open("runs.txt", "a") as fp:
    for row in rows:
        fp.write(",".join([str(_) for _ in row.values()]) + "\n")
# %%
