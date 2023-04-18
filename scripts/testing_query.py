#%%
import polars as pl
from pathlib import Path

#%%
yago_path = Path("/storage/store3/work/jstojano/yago3/")
facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")
fname = "yagoFacts"
filepath = Path(facts2_path, f"{fname}.tsv.parquet")

df = pl.read_parquet(filepath)[:-1]
df.columns = ["id", "subject", "predicate", "cat_object", "num_object"]


#%%
topfreq = df.lazy().groupby("cat_object").agg(pl.count()).top_k(10, by="count")
q=(df.lazy().join(topfreq, on="cat_object").select(pl.col("subject"))
    ).collect()

# %%
topfreq = df.groupby("cat_object").agg(pl.count()).top_k(10, by="count")
q=(df.lazy().filter(pl.col("cat_object").is_in(topfreq["cat_object"])).select(pl.col("subject"))
    ).collect()

# %%
topfreq = df["cat_object"].value_counts().top_k(10, by="counts")
q=(df.lazy().filter(pl.col("cat_object").is_in(topfreq["cat_object"])).select(pl.col("subject"))
    ).collect()
# %%
