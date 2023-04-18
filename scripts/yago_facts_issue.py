#%%
"""
Script for showing how the current YAGO3 format has issues.
"""

import polars as pl
from pathlib import Path

#%%
yago_path = Path("/storage/store3/work/jstojano/yago3/")
facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")
# %%
fname = "yagoTypes"
yagotypes_path = Path(facts1_path, f"{fname}.tsv.parquet")

# %%
yf=pl.read_parquet(yagotypes_path)
# %%
# The first row is used as header, and is still present in the file.
yf.head(2)

# %%
# The last row is a flag to indicate the end of the file
yf.tail(2)
# %%
