from src.yago.utils import *
from pathlib import Path



if __name__ == "__main__":
    yago_path = Path("/storage/store3/work/jstojano/yago3/")
    facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
    facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")
    
    fname = "yagoFacts"
    yagofacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
    
    yagofacts = import_from_yago(yagofacts_path, engine="polars")
    
    fname = "yagoTypes"
    yagotypes_path = Path(facts1_path, f"{fname}.tsv.parquet")
    
    yagotypes = import_from_yago(yagotypes_path, engine="polars")
    # For testing performance
    # triplets_pd = import_from_yago(filepath, engine="pandas")

