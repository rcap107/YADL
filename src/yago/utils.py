import pandas as pd
import polars as pl
from pathlib import Path
from typing import Union


def import_from_yago(filepath: Path, engine="polars"):
    """Given a parquet file, read it assuming YAGO format. The last row is dropped. 
    Polars and Pandas are supported as engines.

    Args:
        filepath (Path): Path to the yago-like file. 
        engine (str, optional): Dataframe engine to use. Defaults to "polars".

    Raises:
        ValueError: Raise ValueError if the supplied engine is not `pandas` or `polars`.

    Returns:
        _type_: Triplets DataFrame. 
    """
    if engine == "polars":
        triplets = pl.read_parquet(filepath)[:-1]
    elif engine == "pandas":
        triplets.drop(triplets.tail(1).index, inplace=True)
        triplets = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unknown engine {engine}")
    triplets.columns = ["id", "subject", "predicate", "cat_object", "num_object"]
    return triplets


def find_unique_predicates(df:Union[pd.DataFrame, pl.DataFrame]):
    """Given a triplet dataframe, return the unique values in column `predicate`. 

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): Input dataframe.

    Raises:
        TypeError: Raise TypeError if `df` has the incorrect type. 

    Returns:
        _type_: A df that contains the unique predicates. 
    """
    if type(df) == pd.DataFrame:
        return df["predicate"].unique()
    elif type(df) == pl.DataFrame:
        return df.lazy().select(
            pl.col("predicate").unique()
        ).collect()
    else:
        raise TypeError("Inappropriate dataframe type.")

def count_occurrences_by_columns(df:Union[pd.DataFrame, pl.DataFrame], column:str=None, descending=True):
    """Given a dataframe `df` and a column `column`, return a dataframe that contains the count of values
    in the given column, sorted by default in descending order. 

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): Dataframe to evaluate.
        column (str, optional): Column to sort by. Defaults to None.
        descending (bool, optional): Whether to sort in descending order or not. Defaults to True.

    Raises:
        ValueError: Raise ValueError if the column is not provided.
        KeyError: Raise KeyError if the column is not found in the table.
        TypeError: Raise TypeError if the dataframe is not `pd.DataFrame` or `pl.DataFrame`.

    Returns:
        _type_: Dataframe that contains the values and their occurences.
    """
    if column is None:
        raise ValueError("Invalid column.")
    if column not in df.columns:
        raise KeyError(f"Column {column} not found. ")
    
    if type(df) == pd.DataFrame:
        return df.value_counts(column)
    elif type(df) == pl.DataFrame:
        return (df.lazy().groupby(
            column
        ).agg(
            [pl.count()]
        ).sort("count",descending=descending)).collect()
    else:
        raise TypeError("Inappropriate dataframe type.")
    