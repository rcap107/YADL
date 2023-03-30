import pandas as pd
from typing import Union
import polars as pl

def query_find_unique_predicates(df: Union[pd.DataFrame, pl.DataFrame]):
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
        return df.lazy().select(pl.col("predicate").unique()).collect()
    else:
        raise TypeError("Inappropriate dataframe type.")


def query_count_occurrences_by_columns(
    df: Union[pd.DataFrame, pl.DataFrame], column: str = None, descending=True
):
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
        return (
            df.lazy()
            .groupby(column)
            .agg([pl.count()])
            .sort("count", descending=descending)
        ).collect()
    else:
        raise TypeError("Inappropriate dataframe type.")

def query_groupby_count_select_first(df: Union[pd.DataFrame, pl.DataFrame]):
    if type(df) == pl.DataFrame:
        q = (df.lazy().groupby(
        "subject"
            ).agg(
                [
                    pl.first("cat_object"),
                    pl.count()
                ]
            ).sort("count", descending=True).collect())
    elif type(df) == pd.DataFrame:
        q=(df.groupby("subject")["cat_object"].agg(
            [
                "first",
                "count"
            ]
        ).reset_index())
    else:
        raise TypeError
    
    return q


def query_most_frequent_types(df: Union[pd.DataFrame, pl.DataFrame]):
    if type(df) == pd.DataFrame:
        top10freq = df.value_counts("cat_object").sort_values(ascending=False)[:10]
        q = df.loc[
            df["cat_object"].isin(top10freq.index)
        ]["subject"]
    elif type(df) == pl.DataFrame:
        q=(df.lazy().filter(
            pl.col("cat_object").is_in(
                df.lazy().groupby(
                    "cat_object"
                    ).agg(
                        [
                            pl.count()
                        ]
                        ).sort(
                            "count", descending=True
                            ).limit(10).select(
                                pl.col("cat_object")
                                ).collect()["cat_object"]
                )
            ).select(pl.col("subject"))
            ).collect()
    else:
        raise TypeError
    
    return q



def select_only_frequent_types(df: Union[pd.DataFrame, pl.DataFrame]):
    if type(df) == pl.DataFrame:
        pass
    elif type(df) == pd.DataFrame:
        pass
    else:
        raise TypeError
    
def select_cooccurring_predicates(df: Union[pd.DataFrame, pl.DataFrame]):
    if type(df) == pl.DataFrame:
        pass
    elif type(df) == pd.DataFrame:
        pass
    else:
        raise TypeError
    
# def most_frequent_types(df: Union[pd.DataFrame, pl.DataFrame]):
#     if type(df) == pl.DataFrame:
#         pass
#     elif type(df) == pd.DataFrame:
#         pass
#     else:
#         raise TypeError
    
# def most_frequent_types(df: Union[pd.DataFrame, pl.DataFrame]):
#     if type(df) == pl.DataFrame:
#         pass
#     elif type(df) == pd.DataFrame:
#         pass
#     else:
#         raise TypeError
    