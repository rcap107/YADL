import re
from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl
import seaborn as sns
from tqdm import tqdm
import os

from itertools import combinations


def cast_features(table: pl.DataFrame):
    """This function takes a given `table` as input and tries to convert all attributes to numeric. Any attribute that
    raises an exception keeps its type.

    Args:
        table (pl.DataFrame): Table to convert.

    Returns:
        pl.DataFrame: Table with casted types.
    """
    if not only_types:
        for col in table.columns:
            try:
                table = table.with_columns(pl.col(col).cast(pl.Float64))
            except pl.ComputeError:
                continue

    return table


def import_from_yago(filepath: Path):
    """Given a parquet file, read it assuming YAGO format. The last row is dropped.

    Args:
        filepath (Path): Path to the yago-like file.

    Returns:
        _type_: Triplets DataFrame.
    """
    triplets = pl.read_parquet(filepath)[:-1]
    triplets.columns = ["id", "subject", "predicate", "cat_object", "num_object"]
    return triplets


def find_unique_predicates(df: Union[pd.DataFrame, pl.DataFrame]):
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


def count_occurrences_by_columns(
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


def get_cooccurring_predicates(df: pl.DataFrame):
    """Given the df, perform a self-join, then return the predicate columns without
    performing any aggregation.

    Args:
        df (pl.DataFrame): Yago-like dataframe to operate on.

    Returns:
        pl.DataFrame: A Dataframe that contains all the pairs of predicates without aggregation.
    """
    return (
        df.lazy()
        .join(df.lazy(), left_on="subject", right_on="subject", how="left")
        .select([pl.col("predicate"), pl.col("predicate_right")])
        .collect()
    )


def get_count_cooccurring_predicates(df: pl.DataFrame):
    return (
        df.lazy()
        .groupby(["predicate", "predicate_right"])
        .agg(pl.count())
        .sort("count", descending=True)
        .collect()
    )


def join_types_predicates(yagotypes, yagofacts, types_subset):
    """Find all the YAGO type/YAGO predicate pairs (i.e. all the predicates that are connected to a subject with the 
    given type).

    Args:
        yagotypes (pl.DataFrame): Dataframe containing the YAGO types.
        yagofacts (pl.DataFrame): Dataframe containing the YAGO facts.
        types_subset (pl.DataFrame): Dataframe containing the selected types. 

    Returns:
        pl.DataFrame: Dataframe containing all pairs type-predicate, and the number of time they appear.
    """
    types_predicates = (
        yagotypes.lazy()
        .filter(pl.col("cat_object").is_in(types_subset["type"]))
        .join(yagofacts.lazy(), left_on="subject", right_on="subject", how="left")
        .select(
            pl.col("subject"),
            pl.col("cat_object").alias("type"),
            pl.col("predicate_right").alias("predicate"),
        )
        .unique()
        .drop_nulls()
        .select([pl.col("type"), pl.col("predicate")])
        .groupby([pl.col("type"), pl.col("predicate")])
        .agg([pl.count()])
        .sort("count", descending=True)
        .collect()
    )
    return types_predicates


def read_yago_files(yago_path=Path("/storage/store3/work/jstojano/yago3/")):
    facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
    facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")

    fname = "yagoTypes"
    yagotypes_path = Path(facts1_path, f"{fname}.tsv.parquet")
    yagotypes = import_from_yago(yagotypes_path)

    fname = "yagoFacts"
    yagofacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
    yagofacts = import_from_yago(yagofacts_path)

    fname = "yagoLiteralFacts"
    yagoliteralfacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
    yagoliteralfacts = import_from_yago(yagoliteralfacts_path)

    fname = "yagoDateFacts"
    yagodatefacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
    yagodatefacts = import_from_yago(yagodatefacts_path)

    yagofacts_overall = pl.concat([yagofacts, yagoliteralfacts, yagodatefacts])

    return yagofacts_overall, yagotypes


def get_selected_types(
    subject_count_sorted: pl.DataFrame,
    yagotypes: pl.DataFrame,
    n_subjects=10000,
    min_count=10,
):
    selected_types = (
        subject_count_sorted.lazy()
        .limit(n_subjects)  # Taking the top 10000 subjects
        .join(yagotypes.lazy(), on="subject")  # Joining on types table
        .select(  # Selecting only subject and type
            [pl.col("subject"), pl.col("cat_object")]
        )
        .join(  # Joining again on yagotypes to get the count of subjects by type
            yagotypes.groupby(  # Grouping to get the count of subjects by type
                "cat_object"
            )
            .agg(pl.count())  # Count of subjects by type
            .lazy(),
            on="cat_object",
        )
        .sort(  # Sorting to have the most frequent types at the start
            ["subject", "count"], descending=True
        )
        .groupby(["subject"])
        .agg(  # Grouping by subject to have all types close to each other
            pl.first("cat_object").suffix(
                "_first"
            )  # Selecting only the first entry from each group (i.e. the most frequent)
        )
        .groupby(  # At this point the selection is done
            "cat_object_first"  # Grouping by type to remove duplicates
        )
        .agg(pl.count())  # counting the number of occurrences of each type
        .sort("count", descending=True)  # Sorting by group (for convenience)
        .select(
            [pl.col("cat_object_first").alias("type"), pl.col("count")],
        )
        .filter(
            pl.col("count") > min_count
        )  # filter to select only the types that are connected to
        # at least `min_count` subjects.
        .collect()
    )
    return selected_types


def prepare_subjects_types_seltab(
    yagofacts, yagotypes, n_subjects=10000, min_count: int = 10
):
    subject_count_sorted = (
        yagofacts.lazy()
        .groupby("subject")
        .agg(pl.count())
        .sort("count", descending=True)
        .collect()
    )

    selected_types = get_selected_types(
        subject_count_sorted, yagotypes, n_subjects=n_subjects, min_count=min_count
    )
    subjects_in_selected_types = (
        yagofacts.lazy()
        .join(
            selected_types.lazy()
            .join(yagotypes.lazy(), left_on=["type"], right_on=["cat_object"])
            .select([pl.col("subject"), pl.col("type")]),
            left_on="subject",
            right_on="subject",
        )
        .collect()
    )

    return subjects_in_selected_types, selected_types.rename({"cat_object": "type"})


def prepare_subjects_types_wordnet(
    yagofacts, yagotypes, top_k=20, cherry_picked=None
):
    """Select only the subjects that are connected to the `top_k` wordnet categories. If `cherry_picked` is not None,
    add subjects that are connected to the categories in `cherry_picked`, if they are not already in the collection of
    subjects.

    Args:
        yagofacts (pl.DataFrame): Dataframe containing YAGO facts.
        yagotypes (pl.DataFrame): Dataframe containing YAGO types (i.e. categories).
        top_k (int, optional): Number of types to consider. Defaults to 20.
        cherry_picked (list, optional): List of types to add, if not found by the heuristic.. Defaults to None.

    Returns:
        (pl.DataFrame, pl.DataFrame): A tuple that contains the subjects to use and the wordnet categories to consider
        for the next steps.
    """

    top_wordnet = (
        yagotypes.lazy()
        .filter(pl.col("cat_object").str.starts_with("<wordnet_"))
        .groupby("cat_object")
        .count()
        .top_k(k=top_k, by="count")
        .collect()
    )

    if cherry_picked is not None:
        cherry_picked_types = (
            yagotypes.lazy()
            .filter(pl.col("cat_object").is_in(cherry_picked))
            .groupby("cat_object")
            .count()
            .top_k(k=top_k, by="count")
            .collect()
        )
        top_wordnet = pl.concat([top_wordnet, cherry_picked_types])

    subjects = (
        yagofacts.lazy()
        .join(
            top_wordnet.lazy()
            .join(yagotypes.lazy(), left_on=["cat_object"], right_on=["cat_object"])
            .select([pl.col("subject"), pl.col("cat_object")]),
            left_on="subject",
            right_on="subject",
        )
        .with_columns(
            pl.when(pl.col("num_object").is_null())
            .then(pl.col("cat_object").alias("cat_object"))
            .otherwise(pl.col("num_object").alias("cat_object"))
        )
        .collect()
    )
    return subjects, top_wordnet.rename({"cat_object": "type"})


def build_adj_dict(types_predicates):
    """Build an adjacency dictionary whose keys are the types, and each
    value is a list of all the predicates that are connected to the type through
    some subject.

    Args:
        types_predicates (pl.DataFrame): Dataframe that contains the number of type-predicate pairs.

    Returns:
        dict: Dictionary with pairs type-predicates.
    """
    adj_dict = {}
    for row in types_predicates.iter_rows():
        left, right, _ = row
        if left not in adj_dict:
            adj_dict[left] = [right]
        else:
            adj_dict[left].append(right)

    for k, v in adj_dict.items():
        adj_dict[k] = set(v)

    return adj_dict


def convert_df(df: pl.DataFrame, predicate: str):
    """Convert dataframe from triplet format to binary format. Given a dataframe
    and a predicate, return a new dataframe where subjects are index keys, the
    predicate is the attribute and the objects fill the attribute.

    Args:
        df (pl.DataFrame): Dataframe to convert.
        predicate (str): Name of the predicate.

    Returns:
        pl.DataFrame: Converted dataframe.
    """
    return df.select(
        pl.col("subject"),
        pl.when(pl.col("num_object").is_not_null())
        .then(pl.col("num_object").alias(predicate.strip("<").rstrip(">")))
        .otherwise(pl.col("cat_object").alias(predicate.strip("<").rstrip(">"))),
    ).lazy()


def get_tabs_by_type(yagotypes, selected_types, group_limit=10):
    """This function takes as input the types dataframe, the selected types and an optional
    limit on the number of types to select and produces the starting tables that
    will be used later to create the full DL tables.

    Args:
        yagotypes (pl.DataFrame): Yago types dataframe.
        selected_types (pl.DataFrame): Dataframe with the selected types.
        group_limit (int, optional): Number of types to use. Defaults to 10.

    Raises:
        ValueError: Raise ValueError if `group_limit` is < 1.

    Returns:
        dict[pl.DataFrame]: Dictionary of dataframes.
    """
    if group_limit < 1:
        raise ValueError(f"`group_limit` must be > 1, got {group_limit}")

    type_groups = (
        yagotypes.lazy()
        .join(
            selected_types.head(group_limit).lazy(),  # Inner join selected types
            left_on="cat_object",
            right_on="type",
        )
        .select(
            [pl.col("subject"), pl.col("cat_object")]
        )  # select only subject and type
        .groupby("cat_object")  # Group by type
        .all()
        .select(
            [
                pl.col("cat_object"),
                pl.col("subject"),
            ]
        )
        .collect()  #
    )

    tabs_by_type = {}
    for tup in type_groups.iter_rows():
        type_str, values = tup
        tab = pl.DataFrame(
            {"type": [type_str] * len(values), "subject": values},
        )
        tabs_by_type[type_str] = tab

    return tabs_by_type


def clean_keys(type_name):
    """Utility function that removes unnecessary symbols in the Yago type names.

    Args:
        type_name (str): String to be reformatted.

    Returns:
        str: Reformatted string.
    """
    pattern = r"(<)([a-zA-Z_]+)[_0-9]*>"
    replacement = r"\g<2>"
    return re.sub(pattern, replacement, type_name).rstrip("_")


def save_tabs_on_file(
    tabs_by_type,
    adj_dict,
    yagofacts: pl.DataFrame,
    dest_path,
    variant_tag: str = "",
    output_format: str = "parquet",
    max_table_width: int = 15,
):
    """Save the required tables in separate files with the given output format.

    Args:
        tabs_by_type (dict): Dictionary that contains the "base tables" to be joined on.
        adj_dict (dict): Dictionary that lists predicates that co-occur, to avoid empty joins.
        yagofacts (pl.DataFrame): Yago facts dataframe.
        dest_path (str): Root folder where all tables will be saved.
        variant_tag (str, optional): If provided, add the given tag to the table names.
        output_format (str, optional): Output format to use, either "parquet" or "csv". Defaults to "parquet".
        max_table_width (int, optional): Maximum width of the final table. Large tables can run out of memory. Defaults to 15.

    Raises:
        ValueError: Raise ValueError if the output format is not known.
    """
    for type_str, tab in tqdm(tabs_by_type.items(), total=len(tabs_by_type)):
        new_tab = tab.clone()
        tqdm.write(type_str)
        for pred_name, pred_group in yagofacts.groupby("predicate"):
            if (
                pred_name in adj_dict[type_str]
                and len(new_tab.columns) < max_table_width
            ):
                transformed_tab = convert_df(pred_group, pred_name)
                new_tab = new_tab.lazy().join(
                    transformed_tab.lazy(), on="subject", how="left"
                )
        clean_type_str = clean_keys(type_str)
        new_tab = cast_features(new_tab.collect())
        if output_format == "csv":
            fname = f"yago{variant_tag}_{clean_type_str}.csv"
            new_tab.write_csv(Path(dest_path, fname))
        elif output_format == "parquet":
            fname = f"yago{variant_tag}_{clean_type_str}.parquet"
            new_tab.write_parquet(Path(dest_path, fname))
        else:
            raise ValueError(f"Unkown output format {output_format}")


def prepare_binary_tables(subjects_in_selected_types: pl.DataFrame, dest_path: str):
    """Prepare a set of binary tables given the selected subjects. All new tables
    will be saved in the column provided in `dest_path`.

    Args:
        subjects_in_selected_types (pl.DataFrame): Selected YAGO subjects prepared in previous steps.
        dest_path (str): Path to the directory where all the subtables will be saved.

    Raises:
        FileNotFoundError: Raise FileNotFoundError if there is no directory in the `dest_path` provided.

    """
    dest_path = Path(dest_path)
    if not dest_path.exists():
        raise FileNotFoundError(f"Directory {dest_path} does not exist. ")

    for gname, group in subjects_in_selected_types.groupby("predicate"):
        print(f"Working on group {gname}")
        dff = group.clone()
        dff = dff[[s.name for s in dff if not (s.null_count() == dff.height)]]
        col_name = gname.replace("<", "").replace(">", "")
        new_df = None

        # If num_object is present, select the numeric version of the parameter.
        if "num_object" in dff.columns:
            new_df = dff.with_columns(
                pl.col("subject"), pl.col("num_object").alias(col_name)
            ).select(pl.col("subject"), pl.col(col_name))
        # Else, use the categorical value.
        else:
            new_df = dff.with_columns(
                pl.col("subject"), pl.col("cat_object").alias(col_name)
            ).select(pl.col("subject"), pl.col(col_name))
        if new_df is not None:
            df_name = f"yago_binary_{col_name}.parquet"
            new_df.write_parquet(Path(dest_path, df_name))
        else:
            print(f"Something wrong with group {gname}")


def explode_table(
    tgt_table: pl.DataFrame, table_name, root_dir_path, comb_size=2, min_occurrences=100
):
    """Explode a target YAGO table into smaller slices with no nulls.
    All column combinations of size `comb_size` are generated, then for each
    combination the starting table is projected over the columns in the combination.
    The number of rows with no nulls is tallied, then only combinations with at
    least `min_occurrences` non-null rows are kept.

    Only cases where all values are non-nulls are kept.

    Finally, each combination is saved into a different table, in a subdir that
    is created starting from `dir_path`, using `table_name` as root.

    Args:
        tgt_table (pl.DataFrame): Table to work on.
        table_name (str): Name of the table, will be used for saving the new tables.
        comb_size (int, optional): Number of columns in each combination. Defaults to 2.
        min_occurrences (int, optional): Minimum number of non-null values to select a pair. Defaults to 100.
    """
    dir_path = Path(root_dir_path, table_name)
    os.makedirs(dir_path, exist_ok=True)
    # Ignore columns `type` and `subject`
    target_columns = tgt_table.columns[2:]
    coords_dict = {}

    # Counting the number of non-null occurrences for each combination of size `comb_size`
    for comb in combinations(target_columns, comb_size):
        tt = tgt_table.select(pl.all(pl.col(comb).is_not_null())).sum().item()
        coords_dict[comb] = tt

    df_coord = pd.DataFrame().from_dict(coords_dict, orient="index", columns=["count"])
    df_coord = df_coord.reset_index()

    # selecting only combinations with more than `min_occurrences` occs
    rich_combs = df_coord[df_coord["count"] >= min_occurrences]

    # For each comb, write a new parquet file.
    written = 0
    skipped = 0
    for _, comb in rich_combs.iterrows():
        sel_col = ["type", "subject"] + list(comb["index"])
        res = (
            tgt_table.filter(pl.all(pl.col(comb["index"]).is_not_null()))
            .select(pl.col(sel_col))
            .unique()
        )

        filename = "_".join(table_name.split("_")[2:]) + "_" + "_".join(comb["index"])
        dest_path = Path(dir_path, filename + ".parquet")
        # print(dest_path)
        if len(res) > min_occurrences:
            res.write_parquet(dest_path)
            written += 1
        else:
            skipped += 1
    print(f"Written: {written} Skipped: {skipped}")


def prepare_combinations(args):
    src_path = Path(args.data_dir)
    for tpath in src_path.iterdir():
        if not tpath.is_dir():
            tab_name = tpath.stem
            tab = pl.read_parquet(tpath)
            for csize in args.comb_size:
                explode_table(
                    tab,
                    tab_name,
                    root_dir_path=args.data_dir,
                    comb_size=csize,
                    min_occurrences=args.min_count,
                )
