import os
import random
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.utils import murmurhash3_32
from tqdm import tqdm

# Replace this path with the path to where your YAGO files are stored.
YAGO_PATH = Path("/storage/store3/work/jstojano/yago3/")


def cast_features(table: pl.DataFrame):
    """This function takes a given `table` as input and tries to convert all attributes to numeric. Any attribute that
    raises an exception keeps its type.

    Args:
        table (pl.DataFrame): Table to convert.

    Returns:
        pl.DataFrame: Table with casted types.
    """
    for col in table.columns:
        try:
            table = table.with_columns(pl.col(col).cast(pl.Float64))
        except pl.ComputeError:
            continue

    return table


def import_from_yago(filepath: Path, debug=False):
    """Given a parquet file, read it assuming YAGO format. The last row is dropped.

    Args:
        filepath (Path): Path to the yago-like file.

    Returns:
        _type_: Triplets DataFrame.
    """
    if debug:
        triplets = pl.scan_parquet(filepath, n_rows=100_000).collect().sample(10_000)
    triplets = pl.read_parquet(filepath)[:-1]
    triplets.columns = ["id", "subject", "predicate", "cat_object", "num_object"]
    return triplets


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


def read_yago_files(
    yago_path=Path("/storage/store3/work/jstojano/yago3/"), debug=False
):
    facts1_path = Path(yago_path, "facts_parquet/yago_updated_2022_part1")
    facts2_path = Path(yago_path, "facts_parquet/yago_updated_2022_part2")

    fname = "yagoTypes"
    yagotypes_path = Path(facts1_path, f"{fname}.tsv.parquet")
    yagotypes = import_from_yago(yagotypes_path, debug=debug)

    fname = "yagoFacts"
    yagofacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
    yagofacts = import_from_yago(yagofacts_path, debug=debug)
    yagofacts = yagofacts.drop("num_object")

    fname = "yagoLiteralFacts"
    yagoliteralfacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
    yagoliteralfacts = import_from_yago(yagoliteralfacts_path, debug=debug)

    fname = "yagoDateFacts"
    yagodatefacts_path = Path(facts2_path, f"{fname}.tsv.parquet")
    yagodatefacts = import_from_yago(yagodatefacts_path, debug=debug)
    if debug:
        yagofacts = yagofacts.sample(10_000)
        yagoliteralfacts = yagoliteralfacts.sample(10_000)
        yagodatefacts = yagodatefacts.sample(10_000)

    # Keeping only complete dates and removing dates with no month/day
    yagodatefacts = (
        yagodatefacts.with_columns(
            pl.col("cat_object")
            .str.split("^^")
            .list.first()
            .str.to_datetime(strict=False)
            .dt.date()
            .cast(pl.Utf8)
        )
        .drop_nulls("cat_object")
        .drop("num_object")
    )

    # Keeping only the numerical version of the fact
    yagoliteralfacts = yagoliteralfacts.with_columns(
        pl.when(pl.col("num_object").is_not_null())
        .then(pl.col("num_object"))
        .otherwise(pl.col("cat_object"))
        .alias("cat_object")
        .cast(pl.Utf8)
    ).drop("num_object")

    yagofacts_overall = pl.concat([yagofacts, yagoliteralfacts, yagodatefacts])

    return yagofacts_overall, yagotypes


def prepare_subjects_types_wordnet(yagofacts, yagotypes, top_k=20, cherry_picked=None):
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
        pl.col("cat_object").alias(predicate.strip("<").rstrip(">")),
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
            fname = f"{variant_tag}-{clean_type_str}.csv"
            new_tab.write_csv(Path(dest_path, fname))
        elif output_format == "parquet":
            fname = f"{variant_tag}-{clean_type_str}.parquet"
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
            df_name = f"binary-{col_name}.parquet"
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
        tt = tgt_table.select(pl.sum(pl.col(comb).is_not_null()) > 0).sum().item()
        coords_dict[comb] = tt

    df_coord = pd.DataFrame().from_dict(coords_dict, orient="index", columns=["count"])
    df_coord = df_coord.reset_index()

    # selecting only combinations with more than `min_occurrences` occs
    rich_combs = df_coord.loc[df_coord["count"] >= min_occurrences]

    # For each comb, write a new parquet file.
    written = 0
    skipped = 0
    for _, comb in rich_combs.iterrows():
        sel_col = ["type", "subject"] + list(comb["index"])
        res = (
            tgt_table.filter(pl.sum(pl.col(comb["index"]).is_not_null()) > 0)
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
    print(
        f"Table {table_name} - {written} sub-tables prepared - {skipped} sub-tables smaller than threshold"
    )


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


def prepare_facts_df():
    """Utility function that reads the YAGO facts, perform some preprocessing and returns a concatenated dataframe that
    includes all selected facts.

    Returns:
        pl.DataFrame: A dataframe that contains all the processed facts.
    """
    # yagoFacts are relations between entities (they are mostly categorical)
    facts2_path = Path(YAGO_PATH, "facts_parquet/yago_updated_2022_part2")
    yagofacts_path = Path(facts2_path, "yagoFacts.tsv.parquet")
    yagofacts = import_from_yago(yagofacts_path)
    # num_object is empty
    yagofacts = yagofacts.drop("num_object")

    # yagoLiteralFacts may contain numerical facts (e.g., population density)
    yagoliteralfacts_path = Path(facts2_path, "yagoLiteralFacts.tsv.parquet")
    yagoliteralfacts = import_from_yago(yagoliteralfacts_path)

    # moving num_object values to cat_object for concatenation
    yagoliteralfacts = yagoliteralfacts.with_columns(
        pl.when(pl.col("num_object").is_not_null())
        .then(pl.col("num_object"))
        .otherwise(pl.col("cat_object"))
        .alias("cat_object")
    ).drop("num_object")

    # yagoDateFacts contains dates
    yagodatefacts_path = Path(facts2_path, "yagoDateFacts.tsv.parquet")
    yagodatefacts = import_from_yago(yagodatefacts_path)
    # some dates are incomplete, so we select only the year for each date
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
    # the different dataframes are concatenated
    return pl.concat([yagofacts, yagoliteralfacts, yagodatefacts]).drop("id")


def get_subjects():
    """Utility function that reads the subjects and executes some preprocessing.

    Returns:
        pl.DataFrame: A dataframe that contains all the subjects.
    """
    facts1_path = Path(YAGO_PATH, "facts_parquet/yago_updated_2022_part1")
    yagotypes_path = Path(facts1_path, "yagoTypes.tsv.parquet")
    df_types = import_from_yago(yagotypes_path)

    subjects_with_wordnet = (
        df_types.filter(pl.col("cat_object").str.starts_with("<wordnet_"))
        .select(pl.col("subject"), pl.col("cat_object"))
        .rename({"cat_object": "type"})
    )
    return subjects_with_wordnet


def clean_string(string_to_clean):
    pattern = re.compile(r"<{1}([a-zA-Z0-9_]+)>{1}")
    m = re.sub(pattern, "\\1", string_to_clean)
    return m


def prepare_long_variant(dest_path: str | Path, max_fields=2):
    df_facts = prepare_facts_df()
    subjects_with_wordnet = get_subjects()
    n_groups = len(subjects_with_wordnet.unique("type"))

    os.makedirs(dest_path, exist_ok=True)
    dest_path = Path(dest_path)

    for this_type, this_df in tqdm(
        subjects_with_wordnet.group_by("type"), total=n_groups
    ):
        clean_type = clean_string(this_type)
        joined_df = df_facts.join(
            this_df.select(pl.col("subject", "type")), on="subject"
        )
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


def generate_table(table, col_comb, subject):
    selected_ = subject + list(col_comb)
    new_df = table.select(selected_).filter(
        pl.any_horizontal(pl.col(col_comb).is_not_null())
    )
    return new_df


def generate_batch(
    dest_dir: Path | str,
    target_columns: list[str],
    base_table: pl.DataFrame,
    table_name: str,
    subject: str,
    case: str,
    col_resample: int = 10,
    row_resample: int = 0,
    min_occurrences: int = 100,
    row_sample_fraction: float = 0.7,
    limit_break=100,
):
    """This function generates a batch of subtables according to the provided
    set of parameters. It takes as input the destination path, the columns that
    should be used for generating subtables, additional parameters for saving the
    file and generating subtables.

    Args:
        dest_dir (Path | str): Path where the resulting subtables will be saved.
        target_columns (list[str]): List of columns that should be used for generating the subtables.
        base_table (pl.DataFrame): Table to be used as seed for the subtables.
        table_name (str): Name of the table (for saving on disk).
        subject (str): Name of the subject, used for selecting the column to keep.
        case (str): String to be added to the filename.
        col_resample (int, optional): Number of subtables to generate for each base table. Defaults to 10.
        row_resample (int, optional): Number of row resamplings to generate to increase row redundancy. Defaults to 0.
        min_occurrences (int, optional): Minimum size of the generated table. Tables smaller than this will be filtered out. Defaults to 100.
        row_sample_fraction (float, optional): Size of the row resamplings. Defaults to 0.7.
        limit_break (int, optional): Number of attempts at generating a viable subtable before skipping to the next. Defaults to 100.

    Raises:
        ValueError: Raises ValueError if the values of row_resample are not correct.

    Returns:
        int: Number of subtables that have successfully been generated.
    """
    if row_resample < 0 or not isinstance(row_resample, int):
        raise ValueError(f"Row resample value must be > 0, found {row_resample}")

    new_dir = Path(dest_dir, table_name)
    break_counter = 0
    col_counter = 0
    good_comb = set()
    bad_comb = set()

    min_sample_size = max(2, len(target_columns) - 2)
    max_sample_size = len(target_columns)

    for _ in tqdm(
        range(col_resample),
        total=col_resample,
        position=0,
        desc=table_name,
        leave=False,
    ):
        if break_counter > limit_break:
            break
        comb = random.sample(
            target_columns, k=random.randint(min_sample_size, max_sample_size)
        )
        comb = tuple(comb)
        if (comb not in good_comb) and (comb not in bad_comb):
            table = generate_table(base_table, comb, subject)
            if len(table) > min_occurrences:
                good_comb.add(comb)
                fname = (
                    "-".join([table_name, str(murmurhash3_32("-".join(comb[:]))), case])
                    + ".parquet"
                )
                col_counter += 1
                destination_path = Path(new_dir, fname)
                table.write_parquet(destination_path)
                for sample_counter in range(row_resample):
                    resampled_table = table.sample(fraction=row_sample_fraction)

                    col_counter += 1
                    fname = (
                        "-".join(
                            [
                                table_name,
                                str(murmurhash3_32("-".join(comb[:]))),
                                str(sample_counter),
                                case,
                            ]
                        )
                        + ".parquet"
                    )

                    destination_path = Path(new_dir, fname)
                    resampled_table.write_parquet(destination_path)
            else:
                bad_comb.add(comb)
                break_counter += 1
        else:
            break_counter += 1

    return col_counter
