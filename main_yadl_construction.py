from pathlib import Path

import src.yago.utils as utils
import os
import argparse
import polars as pl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy",
        action="store",
        default="wordnet",
        help="Strategy to use.",
        choices=["wordnet", "seltab", "binary", "wordnet_cp", "custom"],
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        action="store",
        required=False,
        default="data/yago3-dl",
        help="Root path to save new tables in.",
    )
    parser.add_argument(
        "--top_k",
        action="store",
        type=int,
        default=20,
        help="The top k types by number of occurrences will be selected.",
    )
    parser.add_argument("--min_count", action="store", type=int, default=10)

    parser.add_argument(
        "--explode_tables",
        action="store_true",
        help="After generating the tables, generate new synthetic subtables.",
    )

    parser.add_argument(
        "--comb_size",
        action="store",
        type=int,
        nargs="+",
        default=[2],
        help="Size of the column combinations to be generated in the explode stage. Defaults to 2.",
    )

    parser.add_argument(
        "--min_occurrences",
        action="store",
        type=int,
        default=100,
        help="Minimum number of non-null values to select a pair. Defaults to 100.",
    )

    parser.add_argument(
        "--cherry_pick_path",
        action="store",
        type=str,
        default=None,
        help="If provided, load cherry picked classes from the given file.",
    )

    parser.add_argument(
        "--custom_subjects_path",
        action="store",
        type=str,
        default=None,
        help="If `strategy` is `custom`, path where the custom subjects are stored.",
    )


    parser.add_argument(
        "--custom_types_path",
        action="store",
        type=str,
        default=None,
        help="If `strategy` is `custom`, path where the custom types are stored.",
    )




    args = parser.parse_args()

    return args


def prepare_subtables(
    yagofacts,
    yagotypes,
    selected_types,
    strategy="wordnet",
    dest_path=None,
    output_format="parquet",
    top_k=50,
):
    """Given a subset of types, prepare the master table for each type according to the specified strategy.

    Args:
        yagofacts (pl.DataFrame): Dataframe that contains the YAGO facts.
        yagotypes (pl.DataFrame): Dataframe that contains the YAGO types.
        selected_types (pl.Series): Types to be used to build the tables.
        strategy (str, optional): Strategy to use, can either be "wordnet" or "seltab". Defaults to "wordnet".
        dest_path (str, optional): Location where the new tables will be saved. If "None", default to "data/yago3-dl/{strategy}".
        output_format (str, optional): Output format to use when saving tables. Defaults to "parquet".
        top_k (int, optional): Number of types to consider. Defaults to 20.
    """
    if dest_path is None:
        dest_path = Path("data/yago3-dl/", strategy)
    else:
        dest_path = Path(dest_path)

    os.makedirs(dest_path, exist_ok=True)
    print("Filtering subjects by type")

    print("Preparing adjacency dict")
    types_predicates_selected = utils.join_types_predicates(
        yagotypes, yagofacts, types_subset=selected_types
    )

    adj_dict = utils.build_adj_dict(types_predicates_selected)

    print("Preparing tabs by type")
    tabs_by_type = utils.get_tabs_by_type(yagotypes, selected_types, group_limit=top_k)

    print("Saving tabs on file")
    print(f"Format: {output_format}")
    utils.save_tabs_on_file(
        tabs_by_type,
        adj_dict,
        yagofacts,
        dest_path,
        variant_tag=strategy,
        output_format=output_format,
    )


def get_selected_subjects_types(
    yagofacts, yagotypes, strategy="wordnet", top_k=20, min_count=10, cherry_picked=None
):
    """Extract a subset of types and subjects from YAGO to use when building the tables. Different values for `strategy` 
    can be used. 

    Args:
        yagofacts (_type_): _description_
        yagotypes (_type_): _description_
        strategy (str, optional): _description_. Defaults to "wordnet".
        top_k (int, optional): _description_. Defaults to 20.
        min_count (int, optional): _description_. Defaults to 10.
        cherry_picked (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if strategy in ["wordnet", "wordnet_cp", "binary"]:
        (
            subjects,
            types,
        ) = utils.prepare_subjects_types_wordnet(
            yagofacts, yagotypes, top_k=top_k, cherry_picked=cherry_picked
        )

    elif strategy == "seltab":
        (
            subjects,
            types,
        ) = utils.prepare_subjects_types_seltab(
            yagofacts, yagotypes, min_count=min_count
        )
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    return subjects, types


def read_custom_subjects_types(
    path_subjects,
    path_types
):
    path_subjects = Path(path_subjects)
    path_types = Path(path_types)

    if path_subjects.exists() and path_types.exists():
        subjects = pl.read_parquet(path_subjects)
        types = pl.read_parquet(path_types)
        return subjects, types
    else:
        raise FileNotFoundError("One of the files was not found.")


def read_cherry_picked(path_to_file):
    """Read a list of YAGO predicates to add to the predicates to be used while creating the data lake, to guarantee
    they are present in YADL.

    Note that no error checking over the format of the predicates is done at this step.

    Args:
        path_to_file (str): Path to the file containing the predicates.

    Raises:
        FileNotFoundError: Raise FileNotFoundError if the provided path does not exist.

    Returns:
        list: List of classes to be used.
    """
    if Path(path_to_file).exists():
        cp_list = []
        with open(path_to_file, "r") as fp:
            for idx, row in enumerate(fp):
                cp_list.append(row.strip())
        return cp_list
    else:
        raise FileNotFoundError(f"File {path_to_file} was not found.")


if __name__ == "__main__":

    args = parse_args()
    data_dir = Path(args.data_dir)

    strategy = args.strategy

    print("Reading files")
    yagofacts, yagotypes = utils.read_yago_files()
    if strategy == "wordnet_cp":
        cherry_picked = read_cherry_picked(args.cherry_pick_path)
    else:
        cherry_picked = None

    if strategy != "custom":
        selected_subjects, selected_types = get_selected_subjects_types(
            yagofacts,
            yagotypes,
            strategy=strategy,
            top_k=args.top_k,
            min_count=args.min_count,
            cherry_picked=cherry_picked,
        )
    else:
        selected_subjects, selected_types = read_custom_subjects_types(
            args.custom_subjects_path, args.custom_types_path
        )

    if strategy == "binary":
        utils.prepare_binary_tables(
            selected_subjects,
            dest_path=data_dir,
        )
    elif strategy in ["wordnet", "seltab", "wordnet_cp", "custom"]:
        prepare_subtables(
            yagofacts, yagotypes, selected_types, dest_path=data_dir, strategy=strategy
        )
        if args.explode_tables:
            utils.prepare_combinations(args)
    else:
        raise ValueError(f"Unknown strategy {strategy}")
