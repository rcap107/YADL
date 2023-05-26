#%%
from pathlib import Path

import src.yago.utils as utils
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy",
        action="store",
        default="wordnet",
        help="Strategy to use.",
        choices=["wordnet", "seltab", "binary"],
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

    args = parser.parse_args()

    return args


def prepare_subtables(
    yagofacts,
    yagotypes,
    selected_types,
    strategy="wordnet",
    dest_path=None,
    output_format="parquet",
    top_k=20,
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


def get_selected_types(
    yagofacts, yagotypes, strategy="wordnet", top_k=20, min_count=10
):
    if strategy == "wordnet" or strategy == "binary":
        (
            subjects_in_selected_types,
            selected_types,
        ) = utils.get_subjects_in_wordnet_categories(yagofacts, yagotypes, top_k=top_k)
    elif strategy == "seltab":
        (
            subjects_in_selected_types,
            selected_types,
        ) = utils.get_subjects_in_selected_types(
            yagofacts, yagotypes, min_count=min_count
        )
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    return subjects_in_selected_types, selected_types


if __name__ == "__main__":

    args = parse_args()
    data_dir = args.data_dir

    strategy = args.strategy

    print("Reading files")
    yagofacts, yagotypes = utils.read_yago_files()
    subjects_in_selected_types, selected_types = get_selected_types(
        yagofacts,
        yagotypes,
        strategy=strategy,
        top_k=args.top_k,
        min_count=args.min_count,
    )

    if strategy == "binary":
        utils.prepare_binary_tables(
            subjects_in_selected_types,
            dest_path=Path(data_dir, strategy),
        )
    elif strategy in ["wordnet", "seltab"]:
        prepare_subtables(
            yagofacts, yagotypes, selected_types, dest_path=Path(data_dir, strategy)
        )
    else:
        raise ValueError(f"Unknown strategy {strategy}")
