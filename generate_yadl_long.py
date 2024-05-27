import argparse

from src import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest_path",
        default="data/yago3-dl/wordnet_vldb",
        help="Path where the tables will be saved.",
    )
    parser.add_argument(
        "--max_fields",
        default=2,
        help="Number of argument values to keep when building tables.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.prepare_long_variant(dest_path=args.dest_path, max_fields=args.max_fields)
