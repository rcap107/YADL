import re
from pathlib import Path

import src.yago.utils as utils

if __name__ == "__main__":

    dest_path = Path("data/yago3-dl/seltab")
    output_format = "csv"

    print("Reading files")
    yagofacts, yagotypes = utils.read_yago_files()
    print("Selecting subjects")
    subject_count_sorted = utils.get_subject_count_sorted(yagofacts)

    print("Selecting types")
    selected_types = utils.get_selected_types(
        subject_count_sorted, yagotypes, n_subjects=10000
    )

    print("Filtering subjects by type")
    subjects_in_selected_types, types_subset = utils.get_subjects_in_selected_types(
        yagofacts, selected_types, yagotypes, min_count=10
    )

    print("Prepare adjacency dict")
    types_predicates_selected = utils.join_types_predicates(
        yagotypes, yagofacts, types_subset=types_subset
    )

    adj_dict = utils.build_adj_dict(types_predicates_selected)

    print("Preparing tabs by type")
    tabs_by_type = utils.get_tabs_by_type(yagotypes, selected_types)

    print("Saving tabs on file")
    print(f"Format: {output_format}")
    utils.save_tabs_on_file(tabs_by_type, adj_dict, yagofacts, dest_path, output_format)
