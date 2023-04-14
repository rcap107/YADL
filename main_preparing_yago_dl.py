#%%
from pathlib import Path

import src.yago.utils as utils


dest_path = Path("data/yago3-dl/seltab")
output_format = "parquet"

yagofacts, yagotypes = utils.read_yago_files()
#%%
subject_count_sorted = utils.get_subject_count_sorted(yagofacts, lazy=True)
#%%
selected_types = utils.get_selected_types_decoupled(
    subject_count_sorted, yagotypes, n_subjects=10000, lazy=True
)

#%%
subjects_in_selected_types, types_subset = utils.get_subjects_in_selected_types(
    yagofacts, selected_types, yagotypes, min_count=10, lazy=True
)
# %%
types_predicates_selected = utils.join_types_predicates_v2(
    yagotypes, yagofacts, types_subset=types_subset, lazy=True
)

# %%
adj_dict = utils.build_adj_dict(types_predicates_selected)

# %%
tabs_by_type = utils.get_tabs_by_type(yagotypes, selected_types)

# %%
# utils.save_tabs_on_file(tabs_by_type, adj_dict, yagofacts, dest_path, output_format)

# %%
