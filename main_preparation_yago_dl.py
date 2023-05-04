#%%
import re
from pathlib import Path

import src.yago.utils as utils

#%%
dest_path = Path("data/yago3-dl/wordnet")
output_format = "parquet"

n_type_tabs = 20

print("Reading files")
yagofacts, yagotypes = utils.read_yago_files()


#%%
print("Filtering subjects by type")
# subjects_in_selected_types, selected_types = utils.get_subjects_in_selected_types(
#     yagofacts, yagotypes, min_count=10
# )

subjects_in_selected_types, selected_types = utils.get_subjects_in_wordnet_categories(
    yagofacts, yagotypes, top_k=n_type_tabs
)


#%%
print("Prepare adjacency dict")
types_predicates_selected = utils.join_types_predicates(
    yagotypes, yagofacts, types_subset=selected_types
)

adj_dict = utils.build_adj_dict(types_predicates_selected)

print("Preparing tabs by type")
tabs_by_type = utils.get_tabs_by_type(yagotypes, selected_types, group_limit=n_type_tabs)

print("Saving tabs on file")
print(f"Format: {output_format}")
utils.save_tabs_on_file(tabs_by_type, adj_dict, yagofacts, dest_path, output_format)

# %%
