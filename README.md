# Building YADL

YADL (Yet Another Data Lake) is the benchmark data lake we developed for the paper "A Benchmarking Data Lake for Join Discovery and
Learning with Relational Data". 

We use this repository to "unpack" the YAGO 3 tables in order to go from few, very large tables to a collection of smaller
tables that contain some of the same information, in different formats. 

## Preparing the base tables
Base tables are prepared in notebook `notebooks/Building YADL base tables.ipynb`. The notebook contains all the details
of the pre-processing and how the tables have been modified to be used in the benchmark. 

## Implementing the YADL variants
For the time being, we implement two variants of the YADL data lake in `Binary` and `Wordnet`. Both variants are accessible for download
on [zenodo.org](https://zenodo.org/record/8015298). 

Alternatively, it is possible to build them starting from the YAGO 3 files, provided they are available, by using the script
`main_yadl_construction.py`.

### Usage
```
usage: main_yadl_construction.py [-h] [-d DATA_DIR] [--top_k TOP_K] [--min_count MIN_COUNT] [--explode_tables] [--comb_size COMB_SIZE [COMB_SIZE ...]] [--min_occurrences MIN_OCCURRENCES] [--cherry_pick_path CHERRY_PICK_PATH] {wordnet,seltab,binary,wordnet_cp}

positional arguments:
  {wordnet,seltab,binary,wordnet_cp}
                        Strategy to use.

options:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        Root path to save new tables in.
  --top_k TOP_K         The top k types by number of occurrences will be selected.
  --min_count MIN_COUNT
  --explode_tables      After generating the tables, generate new synthetic subtables.
  --comb_size COMB_SIZE [COMB_SIZE ...]
                        Size of the column combinations to be generated in the explode stage. Defaults to 2.
  --min_occurrences MIN_OCCURRENCES
                        Minimum number of non-null values to select a pair. Defaults to 100.
  --cherry_pick_path CHERRY_PICK_PATH
                        If provided, load cherry picked classes from the given file.
```


To construct the `Wordnet` variant, use the following command and parameters:
```
python main_yadl_construction.py -d destination/path/to/use --explode_tables --comb_size 2 3 --top_k 200 wordnet
```

To construct the `Binary` variant, use the following command and parameters:
```
python main_yadl_construction.py -d destination/path/to/use binary
```

**Details on parameters**
The `strategy` parameter is used to select the specific method to be used to generate the main YADL tables, which are 
then spit over smaller subtables if the parameter `explode_tables` is also provided. 

`cherry_pick_path` is used to force the data lake to include tables that would otherwise be missing because of the 
heuristic used to select the predicates of interest.

## Additional files
Some additional notebooks that were used to perform some preliminary studies are provided to give 
further insight into YAGO. 

## Creating a custom version of YADL
It is possible to create a custom version of YADL by modifying the YAGO 3 tables without following the methods that have
been detailed here. 

The types and subjects used to build each YADL variant can be customized to select a different starting set. Function
`get_selected_subject_types` implements this for variants `binary`, `wordnet`, `seltab`, and provides support for using 
cherry picked types. To create a new set of subjects/types, the user should write a custom function whose return value
is similar to that of function `get_selected_subject_types`, where `subjects` is a view of `yagofacts` containing only 
the selected subjects and their type, while `types` is the list of types to be used, sorted in descending order by their
frequency. 

To generate custom versions, use the `custom` value for parameter `strategy`, then provide the custom selection by using
arguments `custom_subjects_path` and `custom_types_path`. 