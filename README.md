# Building YADL

YADL (Yet Another Data Lake) is the benchmark data lake we developed for the paper "Retrieve, Merge, Predict: Augmenting Tables with Data Lakes".

We use this repository to "unpack" the YAGO 3 tables in order to go from few, very large tables to a collection of smaller
tables that contain some of the same information, in different formats.

## Preparing the base tables
Depending on the method used (refer to the paper for more details), tables are prepared in one of the scripts
- `generate_yadl_vldb.py`
- `generate_yadl_vldb_variants.py`
- `main_yadl_construction.py`

## Implementing the YADL variants
We provide the YADL variants we used in the paper on [Zenodo]().

If YAGO 3 files are available, additional variants may be constructed by using the scripts mentioned above.
Alternatively, it is possible to build them starting from the YAGO 3 files, provided they are available, by using the script
`main_yadl_construction.py`.

### Using `main_yadl_construction.py`
```
usage: main_yadl_construction.py [-h] [-d DATA_DIR] [--top_k TOP_K]
                                 [--min_count MIN_COUNT] [--explode_tables]
                                 [--comb_size COMB_SIZE [COMB_SIZE ...]]
                                 [--min_occurrences MIN_OCCURRENCES] [--debug]
                                 {wordnet,binary}

positional arguments:
  {wordnet,binary}      Strategy to use.

options:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        Root path to save new tables in.
  --top_k TOP_K         The top k types by number of occurrences will be selected.
  --min_count MIN_COUNT
  --explode_tables      After generating the tables, generate new synthetic subtables.
  --comb_size COMB_SIZE [COMB_SIZE ...]
                        Size of the column combinations to be generated in the explode
                        stage. Defaults to 2.
  --min_occurrences MIN_OCCURRENCES
                        Minimum number of non-null values to select a pair. Defaults to
                        100.
  --debug               If set, downsample the yago tables to 100_000 values to reduce
                        runtime.
```


To construct the `YADL Base` variant, use the following command and parameters:
```
python main_yadl_construction.py -d destination/path/to/use --explode_tables --comb_size 2 3 --top_k 200 wordnet
```

To construct the `YADL Binary` variant, use the following command and parameters:
```
python main_yadl_construction.py -d destination/path/to/use binary
```
