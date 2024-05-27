# YADL: Yet Another Data Lake

YADL (Yet Another Data Lake) is the benchmark data lake we developed for the paper "Retrieve, Merge, Predict: Augmenting Tables with Data Lakes".

We use this repository to "unpack" the YAGO 3 tables in order to go from few, very large tables to a collection of smaller
tables that contain some of the same information, in different formats.

## Preparing the base tables
Depending on the method used (refer to the paper for more details), tables are prepared in one of the scripts
- `generate_yadl_long.py`
- `generate_yadl_long_variants.py`
- `main_yadl_construction.py`

## Implementing the YADL variants
We provide the YADL variants we used in the paper on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10600047).

If YAGO 3 files are available, additional variants may be constructed by using the scripts mentioned above.
Alternatively, it is possible to build them starting from the YAGO 3 files, provided they are available, by using the script `main_yadl_construction.py` for YADL Base and YADL Binary, and the scripts `generate_yadl_long.py` and `generate_yadl_long_variants.py` for YADL 10k and 50k.


### Building YADL Base and YADL Binary
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

### Building YADL 10k and 50k variants
This operation is done in two steps, first by creating the long form tables using `generate_yadl_long.py`,
then by building the subtables using `generate_yadl_variants.py`.

```
usage: generate_yadl_long.py [-h] [--dest_path DEST_PATH]
                             [--max_fields MAX_FIELDS]

options:
  -h, --help            show this help message and exit
  --dest_path DEST_PATH
                        Path where the tables will be
                        saved.
  --max_fields MAX_FIELDS
                        Number of argument values to keep
                        when building tables.
```
`dest_path` is the path that will contain the "seed tables": this is the same path that should be provided to the next
script to generate the subtables.
`max_fields` is used to limit the length of the seed tables. Whenever a subject has multiple relations (e.g., a director
with many films), the number of rows in the resulting table will be multiplied by the number of relations; to avoid
having scalability issues and keeping the size of the seed tables "contained", `max_fields` is set to 2. It can be
increased to build much longer tables.

```
usage: generate_yadl_long_variants.py [-h] [--col_resample COL_RESAMPLE] [--row_resample ROW_RESAMPLE]
                                      [--sample_fraction SAMPLE_FRACTION] [--minimum_arity MINIMUM_ARITY]
                                      [--minimum_rows MINIMUM_ROWS]
                                      base_path

positional arguments:
  base_path             Path with base tables to replicate.

options:
  -h, --help            show this help message and exit
  --col_resample COL_RESAMPLE
                        Number of subtables to generate.
  --row_resample ROW_RESAMPLE
                        Number of resamplings for each subtable. Defaults to 2.
  --sample_fraction SAMPLE_FRACTION
                        Fraction of lines to keep for row resampling. Defaults to 0.7.
  --minimum_arity MINIMUM_ARITY
                        Minimum number of columns.
  --minimum_rows MINIMUM_ROWS
                        Minimum number of rows of columns.
```
`col_resample` is the number of random projections that should be generated for each seed table. `row_resample` will
generate `row_resample` new **additional** tables for each random projection by executing a random sampling of the
rows of size `sample_fraction` (so, if `row_resample==2`, two additional subsamplings will be added on top of the single
projection).  `minimum_arity` is the minimum width of columns that a seed table should have to be considered for
generating subtables (other seed tables will be ignored). `minimum_rows` removes all subtables with fewer rows than this
number.


## Estimating the size of the resulting data lake
For the long variants (YADL 10k and 50k), the generation procedure may be time-consuming  and lead to a large disk
footprint. We provide a notebook (`notebooks/Estimate number of tables.ipynb`) to help the user with estimating the
resulting size of the data lake based on the parameters that are provided.

Due to the fact that there is a high degree
of randomness involved in generating the tables (some tables may be filtered out, the fraction of null values is not
constant etc.), the values provided are not exact; we observe that they are however within 10% of the actual values
resulting from preparing the data lake.
