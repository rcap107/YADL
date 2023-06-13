# Building YADL

YADL (Yet Another Data Lake) is the benchmark data lake we developed for the paper "A Benchmarking Data Lake for Join Discovery and
Learning with Relational Data". 

We use this repository to prepare both the base tables and different YADL variants. 

Base tables are prepared in notebook `notebooks/Building YADL base tables.ipynb`.

For the time being, we implement two variants of the YADL data lake in `Binary` and `Wordnet`. Both variants are accessible for download
on [zenodo.org](https://zenodo.org/record/8015298). 

It is also possible to create the variants starting from YAGO 3 files, provided they are available, by using the script
`main_yadl_construction.py`.

## Building the YADL variants
The script `main_yadl_construction.py` must be used to build one of the supported variants. 

To construct the `Wordnet` variant, use the following command and parameters:
```
python main_yadl_construction.py -d destination/path/to/use --explode_tables --comb_size 2 3 --top_k 200 wordnet
```

To construct the `Binary` variant, use the following command and parameters:
```
python main_yadl_construction.py -d destination/path/to/use binary
```

TODO: 
explain the various parameters
explain the step-by-step procedure

## Additional files
Some additional notebooks that were used to perform some preliminary studies are provided to give 
further insight into YAGO. 

