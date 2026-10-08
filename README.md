# Database Localization for Multi-Database Text-to-SQL

Conventional multi-database Text-to-SQL benchmarks do not reveal whether
database localization relies on schema evidence or database contents, because
the two usually vary together. This repository provides the artifacts used to
study that distinction: the construction of content-ambiguous workloads,
Hybrid-DB, the catalog-wide LLM and Iterative-JAR baselines, the downstream
schema-reduction stages, and the archived experimental predictions.

The schema-reduction pipeline performs three successive selections:

```text
question -> database retrieval -> table extraction -> attribute extraction
             DB_result             TAB_result          ATT_result
```

The repository does not generate the final SQL query. Its purpose is to reduce
the search space presented to a downstream Text-to-SQL system while preserving
the schema elements required by the gold query.

## Hybrid-DB

The hybrid retriever assigns each candidate database three query-dependent
scores:

1. cosine similarity between the question and an offline database description;
2. the best cosine similarity between the question and any table document in
   that database;
3. the mean of the two highest BM25 table scores in that database.

The three scores are normalized across the candidate catalog and combined as

```text
score(DB) = 0.25 * description + 0.50 * dense-table + 0.25 * BM25
```

Dense representations use `BAAI/bge-large-en-v1.5`. Table documents contain
the table name, columns, directly joinable tables, and up to five frequent
values per column. BM25 uses the same structural evidence and all retained
textual values in `doc.json`, with English Snowball stemming and character
4-grams. The configuration was selected on BIRD train by an exhaustive search
over 13,970,880 configurations.

Table extraction then receives the predicted database, validated table
descriptions, and direct foreign-key neighbors. Attribute extraction receives
only the selected tables and supports two prompts: reduction-oriented (`R`) and
strict-recall-oriented (`SR`).

## Repository layout

| Path | Purpose |
|---|---|
| [src/DB_retrieval/](src/DB_retrieval/) | Hybrid retrieval, description generation, tuning, ablation, and evaluation |
| [src/schema_linking/tables_extraction/](src/schema_linking/tables_extraction/) | Table-description generation, table selection, and evaluation |
| [src/schema_linking/attributes_extraction/](src/schema_linking/attributes_extraction/) | `R`/`SR` attribute selection and evaluation |
| [src/data_manipulation.py](src/data_manipulation.py) | Schema utilities, `doc.json`, SQL analysis, and content-ambiguous workload construction |
| [src/generate_dev_tables.py](src/generate_dev_tables.py) | Deterministic Spider-style schema generation from SQLite metadata |
| [src/beaver_build.py](src/beaver_build.py) | BEAVER conversion from the published MySQL dumps |
| [src/sqale3_build.py](src/sqale3_build.py) | Reproducible SQaLE3 conversion from pinned Hugging Face revisions |
| [src_DB_retrieval_baseline/](src_DB_retrieval_baseline/) | Catalog-wide LLM database-selection baseline |
| `results_DB_retrieval_*`, `results_gpt-oss-120b-*` | Archived experimental predictions |
| `results/` | Mutable outputs of local runs; intentionally ignored by Git |

`src/` is the authoritative implementation.

## Benchmark contract

A retrieval-ready benchmark has the following layout:

```text
datasets/<benchmark>/
  dev.json
  dev_tables.json
  doc.json
  dev_databases/<db_id>/<db_id>.sqlite
```

- `dev.json` stores the question, gold database identifier, and gold SQL.
- `dev_tables.json` stores aligned semantic and physical schema identifiers,
  types, primary keys, and foreign keys in Spider/BIRD format.
- `doc.json` stores one document per table: schema identifiers, SQLite DDL,
  direct join neighbors, frequent values, and the normalized BM25 text.
- database descriptions used by the dense global signal are stored in
  [src/DB_retrieval/DB_descriptions/](src/DB_retrieval/DB_descriptions/).

Large datasets and SQLite files are not all distributed in Git. The builders
and conversion rules are documented below. Dataset sources, licenses, and
transformation notices are listed in
[datasets/README.md](datasets/README.md).

## Installation

The experiments were developed with Python 3.11 and CUDA-enabled PyTorch. From
the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install nltk==3.9.1 rank-bm25==0.2.2
```

The pinned PyTorch packages in `requirements.txt` target CUDA 12.1; use a
platform-appropriate PyTorch build when CUDA 12.1 is unavailable. The embedding
wrapper falls back to CPU.

LLM stages use AWS Bedrock and `openai.gpt-oss-120b-1:0`. Credentials are read
through botocore's standard credential chain. Optional region settings are
shown in [.env.example](.env.example). Database retrieval itself does not call
Bedrock once the descriptions have been generated.

## Reproduction workflow

The scripts use explicit constants or dataset selections rather than one shared
CLI. The commands below state the required working directory. Before running a
script, inspect its `__main__` block and use a new output path or preserve any
existing artifact that must not be overwritten.

### 1. Build or import a base benchmark

BIRD, Spider, and ARCHER are expected to be placed directly in the benchmark
layout described above. BEAVER and SQaLE3 have dedicated builders.

BEAVER requires authenticated access to its two Hugging Face repositories:

```powershell
# From the repository root; rewrites datasets/BEAVER and removes its download workspace on success.
python src/beaver_build.py
```

The builder creates the complete published BEAVER dataset. The 3,693-question
evaluation subset reported in this repository is an experiment-specific filter
and is not applied automatically by the builder.

SQaLE3 is built from revisions pinned in the script. Both
`datasets/SQALE3/` and `datasets/_sqale3_work/` must be absent before starting:

```powershell
cd src
python sqale3_build.py
cd ..
```

The builder filters verified pairs, materializes the DDL, validates query
results, removes unused databases, and generates `dev_tables.json`.

For any dataset that already contains SQLite files but lacks
`dev_tables.json`, set `DATASET_PATH` and `OUTPUT_PATH` at the top of
`src/generate_dev_tables.py`, then run:

```powershell
cd src
python generate_dev_tables.py
cd ..
```

The generator refuses to overwrite an existing output. `OUTPUT_PATH = None`
writes `<DATASET_PATH>/dev_tables.json`.

### 2. Generate `doc.json`

Run both document functions in this order. They read `dev_tables.json` and the
SQLite files; the second call adds the normalized BM25 text.

```powershell
cd src
python -c "from data_manipulation import create_benchmark_doc, add_bm25_text_to_benchmark_doc; p='../datasets/BIRDdev/'; create_benchmark_doc(p); add_bm25_text_to_benchmark_doc(p)"
cd ..
```

Replace `BIRDdev` with the target directory. `create_benchmark_doc()` writes one
record per table containing schema metadata, declared joinable tables, and at
most 50 frequency-ranked raw values for each eligible non-unique column.
`add_bm25_text_to_benchmark_doc()` builds the indexed text from table names,
columns, joinable tables, and duplicated string values; it applies lowercase,
accent and punctuation removal, digit-token filtering, English Snowball
stemming, and character four-grams. The first call rewrites `doc.json`; preserve
an existing file when it is part of a recorded experiment.

### 3. Generate a content-ambiguous workload

The following command creates `datasets/BIRDdev-ambiguous/` using the default
deterministic split and cyclic suffix rotation:

```powershell
cd src
python -c "from data_manipulation import create_ambiguous_benchmark; create_ambiguous_benchmark('../datasets/BIRDdev/', overwrite=True)"
python -c "from data_manipulation import create_benchmark_doc, add_bm25_text_to_benchmark_doc; p='../datasets/BIRDdev-ambiguous/'; create_benchmark_doc(p); add_bm25_text_to_benchmark_doc(p)"
cd ..
```

The repository versions `dev.json` and `ambiguity_metadata.json` for every
reported content-ambiguous workload. Clone SQLite files and other derived files
are omitted; given the corresponding base benchmark, the first command
deterministically regenerates the split, cyclic clone assignment,
`dev_tables.json`, and SQLite files. `overwrite=True` is required because the
versioned manifests make the destination directory already exist, and it
replaces that directory. The second command regenerates `doc.json`.

### 4. Generate database descriptions

Run from `src/DB_retrieval`, where the prompt path is resolved:

```powershell
cd src/DB_retrieval
python -c "from generate_DB_descriptions import generate_DB_descriptions; generate_DB_descriptions('../../datasets/BIRDdev/dev_tables.json', 'DB_descriptions/BIRDdev_DB_descriptions.json')"
cd ../..
```

For a content-ambiguous workload, pass `replicate_ambiguous_clones=True` only
when every source family contains schema-identical `_1`, `_2`, and `_3` clones:

```powershell
cd src/DB_retrieval
python -c "from generate_DB_descriptions import generate_DB_descriptions; generate_DB_descriptions('../../datasets/BIRDdev-ambiguous/dev_tables.json', 'DB_descriptions/BIRDdev_DB_descriptions_ambiguous.json', replicate_ambiguous_clones=True)"
cd ../..
```

The generator validates existing complete outputs, resumes its atomic
checkpoint, and retries each LLM response up to three times. Each prompt receives
the exact database name and its complete canonical table-to-column mapping from
`dev_tables.json`; the output is one concise `{name, description}` record per
database.

### 5. Run and evaluate database retrieval

From `src/DB_retrieval`, either configure the script's `__main__` block or call
the function directly:

```python
from DB_extractor import extract_DB
from embedder import Embedder

embedder = Embedder("BAAI/bge-large-en-v1.5", "cuda")
extract_DB(embedder, "BIRDdev")
```

For the current SQaLE3 content-ambiguous workload,
`memory_efficient=True` computes the same configured top-1 dense-table score in
batches instead of materializing the full similarity matrix. Results are saved
under `results/DB_retrieval/`.

Evaluate every supported result currently present in that directory with:

```powershell
cd src/DB_retrieval
python evaluation.py
cd ../..
```

Missing result files are reported and skipped. Accuracy always uses all records,
including `ERROR` predictions.

#### Paper Tables 3--7

The following aliases identify the archived predictions used in the paper:

| Alias | Method | Result directory |
|---|---|---|
| `<HYBRID>` | Hybrid-DB | `results_DB_retrieval_tune_BIRDtrain/DB_retrieval` |
| `<ITERATIVE_JAR>` | Iterative-JAR | `results_DB_retrieval_JARiterative/DB_retrieval` |
| `<CATALOG_LLM>` | catalog-wide LLM | `results_DB_retrieval_baseline_gpt-oss-120/DB_retrieval` |
| `<TABLE>` | table extraction (shared by `R` and `SR`) | `results_gpt-oss-120b-R/TAB_retrieval` |
| `<ATTRIBUTE_R>` | attribute extraction (`R`) | `results_gpt-oss-120b-R/ATT_retrieval` |
| `<ATTRIBUTE_SR>` | attribute extraction (`SR`) | `results_gpt-oss-120b-SR/ATT_retrieval` |

Run each command from the repository root once for every result directory listed
for that table, replacing `<RESULTS_DIR>` with its path above.

| Paper table | Reproduced results | Result directories | Command |
|---|---|---|---|
| Table 3 | Exact database-localization accuracy on standard benchmarks | `<HYBRID>`, `<ITERATIVE_JAR>`, `<CATALOG_LLM>` | `python src/DB_retrieval/evaluation.py --results-dir <RESULTS_DIR>` |
| Table 4 | Exact clone-localization accuracy on content-ambiguous workloads | `<HYBRID>`, `<ITERATIVE_JAR>`, `<CATALOG_LLM>` | `python src/DB_retrieval/evaluation.py --results-dir <RESULTS_DIR>` |
| Table 5 | Family Accuracy and exact clone accuracy conditioned on the correct family | `<HYBRID>`, `<ITERATIVE_JAR>`, `<CATALOG_LLM>` | `python src/DB_retrieval/evaluation.py --results-dir <RESULTS_DIR> --family-metrics` |
| Table 6 | Table-extraction metrics | `<TABLE>` | `python src/schema_linking/tables_extraction/evaluation.py --results-dir <RESULTS_DIR>` |
| Table 7 | Attribute-extraction metrics for `R` and `SR` | `<ATTRIBUTE_R>`, `<ATTRIBUTE_SR>` | `python src/schema_linking/attributes_extraction/evaluation.py --results-dir <RESULTS_DIR>` |

One evaluator invocation reports every supported file in the selected archive;
therefore the same run provides the standard results for Table 3 and the exact
clone results for Table 4. Missing results for non-executable method--workload
pairs are reported and skipped. In the table evaluator, `TABLE STRICT RECALL`
corresponds to `Strict E2E` and its `for table extraction only` variant to
`Strict|DB`. In the attribute evaluator, `STRICT RECALL` corresponds to
`Strict E2E` and its `for schema linking only` variant to `Strict|DB`. The
evaluators print fractions; the paper reports percentages rounded to two decimal
places. The `TAB_retrieval` files archived under `R` and `SR` are byte-identical;
Table 6 uses the `R` copy as the canonical path.

### 6. Generate table descriptions, extract tables, and evaluate

Set `SCHEMA_FILE` and `OUTPUT_FILE` in
`src/schema_linking/tables_extraction/generate_table_descriptions.py`, then run
from the repository root:

```powershell
python src/schema_linking/tables_extraction/generate_table_descriptions.py
```

Each prompt receives the exact database name and complete canonical
table-to-column mapping from `dev_tables.json` and must return one concise
description per table containing every column name. The generator validates the
exact database/table keys, checkpoints after each database, and retries invalid
generations up to three times.
Joinable tables are read later from `doc.json` and appended deterministically by
the extractor; they are not invented by the description LLM.

Before a complete extraction, select the dataset in `tables_extractor.py`, then
run:

```powershell
cd src/schema_linking/tables_extraction
python tables_extractor.py
python evaluation.py
cd ../../..
```

The extractor reads `results/DB_retrieval/`, writes
`results/TAB_retrieval/`, and checkpoints after every processed sample. The
evaluator processes BIRD dev, Spider dev, BEAVER, and ARCHER files that exist.

### 7. Extract attributes and evaluate

The current main processes BIRD dev, Spider dev, and ARCHER sequentially. Use
`r` for the reduction-oriented prompt or `sr` for the strict-recall-oriented
prompt:

```powershell
cd src/schema_linking/attributes_extraction
python attributes_extractor.py r
python evaluation.py
cd ../../..
```

Use `sr` in place of `r` for the second prompt. Both variants write the same
filenames under `results/ATT_retrieval/`; archive the first run before starting
the other. The extractor checkpoints every processed sample, including the
`NONE` shortcut.

### 8. Reproduce tuning, ablation, and baselines

The exhaustive BIRD-train tuning and the fixed ablation are run from the
retrieval directory:

```powershell
cd src/DB_retrieval
python tune_DB_retrieval.py
python DB_extractor_ablation.py
cd ../..
```

Tuning writes `DB_retrieval_best_hyperparameters.json`. Ablation requires the
full-hybrid references in `results_DB_retrieval_tune_BIRDtrain/`, verifies exact
prediction equality, and writes per-variant files plus
`results_DB_retrieval_ablation/summary.json`.

The catalog-wide LLM baseline is configured in
`src_DB_retrieval_baseline/DB_extractor_llm.py` and must be run from that
directory. Both the extractor and evaluator use `results/DB_retrieval/`:

```powershell
cd src_DB_retrieval_baseline
python DB_extractor_llm.py
python evaluation.py
cd ..
```

Iterative-JAR is included only as an experimental comparator through its
archived outcomes; it is not part of the documented execution pipeline.

All standard evaluators read the mutable `results/` tree. The tables reported
below were instead recomputed from the named archived directories, which should
be preserved as immutable experimental artifacts.

### 9. Reproduce the paper analyses

The identifier distributions, retrieval-error analysis, and catalog-wide LLM
bias analysis can be reproduced from the repository root:

```powershell
python analysis/spider_identifier_distribution.py
python analysis/spider_error_analysis.py
python analysis/bird_llm_bias_analysis.py
```

The first script reads the two Spider `doc.json` files, lowercases identifiers,
counts each name at most once per database, and writes four log-log rank plots
to `analysis/figures/`. It also records the aggregate counts in
`analysis/results/spider_identifier_distribution.json`. Plotting requires
Matplotlib 3.9 or later.

The second script parses the gold SQL of every Hybrid-DB retrieval error and
measures how often the predicted database contains any or all required table
and column names. Its random control makes one uniform draw per error from all
database IDs except the gold ID, using the fixed seed 42. The complete counts,
denominators, and control results are written to
`analysis/results/spider_error_analysis.json`.

The third script reads the archived BIRD content-ambiguous predictions and
reproduces the clone-selection bias analysis for the catalog-wide LLM. It
reports the suffix distribution over valid LLM outputs and examines the LLM
outcomes on the 103 samples retrieved correctly by Hybrid-DB. Its checked
summary is stored in `analysis/results/bird_llm_bias_analysis.json`.

## Main results

Database-retrieval accuracy is the fraction of questions for which the selected
database identifier exactly equals the gold identifier. `ERROR` predictions are
counted as incorrect.

| Base benchmark | Samples | Hybrid | Iterative-JAR | Catalog-wide LLM |
|---|---:|---:|---:|---:|
| BIRD dev | 1,534 | 98.70 | 97.33 | 99.74 |
| Spider dev 1.0 | 1,034 | 97.39 | 96.52 | 99.03 |
| Spider train | 7,000 | 76.20 | 71.36 | 75.47 |
| ARCHER | 518 | 93.63 | 90.54 | 98.07 |
| BEAVER subset | 3,693 | 99.81 | 99.70 | 95.97 |

| Content-ambiguous workload | Samples | Hybrid | Iterative-JAR | Catalog-wide LLM |
|---|---:|---:|---:|---:|
| BIRD dev | 126 | 81.75 | 40.48 | 52.38 |
| Spider dev 1.0 | 34 | 100.00 | 26.47 | 100.00 |
| Spider train | 151 | 76.16 | 31.13 | 75.50 |
| ARCHER | 28 | 89.29 | 50.00 | 96.43 |
| BEAVER | 499 | 76.15 | 36.07 | -- |
| SQaLE3 | 4,281 | 40.43 | 6.52 | -- |

The BEAVER experiments use the same 3,693-question subset, obtained by excluding
4,285 over-represented `dw` questions. The catalog-wide LLM baseline uses at
most nine values per column on BEAVER and 50 on the other reported datasets. It
was not run on the BEAVER or SQaLE3 content-ambiguous workloads because
serializing the full candidate catalog exceeded the model context window.

The ablation provides the clearest evidence for the value signal. Removing
values from both the dense-table and BM25 inputs changes accuracy by at most
0.49 percentage points on the four tested base benchmarks, but reduces it by
43.89--70.59 points on the content-ambiguous workloads. On these workloads, the
selected database family is usually still correct; the model loses the evidence
needed to select the correct clone.

## Reproducibility notes

- Hyperparameters and their search space are recorded in
  [src/DB_retrieval/tune_DB_retrieval.py](src/DB_retrieval/tune_DB_retrieval.py)
  and [DB_retrieval_best_hyperparameters_train.json](src/DB_retrieval/DB_retrieval_best_hyperparameters_train.json).
- Ablation predictions and their aggregate metrics are stored in
  [results_DB_retrieval_ablation/](results_DB_retrieval_ablation/).
- Hybrid predictions used for the comparison are under
  [results_DB_retrieval_tune_BIRDtrain/](results_DB_retrieval_tune_BIRDtrain/).
- Catalog-wide LLM and Iterative-JAR predictions are under
  `results_DB_retrieval_baseline_gpt-oss-120/` and
  `results_DB_retrieval_JARiterative/`, respectively.
- Current schema-linking outputs are archived separately for the `R` and `SR`
  prompts in `results_gpt-oss-120b-R/` and `results_gpt-oss-120b-SR/`.

When comparing archived results, preserve the associated benchmark revision,
sample count, configuration, and metric denominator: directories with similar
names may refer to different experiment revisions.
