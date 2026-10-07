"""Analyze schema overlap in Hybrid-DB retrieval errors on Spider.

For every wrong prediction, the gold SQL is parsed with the same SQLGlot-based
utility used elsewhere in the project. Gold table and column names are compared
with the schema of the predicted database. A deterministic random control draws
one database per error, uniformly from all database IDs except the gold one.
"""

import json
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_manipulation import extract_tables_and_columns  # noqa: E402


RANDOM_SEED = 42
RESULTS_PATH = ROOT / "analysis" / "results" / "spider_error_analysis.json"
METRICS = (
    "any_gold_table",
    "all_gold_tables",
    "any_gold_column",
    "all_gold_columns",
    "all_gold_tables_and_columns",
)

DATASETS = {
    "spider_dev": {
        "doc": ROOT / "datasets" / "SPIDERdev1.0" / "doc.json",
        "predictions": (
            ROOT
            / "results_DB_retrieval_tune_BIRDtrain"
            / "DB_retrieval"
            / "SPIDERdev1_DB_extractor.json"
        ),
    },
    "spider_train": {
        "doc": ROOT / "datasets" / "SPIDERtrain" / "doc.json",
        "predictions": (
            ROOT
            / "results_DB_retrieval_tune_BIRDtrain"
            / "DB_retrieval"
            / "SPIDERtrain_DB_extractor.json"
        ),
    },
}

EXPECTED_RETRIEVAL = {
    "spider_dev": {
        "samples": 1034,
        "errors": 27,
        "column_evaluable_errors": 23,
        "predicted": {
            "any_gold_table": 6,
            "all_gold_tables": 3,
            "any_gold_column": 19,
            "all_gold_columns": 3,
            "all_gold_tables_and_columns": 0,
        },
        "random_control": {
            "any_gold_table": 0,
            "all_gold_tables": 0,
            "any_gold_column": 6,
            "all_gold_columns": 0,
            "all_gold_tables_and_columns": 0,
        },
    },
    "spider_train": {
        "samples": 7000,
        "errors": 1666,
        "column_evaluable_errors": 1559,
        "predicted": {
            "any_gold_table": 867,
            "all_gold_tables": 685,
            "any_gold_column": 958,
            "all_gold_columns": 390,
            "all_gold_tables_and_columns": 199,
        },
        "random_control": {
            "any_gold_table": 53,
            "all_gold_tables": 29,
            "any_gold_column": 220,
            "all_gold_columns": 41,
            "all_gold_tables_and_columns": 7,
        },
    },
}


def normalize_identifier(identifier):
    """Match SQL and doc identifiers despite spaces, underscores, or quoting."""
    return re.sub(r"[^a-z0-9]", "", identifier.lower())


def load_schemas(path):
    with path.open(encoding="utf-8") as file:
        documents = json.load(file)

    schemas = {}
    for document in documents:
        schema = schemas.setdefault(document["db"], {"tables": set(), "columns": set()})
        schema["tables"].add(normalize_identifier(document["table"]))
        schema["columns"].update(
            normalize_identifier(column) for column in document["columns"]
        )
    return schemas


def overlap_flags(required_tables, required_columns, schema):
    all_tables = bool(required_tables) and required_tables <= schema["tables"]
    all_columns = bool(required_columns) and required_columns <= schema["columns"]
    return {
        "any_gold_table": bool(required_tables & schema["tables"]),
        "all_gold_tables": all_tables,
        "any_gold_column": bool(required_columns & schema["columns"]),
        "all_gold_columns": all_columns,
        "all_gold_tables_and_columns": all_tables and all_columns,
    }


def increment(counts, flags):
    for name, value in flags.items():
        counts[name] += int(value)


def analyze_dataset(config):
    schemas = load_schemas(config["doc"])
    database_ids = list(schemas)
    with config["predictions"].open(encoding="utf-8") as file:
        samples = json.load(file)

    rng = random.Random(RANDOM_SEED)
    counts = {
        "predicted": {name: 0 for name in METRICS},
        "random_control": {name: 0 for name in METRICS},
    }
    errors = 0
    column_evaluable_errors = 0

    for sample in samples:
        if sample["DB_result"] == sample["db_id"]:
            continue

        errors += 1
        identifiers = extract_tables_and_columns(sample["SQL"])
        required_tables = {
            normalize_identifier(table) for table in identifiers["table"]
        }
        required_columns = {
            normalize_identifier(column) for column in identifiers["column"]
        }
        column_evaluable_errors += int(bool(required_columns))

        increment(
            counts["predicted"],
            overlap_flags(required_tables, required_columns, schemas[sample["DB_result"]]),
        )

        candidates = [db_id for db_id in database_ids if db_id != sample["db_id"]]
        random_db = rng.choice(candidates)
        increment(
            counts["random_control"],
            overlap_flags(required_tables, required_columns, schemas[random_db]),
        )

    return {
        "samples": len(samples),
        "errors": errors,
        "column_evaluable_errors": column_evaluable_errors,
        **counts,
    }


def validate(dataset_key, result):
    expected = EXPECTED_RETRIEVAL[dataset_key]
    for field in ("samples", "errors", "column_evaluable_errors"):
        if result[field] != expected[field]:
            raise RuntimeError(
                f"Unexpected {dataset_key} {field}: "
                f"expected {expected[field]}, got {result[field]}."
            )
    if result["predicted"] != expected["predicted"]:
        raise RuntimeError(
            f"Unexpected {dataset_key} predicted-database overlap: "
            f"expected {expected['predicted']}, got {result['predicted']}."
        )
    if result["random_control"] != expected["random_control"]:
        raise RuntimeError(
            f"Unexpected {dataset_key} random-control overlap: "
            f"expected {expected['random_control']}, got {result['random_control']}."
        )


def percentage(count, denominator):
    return 100 * count / denominator


def print_result(dataset_key, result):
    print(
        f"{dataset_key}: {result['errors']} retrieval errors; "
        f"{result['column_evaluable_errors']} with at least one parsed column"
    )
    for metric in METRICS:
        denominator = (
            result["column_evaluable_errors"]
            if metric in ("any_gold_column", "all_gold_columns")
            else result["errors"]
        )
        predicted = result["predicted"][metric]
        control = result["random_control"][metric]
        print(
            f"  {metric}: predicted={predicted}/{denominator} "
            f"({percentage(predicted, denominator):.2f}%), "
            f"random={control}/{denominator} "
            f"({percentage(control, denominator):.2f}%)"
        )


def main():
    results = {}
    for dataset_key, config in DATASETS.items():
        result = analyze_dataset(config)
        validate(dataset_key, result)
        results[dataset_key] = result
        print_result(dataset_key, result)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "random_seed": RANDOM_SEED,
        "random_control": "one uniform draw per error, excluding the gold database",
        "datasets": results,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)


if __name__ == "__main__":
    main()
