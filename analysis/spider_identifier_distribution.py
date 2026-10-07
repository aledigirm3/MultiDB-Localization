"""Reproduce the Spider identifier-frequency statistics and rank plots.

Identifiers are read from ``doc.json`` and normalized only by lowercasing.
Each identifier is counted at most once per database, so the plotted value is
the number of distinct databases in which that table or column name occurs.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "analysis" / "figures"
RESULTS_PATH = ROOT / "analysis" / "results" / "spider_identifier_distribution.json"

DATASETS = {
    "spider_dev": {
        "label": "Spider dev",
        "doc": ROOT / "datasets" / "SPIDERdev1.0" / "doc.json",
    },
    "spider_train": {
        "label": "Spider train",
        "doc": ROOT / "datasets" / "SPIDERtrain" / "doc.json",
    },
}

EXPECTED = {
    "spider_dev": {
        "tables": (81, 80, 79),
        "columns": (441, 307, 279),
    },
    "spider_train": {
        "tables": (737, 545, 467),
        "columns": (3850, 1758, 1446),
    },
}


def load_identifier_sets(path):
    """Return per-database names and their raw schema occurrence counts."""
    with path.open(encoding="utf-8") as file:
        documents = json.load(file)

    tables_by_db = defaultdict(set)
    columns_by_db = defaultdict(set)
    table_occurrences = 0
    column_occurrences = 0
    for document in documents:
        db_id = document["db"]
        tables_by_db[db_id].add(document["table"].lower())
        columns_by_db[db_id].update(column.lower() for column in document["columns"])
        table_occurrences += 1
        column_occurrences += len(document["columns"])
    return (
        (tables_by_db, table_occurrences),
        (columns_by_db, column_occurrences),
    )


def summarize(names_by_db, total_occurrences):
    frequencies = Counter()
    for names in names_by_db.values():
        frequencies.update(names)

    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    distinct_identifiers = len(frequencies)
    singletons = sum(frequency == 1 for frequency in frequencies.values())
    return {
        "database_count": len(names_by_db),
        "total_occurrences": total_occurrences,
        "distinct_identifiers": distinct_identifiers,
        "singletons": singletons,
        "singleton_percentage": 100 * singletons / distinct_identifiers,
        "ranked_frequencies": ranked,
    }


def plot_distribution(summary, title, output_path):
    ranks = range(1, summary["distinct_identifiers"] + 1)
    frequencies = [frequency for _, frequency in summary["ranked_frequencies"]]

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(ranks, frequencies, color="#326fa8", linewidth=2.2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Identifier rank")
    axis.set_ylabel("Distinct databases")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(True, which="major", color="#d8dde3", linewidth=0.8)
    axis.grid(True, which="minor", color="#edf0f2", linewidth=0.5)
    axis.text(
        0.98,
        0.96,
        (
            f"identifiers={summary['distinct_identifiers']}; "
            f"singletons={summary['singleton_percentage']:.1f}%"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for dataset_key, dataset in DATASETS.items():
        tables, columns = load_identifier_sets(dataset["doc"])
        results[dataset_key] = {}

        for identifier_type, (names_by_db, total_occurrences) in (
            ("tables", tables),
            ("columns", columns),
        ):
            summary = summarize(names_by_db, total_occurrences)
            observed = (
                summary["total_occurrences"],
                summary["distinct_identifiers"],
                summary["singletons"],
            )
            if observed != EXPECTED[dataset_key][identifier_type]:
                raise RuntimeError(
                    f"Unexpected {dataset_key} {identifier_type} statistics: "
                    f"expected {EXPECTED[dataset_key][identifier_type]}, got {observed}."
                )

            plot_distribution(
                summary,
                f"{dataset['label']} - {identifier_type}",
                FIGURES_DIR / f"{dataset_key}_{identifier_type}.png",
            )
            results[dataset_key][identifier_type] = {
                key: value
                for key, value in summary.items()
                if key != "ranked_frequencies"
            }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    for dataset_key, dataset_results in results.items():
        for identifier_type, summary in dataset_results.items():
            print(
                f"{dataset_key} {identifier_type}: "
                f"occurrences={summary['total_occurrences']}, "
                f"identifiers={summary['distinct_identifiers']}, "
                f"singletons={summary['singletons']} "
                f"({summary['singleton_percentage']:.2f}%)"
            )


if __name__ == "__main__":
    main()
