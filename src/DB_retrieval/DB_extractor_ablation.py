"""Run the fixed-weight ablations of the hybrid DB retriever.

Each dataset is encoded once. The normalized description, dense-table and
BM25 scores are then recombined without changing any retriever hyperparameter.
The value-free full hybrid uses the same weights but removes top values from
both the dense-table text and the BM25 corpus.
The full-hybrid predictions must match the consolidated reference exactly;
otherwise the dataset is not saved.
"""

import json
from pathlib import Path

from sentence_transformers import util
import torch

import DB_extractor as retriever
from embedder import Embedder


DATASETS = (
    "BIRDdev",
    "SPIDERdev1",
    "SPIDERtrain",
    "ARCHER",
    "BIRDdev-ambiguous",
    "SPIDERdev1-ambiguous",
    "SPIDERtrain-ambiguous",
    "ARCHER-ambiguous",
    "BEAVER-ambiguous",
)

WEIGHT_CONFIGURATIONS = {
    "description_only": (1.0, 0.0, 0.0),
    "dense_table_only": (0.0, 1.0, 0.0),
    "bm25_value_only": (0.0, 0.0, 1.0),
    "dense_description": (1.0 / 3.0, 2.0 / 3.0, 0.0),
    "dense_bm25": (0.0, 2.0 / 3.0, 1.0 / 3.0),
    "description_bm25": (0.5, 0.0, 0.5),
    "full_hybrid": (
        retriever.DESCRIPTION_WEIGHT,
        retriever.TABLE_WEIGHT,
        retriever.BM25_WEIGHT,
    ),
    "full_hybrid_no_values": (
        retriever.DESCRIPTION_WEIGHT,
        retriever.TABLE_WEIGHT,
        retriever.BM25_WEIGHT,
    ),
}

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "results_DB_retrieval_ablation"
REFERENCE_DIR = (
    PROJECT_DIR
    / "results_DB_retrieval_tune_BIRDtrain"
    / "DB_retrieval"
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=4)


def build_value_free_table_docs(table_docs: list) -> list:
    value_free_docs = []
    for table_doc in table_docs:
        text_parts = [
            table_doc.get("table", ""),
            *table_doc.get("columns", []),
            *table_doc.get("joinable_tables", []),
        ]
        normalized_text = retriever.normalize_and_stem_text(
            " ".join(part for part in text_parts if part)
        )
        value_free_docs.append({
            **table_doc,
            "top_values": {},
            "bm25_text": retriever.add_char4_tokens(normalized_text),
        })
    return value_free_docs


def source_family(database_name: str) -> str:
    prefix, separator, suffix = database_name.rpartition("_")
    if separator and suffix in {"1", "2", "3"}:
        return prefix
    return database_name


def evaluate_predictions(predictions: list, ambiguous: bool) -> dict:
    total = len(predictions)
    exact = sum(
        item["DB_result"] == item["db_id"]
        for item in predictions
    )
    metrics = {
        "correct": exact,
        "accuracy": exact / total if total else 0.0,
    }

    if ambiguous:
        family_correct = sum(
            source_family(item["DB_result"]) == source_family(item["db_id"])
            for item in predictions
        )
        metrics.update({
            "family_correct": family_correct,
            "family_accuracy": family_correct / total if total else 0.0,
            "clone_given_family": (
                exact / family_correct
                if family_correct
                else 0.0
            ),
        })

    return metrics


def validate_full_hybrid(predictions: list, reference_path: Path):
    reference = load_json(reference_path)
    if len(predictions) != len(reference):
        raise ValueError(
            f"Full-hybrid length mismatch for {reference_path.name}: "
            f"{len(predictions)} != {len(reference)}"
        )

    mismatches = []
    for index, (current, expected) in enumerate(zip(predictions, reference)):
        current_key = (
            current["question_id"],
            current["db_id"],
            current["DB_result"],
        )
        expected_key = (
            expected["question_id"],
            expected["db_id"],
            expected["DB_result"],
        )
        if current_key != expected_key:
            mismatches.append((index, current_key, expected_key))
            if len(mismatches) == 5:
                break

    if mismatches:
        raise ValueError(
            f"Full-hybrid predictions do not match {reference_path.name}: "
            f"{mismatches}"
        )


def run_dataset(embedder: Embedder, dataset: str) -> dict:
    retriever.validate_hyperparameters()
    dataset_paths = retriever.get_dataset_paths(dataset)
    table_docs = load_json(dataset_paths["doc_path"])
    descriptions = load_json(dataset_paths["descriptions_path"])
    data = load_json(dataset_paths["questions_path"])

    if not data or not table_docs or not descriptions:
        raise ValueError(f"Empty input found for {dataset}.")

    database_names = [description["name"] for description in descriptions]
    value_free_table_docs = build_value_free_table_docs(table_docs)
    bm25 = retriever.build_bm25(table_docs)
    value_free_bm25 = retriever.build_bm25(value_free_table_docs)
    description_embeddings = retriever.build_description_embeddings(
        embedder,
        descriptions,
    )
    table_embeddings = retriever.build_table_embeddings(embedder, table_docs)
    value_free_table_embeddings = retriever.build_table_embeddings(
        embedder,
        value_free_table_docs,
    )
    query_embeddings = embedder.get_sentences_embeddings([
        retriever.QUERY_INSTRUCTION + sample["question"]
        for sample in data
    ])

    description_similarity_matrix = util.cos_sim(
        query_embeddings,
        description_embeddings,
    ).cpu()
    table_similarity_matrix = util.cos_sim(
        query_embeddings,
        table_embeddings,
    ).cpu()
    value_free_table_similarity_matrix = util.cos_sim(
        query_embeddings,
        value_free_table_embeddings,
    ).cpu()
    if not torch.isfinite(description_similarity_matrix).all():
        raise ValueError(f"Non-finite description similarities for {dataset}.")
    if not torch.isfinite(table_similarity_matrix).all():
        raise ValueError(f"Non-finite table similarities for {dataset}.")
    if not torch.isfinite(value_free_table_similarity_matrix).all():
        raise ValueError(
            f"Non-finite value-free table similarities for {dataset}."
        )

    table_indices_by_db = retriever.build_table_indices_by_db(table_docs)
    table_db_score_matrix = retriever.aggregate_table_embedding_scores(
        table_similarity_matrix,
        table_indices_by_db,
        database_names,
    )
    value_free_table_db_score_matrix = (
        retriever.aggregate_table_embedding_scores(
            value_free_table_similarity_matrix,
            table_indices_by_db,
            database_names,
        )
    )
    predictions = {
        configuration: []
        for configuration in WEIGHT_CONFIGURATIONS
    }

    for question_index, sample in enumerate(data):
        description_scores = dict(zip(
            database_names,
            description_similarity_matrix[question_index].tolist(),
        ))
        table_scores = dict(zip(
            database_names,
            table_db_score_matrix[question_index].tolist(),
        ))
        value_free_table_scores = dict(zip(
            database_names,
            value_free_table_db_score_matrix[question_index].tolist(),
        ))
        bm25_scores = retriever.get_bm25_db_scores(
            sample["question"],
            table_docs,
            bm25,
            database_names,
        )
        value_free_bm25_scores = retriever.get_bm25_db_scores(
            sample["question"],
            value_free_table_docs,
            value_free_bm25,
            database_names,
        )
        normalized_description = retriever.min_max_normalize(
            description_scores
        )
        normalized_tables = retriever.min_max_normalize(table_scores)
        normalized_bm25 = retriever.normalize_bm25_scores(bm25_scores)
        normalized_value_free_tables = retriever.min_max_normalize(
            value_free_table_scores
        )
        normalized_value_free_bm25 = retriever.normalize_bm25_scores(
            value_free_bm25_scores
        )

        base_item = {
            "question_id": retriever.sample_question_id(
                sample,
                question_index,
            ),
            "db_id": sample["db_id"],
            "question": sample["question"],
            "SQL": retriever.sample_sql(sample),
        }

        for configuration, weights in WEIGHT_CONFIGURATIONS.items():
            description_weight, table_weight, bm25_weight = weights
            if configuration == "full_hybrid_no_values":
                table_component = normalized_value_free_tables
                bm25_component = normalized_value_free_bm25
            else:
                table_component = normalized_tables
                bm25_component = normalized_bm25
            final_scores = {
                database_name: (
                    description_weight
                    * normalized_description[database_name]
                    + table_weight * table_component[database_name]
                    + bm25_weight * bm25_component[database_name]
                )
                for database_name in database_names
            }
            database_result = min(
                database_names,
                key=lambda database_name: (
                    -final_scores[database_name],
                    database_name,
                ),
            )
            predictions[configuration].append({
                **base_item,
                "DB_result": database_result,
            })

    reference_path = REFERENCE_DIR / dataset_paths["result_file_path"].name
    validate_full_hybrid(predictions["full_hybrid"], reference_path)

    dataset_output_dir = OUTPUT_DIR / dataset
    metrics = {}
    for configuration, configuration_predictions in predictions.items():
        write_json(
            dataset_output_dir / f"{configuration}.json",
            configuration_predictions,
        )
        metrics[configuration] = evaluate_predictions(
            configuration_predictions,
            ambiguous=dataset.endswith("-ambiguous"),
        )

    return {
        "samples": len(data),
        "full_reference": str(reference_path.relative_to(PROJECT_DIR)),
        "full_reference_mismatches": 0,
        "variants": metrics,
    }


def main():
    embedder = Embedder(
        model_name=retriever.EMBEDDING_MODEL_NAME,
        device_name=retriever.EMBEDDING_DEVICE_NAME,
    )
    summary = {
        "weights": {
            name: {
                "description": weights[0],
                "dense_table": weights[1],
                "bm25": weights[2],
            }
            for name, weights in WEIGHT_CONFIGURATIONS.items()
        },
        "datasets": {},
    }

    for dataset in DATASETS:
        print(f"Processing {dataset}...", flush=True)
        summary["datasets"][dataset] = run_dataset(embedder, dataset)
        write_json(OUTPUT_DIR / "summary.json", summary)
        print(f"Completed {dataset}.", flush=True)


if __name__ == "__main__":
    main()
