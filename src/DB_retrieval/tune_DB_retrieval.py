import json
import math
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

import DB_extractor as db_retrieval
from embedder import Embedder


DATASET = "BIRDtrain"
OUTPUT_PATH = SCRIPT_DIR / "DB_retrieval_best_hyperparameters.json"
PROGRESS_EVERY = 1000

# Tuning search space
SEARCH_SPACE = {
    "QUERY_INSTRUCTION": [
        "Represent this sentence for searching relevant passages: ",
        "",
    ],
    "MAX_VALUES_PER_COLUMN": [3, 4, 5, 6, 7, 10],
    "TABLE_EMBEDDING_TABLES_PER_DB": [1, 2, 3, 5, 7, 10],
    "BM25_K1": [1.3, 1.5, 1.8, 1.9, 2, 2.2, 2.3, 2.5, 2.6, 2.7, 3, 3.5, 3.8, 4],
    "BM25_B": [0.5, 0.65, 0.75, 0.85, 0.95,  1.0],
    "BM25_TABLES_PER_DB": [1, 2, 3, 5, 7],
    "BM25_USE_CHAR4": [True, False],
}
WEIGHT_STEP = 0.05


def build_weight_configs() -> list:
    if WEIGHT_STEP <= 0:
        raise ValueError("WEIGHT_STEP must be positive.")

    units = round(1 / WEIGHT_STEP)
    if not math.isclose(units * WEIGHT_STEP, 1.0):
        raise ValueError("WEIGHT_STEP must be a positive divisor of 1.0.")

    return [
        {
            "DESCRIPTION_WEIGHT": description_units / units,
            "TABLE_WEIGHT": table_units / units,
            "BM25_WEIGHT": (units - description_units - table_units) / units,
        }
        for description_units in range(units + 1)
        for table_units in range(units - description_units + 1)
    ]


def load_dataset():
    dataset_paths = db_retrieval.get_dataset_paths(DATASET)

    with open(dataset_paths["doc_path"], "r", encoding="utf-8") as file:
        table_docs = json.load(file)

    with open(dataset_paths["descriptions_path"], "r", encoding="utf-8") as file:
        descriptions = json.load(file)

    with open(dataset_paths["questions_path"], "r", encoding="utf-8") as file:
        data = json.load(file)

    if not table_docs:
        raise ValueError("The table document file is empty.")
    if not descriptions:
        raise ValueError("The database description file is empty.")
    if not data:
        raise ValueError("The questions file is empty.")

    return table_docs, descriptions, data


def apply_hyperparameters(config: dict):
    for name, value in config.items():
        setattr(db_retrieval, name, value)
    db_retrieval.validate_hyperparameters()


def min_max_normalize_matrix(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.double()
    minimum = scores.min(dim=1, keepdim=True).values
    maximum = scores.max(dim=1, keepdim=True).values
    denominator = maximum - minimum

    normalized = torch.zeros_like(scores)
    mask = denominator.squeeze(1) != 0
    normalized[mask] = (
        scores[mask] - minimum[mask]
    ) / denominator[mask]
    return normalized


def normalize_bm25_matrix(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.double()
    maximum = scores.max(dim=1, keepdim=True).values

    normalized = torch.zeros_like(scores)
    mask = maximum.squeeze(1) > 0
    normalized[mask] = scores[mask] / maximum[mask]
    return normalized


def get_query_embeddings(
    cache: dict,
    embedder: Embedder,
    data: list,
    query_instruction: str,
):
    if query_instruction not in cache:
        cache[query_instruction] = embedder.get_sentences_embeddings([
            query_instruction + sample["question"]
            for sample in data
        ])
    return cache[query_instruction]


def get_description_score_matrix(
    cache: dict,
    query_embedding_cache: dict,
    embedder: Embedder,
    data: list,
    description_embeddings,
    query_instruction: str,
) -> torch.Tensor:
    if query_instruction not in cache:
        query_embeddings = get_query_embeddings(
            query_embedding_cache,
            embedder,
            data,
            query_instruction,
        )
        scores = db_retrieval.util.cos_sim(
            query_embeddings,
            description_embeddings,
        ).cpu()
        if not torch.isfinite(scores).all():
            raise ValueError("Description similarities contain non-finite values.")
        cache[query_instruction] = min_max_normalize_matrix(scores)

    return cache[query_instruction]


def get_table_embeddings(
    cache: dict,
    embedder: Embedder,
    table_docs: list,
    max_values_per_column: int,
):
    if max_values_per_column not in cache:
        db_retrieval.MAX_VALUES_PER_COLUMN = max_values_per_column
        cache[max_values_per_column] = db_retrieval.build_table_embeddings(
            embedder,
            table_docs,
        )
    return cache[max_values_per_column]


def get_table_score_matrix(
    cache: dict,
    query_embedding_cache: dict,
    table_embedding_cache: dict,
    embedder: Embedder,
    data: list,
    table_docs: list,
    table_indices_by_db: dict,
    database_names: list,
    query_instruction: str,
    max_values_per_column: int,
    tables_per_db: int,
) -> torch.Tensor:
    key = (query_instruction, max_values_per_column, tables_per_db)
    if key not in cache:
        query_embeddings = get_query_embeddings(
            query_embedding_cache,
            embedder,
            data,
            query_instruction,
        )
        table_embeddings = get_table_embeddings(
            table_embedding_cache,
            embedder,
            table_docs,
            max_values_per_column,
        )
        scores = db_retrieval.util.cos_sim(
            query_embeddings,
            table_embeddings,
        ).cpu()
        if not torch.isfinite(scores).all():
            raise ValueError("Table similarities contain non-finite values.")

        for top_k in SEARCH_SPACE["TABLE_EMBEDDING_TABLES_PER_DB"]:
            db_retrieval.TABLE_EMBEDDING_TABLES_PER_DB = top_k
            top_k_key = (query_instruction, max_values_per_column, top_k)
            db_scores = db_retrieval.aggregate_table_embedding_scores(
                scores,
                table_indices_by_db,
                database_names,
            )
            cache[top_k_key] = min_max_normalize_matrix(db_scores)

    return cache[key]


def build_query_tokens(data: list, use_char4: bool) -> list:
    query_tokens = []
    for sample in data:
        normalized_question = db_retrieval.normalize_and_stem_text(
            sample["question"]
        )
        query_tokens.append(
            db_retrieval.add_char4_tokens(normalized_question).split()
            if use_char4
            else normalized_question.split()
        )
    return query_tokens


def aggregate_bm25_table_scores(
    table_score_matrix: np.ndarray,
    table_indices_by_db: dict,
    database_names: list,
    tables_per_db: int,
) -> torch.Tensor:
    db_score_matrix = np.empty(
        (table_score_matrix.shape[0], len(database_names)),
        dtype=np.float64,
    )

    for db_index, db_name in enumerate(database_names):
        table_indices = table_indices_by_db[db_name]
        table_count = min(tables_per_db, len(table_indices))
        table_scores = table_score_matrix[:, table_indices]
        best_scores = np.partition(
            table_scores,
            -table_count,
            axis=1,
        )[:, -table_count:]
        db_score_matrix[:, db_index] = best_scores.mean(axis=1)

    return torch.from_numpy(db_score_matrix)


def get_bm25_score_matrix(
    cache: dict,
    token_cache: dict,
    data: list,
    table_docs: list,
    table_indices_by_db: dict,
    database_names: list,
    k1: float,
    b: float,
    use_char4: bool,
    tables_per_db: int,
) -> torch.Tensor:
    key = (k1, b, use_char4, tables_per_db)
    if key not in cache:
        db_retrieval.BM25_K1 = k1
        db_retrieval.BM25_B = b
        db_retrieval.BM25_USE_CHAR4 = use_char4
        bm25 = db_retrieval.build_bm25(table_docs)

        if use_char4 not in token_cache:
            token_cache[use_char4] = build_query_tokens(data, use_char4)

        table_score_matrix = np.empty(
            (len(data), len(table_docs)),
            dtype=np.float64,
        )
        for query_index, query_tokens in enumerate(token_cache[use_char4]):
            scores = bm25.get_scores(query_tokens)
            if not np.isfinite(scores).all():
                raise ValueError(
                    f"BM25 produced non-finite scores for query index {query_index}."
                )
            table_score_matrix[query_index] = np.maximum(scores, 0.0)

        for top_k in SEARCH_SPACE["BM25_TABLES_PER_DB"]:
            top_k_key = (k1, b, use_char4, top_k)
            db_scores = aggregate_bm25_table_scores(
                table_score_matrix,
                table_indices_by_db,
                database_names,
                top_k,
            )
            cache[top_k_key] = normalize_bm25_matrix(db_scores)

    return cache[key]


def calculate_accuracy(
    description_scores: torch.Tensor,
    table_scores: torch.Tensor,
    bm25_scores: torch.Tensor,
    weight_config: dict,
    gold_indices: torch.Tensor,
    lexicographic_indices: torch.Tensor,
) -> float:
    final_scores = (
        weight_config["DESCRIPTION_WEIGHT"] * description_scores
        + weight_config["TABLE_WEIGHT"] * table_scores
        + weight_config["BM25_WEIGHT"] * bm25_scores
    )
    predictions = lexicographic_indices[
        final_scores[:, lexicographic_indices].argmax(dim=1)
    ]
    return (predictions == gold_indices).double().mean().item()


def main():
    start = time.perf_counter()
    table_docs, descriptions, data = load_dataset()

    database_names = [
        description["name"]
        for description in descriptions
    ]
    db_index_by_name = {
        db_name: index
        for index, db_name in enumerate(database_names)
    }
    gold_indices = torch.tensor(
        [db_index_by_name[sample["db_id"]] for sample in data],
        dtype=torch.long,
    )
    lexicographic_indices = torch.tensor(
        sorted(range(len(database_names)), key=lambda index: database_names[index]),
        dtype=torch.long,
    )

    table_indices_by_db = db_retrieval.build_table_indices_by_db(table_docs)

    embedder = Embedder(
        model_name=db_retrieval.EMBEDDING_MODEL_NAME,
        device_name=db_retrieval.EMBEDDING_DEVICE_NAME,
    )
    description_embeddings = db_retrieval.build_description_embeddings(
        embedder,
        descriptions,
    )

    query_embedding_cache = {}
    description_score_cache = {}
    table_embedding_cache = {}
    table_score_cache = {}
    bm25_token_cache = {}
    bm25_score_cache = {}

    parameter_names = list(SEARCH_SPACE)
    parameter_combinations = list(product(
        *(SEARCH_SPACE[name] for name in parameter_names)
    ))
    weight_configs = build_weight_configs()
    total_configurations = len(parameter_combinations) * len(weight_configs)

    best_accuracy = -1.0
    best_config = None

    print(f"Testing {total_configurations} configurations...")
    print(
        f"Base configurations: {len(parameter_combinations)}, "
        f"weight configurations: {len(weight_configs)}"
    )

    checked_configurations = 0
    for base_index, parameter_values in enumerate(parameter_combinations, start=1):
        base_config = dict(zip(parameter_names, parameter_values))

        description_scores = get_description_score_matrix(
            description_score_cache,
            query_embedding_cache,
            embedder,
            data,
            description_embeddings,
            base_config["QUERY_INSTRUCTION"],
        )
        table_scores = get_table_score_matrix(
            table_score_cache,
            query_embedding_cache,
            table_embedding_cache,
            embedder,
            data,
            table_docs,
            table_indices_by_db,
            database_names,
            base_config["QUERY_INSTRUCTION"],
            base_config["MAX_VALUES_PER_COLUMN"],
            base_config["TABLE_EMBEDDING_TABLES_PER_DB"],
        )
        bm25_scores = get_bm25_score_matrix(
            bm25_score_cache,
            bm25_token_cache,
            data,
            table_docs,
            table_indices_by_db,
            database_names,
            base_config["BM25_K1"],
            base_config["BM25_B"],
            base_config["BM25_USE_CHAR4"],
            base_config["BM25_TABLES_PER_DB"],
        )

        for weight_config in weight_configs:
            config = {**base_config, **weight_config}
            apply_hyperparameters(config)

            accuracy = calculate_accuracy(
                description_scores,
                table_scores,
                bm25_scores,
                weight_config,
                gold_indices,
                lexicographic_indices,
            )
            checked_configurations += 1

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config.copy()
                print(
                    f"New best accuracy: {best_accuracy:.6f} "
                    f"after {checked_configurations} configurations"
                )

        if base_index % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - start
            print(
                f"Completed {base_index}/{len(parameter_combinations)} "
                f"base configurations in {elapsed:.2f}s"
            )

    result = {
        "accuracy": best_accuracy,
        "hyperparameters": best_config,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

    elapsed = time.perf_counter() - start
    print(json.dumps(result, ensure_ascii=False, indent=4))
    print(f"Best configuration saved to: {OUTPUT_PATH}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
