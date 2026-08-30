import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import util
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedder import Embedder
import paths
from ansi_colors import *
from data_manipulation import add_char4_tokens, normalize_and_stem_text


# Retrieval hyperparameters
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DEVICE_NAME = "cuda"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

MAX_VALUES_PER_COLUMN = 5
TABLE_EMBEDDING_TABLES_PER_DB = 1

BM25_K1 = 1.3
BM25_B = 1.0
BM25_TABLES_PER_DB = 2
BM25_USE_CHAR4 = True

DESCRIPTION_WEIGHT = 0.25
TABLE_WEIGHT = 0.50
BM25_WEIGHT = 0.25


def get_dataset_paths(dataset: str):
    src_dir = Path(__file__).resolve().parents[1]
    extractor_dir = Path(__file__).resolve().parent

    if dataset == "BIRDdev":
        dataset_dir = (src_dir / paths.DATASETS.BIRDdev.value).resolve()
        description_filename = "BIRDdev_DB_descriptions.json"
        result_filename = "BIRDdev_DB_extractor.json"
    elif dataset == "BIRDdev-ambiguous":
        dataset_dir = (src_dir / paths.DATASETS.BIRDdev_ambiguous.value).resolve()
        description_filename = "BIRDdev_DB_descriptions_ambiguous.json"
        result_filename = "BIRDdev_DB_extractor_ambiguous.json"
    elif dataset == "SPIDERdev1":
        dataset_dir = (src_dir / paths.DATASETS.SPIDERdev1.value).resolve()
        description_filename = "SPIDERdev1_DB_descriptions.json"
        result_filename = "SPIDERdev1_DB_extractor.json"
    elif dataset == "SPIDERdev1-ambiguous":
        dataset_dir = (src_dir / paths.DATASETS.SPIDERdev1_ambiguous.value).resolve()
        description_filename = "SPIDERdev1_DB_descriptions_ambiguous.json"
        result_filename = "SPIDERdev1_DB_extractor_ambiguous.json"
    elif dataset == "BIRDtrain":
        dataset_dir = (src_dir / paths.DATASETS.BIRDtrain.value).resolve()
        description_filename = "BIRDtrain_DB_descriptions.json"
        result_filename = "BIRDtrain_DB_extractor.json"
    elif dataset == "BIRDtrain-ambiguous":
        dataset_dir = (src_dir / paths.DATASETS.BIRDtrain_ambiguous.value).resolve()
        description_filename = "BIRDtrain_DB_descriptions_ambiguous.json"
        result_filename = "BIRDtrain_DB_extractor_ambiguous.json"
    elif dataset == "BEAVER":
            dataset_dir = (src_dir / paths.DATASETS.BEAVER.value).resolve()
            description_filename = "BEAVER_DB_descriptions.json"
            result_filename = "BEAVER_DB_extractor.json"
    else:
        raise ValueError(f"Dataset not found: {dataset}")

    results_folder = (src_dir / paths.RESULTS.DB_RETRIEVAL.value).resolve()
    doc_filename = "doc.json"

    return {
        "questions_path": dataset_dir / "dev.json",
        "doc_path": dataset_dir / doc_filename,
        "descriptions_path": extractor_dir / description_filename,
        "results_folder": results_folder,
        "result_file_path": results_folder / result_filename,
    }


def build_bm25(table_docs: list):
    tokenized_corpus = []
    for table_doc in table_docs:
        tokens = table_doc.get("bm25_text", "").split()
        if not BM25_USE_CHAR4:
            tokens = [
                token
                for token in tokens
                if not token.startswith("char4:")
            ]
        tokenized_corpus.append(tokens)

    return BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)


def get_bm25_db_scores(
    question: str,
    table_docs: list,
    bm25: BM25Okapi,
    database_names: list,
):
    normalized_question = normalize_and_stem_text(question)
    query_tokens = (
        add_char4_tokens(normalized_question).split()
        if BM25_USE_CHAR4
        else normalized_question.split()
    )
    scores = bm25.get_scores(query_tokens)
    db_table_scores = defaultdict(list)

    for index, table_doc in enumerate(table_docs):
        db_name = table_doc.get("db")
        score = float(scores[index])
        if not math.isfinite(score):
            raise ValueError(
                f"BM25 produced a non-finite score for table index {index}."
            )
        score = max(0.0, score)
        db_table_scores[db_name].append(score)

    db_scores = {}
    for db_name in database_names:
        table_scores = sorted(db_table_scores.get(db_name, []), reverse=True)
        best_scores = table_scores[:BM25_TABLES_PER_DB]
        db_scores[db_name] = (
            sum(best_scores) / len(best_scores)
            if best_scores
            else 0.0
        )

    return db_scores


def build_description_embeddings(embedder: Embedder, descriptions: list):
    description_texts = [
        description["description"]
        for description in descriptions
    ]
    return embedder.get_sentences_embeddings(description_texts)


def build_table_text(table_doc: dict) -> str:
    table_text = (
        f"Table {table_doc['table']}. "
        f"Columns: {', '.join(table_doc.get('columns', []))}."
    )

    joinable_tables = table_doc.get("joinable_tables", [])
    if joinable_tables:
        table_text += f" Related tables: {', '.join(joinable_tables)}."

    value_parts = []
    for column, values in table_doc.get("top_values", {}).items():
        if not isinstance(values, list):
            values = [values]

        selected_values = [
            str(value)
            for value in values[:MAX_VALUES_PER_COLUMN]
            if value is not None
        ]
        if selected_values:
            value_parts.append(f"{column}: {', '.join(selected_values)}")

    if value_parts:
        table_text += f" Example values: {'; '.join(value_parts)}."

    return table_text


def build_table_embeddings(embedder: Embedder, table_docs: list):
    table_texts = [
        build_table_text(table_doc)
        for table_doc in table_docs
    ]
    return embedder.get_sentences_embeddings(table_texts)


def build_table_indices_by_db(table_docs: list) -> dict:
    table_indices_by_db = defaultdict(list)
    for index, table_doc in enumerate(table_docs):
        table_indices_by_db[table_doc["db"]].append(index)
    return table_indices_by_db


def aggregate_table_embedding_scores(
    table_similarity_matrix,
    table_indices_by_db: dict,
    database_names: list,
):
    aggregated_scores = table_similarity_matrix.new_empty(
        (table_similarity_matrix.shape[0], len(database_names))
    )

    for db_index, db_name in enumerate(database_names):
        table_indices = table_indices_by_db[db_name]

        table_count = min(
            TABLE_EMBEDDING_TABLES_PER_DB,
            len(table_indices),
        )
        best_scores = table_similarity_matrix[:, table_indices].topk(
            table_count,
            dim=1,
        ).values
        aggregated_scores[:, db_index] = best_scores.mean(dim=1)

    return aggregated_scores


def min_max_normalize(scores: dict) -> dict:
    minimum = min(scores.values())
    maximum = max(scores.values())

    if maximum == minimum:
        return {name: 0.0 for name in scores}

    return {
        name: (score - minimum) / (maximum - minimum)
        for name, score in scores.items()
    }


def normalize_bm25_scores(scores: dict) -> dict:
    maximum = max(scores.values())
    if maximum <= 0:
        return {name: 0.0 for name in scores}

    return {
        name: score / maximum
        for name, score in scores.items()
    }


def validate_hyperparameters():
    if MAX_VALUES_PER_COLUMN < 0:
        raise ValueError("MAX_VALUES_PER_COLUMN must be at least 0.")
    if TABLE_EMBEDDING_TABLES_PER_DB < 1:
        raise ValueError("TABLE_EMBEDDING_TABLES_PER_DB must be at least 1.")
    if BM25_TABLES_PER_DB < 1:
        raise ValueError("BM25_TABLES_PER_DB must be at least 1.")
    if BM25_K1 <= 0:
        raise ValueError("BM25_K1 must be greater than 0.")
    if not 0 <= BM25_B <= 1:
        raise ValueError("BM25_B must be between 0 and 1.")

    weights = [DESCRIPTION_WEIGHT, TABLE_WEIGHT, BM25_WEIGHT]
    if any(weight < 0 for weight in weights):
        raise ValueError("Retrieval weights cannot be negative.")
    if not math.isclose(sum(weights), 1.0):
        raise ValueError("Retrieval weights must sum to 1.0.")


def sample_sql(sample: dict) -> str:
    if "SQL" in sample:
        return sample["SQL"]

    return sample["query"]


def sample_question_id(sample: dict, fallback_question_id: int) -> int:
    return sample.get("question_id", fallback_question_id)


def extract_DB(embedder: Embedder, dataset: str):
    validate_hyperparameters()
    dataset_paths = get_dataset_paths(dataset)

    with open(dataset_paths["doc_path"], "r", encoding="utf-8") as f:
        table_docs = json.load(f)

    with open(dataset_paths["descriptions_path"], "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    with open(dataset_paths["questions_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        os.makedirs(dataset_paths["results_folder"], exist_ok=True)
        with open(dataset_paths["result_file_path"], "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        return

    if not table_docs:
        raise ValueError("The table document file is empty.")
    if not descriptions:
        raise ValueError("The database description file is empty.")

    database_names = [
        description["name"]
        for description in descriptions
    ]

    bm25 = build_bm25(table_docs)
    description_embeddings = build_description_embeddings(embedder, descriptions)
    table_embeddings = build_table_embeddings(embedder, table_docs)
    table_indices_by_db = build_table_indices_by_db(table_docs)

    query_embeddings = embedder.get_sentences_embeddings([
        QUERY_INSTRUCTION + sample["question"]
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
    if not torch.isfinite(description_similarity_matrix).all():
        raise ValueError("Description similarities contain non-finite values.")
    if not torch.isfinite(table_similarity_matrix).all():
        raise ValueError("Table similarities contain non-finite values.")

    table_db_score_matrix = aggregate_table_embedding_scores(
        table_similarity_matrix,
        table_indices_by_db,
        database_names,
    )

    result_list = []

    for question_id, sample in enumerate(data):
        description_scores = dict(zip(
            database_names,
            description_similarity_matrix[question_id].tolist(),
        ))
        table_scores = dict(zip(
            database_names,
            table_db_score_matrix[question_id].tolist(),
        ))
        bm25_scores = get_bm25_db_scores(
            sample["question"],
            table_docs,
            bm25,
            database_names,
        )

        normalized_description_scores = min_max_normalize(description_scores)
        normalized_table_scores = min_max_normalize(table_scores)
        normalized_bm25_scores = normalize_bm25_scores(bm25_scores)

        final_scores = {
            db_name: (
                DESCRIPTION_WEIGHT * normalized_description_scores[db_name]
                + TABLE_WEIGHT * normalized_table_scores[db_name]
                + BM25_WEIGHT * normalized_bm25_scores[db_name]
            )
            for db_name in database_names
        }
        db_result = min(
            database_names,
            key=lambda db_name: (-final_scores[db_name], db_name),
        )

        item = {
            "question_id": sample_question_id(sample, question_id),
            "db_id": sample["db_id"],
            "question": sample["question"],
            "SQL": sample_sql(sample),
            "DB_result": db_result,
        }

        result_list.append(item)

    os.makedirs(dataset_paths["results_folder"], exist_ok=True)
    with open(dataset_paths["result_file_path"], "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

    print(f"{GREEN}JSON file saved at {dataset_paths['result_file_path']}{RESET}")


if __name__ == "__main__":
    start = time.perf_counter()
    embedder = Embedder(
        model_name=EMBEDDING_MODEL_NAME,
        device_name=EMBEDDING_DEVICE_NAME,
    )

    print(f"\n{CYAN}Processing BIRDdev...{RESET}")
    extract_DB(embedder, "BIRDdev")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing SPIDERdev1...{RESET}")
    extract_DB(embedder, "SPIDERdev1")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    # Time: 12.84s

    # print(f"\n{CYAN}Processing BIRDdev-ambiguous...{RESET}")
    # extract_DB(embedder, "BIRDdev-ambiguous")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    # print(f"\n{CYAN}Processing SPIDERdev1-ambiguous...{RESET}")
    # extract_DB(embedder, "SPIDERdev1-ambiguous")
    # print(f"{GREEN}Extraction completed!{RESET}\n")
    
    # print(f"\n{CYAN}Processing BIRD_SPIDER_dev...{RESET}")
    # extract_DB(embedder, "BIRD_SPIDER_dev")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    # print(f"\n{CYAN}Processing BIRD_SPIDER_dev-ambiguous...{RESET}")
    # extract_DB(embedder, "BIRD_SPIDER_dev-ambiguous")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    # print(f"\n{CYAN}Processing BIRDtrain...{RESET}")
    # extract_DB(embedder, "BIRDtrain")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    # print(f"\n{CYAN}Processing BIRDtrain-ambiguous...{RESET}")
    # extract_DB(embedder, "BIRDtrain-ambiguous")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    # print(f"\n{CYAN}Processing BEAVER...{RESET}")
    # extract_DB(embedder, "BEAVER")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    end = time.perf_counter()
    print(f"Time: {end - start:.2f}s")
