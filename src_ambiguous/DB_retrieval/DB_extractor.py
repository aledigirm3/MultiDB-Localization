import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedder import Embedder
import paths
from ansi_colors import *
from data_manipulation import normalize_and_stem_text


EMBEDDING_MARGIN_THRESHOLD = 0.02


def get_dataset_paths(dataset: str):
    src_dir = Path(__file__).resolve().parents[1]
    extractor_dir = Path(__file__).resolve().parent

    if dataset == "BIRDdev":
        dataset_dir = (src_dir / paths.DATASETS.BIRDdev.value).resolve()
        doc_filename = "bird_doc.json"
        description_filename = "BIRDdev_DB_descriptions.json"
        result_filename = "BIRDdev_DB_extractor.json"
    elif dataset == "BIRDdev-ambiguous":
        dataset_dir = (src_dir / paths.DATASETS.BIRDdev_ambiguous.value).resolve()
        doc_filename = "bird_doc.json"
        description_filename = "BIRDdev_DB_descriptions_ambiguous.json"
        result_filename = "BIRDdev_DB_extractor_ambiguous.json"
    elif dataset == "SPIDERdev1":
        dataset_dir = (src_dir / paths.DATASETS.SPIDERdev1.value).resolve()
        doc_filename = "spider_doc.json"
        description_filename = "SPIDERdev1_DB_descriptions.json"
        result_filename = "SPIDERdev1_DB_extractor.json"
    elif dataset == "SPIDERdev1-ambiguous":
        dataset_dir = (src_dir / paths.DATASETS.SPIDERdev1_ambiguous.value).resolve()
        doc_filename = "spider_doc.json"
        description_filename = "SPIDERdev1_DB_descriptions_ambiguous.json"
        result_filename = "SPIDERdev1_DB_extractor_ambiguous.json"
    else:
        print(f"{RED} DATASET NOT FOUND, check the name{RESET}")
        sys.exit(1)

    results_folder = (src_dir / paths.RESULTS.DB_RETRIEVAL.value).resolve()

    return {
        "questions_path": dataset_dir / "dev.json",
        "doc_path": dataset_dir / doc_filename,
        "descriptions_path": extractor_dir / description_filename,
        "results_folder": results_folder,
        "result_file_path": results_folder / result_filename,
    }


def build_bm25(table_docs: list):
    tokenized_corpus = [
        table_doc.get("bm25_text", "").split()
        for table_doc in table_docs
    ]
    return BM25Okapi(tokenized_corpus)


def rank_tables(question: str, table_docs: list, bm25: BM25Okapi, top_k: int = 11):
    query_tokens = normalize_and_stem_text(question).split()
    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(table_docs)),
        key=lambda index: (scores[index], table_docs[index].get("table_id", "")),
        reverse=True,
    )

    ranked_tables = []
    for index in ranked_indexes[:top_k]:
        ranked_table = dict(table_docs[index])
        ranked_table["_bm25_score"] = float(scores[index])
        ranked_tables.append(ranked_table)

    return ranked_tables


def get_bm25_top_dbs(ranked_tables: list):
    db_scores = defaultdict(float)
    for table_doc in ranked_tables:
        db_name = table_doc.get("db")
        if not db_name:
            continue

        db_scores[db_name] += table_doc.get("_bm25_score", 0.0)

    if not db_scores:
        return []

    max_score = max(db_scores.values())
    return [
        db for db, score in db_scores.items()
        if score == max_score
    ]


def build_description_embeddings(embedder: Embedder, descriptions: list):
    desc_embeddings = []
    for description in descriptions:
        desc_embedding = embedder.get_sentence_embedding(
            "passage: " + description["description"]
        )
        desc_embeddings.append((description["name"], desc_embedding))

    return desc_embeddings


def rank_databases_with_embeddings(
    question: str,
    desc_embeddings: list,
    embedder: Embedder,
    top_k: int = 3,
):
    query_embedding = embedder.get_sentence_embedding("query: " + question)
    best_score_db = []

    for db_name, desc_embedding in desc_embeddings:
        score = util.cos_sim(query_embedding, desc_embedding).item()
        best_score_db.append((score, db_name))

    return sorted(best_score_db, reverse=True)[:top_k]


def choose_db_result(embedding_ranking: list, bm25_top_dbs: list):
    embedding_top_dbs = [db_name for _, db_name in embedding_ranking]
    intersection = [
        db for db in embedding_top_dbs
        if db in bm25_top_dbs
    ]

    if len(embedding_ranking) > 1:
        margin = embedding_ranking[0][0] - embedding_ranking[1][0]
    else:
        margin = 0

    if intersection and margin <= EMBEDDING_MARGIN_THRESHOLD:
        return intersection[0]

    return embedding_top_dbs[0]


def extract_DB(embedder: Embedder, dataset: str):
    dataset_paths = get_dataset_paths(dataset)

    with open(dataset_paths["doc_path"], "r", encoding="utf-8") as f:
        table_docs = json.load(f)

    with open(dataset_paths["descriptions_path"], "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    with open(dataset_paths["questions_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    bm25 = build_bm25(table_docs)
    desc_embeddings = build_description_embeddings(embedder, descriptions)
    result_list = []

    for question_id, sample in enumerate(data):
        embedding_ranking = rank_databases_with_embeddings(
            sample["question"],
            desc_embeddings,
            embedder,
            top_k=3,
        )
        ranked_tables = rank_tables(sample["question"], table_docs, bm25, top_k=8)
        bm25_top_dbs = get_bm25_top_dbs(ranked_tables)
        db_result = choose_db_result(embedding_ranking, bm25_top_dbs)

        if dataset == "BIRDdev" or dataset == "BIRDdev-ambiguous":
            item = {
                "question_id": sample["question_id"],
                "db_id": sample["db_id"],
                "question": sample["question"],
                "SQL": sample["SQL"],
                "DB_result": db_result,
            }
        else:
            item = {
                "question_id": question_id,
                "db_id": sample["db_id"],
                "question": sample["question"],
                "SQL": sample["query"],
                "DB_result": db_result,
            }

        result_list.append(item)

    os.makedirs(dataset_paths["results_folder"], exist_ok=True)
    with open(dataset_paths["result_file_path"], "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

    print(f"{GREEN}JSON file saved at {dataset_paths['result_file_path']}{RESET}")


if __name__ == "__main__":
    start = time.perf_counter()
    embedder = Embedder(model_name="BAAI/bge-large-en-v1.5", device_name="cuda")

    print(f"\n{CYAN}Processing BIRDdev...{RESET}")
    extract_DB(embedder, "BIRDdev")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing SPIDERdev1...{RESET}")
    extract_DB(embedder, "SPIDERdev1")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing BIRDdev-ambiguous...{RESET}")
    extract_DB(embedder, "BIRDdev-ambiguous")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing SPIDERdev1-ambiguous...{RESET}")
    extract_DB(embedder, "SPIDERdev1-ambiguous")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    end = time.perf_counter()
    print(f"Time: {end - start:.2f}s")
