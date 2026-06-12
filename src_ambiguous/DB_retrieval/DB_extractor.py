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
BM25_TABLES_PER_DB = 5


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
    elif dataset == "BIRD_SPIDER_dev":
        dataset_dir = (src_dir / paths.DATASETS.BIRD_SPIDER_dev.value).resolve()
        doc_filename = "bird_doc.json"
        description_filename = "BIRD_SPIDER_dev_descriptions.json"
        result_filename = "BIRD_SPIDER_dev_DB_extractor.json"
    elif dataset == "BIRD_SPIDER_dev-ambiguous":
        dataset_dir = (src_dir / paths.DATASETS.BIRD_SPIDER_dev_ambiguous.value).resolve()
        doc_filename = "bird_doc.json"
        description_filename = "BIRD_SPIDER_dev_descriptions_ambiguous.json"
        result_filename = "BIRD_SPIDER_dev_DB_extractor_ambiguous.json"
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


def get_candidate_bm25_db_scores(
    question: str,
    table_docs: list,
    bm25: BM25Okapi,
    candidate_dbs: list,
):
    query_tokens = normalize_and_stem_text(question).split()
    scores = bm25.get_scores(query_tokens)
    candidate_db_set = set(candidate_dbs)
    db_table_scores = defaultdict(list)

    for index, table_doc in enumerate(table_docs):
        db_name = table_doc.get("db")
        if db_name not in candidate_db_set:
            continue

        score = max(0.0, float(scores[index]))
        db_table_scores[db_name].append(score)

    db_scores = {}
    for db_name in candidate_dbs:
        table_scores = sorted(db_table_scores.get(db_name, []), reverse=True)
        db_scores[db_name] = sum(table_scores[:BM25_TABLES_PER_DB])

    return db_scores


def get_best_bm25_db(embedding_top_dbs: list, bm25_db_scores: dict):
    if not bm25_db_scores:
        return embedding_top_dbs[0]

    max_score = max(bm25_db_scores.values())
    if max_score <= 0:
        return embedding_top_dbs[0]

    best_dbs = {
        db for db, score in bm25_db_scores.items()
        if score == max_score
    }

    for db_name in embedding_top_dbs:
        if db_name in best_dbs:
            return db_name

    return embedding_top_dbs[0]


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


def get_embedding_margin(embedding_ranking: list):
    if len(embedding_ranking) > 1:
        return embedding_ranking[0][0] - embedding_ranking[1][0]

    return 0


def should_rerank_with_bm25(embedding_ranking: list):
    return get_embedding_margin(embedding_ranking) <= EMBEDDING_MARGIN_THRESHOLD


def choose_db_result(embedding_ranking: list, bm25_db_scores: dict):
    embedding_top_dbs = [db_name for _, db_name in embedding_ranking]
    if not embedding_top_dbs:
        return None

    if should_rerank_with_bm25(embedding_ranking):
        return get_best_bm25_db(embedding_top_dbs, bm25_db_scores)

    return embedding_top_dbs[0]

def sample_sql(sample: dict) -> str:
    if "SQL" in sample:
        return sample["SQL"]

    return sample["query"]


def sample_question_id(sample: dict, fallback_question_id: int) -> int:
    return sample.get("question_id", fallback_question_id)


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
        candidate_dbs = [db_name for _, db_name in embedding_ranking]
        bm25_db_scores = {}
        if should_rerank_with_bm25(embedding_ranking):
            bm25_db_scores = get_candidate_bm25_db_scores(
                sample["question"],
                table_docs,
                bm25,
                candidate_dbs,
            )
        db_result = choose_db_result(embedding_ranking, bm25_db_scores)

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

    print(f"\n{CYAN}Processing BIRD_SPIDER_dev...{RESET}")
    extract_DB(embedder, "BIRD_SPIDER_dev")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing BIRD_SPIDER_dev-ambiguous...{RESET}")
    extract_DB(embedder, "BIRD_SPIDER_dev-ambiguous")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    end = time.perf_counter()
    print(f"Time: {end - start:.2f}s")
