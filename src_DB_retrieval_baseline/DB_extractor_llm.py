import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ansi_colors import *
from llm import query_bedrock
import paths


SYSTEM_PROMPT = """You are a database retrieval assistant for a Text-to-SQL pipeline.

Your task is to choose the single database that is most likely needed to answer
the user's natural-language question.

You will receive:
- one natural-language question;
- a JSON catalog of candidate databases;
- for each database, its tables;
- for each table, the SQL table definition and frequent textual values.

Rules:
1. Output exactly one database id from the provided candidate database ids.
2. Output only the database id, with no explanation, no punctuation, no quotes,
   no markdown, and no extra text.
3. Use schema_sql to understand tables and columns.
4. Use top_values as evidence when the question mentions concrete entities,
   categories, names, places, codes, or other values.
5. If multiple databases look plausible, choose the one with the strongest
   evidence from the question, schema_sql, and top_values.
"""

WRITE_MAX_ATTEMPTS = 7
WRITE_RETRY_DELAY_SECONDS = 0.5


def get_dataset_paths(dataset: str) -> Dict[str, Path]:

    if dataset == "BIRDdev":
        dataset_dir = paths.DATASETS.BIRDdev.value
        doc_filename = "doc.json"
        result_filename = "BIRDdev_DB_extractor.json"
    elif dataset == "BIRDdev-ambiguous":
        dataset_dir = paths.DATASETS.BIRDdev_ambiguous.value
        doc_filename = "doc.json"
        result_filename = "BIRDdev_DB_extractor_ambiguous.json"
    elif dataset == "SPIDERdev1":
        dataset_dir = paths.DATASETS.SPIDERdev1.value
        doc_filename = "doc.json"
        result_filename = "SPIDERdev1_DB_extractor.json"
    elif dataset == "SPIDERdev1-ambiguous":
        dataset_dir = paths.DATASETS.SPIDERdev1_ambiguous.value
        doc_filename = "doc.json"
        result_filename = "SPIDERdev1_DB_extractor_ambiguous.json"
    elif dataset == "BEAVER":
        dataset_dir = paths.DATASETS.BEAVER.value
        doc_filename = "doc.json"
        result_filename = "BEAVER_DB_extractor.json"
    elif dataset == "ARCHER":
        dataset_dir = paths.DATASETS.ARCHER.value
        doc_filename = "doc.json"
        result_filename = "ARCHER_DB_extractor.json"
    elif dataset == "ARCHER-ambiguous":
        dataset_dir = paths.DATASETS.ARCHER_ambiguous.value
        doc_filename = "doc.json"
        result_filename = "ARCHER_DB_extractor_ambiguous.json"
    elif dataset == "SPIDERtrain":
        dataset_dir = paths.DATASETS.SPIDERtrain.value
        doc_filename = "doc.json"
        result_filename = "SPIDERtrain_DB_extractor.json"
    elif dataset == "SPIDERtrain-ambiguous":
        dataset_dir = paths.DATASETS.SPIDERtrain_ambiguous.value
        doc_filename = "doc.json"
        result_filename = "SPIDERtrain_DB_extractor_ambiguous.json"

    else:
        print(f"{RED}DATASET NOT FOUND, check the name{RESET}")
        sys.exit(1)

    results_folder = paths.RESULTS.DB_RETRIEVAL.value

    return {
        "questions_path": dataset_dir + "/dev.json",
        "doc_path": dataset_dir + '/' + doc_filename,
        "results_folder": results_folder,
        "result_file_path": os.path.join(results_folder, result_filename),
    }


def is_number_like(value: str) -> bool:
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return False

    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def clean_top_values(top_values: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    cleaned_top_values = {}

    for column_name, values in top_values.items():
        cleaned_values = []
        for value in values:
            if not isinstance(value, str):
                continue

            value = value.strip()
            if not value or is_number_like(value):
                continue

            cleaned_values.append(value)

        if cleaned_values:
            cleaned_top_values[column_name] = cleaned_values

    return cleaned_top_values


def build_database_catalog(table_docs: List[dict]) -> Dict[str, Dict[str, dict]]:
    catalog = {}

    for table_doc in table_docs:
        db_id = table_doc["db"]
        table_name = table_doc["table"]

        catalog.setdefault(db_id, {})[table_name] = {
            "schema_sql": table_doc.get("schema_sql", ""),
            "top_values": clean_top_values(table_doc.get("top_values", {})),
        }

    return catalog


def build_user_prompt(question: str, catalog: Dict[str, Dict[str, dict]]) -> str:
    candidate_db_ids = list(catalog.keys())
    catalog_json = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))

    return f"""[QUESTION]
{question}

[CANDIDATE_DATABASE_IDS]
{json.dumps(candidate_db_ids, ensure_ascii=False)}

[DATABASE_CATALOG_JSON]
{catalog_json}

Return exactly one database id from [CANDIDATE_DATABASE_IDS]."""


def clean_llm_db_result(response: str, valid_db_ids: List[str]) -> str:
    if response is None:
        return "ERROR"

    cleaned = response.strip()
    cleaned = cleaned.strip("`").strip()
    cleaned = cleaned.strip("\"'").strip()

    valid_db_id_set = set(valid_db_ids)
    if cleaned in valid_db_id_set:
        return cleaned

    matches = []
    for db_id in valid_db_ids:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(db_id)}(?![A-Za-z0-9_])"
        if re.search(pattern, response):
            matches.append(db_id)

    if len(matches) == 1:
        return matches[0]

    return "ERROR"


def get_llm_response(question: str, catalog: Dict[str, Dict[str, dict]]) -> str:
    return query_bedrock(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(question, catalog),
            },
        ],
        model_id="openai.gpt-oss-120b-1:0",
        temperature=0.0,
    )


def sample_sql(sample: dict) -> str:
    if "SQL" in sample:
        return sample["SQL"]

    return sample["query"]


def sample_question_id(sample: dict, fallback_question_id: int) -> int:
    return sample.get("question_id", fallback_question_id)


def write_results(path: str, data: List[dict]):
    payload = json.dumps(data, ensure_ascii=False, indent=4)

    for attempt in range(1, WRITE_MAX_ATTEMPTS + 1):
        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(payload)
            return
        except OSError as error:
            if attempt == WRITE_MAX_ATTEMPTS:
                raise
            print(
                f"[write retry] attempt {attempt}/{WRITE_MAX_ATTEMPTS} failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            time.sleep(WRITE_RETRY_DELAY_SECONDS)


def extract_DB(dataset: str, start_index: int = 0):
    dataset_paths = get_dataset_paths(dataset)

    with open(dataset_paths["doc_path"], "r", encoding="utf-8") as file:
        table_docs = json.load(file)

    with open(dataset_paths["questions_path"], "r", encoding="utf-8") as file:
        data = json.load(file)

    if not 0 <= start_index <= len(data):
        raise ValueError(f"Invalid start_index {start_index} for {len(data)} samples.")

    catalog = build_database_catalog(table_docs)
    # print(json.dumps(catalog, ensure_ascii=False, indent=2))
    # return
    valid_db_ids = list(catalog.keys())
    result_list = []

    os.makedirs(dataset_paths["results_folder"], exist_ok=True)

    for question_id, sample in enumerate(data[start_index:], start=start_index):
        question = sample["question"]
        llm_response = get_llm_response(question, catalog)
        db_result = clean_llm_db_result(llm_response, valid_db_ids)

        item = {
            "question_id": sample_question_id(sample, question_id),
            "db_id": sample["db_id"],
            "question": question,
            "SQL": sample_sql(sample),
            "DB_result": db_result,
        }
        result_list.append(item)

        write_results(dataset_paths["result_file_path"], result_list)

    print(f"{GREEN}JSON file saved at {dataset_paths['result_file_path']}{RESET}")


if __name__ == "__main__":
    start = time.perf_counter()

    # print(f"\n{CYAN}Processing BIRDdev...{RESET}")
    # extract_DB("BIRDdev")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    # print(f"\n{CYAN}Processing SPIDERdev1...{RESET}")
    # extract_DB("SPIDERdev1")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing BIRDdev-ambiguous...{RESET}")
    extract_DB("BIRDdev-ambiguous")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing SPIDERdev1-ambiguous...{RESET}")
    extract_DB("SPIDERdev1-ambiguous")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing ARCHER-ambiguous...{RESET}")
    extract_DB("ARCHER-ambiguous")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    # print(f"\n{CYAN}Processing BEAVER...{RESET}")
    # extract_DB("BEAVER")
    # print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing SPIDERtrain-ambiguous...{RESET}")
    extract_DB("SPIDERtrain-ambiguous")
    print(f"{GREEN}Extraction completed!{RESET}\n")

    end = time.perf_counter()
    print(f"Time: {end - start:.2f}s")





#     Processing BIRDdev...
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=37914 output=1 total=37915 max_output=1 stopReason=max_tokens (max_output=1 was used only for see input token)
# JSON file saved at ../results/DB_retrieval//BIRDdev_DB_extractor.json
# Extraction completed!


# Processing SPIDERdev1...
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=9825 output=1024 total=10849 max_output=1024 stopReason=max_tokens
# JSON file saved at ../results/DB_retrieval//SPIDERdev1_DB_extractor.json
# Extraction completed!


# Processing BIRDdev-ambiguous...
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=98203 output=1024 total=99227 max_output=1024 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=98187 output=1024 total=99211 max_output=1024 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=98205 output=1024 total=99229 max_output=1024 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=98187 output=1024 total=99211 max_output=1024 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=98201 output=1024 total=99225 max_output=1024 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=98193 output=1024 total=99217 max_output=1024 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=98193 output=1024 total=99217 max_output=1024 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=98189 output=1024 total=99213 max_output=1024 stopReason=max_tokens
# JSON file saved at ../results/DB_retrieval//BIRDdev_DB_extractor_ambiguous.json
# Extraction completed!


# Processing SPIDERdev1-ambiguous...
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=8919 output=1 total=8920 max_output=1 stopReason=max_tokens (max_output=1 was used only for see input token)
# JSON file saved at ../results/DB_retrieval//SPIDERdev1_DB_extractor_ambiguous.json
# Extraction completed!

# Processing BIRD_SPIDER_dev...
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=47446 output=1 total=47447 max_output=1 stopReason=max_tokens (max_output=1 was used only for see input token)
# JSON file saved at ../results/DB_retrieval//BIRD_SPIDER_dev_DB_extractor.json
# Extraction completed!

# Processing BIRD_SPIDER_dev-ambiguous...
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=106804 output=2048 total=108852 max_output=2048 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=106809 output=2048 total=108857 max_output=2048 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=106810 output=2048 total=108858 max_output=2048 stopReason=max_tokens
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=106801 output=2048 total=108849 max_output=2048 stopReason=max_tokens
# JSON file saved at ../results/DB_retrieval//BIRD_SPIDER_dev_DB_extractor_ambiguous.json
# Extraction completed!

# Time: 9053.09s + 11573.60s + 1746.31s

# Processing BEAVER...
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=122017 output=2048 total=124065 max_output=2048 stopReason=max_tokens
# JSON file saved at ../results/DB_retrieval//BEAVER_DB_extractor.json
# Extraction completed!
#(w if cleaned_values:
            #cleaned_top_values[column_name] = cleaned_values[:9])


# Processing SPIDERtrain-ambiguous...
# [Bedrock token saturation] model=openai.gpt-oss-120b-1:0 input=112963 output=1 total=112964 max_output=1 stopReason=max_tokens

# Time: 23431.30s

# BIRD dev evaluation...

# Accuracy: 0.9973924380704041

# SPIDER dev 1.0 evaluation...

# Accuracy: 0.9903288201160542

# BIRD dev AMBIGUOUS evaluation...

# Accuracy: 0.5793650793650794

# SPIDER dev 1.0 AMBIGUOUS evaluation...

# Accuracy: 1.0

# BIRD SPIDER dev evaluation...

# Accuracy: 0.9875389408099688

# BIRD SPIDER dev AMBIGUOUS evaluation...

# Accuracy: 0.543750
