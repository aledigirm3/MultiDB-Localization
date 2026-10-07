"""Reproduce the catalog-wide LLM clone-bias analysis on BIRD.

The analysis uses the archived BIRD content-ambiguous predictions. It measures
which clone suffix the LLM selects and, on the 103 samples correctly retrieved
by Hybrid-DB, separates valid LLM errors from invalid ``ERROR`` outputs.
"""

from collections import Counter
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LLM_RESULTS_PATH = (
    ROOT
    / "results_DB_retrieval_baseline_gpt-oss-120"
    / "DB_retrieval"
    / "BIRDdev_DB_extractor_ambiguous.json"
)
HYBRID_RESULTS_PATH = (
    ROOT
    / "results_DB_retrieval_tune_BIRDtrain"
    / "DB_retrieval"
    / "BIRDdev_DB_extractor_ambiguous.json"
)
RESULTS_PATH = ROOT / "analysis" / "results" / "bird_llm_bias_analysis.json"

EXPECTED = {
    "samples": 126,
    "valid_llm_outputs": 120,
    "invalid_llm_outputs": 6,
    "llm_clone_selections": {"1": 78, "2": 19, "3": 23},
    "hybrid_correct_samples": 103,
    "on_hybrid_correct": {
        "llm_correct": 59,
        "llm_valid_wrong": 39,
        "llm_invalid": 5,
        "wrong_clone_selections": {"1": 37, "2": 1, "3": 1},
    },
}


def load_results(path):
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)

    indexed = {row["question_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate question_id in {path}.")
    return indexed


def clone_suffix(db_id):
    match = re.search(r"_([123])$", db_id)
    if not match:
        raise ValueError(f"Database ID has no clone suffix: {db_id}")
    return match.group(1)


def analyze():
    llm = load_results(LLM_RESULTS_PATH)
    hybrid = load_results(HYBRID_RESULTS_PATH)
    if llm.keys() != hybrid.keys():
        raise ValueError("LLM and Hybrid-DB predictions contain different question IDs.")

    clone_selections = Counter()
    invalid_llm_outputs = 0
    hybrid_correct_ids = []

    for question_id, llm_row in llm.items():
        hybrid_row = hybrid[question_id]
        if llm_row["db_id"] != hybrid_row["db_id"]:
            raise ValueError(f"Gold DB mismatch for question {question_id}.")

        if llm_row["DB_result"] == "ERROR":
            invalid_llm_outputs += 1
        else:
            clone_selections[clone_suffix(llm_row["DB_result"])] += 1

        if hybrid_row["DB_result"] == hybrid_row["db_id"]:
            hybrid_correct_ids.append(question_id)

    on_hybrid_correct = {
        "llm_correct": 0,
        "llm_valid_wrong": 0,
        "llm_invalid": 0,
        "wrong_clone_selections": Counter(),
    }
    for question_id in hybrid_correct_ids:
        row = llm[question_id]
        if row["DB_result"] == "ERROR":
            on_hybrid_correct["llm_invalid"] += 1
        elif row["DB_result"] == row["db_id"]:
            on_hybrid_correct["llm_correct"] += 1
        else:
            on_hybrid_correct["llm_valid_wrong"] += 1
            on_hybrid_correct["wrong_clone_selections"][
                clone_suffix(row["DB_result"])
            ] += 1

    suffix_counts = {suffix: clone_selections[suffix] for suffix in ("1", "2", "3")}
    wrong_suffix_counts = {
        suffix: on_hybrid_correct["wrong_clone_selections"][suffix]
        for suffix in ("1", "2", "3")
    }
    return {
        "samples": len(llm),
        "valid_llm_outputs": len(llm) - invalid_llm_outputs,
        "invalid_llm_outputs": invalid_llm_outputs,
        "llm_clone_selections": suffix_counts,
        "hybrid_correct_samples": len(hybrid_correct_ids),
        "on_hybrid_correct": {
            "llm_correct": on_hybrid_correct["llm_correct"],
            "llm_valid_wrong": on_hybrid_correct["llm_valid_wrong"],
            "llm_invalid": on_hybrid_correct["llm_invalid"],
            "wrong_clone_selections": wrong_suffix_counts,
        },
    }


def main():
    result = analyze()
    if result != EXPECTED:
        raise RuntimeError(f"Unexpected BIRD LLM-bias statistics: {result}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    valid = result["valid_llm_outputs"]
    hybrid_correct = result["hybrid_correct_samples"]
    valid_wrong = result["on_hybrid_correct"]["llm_valid_wrong"]
    wrong_suffix_1 = result["on_hybrid_correct"]["wrong_clone_selections"]["1"]
    print(
        f"LLM clone _1 selections: {result['llm_clone_selections']['1']}/{valid} "
        f"({100 * result['llm_clone_selections']['1'] / valid:.2f}%)"
    )
    print(
        f"Valid LLM errors on Hybrid-DB-correct samples: "
        f"{valid_wrong}/{hybrid_correct} ({100 * valid_wrong / hybrid_correct:.2f}%)"
    )
    print(
        f"Those valid errors selecting clone _1: {wrong_suffix_1}/{valid_wrong} "
        f"({100 * wrong_suffix_1 / valid_wrong:.2f}%)"
    )


if __name__ == "__main__":
    main()
