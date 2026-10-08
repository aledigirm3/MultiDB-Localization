import argparse
import json
from pathlib import Path
import re
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ansi_colors import *


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "DB_retrieval"


def source_family(database_id):
    return re.sub(r"_[123]$", "", database_id)


def DBs_extraction_eval(
    filename,
    results_dir=DEFAULT_RESULTS_DIR,
    family_metrics=False,
):
    file_path = Path(results_dir) / filename

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"{RED}File not found: {file_path}. Skipping evaluation.{RESET}\n")
        return

    correct_samples = 0
    for sample in data:
        if sample['DB_result'] == "ERROR":
            continue
        if sample['db_id'] == sample['DB_result']:
            correct_samples += 1

    accuracy = correct_samples / len(data)
    print(f"\n{GREEN}Accuracy:{RESET} {accuracy}")

    if family_metrics and filename.endswith("_ambiguous.json"):
        family_correct = sum(
            source_family(sample["DB_result"]) == source_family(sample["db_id"])
            for sample in data
        )
        family_accuracy = family_correct / len(data)
        clone_given_family = (
            correct_samples / family_correct if family_correct else 0.0
        )
        print(f"{GREEN}Family accuracy:{RESET} {family_accuracy}")
        print(f"{GREEN}Clone|Family:{RESET} {clone_given_family}")

    print()
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate database-retrieval predictions."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory containing the prediction JSON files "
            f"(default: {DEFAULT_RESULTS_DIR})."
        ),
    )
    parser.add_argument(
        "--family-metrics",
        action="store_true",
        help=(
            "For content-ambiguous files, also report family accuracy and "
            "exact clone accuracy conditioned on the correct family."
        ),
    )
    return parser.parse_args()

        

if __name__ == '__main__':

    args = parse_args()

    filename = 'SPIDERdev1_DB_extractor.json'
    print(f"{CYAN}SPIDER dev 1.0 evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'BIRDdev_DB_extractor.json'
    print(f"{CYAN}BIRD dev evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'SPIDERtrain_DB_extractor.json'
    print(f"{CYAN}SPIDERtrain evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'ARCHER_DB_extractor.json'
    print(f"{CYAN}ARCHER evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'BEAVER_DB_extractor.json'
    print(f"{CYAN}BEAVER evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'SPIDERdev1_DB_extractor_ambiguous.json'
    print(f"{CYAN}SPIDER dev 1.0 AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'BIRDdev_DB_extractor_ambiguous.json'
    print(f"{CYAN}BIRD dev AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'SPIDERtrain_DB_extractor_ambiguous.json'
    print(f"{CYAN}SPIDERtrain AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'ARCHER_DB_extractor_ambiguous.json'
    print(f"{CYAN}ARCHER AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'BEAVER_DB_extractor_ambiguous.json'
    print(f"{CYAN}BEAVER AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    filename = 'SQALE3_DB_extractor_ambiguous.json'
    print(f"{CYAN}SQALE3 AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename, args.results_dir, args.family_metrics)

    # filename = 'BIRDtrain_DB_extractor.json'
    # print(f"{CYAN}BIRD train  evaluation...{RESET}")
    # DBs_extraction_eval(filename)

