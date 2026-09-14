import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ansi_colors import *
import paths

def DBs_extraction_eval(filename):
    file_path = '../' + paths.RESULTS.DB_RETRIEVAL.value + filename

    try:
        with open(file_path, "r", encoding="utf-8") as f:
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

    print(f"\n{GREEN}Accuracy:{RESET} {correct_samples/(len(data))}\n")
    return (correct_samples/(len(data)))

        

if __name__ == '__main__':

    filename = 'BIRDdev_DB_extractor.json'
    print(f"{CYAN}BIRD dev evaluation...{RESET}")
    DBs_extraction_eval(filename)

    filename = 'SPIDERdev1_DB_extractor.json'
    print(f"{CYAN}SPIDER dev 1.0 evaluation...{RESET}")
    DBs_extraction_eval(filename)

    filename = 'BIRDdev_DB_extractor_ambiguous.json'
    print(f"{CYAN}BIRD dev AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename)

    filename = 'SPIDERdev1_DB_extractor_ambiguous.json'
    print(f"{CYAN}SPIDER dev 1.0 AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename)

    filename = 'BIRDtrain_DB_extractor.json'
    print(f"{CYAN}BIRD train  evaluation...{RESET}")
    DBs_extraction_eval(filename)

    filename = 'BIRDtrain_DB_extractor_ambiguous.json'
    print(f"{CYAN}BIRD train AMBIGUOUS evaluation...{RESET}")
    DBs_extraction_eval(filename)

    filename = 'BEAVER_DB_extractor.json'
    print(f"{CYAN}BEAVER evaluation...{RESET}")
    DBs_extraction_eval(filename)

    filename = 'ARCHER_DB_extractor.json'
    print(f"{CYAN}ARCHER evaluation...{RESET}")
    DBs_extraction_eval(filename)

    filename = 'SPIDERtrain_DB_extractor.json'
    print(f"{CYAN}SPIDERtrain evaluation...{RESET}")
    DBs_extraction_eval(filename)
