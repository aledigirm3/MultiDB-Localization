import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ansi_colors import *
import paths

def DBs_extraction_eval(filename):
    with open(paths.RESULTS.DB_RETRIEVAL.value + filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_samples = 0
    err = 0
    for sample in data:
        if sample['DB_result'] == "ERROR":
            err += 1
            continue
        if sample['db_id'] == sample['DB_result']:
            correct_samples += 1

    print(f"\n{GREEN}Accuracy:{RESET} {correct_samples/len(data)}\n")

        

if __name__ == '__main__':

    # filename = 'BIRDdev_DB_extractor.json'
    # print(f"{CYAN}BIRD dev evaluation...{RESET}")
    # DBs_extraction_eval(filename)

    # filename = 'SPIDERdev1_DB_extractor.json'
    # print(f"{CYAN}SPIDER dev 1.0 evaluation...{RESET}")
    # DBs_extraction_eval(filename)

    # filename = 'BIRDdev_DB_extractor_ambiguous.json'
    # print(f"{CYAN}BIRD dev AMBIGUOUS evaluation...{RESET}")
    # DBs_extraction_eval(filename)

    # filename = 'SPIDERdev1_DB_extractor_ambiguous.json'
    # print(f"{CYAN}SPIDER dev 1.0 AMBIGUOUS evaluation...{RESET}")
    # DBs_extraction_eval(filename)

    # filename = 'BIRD_SPIDER_dev_DB_extractor.json'
    # print(f"{CYAN}BIRD SPIDER dev evaluation...{RESET}")
    # DBs_extraction_eval(filename)

    # filename = 'BIRD_SPIDER_dev_DB_extractor_ambiguous.json'
    # print(f"{CYAN}BIRD SPIDER dev AMBIGUOUS evaluation...{RESET}")
    # DBs_extraction_eval(filename)

    filename = 'BEAVER_DB_extractor.json'
    print(f"{CYAN}BEAVER evaluation...{RESET}")
    DBs_extraction_eval(filename)