import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ansi_colors import *

def DBs_extraction_eval():
    with open('../../results/DB_retrieval/sim_DBs_extractor@3.json', "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_samples = 0
    for sample in data:
        if sample['db_id'] in sample['result']:
            correct_samples += 1
        else:
            print(sample['db_id'])
                

    print(f"\n{GREEN}Accuracy:{RESET} {correct_samples/len(data)}\n")

def DB_extraction_eval():
    with open('../../results/DB_retrieval/DB_extractor@3.json', "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_samples = 0
    for sample in data:
        if sample['db_id'] == sample['result']:
            correct_samples += 1
        else:
            print(sample['db_id'])
    
    print(f"\n{GREEN}Accuracy:{RESET} {correct_samples/len(data)}\n")
        

if __name__ == '__main__':
    DBs_extraction_eval()
    #DB_extraction_eval()