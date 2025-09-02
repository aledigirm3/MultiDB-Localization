import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ansi_colors import *
import paths

def DBs_extraction_eval(filename):
    with open('../' + paths.RESULTS.DB_RETRIEVAL.value + filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_samples = 0
    for sample in data:
        if sample['db_id'] in sample['result']:
            correct_samples += 1
        else:
            print(sample['question_id'])
                

    print(f"\n{GREEN}Accuracy:{RESET} {correct_samples/len(data)}\n")

def DB_extraction_eval():
    with open('../../results/DB_retrieval/DB_extractor@3.json', "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_samples = 0
    for sample in data:
        if sample['db_id'] == sample['result']:
            correct_samples += 1
        else:
            print(sample['question_id'])
    
    print(f"\n{GREEN}Accuracy:{RESET} {correct_samples/len(data)}\n")
        

if __name__ == '__main__':

    filename = 'sim_DBs_extractor.json'

    DBs_extraction_eval(filename)
    #DB_extraction_eval()