import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ansi_colors import *

with open('../../results/sim_DB_retrieval.json', "r", encoding="utf-8") as f:
        data = json.load(f)

correect_samples = 0
for sample in data:
    if sample['db_id'] == sample['result']:
            correect_samples += 1
            

print(f"\n{GREEN}Accuracy:{RESET} {correect_samples/len(data)}\n")