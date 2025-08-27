import os
import sys
import json
from embedder import Embedder
from sentence_transformers import util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ansi_colors import *

embedder = Embedder(model_name='all-MiniLM-L6-v2', device_name='cuda')

with open('./DB_descriptions.json', "r", encoding="utf-8") as f:
        descriptions = json.load(f)

with open('../../BIRDdev/dev.json', "r", encoding="utf-8") as f:
        data = json.load(f)

results_folder = '../../results'
os.makedirs(results_folder, exist_ok=True)
result_file_path = os.path.join(results_folder, "sim_DB_retrieval.json")

result_list = []

for sample in data:
    best_db = ''
    best_score = 0
    query_embedding = embedder.get_sentence_embedding(sample['question'])

    for desc in descriptions:
            desc_embedding = embedder.get_sentence_embedding(desc['description'])
            score = util.cos_sim(query_embedding, desc_embedding).item()

            if score > best_score:
                    best_score = score
                    best_db = desc['name']
    
    item = {
            "question_id": sample['question_id'],
            "db_id": sample['db_id'],
            "question": sample['question'],
            "result": best_db,
            "score": best_score
    }

    result_list.append(item)
    
with open(result_file_path, "w", encoding="utf-8") as f:
    json.dump(result_list, f, ensure_ascii=False, indent=4)

print(f"{GREEN}JSON file saved at {result_file_path}{RESET}")