import os
import sys
import json
from sentence_transformers import util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embedder import Embedder
import paths
from ansi_colors import *

# Extracts the most relevant databases using embeddings similarity
def extract_DB(embedder: Embedder, dataset: str):
	
	if dataset == 'BIRDdev':

		with open('./BIRDdev_DB_descriptions.json', "r", encoding="utf-8") as f:
			descriptions = json.load(f)

		with open('../' + paths.DATASETS.BIRDdev.value + 'dev.json', "r", encoding="utf-8") as f:
			data = json.load(f)

		results_folder = '../' + paths.RESULTS.DB_RETRIEVAL.value
		os.makedirs(results_folder, exist_ok=True)
		result_file_path = os.path.join(results_folder, "BIRDdev_DB_extractor.json")

	elif dataset == 'SPIDERdev1': 
		with open('./SPIDERdev1_DB_descriptions.json', "r", encoding="utf-8") as f:
			descriptions = json.load(f)

		with open('../' + paths.DATASETS.SPIDERdev1.value + 'dev.json', "r", encoding="utf-8") as f:
			data = json.load(f)

		results_folder = '../' + paths.RESULTS.DB_RETRIEVAL.value
		os.makedirs(results_folder, exist_ok=True)
		result_file_path = os.path.join(results_folder, "SPIDERdev1_DB_extractor.json")

	else:
		print(f"{RED} DATASET NOT FOUND, check the name{RESET}")
		sys.exit(1)
	

	result_list = []
	
	desc_embeddings = []
	for desc in descriptions:
		desc_embedding = embedder.get_sentence_embedding("passage: " + desc['description'])
		desc_embeddings.append((desc['name'], desc_embedding))

	i = 0
	for sample in data:
		best_score_DB = []
		query_embedding = embedder.get_sentence_embedding("query: " + sample['question'])
		
		for desc in desc_embeddings:

			score = util.cos_sim(query_embedding, desc[1]).item()
			
			best_score_DB.append((score, desc[0]))

		sorted_best_score_DB = sorted(best_score_DB, reverse=True)
		best_db = sorted_best_score_DB[0][1]


		if dataset == 'BIRDdev':
			item = {
				"question_id": sample['question_id'],
				"db_id": sample['db_id'],
				"question": sample['question'],
				"SQL": sample['SQL'],
				"DB_result": best_db,
				}

			result_list.append(item)
		else:
			item = {
				"question_id": i,
				"db_id": sample['db_id'],
				"question": sample['question'],
				"SQL": sample['query'],
				"DB_result": best_db,
				}
			i += 1
			result_list.append(item)
			

	with open(result_file_path, "w", encoding="utf-8") as f:
		json.dump(result_list, f, ensure_ascii=False, indent=4)
	
	print(f"{GREEN}JSON file saved at {result_file_path}{RESET}")



if __name__ == '__main__':

	embedder = Embedder(model_name='BAAI/bge-large-en-v1.5', device_name='cuda')

	dataset = "BIRDdev"
	extract_DB(embedder, dataset)

	dataset = "SPIDERdev1"
	extract_DB(embedder, dataset)



