import os
import sys
import json
from sentence_transformers import util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embedder import Embedder
from llm import query_groq
from ansi_colors import *
from prompts import DB_SYSTEM_PROMPT

# Extracts the 3 most relevant databases using embeddings similarity
def extract_DBs(embedder: Embedder):
	
	with open('./DB_descriptions.json', "r", encoding="utf-8") as f:
		descriptions = json.load(f)

	with open('../../BIRDdev/dev.json', "r", encoding="utf-8") as f:
		data = json.load(f)

	results_folder = '../../results/DB_retrieval'
	os.makedirs(results_folder, exist_ok=True)
	result_file_path = os.path.join(results_folder, "sim_DBs_extractor.json")

	result_list = []

	for sample in data:
		best_score_DB = []
		query_embedding = embedder.get_sentence_embedding(sample['question'])
		
		for desc in descriptions:
			desc_embedding = embedder.get_sentence_embedding(desc['description'])
			score = util.cos_sim(query_embedding, desc_embedding).item()
			
			best_score_DB.append((score, desc['name']))

		sorted_best_score_DB = sorted(best_score_DB, reverse=True)
		best_dbs = [sorted_best_score_DB[0][1], sorted_best_score_DB[1][1], sorted_best_score_DB[2][1]]



		item = {
			"question_id": sample['question_id'],
			"db_id": sample['db_id'],
			"question": sample['question'],
			"result": best_dbs,
			}

		result_list.append(item)

	with open(result_file_path, "w", encoding="utf-8") as f:
		json.dump(result_list, f, ensure_ascii=False, indent=4)
	
	print(f"{GREEN}JSON file saved at {result_file_path}{RESET}")

def DB_descriptions_to_dict():
	"""
	Create a dictionary mapping database names to their descriptions.
	"""
	with open('./DB_descriptions.json', "r", encoding="utf-8") as f:
		descriptions = json.load(f)

	db_dict = {entry["name"]: entry["description"] for entry in descriptions}

	return db_dict

# Extracts the most relevant database using LLM
def extract_DB():
	
	results_folder = '../../results/DB_retrieval'
	os.makedirs(results_folder, exist_ok=True)
	result_file_path = os.path.join(results_folder, "DB_extractor.json")

	with open('../../results/DB_retrieval/sim_DBs_extractor@3.json', "r", encoding="utf-8") as f:
			data = json.load(f)

	result_list = []
	db_dict = DB_descriptions_to_dict()

	for sample in data:
		user_prompt = ''
		user_prompt += sample['question'] + '\n\n'
		dbs = sample['result']
		for db in dbs:
			user_prompt += f"{db}: {db_dict[db]}\n"

		messages = [
			{
				"role": "system", 
				"content": DB_SYSTEM_PROMPT
			},
			{
				"role": "user", 
				"content": user_prompt
			}
		]

		llm_reply = query_groq(messages, model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1, maxTokens=150)

		item = {
			"question_id": sample['question_id'],
			"db_id": sample['db_id'],
			"question": sample['question'],
			"result": llm_reply,
			}

		result_list.append(item)

		# This section writes the entire result_list to the JSON file after each iteration.
		# It ensures that no results are lost in case the LLM reaches the token limit
		# or the process is interrupted, even though it incurs a slightly higher
		# computational cost due to rewriting the file multiple times.
		with open(result_file_path, "w", encoding="utf-8") as f:
			json.dump(result_list, f, ensure_ascii=False, indent=4)

	print(f"{GREEN}JSON file saved at {result_file_path}{RESET}")

if __name__ == '__main__':

	extract_DB()
	#embedder = Embedder(model_name='all-MiniLM-L12-v2', device_name='cuda')
	#extract_DBs(embedder)


