import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data_manipulation import create_db_schema_dictionary
from llm import query_bedrock
import paths
from ansi_colors import *
from prompts import REDUCTION_ORIENTED, SR_ORIENTED
import time


def get_llm_response(query: str, schema: str, prompt_type: str) -> str:
    """
    Function to extract relevant attributes from a natural language query and a given schema subset.
    
    Args:
        query (str): Natural language query.
        schema (str): The schema of pre-selected relevant tables.
    Returns:
        str: Relevant attributes (comma separated).
    """

    if prompt_type == "r":
        system_prompt = REDUCTION_ORIENTED
    elif prompt_type == "sr":
        system_prompt = SR_ORIENTED
    else:
        print(f"{RED}INVALID PROMPT TYPE (use 'r' or 'sr'){RESET}")
        sys.exit(1)
    
    content = f"""Now, receive the actual `[QUERY]` and `[RELEVANT TABLES SCHEMA]` and produce the single-line answer only.

[QUERY]:
{query}

[RELEVANT TABLES SCHEMA]:
{schema}"""
    
    return query_bedrock(messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ])



def extract_attributes(dataset, prompt_type):

    if dataset == 'BIRDdev':
        with open('../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'BIRDdev_TAB_extractor.json', "r", encoding="utf-8") as f:
            data = json.load(f)

        results_folder = '../../' + paths.RESULTS.ATT_RETRIEVAL.value
        os.makedirs(results_folder, exist_ok=True)
        result_file_path = os.path.join(results_folder, "BIRDdev_ATT_extractor.json")

        db_schema_dictionary = create_db_schema_dictionary('../../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json')

    elif dataset == 'SPIDERdev1':
        with open('../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'SPIDERdev1_TAB_extractor.json', "r", encoding="utf-8") as f:
            data = json.load(f)

        results_folder = '../../' + paths.RESULTS.ATT_RETRIEVAL.value
        os.makedirs(results_folder, exist_ok=True)
        result_file_path = os.path.join(results_folder, "SPIDERdev1_ATT_extractor.json")

        db_schema_dictionary = create_db_schema_dictionary('../../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json')

    elif dataset == 'ARCHER':
        with open('../../' + paths.RESULTS.TAB_RETRIEVAL.value + 'ARCHER_TAB_extractor.json', "r", encoding="utf-8") as f:
            data = json.load(f)

        results_folder = '../../' + paths.RESULTS.ATT_RETRIEVAL.value
        os.makedirs(results_folder, exist_ok=True)
        result_file_path = os.path.join(results_folder, "ARCHER_ATT_extractor.json")

        db_schema_dictionary = create_db_schema_dictionary('../../' + paths.DATASETS.ARCHER.value + 'dev_tables.json')
        
    else:
        print(f"{RED}INVALID DATASET!{RESET}")
        sys.exit(1)

    result_list = []

    for sample in data:

        query = sample['question']
        db = sample['DB_result']

        tables = sample['TAB_result']
        if 'NONE' in tables:
            item = {
				"question_id": sample['question_id'],
				"db_id": sample['db_id'],
				"question": sample['question'],
				"SQL": sample['SQL'],
				"DB_result": sample['DB_result'],
                "TAB_result": sample['TAB_result'],
                "ATT_result": sample['TAB_result']
				}
            result_list.append(item)
            with open(result_file_path, "w", encoding="utf-8") as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)
            continue

        schema_text = ""
        for table in tables:
            schema_text += "TABLE: " + table + "\n"
            columns = db_schema_dictionary[db][table]
            schema_text += 'COLUMNS:\n'
            for col in columns:
                schema_text += f"- {col}\n"
        
        llm_response = get_llm_response(query, schema_text, prompt_type)

        # llm_response to list
        llm_response_list = [w.strip() for w in llm_response.split(',')]

        # list_to_lower = {w.lower() for w in llm_response_list}


        item = {
				"question_id": sample['question_id'],
				"db_id": sample['db_id'],
				"question": sample['question'],
				"SQL": sample['SQL'],
				"DB_result": sample['DB_result'],
                "TAB_result": sample['TAB_result'],
                "ATT_result": list(set(llm_response_list))
				}
        result_list.append(item)

        # Overwrite the file at every iteration.
        # This way, even if the process fails or hits token limits, 
        # we always have a checkpoint with partial results saved.
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':

    prompt_type = ""
    
    if len(sys.argv) > 1:
        prompt_type = sys.argv[1]
    else:
        print(f"{RED}Please, select the prompt!{RESET}")
        sys.exit(1)
    
    start = time.perf_counter()

    print(f"\n{CYAN}Processing BIRDdev...{RESET}")
    dataset = 'BIRDdev'
    extract_attributes(dataset, prompt_type)
    print(f"{GREEN}Extraction completed!{RESET}\n")

    end = time.perf_counter()
    print(f"BIRDdev time: {end - start:.2f}s")


    start = time.perf_counter()

    print(f"\n{CYAN}Processing SPIDERdev1.0...{RESET}")
    dataset = 'SPIDERdev1'
    extract_attributes(dataset, prompt_type)
    print(f"{GREEN}Extraction completed!{RESET}\n")

    end = time.perf_counter()
    print(f"SPIDERdev time: {end - start:.2f}s")


    start = time.perf_counter()

    print(f"\n{CYAN}Processing ARCHER...{RESET}")
    dataset = 'ARCHER'
    extract_attributes(dataset, prompt_type)
    print(f"{GREEN}Extraction completed!{RESET}\n")

    end = time.perf_counter()
    print(f"ARCHER time: {end - start:.2f}s")

# Processing BIRDdev...
# Extraction completed!
# BIRDdev time: 8166.66s

# Processing SPIDERdev1.0...
# Extraction completed!
# SPIDERdev time: 4152.36s

# Processing ARCHER...
# Extraction completed!
# ARCHER time: 2904.24s
