import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_manipulation import create_db_schema_dictionary
from llm import query_groq
import paths
from ansi_colors import *
from prompts import REDUCTION_ORIENTED, SR_ORIENTED


def get_llm_response(query: str, attributes: str, prompt_type: str) -> str:
    """
    Function to extract relevant attributes from a natural language query and a given available attributes.
    
    Args:
        query (str): Natural language query.
        attributes (str): All possible tables and columns with this format: "table name.column name".
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

    
    content = f"""Now, receive the actual `[QUERY]` and `[AVAILABLE ATTRIBUTES]` and produce the single-line answer only.

[QUERY]:
{query}

[AVAILABLE ATTRIBUTES]:
{attributes}"""
    
    # print(f"{CYAN}{system_prompt}{RESET}")
    # print(f"{GREEN}{content}{RESET}")
    # sys.exit(0)
    
    return query_groq(messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ])



def extract_table_and_attributes(dataset, prompt_type):

    if dataset == 'BIRDdev':
        with open('../' + paths.RESULTS.DB_RETRIEVAL.value + 'BIRDdev_DB_extractor.json', "r", encoding="utf-8") as f:
            data = json.load(f)

        results_folder = '../' + paths.RESULTS.SL_RETRIEVAL.value
        os.makedirs(results_folder, exist_ok=True)
        result_file_path = os.path.join(results_folder, "BIRDdev_SL_extractor.json")

        db_schema_dictionary = create_db_schema_dictionary('../' + paths.DATASETS.BIRDdev.value + 'dev_tables.json')

    elif dataset == 'SPIDERdev1':
        with open('../' + paths.RESULTS.DB_RETRIEVAL.value + 'SPIDERdev1_DB_extractor.json', "r", encoding="utf-8") as f:
            data = json.load(f)

        results_folder = '../' + paths.RESULTS.SL_RETRIEVAL.value
        os.makedirs(results_folder, exist_ok=True)
        result_file_path = os.path.join(results_folder, "SPIDERdev1_SL_extractor.json")

        db_schema_dictionary = create_db_schema_dictionary('../' + paths.DATASETS.SPIDERdev1.value + 'dev_tables.json')
        
    else:
        print(f"{RED}INVALID DATASET!{RESET}")
        sys.exit(1)

    result_list = []

    for sample in data:
        query = sample['question']
        db = sample['DB_result']

        tables = db_schema_dictionary[db]
        lines = []
        for table in sorted(tables.keys()):
            for col in tables[table]:
                lines.append(f"{table}.{col}")
        attributes = "\n".join(lines)
        
        llm_response = get_llm_response(query, attributes, prompt_type)

        # llm_response to list
        llm_response_list = [w.strip() for w in llm_response.split(',')]

        # list_to_lower = {w.lower() for w in llm_response_list} # Done in the evaluation


        item = {
				"question_id": sample['question_id'],
				"db_id": sample['db_id'],
				"question": sample['question'],
				"SQL": sample['SQL'],
				"DB_result": sample['DB_result'],
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

    print(f"\n{CYAN}Processing BIRDdev...{RESET}")
    dataset = 'BIRDdev'
    extract_table_and_attributes(dataset, prompt_type)
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing SPIDERdev1.0...{RESET}")
    dataset = 'SPIDERdev1'
    extract_table_and_attributes(dataset, prompt_type)
    print(f"{GREEN}Extraction completed!{RESET}\n")