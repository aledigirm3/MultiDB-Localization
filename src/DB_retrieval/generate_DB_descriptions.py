import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import paths
from data_manipulation import get_DB_table_att
from llm import query_bedrock
from ansi_colors import *


PROMPT_PATH = "./DB_description_prompt.txt"


def _parse_llm_response(response: str, database_name: str) -> dict:
    response = response.strip()
    if response.startswith("```") and response.endswith("```"):
        response = response[3:-3].strip()
        if response.startswith("json"):
            response = response[4:].strip()

    result = json.loads(response)
    description = result.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"The LLM returned an invalid description for database '{database_name}'."
        )

    return {
        "name": database_name,
        "description": description.strip(),
    }


def generate_DB_descriptions(file_path: str, output_file_name: str) -> None:
    """Generates one LLM description for each database in a schema JSON file."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
        system_prompt = file.read()

    #i = 0
    descriptions = []
    for database_schema in get_DB_table_att(file_path):
        response = query_bedrock(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(database_schema, indent=2),
                },
            ]
        )
        descriptions.append(
            _parse_llm_response(response, database_schema["database_name"])
        )
        #i+=1
        #print(i)


    with open(output_file_name, "w", encoding="utf-8") as file:
        json.dump(descriptions, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":


    file_path = '../' + paths.DATASETS.BIRDtrain.value + "dev_tables.json"
    output_file_name = './BIRDtrain_DB_descriptions.json'
    print(f"{GREEN}Start generation DBs descriptions...{RESET} ({file_path})")
    generate_DB_descriptions(file_path, output_file_name)
    print(f"{GREEN}DBs descriptions generated!{RESET} ({file_path})")
