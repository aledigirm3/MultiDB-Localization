import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data_manipulation import create_db_schema_dictionary
from llm import query_groq
import paths
from ansi_colors import *


def get_llm_response(query: str, schema: str) -> str:
    """
    Function to extract relevant attributes from a natural language query and a given schema subset.
    
    Args:
        query (str): Natural language query.
        schema (str): The schema of pre-selected relevant tables.
    Returns:
        str: Relevant attributes (comma separated).
    """

    system_prompt = """You are a specialized attribute-selector assistant. Your only job is: given a natural language query and the schema of pre-selected relevant database tables, return the comma-separated list of fully-qualified column names (table.column) that are necessary to build the SQL query. Nothing else.
    
- Always output attributes in the exact format: table_name.column_name
- **Use EXACT names from the provided schema**. Do not rename, modify, singularize, pluralize, or shorten any names.
- Always include primary/foreign keys required to connect tables if multiple tables are involved.
- **If a column's name, or a VARIATION (synonym, singular/plural), appears in the query, you MUST include that column from EVERY table where it exists**, but only if the table and column actually exist in the provided schema.
- If you include a column, you MUST also include any other column whose name shares the same key terms or structural components (such as repeated words, numeric markers, or bracketed segments), even if the wording is not identical.
- **Be permissive**: if uncertain whether an attribute might be needed, INCLUDE IT rather than risk excluding it.
- **If a table is included in the provided schema, assume its attributes may be needed unless clearly irrelevant**. Exclude ONLY columns that are CLEARLY IRRELEVANT to the question.

### OUTPUT FORMAT: 
1. Single line, only values: table.column,table.column,...
2. No spaces, no explanations, no comments. Any extra output will be treated as an error.
3. If ABSOLUTELY NOTHING from the schema can answer the query, return exactly: NONE

### INPUT FORMAT:
[QUERY]: 
<natural language question> 

[RELEVANT TABLES SCHEMA]: 
TABLE: <table_name_1>
COLUMNS:
- <column_1>
- <column_2>
TABLE: <table_name_2>
COLUMNS:
- <column_3>
...

EXAMPLES (generic):
Example 1 (Selection and Filtering)
[QUERY]:
Show the names and emails of customers who live in Rome.
    
[RELEVANT TABLES SCHEMA]:
TABLE: customers
COLUMNS:
- customer_id
- first name
- last_name
- email
- city
- registration_date

EXPECTED OUTPUT:
customers.first name,customers.last_name,customers.email,customers.city

---

Example 2 (Join and Aggregation)
[QUERY]:
What is the total number of orders for each customer? Show the customer's name and the count.

[RELEVANT TABLES SCHEMA]:
TABLE: customers
COLUMNS:
- customer_id
- name
- email
TABLE: orders
COLUMNS:
- order_id
- customer id
- order_date
- amount

EXPECTED OUTPUT:
customers.customer_id,customers.name,orders.customer id,orders.order_id

---

Example 3 (Filtering by Attribute)
[QUERY]:
List the products released after 2022 and their price.

[RELEVANT TABLES SCHEMA]:
TABLE: products
COLUMNS:
- product_id 
- product_name
- price
- release date
- supplier_id

EXPECTED OUTPUT:
products.product name,products.price,products.release date

---

Example 4 (Irrelevant Query)
[QUERY]:
What is the speed of light?

[RELEVANT TABLES SCHEMA]:
TABLE: customers
COLUMNS:
- customer_id
- name
- email
TABLE: orders
COLUMNS:
- order id
- customer id
- order date

EXPECTED OUTPUT:
NONE
"""
    
    content = f"""Now, receive the actual `[QUERY]` and `[RELEVANT TABLES SCHEMA]` and produce the single-line answer only.

[QUERY]:
{query}

[RELEVANT TABLES SCHEMA]:
{schema}"""
    
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



def extract_attributes(dataset):

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
            continue

        schema_text = ""
        for table in tables:
            schema_text += "TABLE: " + table + "\n"
            columns = db_schema_dictionary[db][table]
            schema_text += 'COLUMNS:\n'
            for col in columns:
                schema_text += f"- {col}\n"
        
        llm_response = get_llm_response(query, schema_text)

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
                "ATT_result": llm_response_list
				}
        result_list.append(item)

        # Overwrite the file at every iteration.
        # This way, even if the process fails or hits token limits, 
        # we always have a checkpoint with partial results saved.
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':

    print(f"\n{CYAN}Processing BIRDdev...{RESET}")
    dataset = 'BIRDdev'
    extract_attributes(dataset)
    print(f"{GREEN}Extraction completed!{RESET}\n")

    print(f"\n{CYAN}Processing SPIDERdev1.0...{RESET}")
    dataset = 'SPIDERdev1'
    extract_attributes(dataset)
    print(f"{GREEN}Extraction completed!{RESET}\n")