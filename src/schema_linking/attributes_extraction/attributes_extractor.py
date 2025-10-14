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

    system_prompt = """You are a specialized attribute-selector assistant. Your only job is: given a natural language query and the schema of pre-selected relevant database tables, return EXACTLY and ONLY the comma-separated list of column names that are necessary to build the SQL query. Nothing else.

- Analyze the query to identify the specific information requested. Think about which columns will be needed for the `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, and `JOIN` clauses.
- To connect the provided tables, always include the primary and/or foreign keys needed for the join (e.g., if the query needs data from `orders` and `customers`, include `customers.customer_id` and `orders.customer_id`).
- **If a column's name, or a close variation of it (synonym, singular/plural form), is explicitly mentioned in the query, you MUST include that column.**

OUTPUT RULES:
1. Output must be a single line containing only column names separated by commas, with NO SPACES (example: `name,email,order_date`).
2. Use only the column names exactly as they appear in the provided schema. Do not invent, abbreviate, or change names.
3. **If you are uncertain whether a column is needed, INCLUDE IT.** IT IS BETTER TO INCLUDE AN EXTRA COLUMN THAN TO OMIT A NECESSARY ONE.
4. If you are **100% certain** that no column from the provided schema is useful to answer the query (because the question is completely unrelated), return exactly: `NONE`
5. Do NOT output any reasoning, internal chain-of-thought, table names (e.g., `customers.name`), or any other text. Any extra output will be treated as an error.

INPUT FORMAT (this exact structure will be provided):
[QUERY]:
<natural language question>

[RELEVANT TABLES SCHEMA]:
TABLE: <table_name_1>
COLUMNS:
- <column_1>
- <column_2>
...
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
- first_name
- last_name
- email
- city
- registration_date

EXPECTED OUTPUT (single-line):
first_name,last_name,email,city

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
- customer_id
- order_date
- amount

EXPECTED OUTPUT (single-line):
name,customer_id,order_id

---

Example 3
[QUERY]:
List the products released after 2022 and their price.

[RELEVANT TABLES SCHEMA]:
TABLE: products
COLUMNS:
- product_id
- product_name
- price
- release_date
- supplier_id

EXPECTED OUTPUT (single-line):
product_name,price,release_date

---

Example 4 (Query is irrelevant to the schema)
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
- order_id
- customer_id
- order_date

EXPECTED OUTPUT (single-line):
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

    dataset = 'BIRDdev'
    extract_attributes(dataset)