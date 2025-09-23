import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from llm import query_groq
import paths
from ansi_colors import *

def get_llm_response(query: str, tables: str) -> str:
    """
    Function to extract relevant tables from natural language query.
    
    Args:
        query (str): Natural language query.
        tables (str): tables in the format preferred by the llm (tables + descriptions)
    Returns:
        str: relevant tables (comma separated).
    """

    system_prompt = """You are a schema-selector assistant. Your only job is: given a natural language query and a database description (table names + short descriptions and attributes), return EXACTLY and ONLY the comma-separated list of table names that are necessary to answer the query. Nothing else.

- When analyzing the query identify the attributes the user asks for (explicit columns or implied attributes).
- If an attribute required or explicitly mentioned in the query appears in multiple tables, include all of those tables, even if one table alone would be sufficient. Do not try to optimize or exclude. Always include every table that contains that attribute.
- **If a table is explicitly mentioned in the query, or a close variation of its name (including singular/plural forms, abbreviations, or similar naming), always include that table.**

OUTPUT RULES:
1. Output must be a single line containing only table names separated by commas, with NO SPACES (example: customers,orders). Do NOT include any labels, punctuation, explanation, or code fences.
2. Use only the table names exactly as they appear on the left-hand side of the database description lines (the canonical names). Do not invent, abbreviate, or change names.
3. Return the set of tables that together contain the information required to satisfy the query.
4. If no table is needed to answer the query (e.g., query is about general facts not in the DB), return exactly: NONE
5. Do NOT output any reasoning, internal chain-of-thought, or extra metadata. Any extra output will be treated as an error.
6. **If you are uncertain whether a table is needed, include it** to avoid missing important information.

INPUT FORMAT (this exact structure will be provided):
[QUERY]:
<single natural-language question>

[DATABASE WITH TABLE DESCRIPTIONS]:
database: <DatabaseName>
tableA: <short description including columns if present>
tableB: <short description>
...
(Each table line begins with the canonical table name, then a colon, then description.)

EXAMPLES (generic):

Example 1
[QUERY]:
List all active customers who placed orders last month.

[DATABASE WITH TABLE DESCRIPTIONS]:
database: shop_db
customers: The `customers` table stores information about people registered in the system. Columns include: customer_id, name, email, status (active/inactive).
orders: The `orders` table represents purchases made by users. Columns include: order_id, customer_id, order_date, total.
products: The `products` table contains information about products available in the system. Columns include: product_id, name, price.

EXPECTED OUTPUT (single-line):
customers,orders

Example 2
[QUERY]:
How many products are currently in the catalog?

[DATABASE WITH TABLE DESCRIPTIONS]:
database: shop_db
customers: The `customers` table stores information about people registered in the system. Columns include: customer_id, name, email.
orders: The `orders` table represents purchases made by users. Columns include: order_id, customer_id, order_date.
products: The `products` table contains information about products available in the system. Columns include: product_id, name, price.

EXPECTED OUTPUT (single-line):
products

Example 3 (Attribute appears in multiple tables)
[QUERY]:
List the email addresses of all users who completed the survey last month.

[DATABASE WITH TABLE DESCRIPTIONS]:
database: survey_db
users: The `users` table stores information about registered users. Columns include: user_id, name, email, signup_date.
survey_responses: The `survey_responses` table records all survey submissions. Columns include: response_id, user_id, email, survey_id, completion_date.
surveys: The `surveys` table contains details about the surveys. Columns include: survey_id, survey_name, created_date.

EXPECTED OUTPUT (single-line):
users,survey_responses

Example 4 (no DB data required)
[QUERY]:
What is the capital of France?

[DATABASE WITH TABLE DESCRIPTIONS]:
database: dummy
customers: The `customers` table stores information about people registered in the system. Columns include: customer_id, name, email.
orders: The `orders` table represents purchases made by users. Columns include: order_id, customer_id, order_date.

EXPECTED OUTPUT (single-line):
NONE

Example 5 (tables explicitly mentioned in the query or with close variations)
[QUERY]:
Give me the product code of the item released in January 2021.

[DATABASE WITH TABLE DESCRIPTIONS]:
database: inventory_db
items: The `items` table stores information about products. Columns include: item_id, name, release_date, product_code.
product: The `product` table contains general metadata about product categories. Columns include: id, category_name, description.
product labels: The `product_labels` table stores internal labeling data for products. Columns include: id, product_code, label_type.
suppliers: The `suppliers` table contains supplier details. Columns include: supplier_id, name, country.

EXPECTED OUTPUT (single-line):
items,product,product labels
"""

    content = f"""Now receive the actual `[QUERY]` and `[DATABASE WITH TABLE DESCRIPTIONS]` and produce the single-line answer only.
    
[QUERY]:
{query}

[DATABASE WITH TABLE DESCRIPTIONS]:
{tables}"""
    
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


def extract_tables(dataset):

    if dataset == 'BIRDdev':
        with open('./BIRDdev_table_descriptions.json', "r", encoding="utf-8") as f:
            table_descriptions = json.load(f)

        with open('../../' + paths.RESULTS.DB_RETRIEVAL.value + 'BIRDdev_DB_extractor.json', "r", encoding="utf-8") as f:
            data = json.load(f)

        results_folder = '../../' + paths.RESULTS.TAB_RETRIEVAL.value
        os.makedirs(results_folder, exist_ok=True)
        result_file_path = os.path.join(results_folder, "BIRDdev_TAB_extractor.json")

    elif dataset == 'SPIDERdev1':
        with open('./SPIDERdev1_table_descriptions.json', "r", encoding="utf-8") as f:
            table_descriptions = json.load(f)

        with open('../../' + paths.RESULTS.DB_RETRIEVAL.value + 'SPIDERdev1_DB_extractor.json', "r", encoding="utf-8") as f:
            data = json.load(f)

        results_folder = '../../' + paths.RESULTS.TAB_RETRIEVAL.value
        os.makedirs(results_folder, exist_ok=True)
        result_file_path = os.path.join(results_folder, "SPIDERdev1_TAB_extractor.json")
        
    else:
        print(f"{RED}INVALID DATASET!{RESET}")
        sys.exit(1)

    result_list = []

    for sample in data:
        query = sample['question']
        db = sample['DB_result']

        tables = f"database: {db}\n"
        for table, desc in table_descriptions[db].items():
            tables += f"{table}: {desc}\n"

        llm_response = get_llm_response(query, tables)

        # llm_response to list
        llm_response_list = [w.strip() for w in llm_response.split(',')]

        item = {
				"question_id": sample['question_id'],
				"db_id": sample['db_id'],
				"question": sample['question'],
				"SQL": sample['SQL'],
				"DB_result": sample['DB_result'],
                "TAB_result": llm_response_list
				}
        result_list.append(item)

        # Overwrite the file at every iteration.
        # This way, even if the process fails or hits token limits, 
        # we always have a checkpoint with partial results saved.
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':

    dataset = 'SPIDERdev1'
    extract_tables(dataset)