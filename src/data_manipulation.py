import json
import paths
import re
from typing import Dict, List


def get_databases(file_path: str) -> list:

    """
    Opens a JSON file, reads it, and creates a list of databases.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        A list of databases
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f) 


    db_ids = [item["db_id"] for item in data]

    return db_ids

def get_DB_tables_dict(file_path: str) -> dict:
    """
    Opens a JSON file, reads it, and creates a dictionary with 'db_id' 
    as the key and the list of 'table_names' as the value.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        A dictionary with 'db_id' as keys and lists of 'table_names' as values,
        or an empty dictionary if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            db_dict = {}
            for item in data:
                db_id = item.get("db_id")
                table_names = item.get("table_names")
                if db_id and table_names:
                    db_dict[db_id] = table_names
            return db_dict
            
    except FileNotFoundError:
        print(f"Error: The file was not found at path '{file_path}'")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON.")
        return {}
    
def print_DB_tables_dict(file_path: str) -> str:

    resulting_dictionary = get_DB_tables_dict(file_path)

    if resulting_dictionary:
        print("--- Formatted Database Info ---\n")
        for db_id, tables in resulting_dictionary.items():
            print(f"Database name: {db_id}")
            
            tables_string = ", ".join(tables)
            
            print(f"Tables: {tables_string}\n")

def print_DB_table_att(file_path):
    """Formats and prints database schemas from a list of dictionaries.

    This function iterates through a list of database schema objects. For each
    schema, it restructures the data to map column names directly to their
    corresponding table names and then prints the simplified structure to the
    console as a formatted JSON string.

    Args:
        data: A list of dictionaries, where each dictionary represents a
              database schema. Each dictionary must contain 'db_id',
              'table_names', and 'column_names' keys.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            for db_schema in data:

                output_structure = {
                    "database_name": db_schema["db_id"],
                    "tables": {}
                }

                table_names = db_schema["table_names"]
                for table_name in table_names:
                    output_structure["tables"][table_name] = []

                for column_info in db_schema["column_names"][1:]:
                    table_index = column_info[0]
                    column_name = column_info[1]

                    corresponding_table = table_names[table_index]

                    output_structure["tables"][corresponding_table].append(column_name)

                print(json.dumps(output_structure, indent=2))
                print("-" * 40)
            
    except FileNotFoundError:
        print(f"Error: The file was not found at path '{file_path}'")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON.")



def print_sql_queries(file_path):

    "Print all sql queries in the dataset."

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("\n")
        for sample in data:

            print("- " + sample['SQL'])

        print("\n")

    except FileNotFoundError:
        print(f"Error: The file was not found at path '{file_path}'")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON.")


def remove_unused_databases_spider1():
	
    question_path = paths.DATASETS.SPIDERdev1.value + 'dev.json'
    tables_path = paths.DATASETS.SPIDERdev1.value + 'dev_tables.json'

    with open(question_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    db_ids = {sample['db_id'] for sample in data}  # set comprehension

    with open(tables_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = [sample for sample in data if sample['db_id'] in db_ids]

    with open(tables_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(len(filtered_data))


def create_table_name_mapping(filename: str) -> Dict[str, Dict[str, str]]:
    """
    Parses a JSON file to create a mapping between user-friendly table names
    and their original names in the database schema.

    The input JSON file is expected to be a list of objects, where each object
    represents a database and contains 'db_id', 'table_names', and
    'table_names_original' keys.

    Args:
        filename: The path to the input JSON file.

    Returns:
        A dictionary where each key is a 'db_id' and its value is another
        dictionary that maps the friendly table names from 'table_names'
        to the original names from 'table_names_original'.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
        KeyError: If an object in the JSON is missing one of the required keys
                  ('db_id', 'table_names', 'table_names_original').
    """
    output_dict = {}

    with open(filename, 'r') as f:
        data = json.load(f)

    for item in data:
        db_id = item['db_id']
        friendly_names = item['table_names']
        original_names = item['table_names_original']

        # The zip function pairs elements from both lists, and dict()
        # converts these pairs into a key-value dictionary.
        output_dict[db_id] = dict(zip(friendly_names, original_names))

    return output_dict

def get_sql_table_names(sql_query: str) -> List[str]:
    """
    Extracts a list of table names from any given SQL query.
    Args:
        sql_query: A string containing the SQL query.

    Returns:
        A list of unique table names found in the query. Returns an
        empty list if no tables are found.
    """
    cte_pattern = re.compile(r"""
    WITH\s+([\w`"]+)
    \s+AS\s*\(
""", re.IGNORECASE | re.VERBOSE)

    ctes = set(m.group(1) for m in cte_pattern.finditer(sql_query))

    table_pattern = re.compile(r"""
        (?:FROM|JOIN)\s+
        ([`"]?[a-zA-Z_][\w$]*[`"]?)
    """, re.IGNORECASE | re.VERBOSE)

    tables = table_pattern.findall(sql_query)

    result = []
    for t in tables:
        if t not in ctes and t not in result:
            result.append(t)

    return result


if __name__ == '__main__':

    "Used to call utility functions."

    #file_path = paths.DATASETS.SPIDERdev1.value + 'dev_tables.json'
    #print_DB_table_att(file_path)

    #file_path = paths.DATASETS.BIRDdev.value + 'dev.json'
    #print_sql_queries(file_path)
    #print(get_sql_table_names("SELECT product_name, order_date FROM marketing.orders WHERE status = 'shipped';"))

    #remove_unused_databases_spider1()

    #filename = paths.DATASETS.SPIDERdev1.value + 'dev_tables.json'
    #dict = create_table_name_mapping(filename)
    #print(dict)



