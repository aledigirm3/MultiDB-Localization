import json
import paths


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

if __name__ == '__main__':

    "Used to call utility functions."

    #file_path = paths.DATASETS.BIRDdev.value + 'dev_tables.json'
    #print_DB_table_att(file_path)

    #file_path = paths.DATASETS.BIRDdev.value + 'dev.json'
    #print_sql_queries(file_path)

    #remove_unused_databases_spider1()



