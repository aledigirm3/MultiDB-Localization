import json


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
    
if __name__ == '__main__':

    file_path = '../BIRDdev/dev_tables.json'
    print_DB_tables_dict(file_path)
    



