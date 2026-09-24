import json
import sqlite3
import re
import unicodedata
import paths
import copy
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import sqlglot
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope
from collections import Counter, defaultdict
from nltk.stem import SnowballStemmer

from ansi_colors import *

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

def get_DB_table_att(file_path: str) -> list:
    """Formats database schemas as database/table/attribute dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    formatted_schemas = []
    for db_schema in data:
        output_structure = {
            "database_name": db_schema["db_id"],
            "tables": {}
        }

        table_names = db_schema["table_names"]
        for table_name in table_names:
            output_structure["tables"][table_name] = []

        for table_index, column_name in db_schema["column_names"][1:]:
            corresponding_table = table_names[table_index]
            output_structure["tables"][corresponding_table].append(column_name)

        formatted_schemas.append(output_structure)

    return formatted_schemas


def print_DB_table_att(file_path):
    """Prints database schemas in database/table/attribute format."""
    try:
        for output_structure in get_DB_table_att(file_path):
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


def print_ambiguous_distribution(dataset_path):
    """Print sample distribution and distinct base/clone DB counts from dev.json.

    Count only DBs with samples. Reject invalid suffixes; empty data reports zeros.
    """
    dataset_path = Path(dataset_path)
    with open(dataset_path / 'dev.json', 'r', encoding='utf-8') as file:
        samples = json.load(file)
    if not isinstance(samples, list):
        raise ValueError('dev.json must contain a list of samples.')

    counts = Counter()
    db_ids = set()
    for index, sample in enumerate(samples, start=1):
        db_id = sample.get('db_id') if isinstance(sample, dict) else None
        if not isinstance(db_id, str) or not re.fullmatch(r'.+_[123]', db_id):
            raise ValueError(f'Invalid ambiguous db_id at sample {index}: {db_id!r}')
        counts[db_id[-1]] += 1
        db_ids.add(db_id)

    total = len(samples)
    print(f'{dataset_path.name}: {total} samples')
    print(f'Original DBs (in dev.json): {len({db_id.rsplit("_", 1)[0] for db_id in db_ids})}')
    print(f'Split DBs (in dev.json): {len(db_ids)}')
    for split in ('1', '2', '3'):
        percentage = 100 * counts[split] / total if total else 0.0
        print(f'_{split}: {counts[split]} samples ({percentage:.2f}%)')


def remove_unused_databases(questions_path, tables_path):
    """Keep only schemas and SQLite folders for DBs referenced in questions_path.

    Overwrite tables_path and remove unused dev_databases/<db_id> folders beside
    it, if they contain <db_id>.sqlite. Print the number of retained schemas.
    """

    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    db_ids = {sample['db_id'] for sample in data}  # set comprehension

    with open(tables_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = [sample for sample in data if sample['db_id'] in db_ids]

    databases_path = Path(tables_path).resolve().parent / 'dev_databases'
    unused_directories = []
    if databases_path.is_dir():
        if databases_path.resolve() != databases_path:
            raise ValueError(f"Refusing to clean a linked directory: {databases_path}")
        unused_directories = [
            directory for directory in sorted(databases_path.iterdir(), key=lambda p: p.name)
            if directory.is_dir() and directory.name not in db_ids
            and (directory / f'{directory.name}.sqlite').is_file()
        ]
        for directory in unused_directories:
            if directory.resolve() != directory:
                raise ValueError(f"Refusing to delete a linked directory: {directory}")

    with open(tables_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    for directory in unused_directories:
        shutil.rmtree(directory)

    print(f"Number of DBs: {len(filtered_data)}")


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


def create_db_schema_dictionary(file_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parses a JSON file from the BIRD dataset to create a nested dictionary
    representing database schemas.

    The function reads a file containing a list of database schemas. For each
    database, it maps its tables to a list of their corresponding column names.
    It uses the 'table_names' for table names and 'column_names'
    for column mappings.

    The structure of the output dictionary is:
    {
        'db_id_1': {
            'table_name_1': ['column_1', 'column_2', ...],
            'table_name_2': ['column_A', 'column_B', ...],
            ...
        },
        'db_id_2': { ... },
        ...
    }

    Args:
        file_path (str): The full path to the input JSON file (e.g., 'tables.json').

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary where each key is a 'db_id'
        and its value is another dictionary. This inner dictionary's keys are
        table names, and its values are lists of column names for that table.
        Returns an empty dictionary if the file cannot be found or is not valid JSON.
    """
    db_schemas = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file at '{file_path}' is not a valid JSON file.")
        return {}

    for db_info in data:
        db_id = db_info.get("db_id")
        table_names = db_info.get("table_names")
        columns = db_info.get("column_names")

        if not all([db_id, table_names, columns]):
            print(f"Warning: Skipping an entry due to missing 'db_id', "
                  f"'table_names', or 'column_names' keys.")
            continue

        current_db_schema = {table: [] for table in table_names}

        for table_index, column_name in columns:

            if table_index >= 0:
                if table_index < len(table_names):
                    target_table = table_names[table_index]
                    current_db_schema[target_table].append(column_name)
                else:
                    print(f"Warning: Found an invalid table_index '{table_index}'"
                          f"for db_id '{db_id}'.")

        db_schemas[db_id] = current_db_schema

    return db_schemas

def create_db_original_schema_dictionary(file_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parses a JSON file from the BIRD dataset to create a nested dictionary
    representing database schemas.

    The function reads a file containing a list of database schemas. For each
    database, it maps its tables to a list of their corresponding column names.
    It uses the 'table_names_original' for table names and 'column_names_original'
    for column mappings.

    The structure of the output dictionary is:
    {
        'db_id_1': {
            'table_name_1': ['column_1', 'column_2', ...],
            'table_name_2': ['column_A', 'column_B', ...],
            ...
        },
        'db_id_2': { ... },
        ...
    }

    Args:
        file_path (str): The full path to the input JSON file (e.g., 'tables.json').

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary where each key is a 'db_id'
        and its value is another dictionary. This inner dictionary's keys are
        table names, and its values are lists of column names for that table.
        Returns an empty dictionary if the file cannot be found or is not valid JSON.
    """
    db_schemas = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file at '{file_path}' is not a valid JSON file.")
        return {}

    for db_info in data:
        db_id = db_info.get("db_id")
        table_names = db_info.get("table_names_original")
        columns = db_info.get("column_names_original")

        if not all([db_id, table_names, columns]):
            print(f"Warning: Skipping an entry due to missing 'db_id', "
                  f"'table_names', or 'column_names' keys.")
            continue

        current_db_schema = {table: [] for table in table_names}

        for table_index, column_name in columns:

            if table_index >= 0:
                if table_index < len(table_names):
                    target_table = table_names[table_index]
                    current_db_schema[target_table].append(column_name)
                else:
                    print(f"Warning: Found an invalid table_index '{table_index}'"
                          f"for db_id '{db_id}'.")

        db_schemas[db_id] = current_db_schema

    return db_schemas

# 利用sqlglot工具，从一个sql语句中，提取出其中涉及的所有表和列（但是无法确认表与列之间的严格对应关系）
def extract_tables_and_columns(sql_query):
    parsed_query = sqlglot.parse_one(sql_query, read="sqlite")
    table_names = parsed_query.find_all(sqlglot.exp.Table)
    column_names = parsed_query.find_all(sqlglot.exp.Column)
    return {
        'table': {_table.name for _table in table_names},
        'column': {_column.alias_or_name for _column in column_names}
    }

def extract_qualified_columns(
    sql_query: str,
    db_schema: Dict[str, Dict[str, str]],
) -> Set[Tuple[str, str]]:
    """Extract lowercase (table, column) pairs resolved against a SQLGlot schema."""
    parsed_query = qualify(
        sqlglot.parse_one(sql_query, read="sqlite"),
        dialect="sqlite",
        schema=db_schema,
        validate_qualify_columns=False,
        identify=False,
    )
    qualified_columns = set()
    known_columns = {column for columns in db_schema.values() for column in columns}

    for scope in traverse_scope(parsed_query):
        for column in scope.columns:
            source_scope = scope
            source = source_scope.sources.get(column.table)

            # Correlated subqueries can reference a table from a parent scope.
            while source is None and source_scope.parent is not None:
                source_scope = source_scope.parent
                source = source_scope.sources.get(column.table)

            if isinstance(source, sqlglot.exp.Table):
                attribute = (source.name.lower(), column.name.lower())
                if attribute[1] in db_schema.get(attribute[0], {}):
                    qualified_columns.add(attribute)
                elif attribute[1] in known_columns:
                    raise ValueError(f"Unable to resolve column '{column.sql()}' against its source table")
            elif not isinstance(source, Scope) and column.name.lower() in known_columns:
                raise ValueError(f"Unable to resolve column '{column.sql()}' against the database schema")

    return qualified_columns

def create_attribute_mapping(file_path: str) -> dict:
    """
    Parses a JSON file containing database schemas to create a mapping from
    normalized attribute names to their original names.

    The function reads a list of database schemas, and for each schema, it maps
    the 'normal' table and column names to the 'original' ones. The final
    structure is a nested dictionary.

    Args:
        file_path (str): The path to the input JSON file.

    Returns:
        dict: A dictionary with the following structure:
              {
                  db_id: {
                      table_name: {
                          column_name: column_name_original
                      }
                  }
              }
              Returns an empty dictionary if the file cannot be read or is empty.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading the file: {e}")
        return {}

    # The main dictionary to store the final mapping.
    # We use defaultdict to simplify the creation of nested dictionaries.
    mapping = defaultdict(lambda: defaultdict(dict))

    # Iterate over each database schema in the JSON data.
    for db_schema in data:
        db_id = db_schema['db_id']
        
        # Get the lists of normal and original names.
        table_names_normal = db_schema['table_names']
        column_names_normal = db_schema['column_names']
        column_names_original = db_schema['column_names_original']

        # Iterate through the columns to build the mapping.
        # We use zip to pair up the normal and original column entries.
        for normal_col_info, original_col_info in zip(column_names_normal, column_names_original):
            table_index, col_name_normal = normal_col_info
            _, col_name_original = original_col_info

            # A table_index of -1 usually refers to a wildcard '*' and is not
            # associated with a specific table, so we can skip it.
            if table_index == -1:
                continue

            # Get the 'normal' table name using its index.
            table_name_normal = table_names_normal[table_index]

            # Populate the nested dictionary with the mapping.
            # {db_id: {table_name: {column_name: column_name_original}}}
            mapping[db_id][table_name_normal][col_name_normal] = col_name_original

    # Convert defaultdict back to a regular dict for the final output.
    # This is optional but can be cleaner for the user.
    return json.loads(json.dumps(mapping))

def lowercase_dict(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            # Trasforma la chiave in minuscolo se è stringa
            new_key = k.lower() if isinstance(k, str) else k
            # Ricorsione per valori
            new_dict[new_key] = lowercase_dict(v)
        return new_dict
    elif isinstance(d, list):
        # Se è lista, applica ricorsione a ogni elemento
        return [lowercase_dict(item) for item in d]
    elif isinstance(d, str):
        # Se è stringa, trasformala in minuscolo
        return d.lower()
    else:
        # Se è altro tipo (int, float, etc.), lascialo invariato
        return d


def normalize_and_stem_text(text: str) -> str:
    """
    Normalizes text for BM25 indexing/querying and applies English Snowball stemming.
    """

    if not isinstance(text, str):
        return ""

    stemmer = SnowballStemmer("english")
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    normalized = re.sub(r"[\W_]+", " ", normalized.lower(), flags=re.UNICODE)
    tokens = [
        token for token in normalized.split()
        if not any(char.isdigit() for char in token)
    ]

    return " ".join(stemmer.stem(token) for token in tokens)


def add_char4_tokens(text: str) -> str:
    tokens = text.split()
    char4_tokens = []

    for token in tokens:
        if len(token) < 4:
            continue
        char4_tokens.extend(
            f"char4:{token[index:index + 4]}"
            for index in range(len(token) - 3)
        )

    return " ".join(tokens + char4_tokens)


def create_benchmark_doc(dataset_path):
    """
    Creates a benchmark documentation JSON file from a Spider/BIRD-like dataset.

    Args:
        dataset_path: A paths.DATASETS enum member, its .value path, or a direct
                      dataset directory path (e.g. '../datasets/BIRDdev/').

    Returns:
        The path of the generated JSON file.
    """
    max_top_values = 50
    max_distinct_ratio = 0.50

    def resolve_dataset_dir(path_value):
        raw_path = getattr(path_value, "value", path_value)
        dataset_dir = Path(raw_path)
        candidates = [dataset_dir]

        if not dataset_dir.is_absolute():
            parts = dataset_dir.parts
            if parts and parts[0] == "..":
                candidates.append(Path(*parts[1:]))
            candidates.append(Path(__file__).parent / dataset_dir)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return dataset_dir

    def quote_identifier(identifier):
        return '"' + str(identifier).replace('"', '""') + '"'

    def json_safe_value(value):
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        return value

    def add_joinable(joinables, source_idx, target_idx, table_names):
        if source_idx == target_idx:
            return
        if source_idx < 0 or target_idx < 0:
            return
        if source_idx >= len(table_names) or target_idx >= len(table_names):
            return

        target_table = table_names[target_idx]
        if target_table not in joinables[source_idx]:
            joinables[source_idx].append(target_table)

    def get_sqlite_path(dataset_dir, db_id):
        db_dir = dataset_dir / "dev_databases" / db_id
        expected_path = db_dir / f"{db_id}.sqlite"

        if expected_path.exists():
            return expected_path

        sqlite_files = list(db_dir.glob("*.sqlite")) if db_dir.exists() else []
        return sqlite_files[0] if sqlite_files else None

    def get_table_schema_sql(connection, table_name):
        try:
            row = connection.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (table_name,),
            ).fetchone()
        except sqlite3.Error:
            return ""

        return " ".join(row[0].split()) if row and row[0] else ""

    def get_column_top_values(connection, table_name, column_name):
        quoted_table = quote_identifier(table_name)
        quoted_column = quote_identifier(column_name)

        try:
            non_null_count, distinct_count = connection.execute(
                f"""
                SELECT COUNT({quoted_column}), COUNT(DISTINCT {quoted_column})
                FROM {quoted_table}
                """
            ).fetchone()
        except sqlite3.Error:
            return []

        if not non_null_count or not distinct_count:
            return []
        if distinct_count == non_null_count:
            return []
        if distinct_count / non_null_count > max_distinct_ratio:
            return []

        try:
            rows = connection.execute(
                f"""
                SELECT {quoted_column}, COUNT(*) AS value_frequency
                FROM {quoted_table}
                WHERE {quoted_column} IS NOT NULL
                GROUP BY {quoted_column}
                ORDER BY value_frequency DESC, CAST({quoted_column} AS TEXT) ASC
                LIMIT {max_top_values}
                """
            ).fetchall()
        except sqlite3.Error:
            return []

        return [json_safe_value(row[0]) for row in rows]

    dataset_dir = resolve_dataset_dir(dataset_path)
    tables_path = dataset_dir / "dev_tables.json"

    with open(tables_path, "r", encoding="utf-8") as f:
        schemas = json.load(f)

    output_path = dataset_dir / "doc.json"
    benchmark_doc = []

    for db_schema in schemas:
        db_id = db_schema.get("db_id")
        table_names = db_schema.get("table_names", [])
        table_names_original = db_schema.get("table_names_original", [])
        column_names = db_schema.get("column_names", [])
        column_names_original = db_schema.get("column_names_original", [])

        if not db_id or not table_names or not table_names_original:
            continue

        columns_by_table = {idx: [] for idx in range(len(table_names))}
        original_columns_by_table = {idx: [] for idx in range(len(table_names))}

        for normal_column_info, original_column_info in zip(column_names, column_names_original):
            table_idx, column_name = normal_column_info
            _, original_column_name = original_column_info

            if table_idx < 0 or table_idx >= len(table_names):
                continue

            columns_by_table[table_idx].append(column_name)
            original_columns_by_table[table_idx].append((column_name, original_column_name))

        joinables_by_table = {idx: [] for idx in range(len(table_names))}
        for foreign_key in db_schema.get("foreign_keys", []):
            if len(foreign_key) != 2:
                continue

            left_column_idx, right_column_idx = foreign_key
            if left_column_idx < 0 or right_column_idx < 0:
                continue
            if left_column_idx >= len(column_names) or right_column_idx >= len(column_names):
                continue

            left_table_idx = column_names[left_column_idx][0]
            right_table_idx = column_names[right_column_idx][0]

            add_joinable(joinables_by_table, left_table_idx, right_table_idx, table_names)
            add_joinable(joinables_by_table, right_table_idx, left_table_idx, table_names)

        sqlite_path = get_sqlite_path(dataset_dir, db_id)
        connection = None
        if sqlite_path:
            connection = sqlite3.connect(sqlite_path)

        try:
            for table_idx, table_name in enumerate(table_names):
                if table_idx >= len(table_names_original):
                    continue

                original_table_name = table_names_original[table_idx]
                schema_sql = ""
                top_values = {}

                if connection:
                    schema_sql = get_table_schema_sql(connection, original_table_name)
                    for column_name, original_column_name in original_columns_by_table[table_idx]:
                        values = get_column_top_values(
                            connection,
                            original_table_name,
                            original_column_name,
                        )
                        if values:
                            top_values[column_name] = values

                benchmark_doc.append(
                    {
                        "table_id": f"{db_id}.{table_name}",
                        "db": db_id,
                        "table": table_name,
                        "columns": columns_by_table[table_idx],
                        "schema_sql": schema_sql,
                        "joinable_tables": joinables_by_table[table_idx],
                        "top_values": top_values,
                    }
                )
        finally:
            if connection:
                connection.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_doc, f, indent=2, ensure_ascii=False)

    print(f"{GREEN}File saved: {output_path}{RESET} ({len(benchmark_doc)} tables)")
    return str(output_path)


def add_bm25_text_to_benchmark_doc(dataset_path):
    """
    Adds a normalized bm25_text field to each table entry in a benchmark doc file.

    Args:
        dataset_path: A paths.DATASETS enum member, its .value path, or a direct
                      dataset directory path (e.g. '../datasets/BIRDdev/').
    """
    def resolve_dataset_dir(path_value):
        raw_path = getattr(path_value, "value", path_value)
        dataset_dir = Path(raw_path)
        candidates = [dataset_dir]

        if not dataset_dir.is_absolute():
            parts = dataset_dir.parts
            if parts and parts[0] == "..":
                candidates.append(Path(*parts[1:]))
            candidates.append(Path(__file__).parent / dataset_dir)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return dataset_dir

    def add_text_part(parts, value):
        if isinstance(value, str) and value.strip():
            parts.append(value)

    def build_bm25_text(table_doc):
        parts = []

        add_text_part(parts, table_doc.get("table"))

        for column_name in table_doc.get("columns", []):
            add_text_part(parts, column_name)

        for joinable_table in table_doc.get("joinable_tables", []):
            add_text_part(parts, joinable_table)

        top_values = table_doc.get("top_values", {})
        if isinstance(top_values, dict):
            for values in top_values.values():
                if not isinstance(values, list):
                    continue

                string_values = [
                    value for value in values
                    if isinstance(value, str) and value.strip()
                ]

                if string_values:
                    parts.extend(string_values)
                    parts.extend(string_values)

        return add_char4_tokens(normalize_and_stem_text(" ".join(parts)))

    dataset_dir = resolve_dataset_dir(dataset_path)
    doc_path = dataset_dir / "doc.json"

    with open(doc_path, "r", encoding="utf-8") as f:
        benchmark_doc = json.load(f)

    updated_doc = []
    for table_doc in benchmark_doc:
        bm25_text = build_bm25_text(table_doc)
        updated_table_doc = {}
        bm25_inserted = False

        for key, value in table_doc.items():
            if key == "bm25_text":
                continue

            updated_table_doc[key] = value
            if key == "top_values":
                updated_table_doc["bm25_text"] = bm25_text
                bm25_inserted = True

        if not bm25_inserted:
            updated_table_doc["bm25_text"] = bm25_text

        updated_doc.append(updated_table_doc)

    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(updated_doc, f, indent=2, ensure_ascii=False)

    print(f"{GREEN}File updated: {doc_path}{RESET} ({len(updated_doc)} tables)")


def create_ambiguous_benchmark(
    dataset_path,
    output_path=None,
    overwrite: bool = False,
    distinct_ratio_threshold: float = 0.30,
):
    """
    Creates a Spider/BIRD-like benchmark with three split SQLite clones for each
    database that has a suitable low-cardinality text column referenced by SQL.

    The selected column is chosen by counting SQL equality/IN predicates whose
    literal values match normalized database values and whose normalized length
    is greater than 3 characters. A split column must have at least 3 distinct
    usable normalized values. To avoid a systematic suffix imbalance, duplicated
    databases are ordered by `db_id` and the primary, secondary, and remaining
    value groups are cyclically assigned to suffixes `_1`, `_2`, and `_3`. This
    rotation only relabels the three groups; it does not change their rows. Child
    rows that would reference removed parent rows are removed recursively
    according to `dev_tables.json` foreign keys. Sample assignment is
    intentionally strict:
    a sample must explicitly reference the split column in its SQL with a value
    longer than 3 characters, then the SQL query is executed on the original DB
    and all clones; the sample is kept only when exactly one clone preserves the
    original query result. Rows use the first unshadowed SQLite rowid alias;
    WITHOUT ROWID tables use their ordered primary-key columns instead.
    """

    min_split_value_chars = 4
    min_split_distinct_values = 3

    def resolve_dataset_dir(path_value):
        raw_path = getattr(path_value, "value", path_value)
        dataset_dir = Path(raw_path)
        candidates = [dataset_dir]

        if not dataset_dir.is_absolute():
            candidates.append(Path.cwd() / dataset_dir)
            parts = dataset_dir.parts
            if parts and parts[0] == "..":
                candidates.append(Path.cwd() / Path(*parts[1:]))
            candidates.append(Path(__file__).parent / dataset_dir)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return dataset_dir.resolve()

    def quote_identifier(identifier):
        return '"' + str(identifier).replace('"', '""') + '"'

    def normalize_identifier(identifier):
        return str(identifier).strip().strip('`"[]').lower()

    def candidate_key(table_name, column_name):
        return (
            normalize_identifier(table_name),
            normalize_identifier(column_name),
        )

    def split_value_length(normalized_value):
        return len(str(normalized_value).replace(" ", ""))

    def is_usable_split_value(normalized_value):
        return split_value_length(normalized_value) >= min_split_value_chars

    def json_safe_value(value):
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        return value

    def normalized_db_value(raw_value):
        return normalize_and_stem_text(json_safe_value(raw_value))

    def normalized_phrase_in_text(normalized_phrase, normalized_text):
        return f" {normalized_phrase} " in f" {normalized_text} "

    def build_clone_split_rules(split_info, rotation_offset):
        primary_value = split_info["primary_split_value"]
        secondary_value = split_info["secondary_split_value"]

        logical_rules = [
            {
                "kind": "primary_value",
                "keep_values": [primary_value],
                "excluded_values": [],
            },
            {
                "kind": "secondary_value",
                "keep_values": [secondary_value],
                "excluded_values": [],
            },
            {
                "kind": "remaining_values",
                "keep_values": [],
                "excluded_values": [primary_value, secondary_value],
            },
        ]
        rotated_rules = {
            ((logical_index + rotation_offset) % 3) + 1: rule
            for logical_index, rule in enumerate(logical_rules)
        }
        return dict(sorted(rotated_rules.items()))

    def sql_field_name(sample):
        if "SQL" in sample:
            return "SQL"
        if "query" in sample:
            return "query"
        raise KeyError("Sample does not contain SQL or query.")

    def get_sqlite_path(dataset_dir, db_id):
        db_dir = dataset_dir / "dev_databases" / db_id
        expected_path = db_dir / f"{db_id}.sqlite"

        if expected_path.exists():
            return expected_path

        sqlite_files = list(db_dir.glob("*.sqlite")) if db_dir.exists() else []
        return sqlite_files[0] if sqlite_files else None

    def literal_text(expression):
        if isinstance(expression, sqlglot.exp.Literal):
            return str(expression.this)
        if isinstance(expression, sqlglot.exp.Cast):
            return literal_text(expression.this)
        if isinstance(expression, sqlglot.exp.Paren):
            return literal_text(expression.this)
        if isinstance(expression, sqlglot.exp.Neg):
            inner_value = literal_text(expression.this)
            return f"-{inner_value}" if inner_value is not None else None
        if expression.__class__.__name__ == "Boolean":
            return str(expression.this)
        return None

    def build_schema_lookup(db_schema):
        table_names = db_schema.get("table_names_original", [])
        column_names = db_schema.get("column_names_original", [])
        table_lookup = {
            normalize_identifier(table_name): table_name
            for table_name in table_names
        }
        columns_by_table = defaultdict(dict)
        column_locations = defaultdict(list)

        for table_idx, column_name in column_names:
            if table_idx < 0 or table_idx >= len(table_names):
                continue

            table_name = table_names[table_idx]
            table_key = normalize_identifier(table_name)
            column_key = normalize_identifier(column_name)
            columns_by_table[table_key][column_key] = column_name
            column_locations[column_key].append((table_name, column_name))

        return table_lookup, columns_by_table, column_locations

    def resolve_sql_column(column_expression, alias_map, db_schema):
        table_lookup, columns_by_table, column_locations = build_schema_lookup(db_schema)
        column_name = column_expression.name
        column_key = normalize_identifier(column_name)
        table_reference = column_expression.table

        if table_reference:
            table_key = normalize_identifier(table_reference)
            table_name = alias_map.get(table_key) or table_lookup.get(table_key)
            if not table_name:
                return None

            original_column = columns_by_table.get(
                normalize_identifier(table_name), {}
            ).get(column_key)
            if not original_column:
                return None

            return table_name, original_column

        query_table_keys = {
            normalize_identifier(table_name)
            for table_name in alias_map.values()
        }
        matching_locations = [
            (table_name, original_column)
            for table_name, original_column in column_locations.get(column_key, [])
            if normalize_identifier(table_name) in query_table_keys
        ]

        if len(matching_locations) == 1:
            return matching_locations[0]

        all_locations = column_locations.get(column_key, [])
        if len(all_locations) == 1:
            return all_locations[0]

        return None

    def extract_sql_value_refs(sql_query, db_schema):
        try:
            parsed_query = sqlglot.parse_one(sql_query, read="sqlite")
        except Exception as error:
            return [], str(error)

        table_lookup, _, _ = build_schema_lookup(db_schema)
        alias_map = {}
        for table_expression in parsed_query.find_all(sqlglot.exp.Table):
            table_name = table_expression.name
            original_table = table_lookup.get(
                normalize_identifier(table_name),
                table_name,
            )
            alias_map[normalize_identifier(table_name)] = original_table

            if table_expression.alias:
                alias_map[normalize_identifier(table_expression.alias)] = original_table

        refs = []

        def unresolved_identifier_text(expression):
            if not isinstance(expression, sqlglot.exp.Column):
                return None
            if expression.table:
                return None
            if resolve_sql_column(expression, alias_map, db_schema):
                return None
            return expression.name

        def expression_value_text(expression):
            value = literal_text(expression)
            if value is not None:
                return value
            return unresolved_identifier_text(expression)

        def add_resolved_ref(resolved_column, value):
            normalized_value = normalize_and_stem_text(value)
            if not normalized_value:
                return

            table_name, column_name = resolved_column
            refs.append(
                {
                    "key": candidate_key(table_name, column_name),
                    "table": table_name,
                    "column": column_name,
                    "normalized_value": normalized_value,
                    "raw_value": value,
                }
            )

        for equality in parsed_query.find_all(sqlglot.exp.EQ):
            left_expression = equality.left
            right_expression = equality.right

            if isinstance(left_expression, sqlglot.exp.Column):
                resolved_column = resolve_sql_column(
                    left_expression,
                    alias_map,
                    db_schema,
                )
                value = expression_value_text(right_expression)
                if resolved_column and value is not None:
                    add_resolved_ref(resolved_column, value)

            if isinstance(right_expression, sqlglot.exp.Column):
                resolved_column = resolve_sql_column(
                    right_expression,
                    alias_map,
                    db_schema,
                )
                value = expression_value_text(left_expression)
                if resolved_column and value is not None:
                    add_resolved_ref(resolved_column, value)

        for in_expression in parsed_query.find_all(sqlglot.exp.In):
            column_expression = in_expression.this
            if not isinstance(column_expression, sqlglot.exp.Column):
                continue

            resolved_column = resolve_sql_column(
                column_expression,
                alias_map,
                db_schema,
            )
            if not resolved_column:
                continue

            for value_expression in in_expression.expressions:
                value = expression_value_text(value_expression)
                if value is not None:
                    add_resolved_ref(resolved_column, value)

        return refs, None

    def get_candidate_columns(sqlite_path, db_schema):
        candidates = {}
        candidate_stats = Counter()
        table_names = db_schema.get("table_names_original", [])
        column_names = db_schema.get("column_names_original", [])
        columns_by_table_idx = defaultdict(list)

        for table_idx, column_name in column_names:
            if table_idx >= 0:
                columns_by_table_idx[table_idx].append(column_name)

        connection = sqlite3.connect(sqlite_path)
        try:
            for table_idx, table_name in enumerate(table_names):
                quoted_table = quote_identifier(table_name)

                try:
                    row_count = connection.execute(
                        f"SELECT COUNT(*) FROM {quoted_table}"
                    ).fetchone()[0]
                except sqlite3.Error:
                    continue

                if not row_count:
                    continue

                for column_name in columns_by_table_idx[table_idx]:
                    quoted_column = quote_identifier(column_name)

                    try:
                        raw_distinct_count = connection.execute(
                            f"""
                            SELECT COUNT(DISTINCT {quoted_column})
                            FROM {quoted_table}
                            WHERE {quoted_column} IS NOT NULL
                            """
                        ).fetchone()[0]
                    except sqlite3.Error:
                        continue

                    if raw_distinct_count <= 1:
                        continue

                    distinct_ratio = raw_distinct_count / row_count
                    if distinct_ratio >= distinct_ratio_threshold:
                        continue

                    try:
                        rows = connection.execute(
                            f"""
                            SELECT {quoted_column}, COUNT(*) AS value_frequency
                            FROM {quoted_table}
                            WHERE {quoted_column} IS NOT NULL
                            GROUP BY {quoted_column}
                            """
                        ).fetchall()
                    except sqlite3.Error:
                        continue

                    normalized_counts = Counter()
                    raw_values_by_normalized = defaultdict(list)

                    for raw_value, frequency in rows:
                        safe_raw_value = json_safe_value(raw_value)
                        normalized_value = normalize_and_stem_text(safe_raw_value)

                        if not normalized_value:
                            continue

                        normalized_counts[normalized_value] += frequency
                        if len(raw_values_by_normalized[normalized_value]) < 10:
                            raw_values_by_normalized[normalized_value].append(
                                safe_raw_value
                            )

                    if len(normalized_counts) <= 1:
                        continue

                    candidate_stats["low_cardinality_text_candidates"] += 1
                    usable_normalized_values = {
                        value for value in normalized_counts
                        if is_usable_split_value(value)
                    }
                    if len(usable_normalized_values) < min_split_distinct_values:
                        candidate_stats["insufficient_usable_value_candidates"] += 1
                        continue

                    key = candidate_key(table_name, column_name)
                    candidates[key] = {
                        "key": key,
                        "table_index": table_idx,
                        "table_name": table_name,
                        "column_name": column_name,
                        "row_count": row_count,
                        "raw_distinct_count": raw_distinct_count,
                        "distinct_ratio": distinct_ratio,
                        "normalized_value_counts": normalized_counts,
                        "usable_normalized_values": usable_normalized_values,
                        "raw_values_by_normalized": raw_values_by_normalized,
                    }
        finally:
            connection.close()

        return candidates, candidate_stats

    def select_split_column(sqlite_path, db_schema, db_samples):
        candidates, candidate_stats = get_candidate_columns(sqlite_path, db_schema)

        if not candidates:
            if candidate_stats["insufficient_usable_value_candidates"]:
                return None, "no_low_cardinality_text_candidate_with_three_usable_values"
            return None, "no_low_cardinality_text_candidate"

        usage = {
            key: {
                "sample_hits": 0,
                "value_mentions": Counter(),
                "question_hits": 0,
                "question_value_mentions": Counter(),
            }
            for key in candidates
        }
        parse_errors = []

        for sample in db_samples:
            normalized_question = normalize_and_stem_text(sample.get("question", ""))
            for key, candidate in candidates.items():
                question_values = [
                    value for value in candidate["usable_normalized_values"]
                    if normalized_phrase_in_text(value, normalized_question)
                ]
                if question_values:
                    usage[key]["question_hits"] += 1
                    usage[key]["question_value_mentions"].update(question_values)

            sql_query = sample[sql_field_name(sample)]
            refs, parse_error = extract_sql_value_refs(sql_query, db_schema)

            if parse_error:
                parse_errors.append(
                    {
                        "question_id": sample.get("question_id"),
                        "error": parse_error,
                    }
                )
                continue

            sample_hit_keys = set()
            for ref in refs:
                ref_key = ref["key"]
                normalized_value = ref["normalized_value"]

                if ref_key not in candidates:
                    continue
                if normalized_value not in candidates[ref_key]["normalized_value_counts"]:
                    continue
                if normalized_value not in candidates[ref_key]["usable_normalized_values"]:
                    continue

                usage[ref_key]["value_mentions"][normalized_value] += 1
                sample_hit_keys.add(ref_key)

            for ref_key in sample_hit_keys:
                usage[ref_key]["sample_hits"] += 1

        best_key = None
        best_score = None
        for key, candidate in candidates.items():
            sample_hits = usage[key]["sample_hits"]
            total_mentions = sum(usage[key]["value_mentions"].values())

            if not usage[key]["question_hits"]:
                continue
            if not total_mentions:
                continue

            # Keep tie-breaking deterministic and prefer stronger SQL evidence.
            score = (
                sample_hits,
                total_mentions,
                -candidate["raw_distinct_count"],
                candidate["table_name"],
                candidate["column_name"],
            )

            if best_score is None or score > best_score:
                best_key = key
                best_score = score

        if best_key is None:
            if any(
                sum(candidate_usage["value_mentions"].values())
                for candidate_usage in usage.values()
            ):
                return None, "no_candidate_with_value_mentioned_in_question"
            return None, "no_candidate_referenced_by_sql_equality_long_value"

        selected_candidate = candidates[best_key]
        selected_usage = usage[best_key]
        db_value_counts = selected_candidate["normalized_value_counts"]
        usable_values = selected_candidate["usable_normalized_values"]
        sql_value_mentions = selected_usage["value_mentions"]

        sql_ordered_values = sorted(
            sql_value_mentions,
            key=lambda value: (
                -sql_value_mentions[value],
                -db_value_counts[value],
                value,
            ),
        )
        db_ordered_values = sorted(
            usable_values,
            key=lambda value: (-db_value_counts[value], value),
        )

        primary_value = sql_ordered_values[0]
        secondary_value = next(
            (value for value in sql_ordered_values[1:] if value != primary_value),
            None,
        )
        if secondary_value is None:
            secondary_value = next(
                (value for value in db_ordered_values if value != primary_value),
                None,
            )

        if secondary_value is None:
            return None, "selected_candidate_has_no_secondary_value"

        raw_values_by_normalized = selected_candidate["raw_values_by_normalized"]
        remaining_values = [
            value for value in db_ordered_values
            if value not in {primary_value, secondary_value}
        ]

        return {
            "candidate_key": best_key,
            "table_name": selected_candidate["table_name"],
            "column_name": selected_candidate["column_name"],
            "normalized_value_set": set(usable_values),
            "row_count": selected_candidate["row_count"],
            "raw_distinct_count": selected_candidate["raw_distinct_count"],
            "distinct_ratio": selected_candidate["distinct_ratio"],
            "min_split_value_chars": min_split_value_chars,
            "min_split_distinct_values": min_split_distinct_values,
            "usable_distinct_count": len(usable_values),
            "sql_sample_hits": selected_usage["sample_hits"],
            "sql_total_mentions": sum(sql_value_mentions.values()),
            "sql_value_mentions": dict(sql_value_mentions),
            "question_sample_hits": selected_usage["question_hits"],
            "question_value_mentions": dict(
                selected_usage["question_value_mentions"]
            ),
            "primary_split_value": primary_value,
            "primary_split_raw_values": raw_values_by_normalized[primary_value],
            "secondary_split_value": secondary_value,
            "secondary_split_raw_values": raw_values_by_normalized[secondary_value],
            "remaining_split_value_count": len(remaining_values),
            "remaining_split_values_preview": [
                {
                    "normalized_value": value,
                    "frequency": db_value_counts[value],
                    "raw_values": raw_values_by_normalized[value],
                }
                for value in remaining_values[:10]
            ],
            "top_db_values": [
                {
                    "normalized_value": value,
                    "frequency": db_value_counts[value],
                    "raw_values": raw_values_by_normalized[value],
                }
                for value in db_ordered_values[:10]
            ],
            "parse_errors": parse_errors[:20],
        }, None

    def foreign_keys_from_schema(db_schema):
        table_names = db_schema.get("table_names_original", [])
        column_names = db_schema.get("column_names_original", [])
        foreign_keys = []
        seen = set()

        for foreign_key in db_schema.get("foreign_keys", []):
            if len(foreign_key) != 2:
                continue

            child_column_idx, parent_column_idx = foreign_key
            if child_column_idx < 0 or parent_column_idx < 0:
                continue
            if child_column_idx >= len(column_names) or parent_column_idx >= len(column_names):
                continue

            child_table_idx, child_column_name = column_names[child_column_idx]
            parent_table_idx, parent_column_name = column_names[parent_column_idx]
            if child_table_idx < 0 or parent_table_idx < 0:
                continue
            if child_table_idx >= len(table_names) or parent_table_idx >= len(table_names):
                continue

            edge = (
                table_names[child_table_idx],
                child_column_name,
                table_names[parent_table_idx],
                parent_column_name,
            )

            if edge in seen:
                continue

            seen.add(edge)
            foreign_keys.append(
                {
                    "child_table": edge[0],
                    "child_column": edge[1],
                    "parent_table": edge[2],
                    "parent_column": edge[3],
                }
            )

        return foreign_keys

    def get_row_key_spec(connection, table_name):
        columns = list(connection.execute(
            "SELECT name, pk FROM pragma_table_xinfo(?) ORDER BY cid",
            (table_name,),
        ))
        column_names = {normalize_identifier(name) for name, _ in columns}
        quoted_table = quote_identifier(table_name)

        # Preserve the historical path whenever possible. SQLite's three
        # aliases identify the same internal row, unless a real column uses
        # that name. WITHOUT ROWID tables reject all three aliases.
        for alias in ("rowid", "_rowid_", "oid"):
            if normalize_identifier(alias) in column_names:
                continue
            try:
                connection.execute(
                    f"SELECT {alias} FROM {quoted_table} LIMIT 0"
                )
            except sqlite3.OperationalError:
                continue
            return {
                "expressions": (alias,),
                "uses_internal_rowid": True,
            }

        primary_key = [
            name
            for _, name in sorted(
                (position, name) for name, position in columns if position
            )
        ]
        if not primary_key:
            raise ValueError(
                f"Cannot identify rows in {table_name}: no usable SQLite rowid "
                "alias or primary key"
            )
        return {
            "expressions": tuple(quote_identifier(name) for name in primary_key),
            "uses_internal_rowid": False,
        }

    def load_table_row_keys(connection, table_name, row_key_spec):
        expressions = row_key_spec["expressions"]
        rows = connection.execute(
            f"SELECT {', '.join(expressions)} FROM {quote_identifier(table_name)}"
        ).fetchall()
        if len(expressions) == 1:
            row_keys = {row[0] for row in rows}
        else:
            row_keys = {tuple(row) for row in rows}
        if len(row_keys) != len(rows):
            raise ValueError(f"Non-unique row key in {table_name}")
        return row_keys

    def load_column_values(connection, table_name, column_name, row_key_spec):
        expressions = row_key_spec["expressions"]
        rows = connection.execute(
            f"""
            SELECT {', '.join(expressions)}, {quote_identifier(column_name)}
            FROM {quote_identifier(table_name)}
            """
        ).fetchall()
        if len(expressions) == 1:
            return {row[0]: row[1] for row in rows}
        return {tuple(row[:-1]): row[-1] for row in rows}

    def build_allowed_row_keys(sqlite_path, db_schema, split_info, clone_rule):
        table_names = db_schema.get("table_names_original", [])
        selected_table = split_info["table_name"]
        selected_column = split_info["column_name"]
        value_cache = {}

        connection = sqlite3.connect(sqlite_path)
        try:
            row_key_specs = {
                table_name: get_row_key_spec(connection, table_name)
                for table_name in table_names
            }
            all_row_keys = {
                table_name: load_table_row_keys(
                    connection,
                    table_name,
                    row_key_specs[table_name],
                )
                for table_name in table_names
            }
            allowed_row_keys = {
                table_name: set(row_keys)
                for table_name, row_keys in all_row_keys.items()
            }
            selected_values = load_column_values(
                connection,
                selected_table,
                selected_column,
                row_key_specs[selected_table],
            )

            keep_values = set(clone_rule["keep_values"])
            excluded_values = set(clone_rule["excluded_values"])
            if keep_values:
                selected_row_keys = {
                    row_key
                    for row_key, raw_value in selected_values.items()
                    if normalized_db_value(raw_value) in keep_values
                }
            else:
                selected_row_keys = {
                    row_key
                    for row_key in all_row_keys[selected_table]
                    if normalized_db_value(selected_values.get(row_key)) not in excluded_values
                }

            allowed_row_keys[selected_table] = selected_row_keys

            def cached_values(table_name, column_name):
                cache_key = (table_name, column_name)
                if cache_key not in value_cache:
                    value_cache[cache_key] = load_column_values(
                        connection,
                        table_name,
                        column_name,
                        row_key_specs[table_name],
                    )
                return value_cache[cache_key]

            foreign_keys = foreign_keys_from_schema(db_schema)
            changed = True
            while changed:
                changed = False
                for foreign_key in foreign_keys:
                    parent_table = foreign_key["parent_table"]
                    parent_column = foreign_key["parent_column"]
                    child_table = foreign_key["child_table"]
                    child_column = foreign_key["child_column"]

                    if parent_table not in allowed_row_keys or child_table not in allowed_row_keys:
                        continue

                    if len(allowed_row_keys[parent_table]) == len(all_row_keys[parent_table]):
                        continue

                    parent_values_by_row_key = cached_values(parent_table, parent_column)
                    allowed_parent_values = {
                        parent_values_by_row_key.get(row_key)
                        for row_key in allowed_row_keys[parent_table]
                        if parent_values_by_row_key.get(row_key) is not None
                    }
                    child_values_by_row_key = cached_values(child_table, child_column)
                    child_row_keys_to_keep = {
                        row_key
                        for row_key in allowed_row_keys[child_table]
                        if child_values_by_row_key.get(row_key) is None
                        or child_values_by_row_key.get(row_key) in allowed_parent_values
                    }

                    if len(child_row_keys_to_keep) < len(allowed_row_keys[child_table]):
                        allowed_row_keys[child_table] = child_row_keys_to_keep
                        changed = True

            return allowed_row_keys, {
                table_name: len(row_keys)
                for table_name, row_keys in all_row_keys.items()
            }, len(selected_row_keys), row_key_specs
        finally:
            connection.close()

    def apply_allowed_row_keys(
        sqlite_path,
        table_names,
        allowed_row_keys,
        original_counts,
        row_key_specs,
    ):
        connection = sqlite3.connect(sqlite_path)
        table_counts = {}
        foreign_key_check_rows = []
        foreign_key_check_error = None

        try:
            connection.execute("PRAGMA foreign_keys = OFF")
            for table_name in table_names:
                keep_row_keys = allowed_row_keys.get(table_name, set())
                if len(keep_row_keys) == original_counts.get(table_name, 0):
                    table_counts[table_name] = len(keep_row_keys)
                    continue

                row_key_spec = row_key_specs[table_name]
                expressions = row_key_spec["expressions"]
                quoted_table = quote_identifier(table_name)
                connection.execute("DROP TABLE IF EXISTS temp._ambiguous_keep_row_keys")

                if row_key_spec["uses_internal_rowid"]:
                    expression = expressions[0]
                    connection.execute(
                        "CREATE TEMP TABLE _ambiguous_keep_row_keys "
                        "(row_key INTEGER PRIMARY KEY)"
                    )
                    connection.executemany(
                        "INSERT INTO temp._ambiguous_keep_row_keys(row_key) VALUES (?)",
                        ((row_key,) for row_key in keep_row_keys),
                    )
                    connection.execute(
                        f"""
                        DELETE FROM {quoted_table}
                        WHERE {expression} NOT IN (
                            SELECT row_key FROM temp._ambiguous_keep_row_keys
                        )
                        """
                    )
                else:
                    temp_columns = [
                        f"row_key_{index}" for index in range(len(expressions))
                    ]
                    quoted_temp_columns = [
                        quote_identifier(column) for column in temp_columns
                    ]
                    connection.execute(
                        "CREATE TEMP TABLE _ambiguous_keep_row_keys ("
                        + ", ".join(quoted_temp_columns)
                        + ")"
                    )
                    placeholders = ", ".join("?" for _ in expressions)
                    connection.executemany(
                        "INSERT INTO temp._ambiguous_keep_row_keys ("
                        + ", ".join(quoted_temp_columns)
                        + f") VALUES ({placeholders})",
                        (
                            (row_key,) if len(expressions) == 1 else row_key
                            for row_key in keep_row_keys
                        ),
                    )
                    connection.execute(
                        "CREATE INDEX temp._ambiguous_keep_row_keys_index ON "
                        "_ambiguous_keep_row_keys ("
                        + ", ".join(quoted_temp_columns)
                        + ")"
                    )
                    comparisons = " AND ".join(
                        f"kept.{temp_column} IS {quoted_table}.{expression}"
                        for temp_column, expression in zip(
                            quoted_temp_columns,
                            expressions,
                        )
                    )
                    connection.execute(
                        f"""
                        DELETE FROM {quoted_table}
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM temp._ambiguous_keep_row_keys AS kept
                            WHERE {comparisons}
                        )
                        """
                    )

                connection.execute("DROP TABLE temp._ambiguous_keep_row_keys")
                table_counts[table_name] = len(keep_row_keys)

            connection.commit()
            connection.execute("PRAGMA foreign_keys = ON")
            try:
                foreign_key_check_rows = [
                    tuple(row)
                    for row in connection.execute("PRAGMA foreign_key_check").fetchall()
                ]
            except sqlite3.Error as error:
                foreign_key_check_error = str(error)
        finally:
            connection.close()

        return table_counts, foreign_key_check_rows, foreign_key_check_error

    def create_filtered_clone(source_sqlite_path, clone_sqlite_path, db_schema, split_info, clone_rule):
        shutil.copy2(source_sqlite_path, clone_sqlite_path)

        (
            allowed_row_keys,
            original_counts,
            selected_table_row_count,
            row_key_specs,
        ) = build_allowed_row_keys(source_sqlite_path, db_schema, split_info, clone_rule)
        table_counts, foreign_key_check_rows, foreign_key_check_error = apply_allowed_row_keys(
            clone_sqlite_path,
            db_schema.get("table_names_original", []),
            allowed_row_keys,
            original_counts,
            row_key_specs,
        )

        return {
            "split_kind": clone_rule["kind"],
            "split_keep_values": clone_rule["keep_values"],
            "split_excluded_values": clone_rule["excluded_values"],
            "selected_table_row_count": selected_table_row_count,
            "table_counts": table_counts,
            "foreign_key_check": [
                list(row)
                for row in foreign_key_check_rows[:20]
            ],
            "foreign_key_check_count": len(foreign_key_check_rows),
            "foreign_key_check_error": foreign_key_check_error,
        }

    def execute_sql(sqlite_path, sql_query):
        connection = sqlite3.connect(sqlite_path)
        try:
            rows = connection.execute(sql_query).fetchall()
            return {
                "ok": True,
                "rows": [
                    [json_safe_value(value) for value in row]
                    for row in rows
                ],
                "error": None,
            }
        except sqlite3.Error as error:
            return {
                "ok": False,
                "rows": [],
                "error": str(error),
            }
        finally:
            connection.close()

    def query_is_order_sensitive(sql_query):
        try:
            parsed_query = sqlglot.parse_one(sql_query, read="sqlite")
        except Exception:
            return True

        return parsed_query.find(sqlglot.exp.Order) is not None

    def result_signature(rows):
        return Counter(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
            for row in rows
        )

    def results_match(original_result, clone_result, order_sensitive):
        if not original_result["ok"] or not clone_result["ok"]:
            return False

        original_rows = original_result["rows"]
        clone_rows = clone_result["rows"]

        if order_sensitive:
            return clone_rows == original_rows

        return result_signature(clone_rows) == result_signature(original_rows)

    def sample_has_explicit_split_reference(sample, db_schema, split_info):
        sql_query = sample[sql_field_name(sample)]
        refs, parse_error = extract_sql_value_refs(sql_query, db_schema)

        if parse_error:
            return (
                False,
                "skipped_sql_parse_error_for_split_reference",
                {"error": parse_error},
            )

        split_key = split_info["candidate_key"]
        valid_values = split_info["normalized_value_set"]
        invalid_values = []

        for ref in refs:
            if ref["key"] != split_key:
                continue

            normalized_value = ref["normalized_value"]
            if normalized_value in valid_values:
                return True, None, None

            invalid_values.append(normalized_value)

        if invalid_values:
            return (
                False,
                "skipped_split_reference_value_not_in_column",
                {"normalized_values": invalid_values[:10]},
            )

        return False, "skipped_no_explicit_split_reference", None

    def assign_sample_to_clone(sample, source_sqlite_path, clone_paths):
        sql_query = sample[sql_field_name(sample)]
        order_sensitive = query_is_order_sensitive(sql_query)
        original_result = execute_sql(source_sqlite_path, sql_query)
        clone_results = {
            clone_number: execute_sql(clone_path, sql_query)
            for clone_number, clone_path in sorted(clone_paths.items())
        }

        # A clone is valid for this sample only if it preserves the original answer.
        matches_original = {
            clone_number: results_match(original_result, clone_result, order_sensitive)
            for clone_number, clone_result in clone_results.items()
        }
        matching_clones = [
            clone_number
            for clone_number, matches in matches_original.items()
            if matches
        ]

        if len(matching_clones) == 1:
            clone_number = matching_clones[0]
            return clone_number, f"clone_{clone_number}_matches_original", None

        note = {
            "original_ok": original_result["ok"],
            "original_error": original_result["error"],
            "original_row_count": len(original_result["rows"]),
            "order_sensitive": order_sensitive,
            "matching_clones": matching_clones,
            "clones": {
                str(clone_number): {
                    "ok": clone_result["ok"],
                    "error": clone_result["error"],
                    "row_count": len(clone_result["rows"]),
                    "matches_original": matches_original[clone_number],
                }
                for clone_number, clone_result in clone_results.items()
            },
        }

        if matching_clones:
            return None, "skipped_multiple_clones_match_original", note

        return None, "skipped_no_clone_matches_original", note

    dataset_dir = resolve_dataset_dir(dataset_path)
    tables_path = dataset_dir / "dev_tables.json"
    dev_path = dataset_dir / "dev.json"

    if output_path is None:
        output_dir = dataset_dir.with_name(f"{dataset_dir.name}-ambiguous")
    else:
        output_dir = Path(output_path)
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_dir}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(output_dir)

    with open(tables_path, "r", encoding="utf-8") as file:
        schemas = json.load(file)

    with open(dev_path, "r", encoding="utf-8") as file:
        dev_samples = json.load(file)

    samples_by_db = defaultdict(list)
    for sample in dev_samples:
        samples_by_db[sample.get("db_id")].append(sample)

    output_databases_dir = output_dir / "dev_databases"
    output_databases_dir.mkdir(parents=True, exist_ok=False)

    output_schemas = []
    created_databases = {}
    metadata = {
        "source_dataset": str(dataset_dir),
        "output_dataset": str(output_dir),
        "distinct_ratio_threshold": distinct_ratio_threshold,
        "min_split_value_chars": min_split_value_chars,
        "min_split_distinct_values": min_split_distinct_values,
        "selection_method": (
            "low-cardinality normalized text columns ranked by SQL equality/IN "
            "literal matches whose normalized value length is greater than 3 "
            "and whose usable distinct value count is at least 3. A selected "
            "column must also have at least one usable normalized value mentioned "
            "in a normalized natural-language question for the same database"
        ),
        "split_policy": (
            "Duplicated databases are ordered by db_id. Their primary, secondary, "
            "and remaining value groups are assigned cyclically to suffixes _1, "
            "_2, and _3 using rotation offsets 0, 1, and 2. The rotation only "
            "relabels the groups; child rows referencing removed parent rows are "
            "removed recursively."
        ),
        "sample_assignment_policy": (
            "Each sample must contain an explicit SQL equality/IN reference to "
            "the selected split column with a value present in that column and "
            "longer than 3 normalized characters. Then the SQL query is executed "
            "on the original database and on all clones. A sample is written to "
            "dev.json only when exactly one clone returns the same result as the "
            "original query. Samples without an explicit split reference, with "
            "only short split values, where multiple clones match, or where no "
            "clone matches, are skipped."
        ),
        "databases": [],
        "dev_json": {},
    }

    database_plans = []
    for db_schema in schemas:
        db_id = db_schema.get("db_id")
        sqlite_path = get_sqlite_path(dataset_dir, db_id)
        if not sqlite_path:
            database_plans.append(
                (db_schema, None, None, "sqlite_not_found")
            )
            continue

        split_info, skip_reason = select_split_column(
            sqlite_path,
            db_schema,
            samples_by_db.get(db_id, []),
        )
        database_plans.append(
            (db_schema, sqlite_path, split_info, skip_reason)
        )

    eligible_db_ids = sorted(
        db_schema.get("db_id")
        for db_schema, _, split_info, _ in database_plans
        if split_info
    )
    rotation_by_db = {
        db_id: index % 3
        for index, db_id in enumerate(eligible_db_ids)
    }

    for db_schema, sqlite_path, split_info, skip_reason in database_plans:
        db_id = db_schema.get("db_id")
        db_metadata = {
            "db_id": db_id,
            "status": "skipped",
        }

        if not sqlite_path:
            db_metadata["reason"] = skip_reason
            metadata["databases"].append(db_metadata)
            continue

        if not split_info:
            db_metadata["reason"] = skip_reason
            metadata["databases"].append(db_metadata)
            continue

        clone_paths = {}
        clone_stats = {}
        rotation_offset = rotation_by_db[db_id]
        clone_split_rules = build_clone_split_rules(
            split_info,
            rotation_offset,
        )
        for clone_number, clone_rule in clone_split_rules.items():
            clone_db_id = f"{db_id}_{clone_number}"
            clone_dir = output_databases_dir / clone_db_id
            clone_dir.mkdir(parents=True, exist_ok=False)
            clone_sqlite_path = clone_dir / f"{clone_db_id}.sqlite"
            clone_paths[clone_number] = clone_sqlite_path
            clone_stats[clone_number] = create_filtered_clone(
                sqlite_path,
                clone_sqlite_path,
                db_schema,
                split_info,
                clone_rule,
            )

            cloned_schema = copy.deepcopy(db_schema)
            cloned_schema["db_id"] = clone_db_id
            output_schemas.append(cloned_schema)

        db_metadata.update(
            {
                "status": "created",
                "source_sqlite": str(sqlite_path),
                "split_rotation_offset": rotation_offset,
                "selected_table": split_info["table_name"],
                "selected_column": split_info["column_name"],
                "row_count": split_info["row_count"],
                "raw_distinct_count": split_info["raw_distinct_count"],
                "distinct_ratio": split_info["distinct_ratio"],
                "min_split_value_chars": split_info["min_split_value_chars"],
                "min_split_distinct_values": split_info["min_split_distinct_values"],
                "usable_distinct_count": split_info["usable_distinct_count"],
                "sql_sample_hits": split_info["sql_sample_hits"],
                "sql_total_mentions": split_info["sql_total_mentions"],
                "sql_value_mentions": split_info["sql_value_mentions"],
                "question_sample_hits": split_info["question_sample_hits"],
                "question_value_mentions": split_info["question_value_mentions"],
                "primary_split_value": split_info["primary_split_value"],
                "primary_split_raw_values": split_info["primary_split_raw_values"],
                "secondary_split_value": split_info["secondary_split_value"],
                "secondary_split_raw_values": split_info["secondary_split_raw_values"],
                "remaining_split_value_count": split_info["remaining_split_value_count"],
                "remaining_split_values_preview": split_info[
                    "remaining_split_values_preview"
                ],
                "top_db_values": split_info["top_db_values"],
                "parse_errors": split_info["parse_errors"],
                "clones": {
                    f"{db_id}_{clone_number}": {
                        "sqlite_path": str(clone_paths[clone_number]),
                        **clone_stats[clone_number],
                    }
                    for clone_number in clone_split_rules
                },
                "dev_assignment_counts": {},
                "dev_skipped_counts": {},
            }
        )
        metadata["databases"].append(db_metadata)
        created_databases[db_id] = {
            "schema": db_schema,
            "split_info": split_info,
            "source_sqlite_path": sqlite_path,
            "clone_paths": clone_paths,
            "metadata": db_metadata,
        }

    output_dev_samples = []
    assignment_counts = Counter()
    skipped_assignment_counts = Counter()
    skipped_samples = []
    assignment_notes = []

    for sample in dev_samples:
        db_id = sample.get("db_id")

        if db_id not in created_databases:
            skipped_samples.append(
                {
                    "question_id": sample.get("question_id"),
                    "db_id": db_id,
                    "reason": "db_not_duplicated",
                }
            )
            continue

        db_info = created_databases[db_id]
        has_split_reference, reference_skip_reason, reference_note = (
            sample_has_explicit_split_reference(
                sample,
                db_info["schema"],
                db_info["split_info"],
            )
        )
        if not has_split_reference:
            skipped_assignment_counts[reference_skip_reason] += 1
            db_info["metadata"]["dev_skipped_counts"][reference_skip_reason] = (
                db_info["metadata"]["dev_skipped_counts"].get(
                    reference_skip_reason,
                    0,
                )
                + 1
            )
            skipped_sample = {
                "question_id": sample.get("question_id"),
                "db_id": db_id,
                "reason": reference_skip_reason,
            }
            if reference_note:
                skipped_sample["note"] = reference_note
            skipped_samples.append(skipped_sample)
            continue

        clone_number, assignment_method, note = assign_sample_to_clone(
            sample,
            db_info["source_sqlite_path"],
            db_info["clone_paths"],
        )

        if clone_number is None:
            skipped_assignment_counts[assignment_method] += 1
            db_info["metadata"]["dev_skipped_counts"][assignment_method] = (
                db_info["metadata"]["dev_skipped_counts"].get(assignment_method, 0) + 1
            )
            skipped_sample = {
                "question_id": sample.get("question_id"),
                "db_id": db_id,
                "reason": assignment_method,
            }
            if note:
                skipped_sample["note"] = note
            skipped_samples.append(skipped_sample)
            continue

        updated_sample = copy.deepcopy(sample)
        updated_sample["db_id"] = f"{db_id}_{clone_number}"
        output_dev_samples.append(updated_sample)
        assignment_counts[assignment_method] += 1
        db_info["metadata"]["dev_assignment_counts"][assignment_method] = (
            db_info["metadata"]["dev_assignment_counts"].get(assignment_method, 0) + 1
        )

        if note:
            assignment_notes.append(
                {
                    "question_id": sample.get("question_id"),
                    "db_id": db_id,
                    "assigned_db_id": updated_sample["db_id"],
                    "method": assignment_method,
                    "note": note,
                }
            )

    metadata["dev_json"] = {
        "samples_read": len(dev_samples),
        "samples_written": len(output_dev_samples),
        "samples_skipped": len(skipped_samples),
        "assignment_counts": dict(assignment_counts),
        "skipped_assignment_counts": dict(skipped_assignment_counts),
        "skipped_samples": skipped_samples[:100],
        "assignment_notes": assignment_notes[:100],
    }

    with open(output_dir / "dev_tables.json", "w", encoding="utf-8") as file:
        json.dump(output_schemas, file, indent=2, ensure_ascii=False)

    with open(output_dir / "dev.json", "w", encoding="utf-8") as file:
        json.dump(output_dev_samples, file, indent=2, ensure_ascii=False)

    with open(output_dir / "ambiguity_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(
        f"{GREEN}Ambiguous benchmark saved: {output_dir}{RESET} "
        f"({len(created_databases)} databases, {len(output_dev_samples)} samples)"
    )
    return str(output_dir)


if __name__ == '__main__':

    "Used to call utility functions."

    # file_path = paths.DATASETS.ARCHER_ambiguous.value
    # print_ambiguous_distribution(file_path)

    #file_path = paths.DATASETS.SPIDERdev1.value + 'dev_tables.json'
    #print_DB_table_att(file_path)

    #file_path = paths.DATASETS.BIRDdev.value + 'dev.json'
    #print_sql_queries(file_path)
    #print(get_sql_table_names("SELECT product_name, order_date FROM marketing.orders WHERE status = 'shipped';"))

    # questions_path = paths.DATASETS.SPIDERtrain.value + 'dev.json'
    # tables_path = paths.DATASETS.SPIDERtrain.value + 'dev_tables.json'
    # remove_unused_databases(questions_path, tables_path)

    #filename = paths.DATASETS.SPIDERdev1.value + 'dev_tables.json'
    #dict = create_table_name_mapping(filename)
    #print(dict)

    # file_name = paths.DATASETS.BIRDdev.value + 'dev_tables.json'
    # database_schemas = create_db_original_schema_dictionary(file_name)
    # print(database_schemas['formula_1'])

    # file_name = paths.DATASETS.BIRDdev.value + 'dev_tables.json'
    # dict = create_attribute_mapping(file_name)
    # print(dict['debit_card_specializing']['year and month']['Customer ID'])
    
    # file_name = paths.DATASETS.SPIDERdev1.value + "dev_tables.json"
    # print_DB_table_att(file_name)
# ===================== Doc & BM25 text ================================================ #
    # dataset_path = paths.DATASETS.SPIDERdev1.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # dataset_path = paths.DATASETS.BIRDdev.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # dataset_path = paths.DATASETS.BIRDtrain.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # dataset_path = paths.DATASETS.BEAVER.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # dataset_path = paths.DATASETS.ARCHER.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # dataset_path = paths.DATASETS.SPIDERtrain.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)
# ====================================================================================== #

# ===================== AMBIGUOUS ====================================================== #

    # SPIDER
    # dataset_path = paths.DATASETS.SPIDERdev1.value
    # create_ambiguous_benchmark(dataset_path, overwrite=True)
    # dataset_path = paths.DATASETS.SPIDERdev1_ambiguous.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # BIRD
    # dataset_path = paths.DATASETS.BIRDdev.value
    # create_ambiguous_benchmark(dataset_path, overwrite=True)
    # dataset_path = paths.DATASETS.BIRDdev_ambiguous.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # BIRDtrain
    # dataset_path = paths.DATASETS.BIRDtrain.value
    # create_ambiguous_benchmark(dataset_path, overwrite=True)
    # dataset_path = paths.DATASETS.BIRDtrain_ambiguous.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # ARCHER
    # dataset_path = paths.DATASETS.ARCHER.value
    # create_ambiguous_benchmark(dataset_path, overwrite=True)
    # dataset_path = paths.DATASETS.ARCHER_ambiguous.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # BEAVER
    # dataset_path = paths.DATASETS.BEAVER.value
    # create_ambiguous_benchmark(dataset_path, overwrite=True)
    # dataset_path = paths.DATASETS.BEAVER_ambiguous.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # SPIDERtrain
    # dataset_path = paths.DATASETS.SPIDERtrain.value
    # create_ambiguous_benchmark(dataset_path, overwrite=True)
    # dataset_path = paths.DATASETS.SPIDERtrain_ambiguous.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

    # SQALE3
    # dataset_path = paths.DATASETS.SQALE3.value
    # create_ambiguous_benchmark(dataset_path, overwrite=True)
    # dataset_path = paths.DATASETS.SQALE3_ambiguous.value
    # create_benchmark_doc(dataset_path)
    # add_bm25_text_to_benchmark_doc(dataset_path)

# ====================================================================================== #
