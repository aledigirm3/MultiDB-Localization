import json
import os
import sys
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[1]
sys.path.append(str(SRC_DIR))

from ansi_colors import GREEN, RED, RESET
from data_manipulation import get_DB_table_att
from llm import query_bedrock
import paths


PROMPT_PATH = SCRIPT_DIR / "table_description_prompt.txt"
SCHEMA_FILE = (SRC_DIR / paths.DATASETS.ARCHER.value / "dev_tables.json").resolve()
OUTPUT_FILE = SCRIPT_DIR / "ARCHER_table_descriptions.json"
DEFAULT_MAX_ATTEMPTS = 3
MAX_OUTPUT_TOKENS = 40000


def _parse_llm_response(response: str, database_schema: dict) -> dict:
    """Parses and validates one database's table descriptions."""
    if not isinstance(response, str):
        raise ValueError("The LLM response is not text.")

    response = response.strip()
    if response.startswith("```") and response.endswith("```"):
        response = response[3:-3].strip()
        if response.lower().startswith("json"):
            response = response[4:].strip()

    result = json.loads(response)
    database_name = database_schema["database_name"]
    expected_tables = list(database_schema["tables"])

    if not isinstance(result, dict) or list(result) != [database_name]:
        received_names = list(result) if isinstance(result, dict) else result
        raise ValueError(
            f"Expected only database '{database_name}', received {received_names!r}."
        )

    table_descriptions = result[database_name]
    if not isinstance(table_descriptions, dict):
        raise ValueError(f"Database '{database_name}' does not contain a JSON object.")

    missing_tables = [table for table in expected_tables if table not in table_descriptions]
    unexpected_tables = [table for table in table_descriptions if table not in expected_tables]
    if missing_tables or unexpected_tables:
        raise ValueError(
            f"Invalid tables for database '{database_name}': "
            f"missing={missing_tables}, unexpected={unexpected_tables}."
        )

    validated_descriptions = {}
    for table in expected_tables:
        description = table_descriptions[table]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Invalid description for table '{table}' in database '{database_name}'."
            )
        validated_descriptions[table] = description.strip()

    return validated_descriptions


def _write_checkpoint(descriptions: dict, output_file: Path) -> None:
    """Atomically saves all descriptions generated so far."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(output_file.name + ".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(descriptions, file, indent=4, ensure_ascii=False)
    os.replace(temporary_file, output_file)


def generate_table_descriptions(
    schema_file: str,
    output_file: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    query_function: Callable = query_bedrock,
) -> list[str]:
    """Generates and validates descriptions for every database in a schema file."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    descriptions = {}
    failed_databases = []
    output_path = Path(output_file)

    for database_schema in get_DB_table_att(schema_file):
        database_name = database_schema["database_name"]
        previous_error = None

        for attempt in range(1, max_attempts + 1):
            user_content = json.dumps(database_schema, indent=2, ensure_ascii=False)
            if previous_error:
                user_content += (
                    "\n\nThe previous response was invalid: "
                    f"{previous_error}\nReturn the complete corrected JSON object."
                )

            try:
                response = query_function(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                descriptions[database_name] = _parse_llm_response(
                    response, database_schema
                )
            except Exception as error:
                previous_error = f"{type(error).__name__}: {error}"
                print(
                    f"{RED}[{database_name}] attempt {attempt}/{max_attempts} "
                    f"failed: {previous_error}{RESET}"
                )
                continue

            _write_checkpoint(descriptions, output_path)
            print(
                f"{GREEN}[{database_name}] descriptions generated "
                f"(attempt {attempt}/{max_attempts}).{RESET}"
            )
            break
        else:
            failed_databases.append(database_name)
            print(
                f"{RED}[{database_name}] skipped after "
                f"{max_attempts} failed attempts.{RESET}"
            )

    return failed_databases


def main() -> int:
    try:
        failed_databases = generate_table_descriptions(
            str(SCHEMA_FILE), str(OUTPUT_FILE), DEFAULT_MAX_ATTEMPTS
        )
    except (OSError, json.JSONDecodeError, KeyError, IndexError, ValueError) as error:
        print(f"{RED}Generation stopped: {type(error).__name__}: {error}{RESET}")
        return 1

    if failed_databases:
        print(
            f"{RED}Generation completed with errors. Failed databases: "
            f"{', '.join(failed_databases)}{RESET}"
        )
        return 1

    print(f"{GREEN}All table descriptions generated successfully.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
