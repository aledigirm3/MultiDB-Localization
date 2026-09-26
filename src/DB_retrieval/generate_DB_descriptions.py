import json
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import paths
from data_manipulation import get_DB_table_att
from llm import query_bedrock
from ansi_colors import *


PROMPT_PATH = "./DB_description_prompt.txt"
DEFAULT_MAX_ATTEMPTS = 3
REPLACE_RETRIES = 7
REPLACE_RETRY_DELAY_SECONDS = 0.5


def _write_json_atomic(path: Path, payload: list) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
    for attempt in range(REPLACE_RETRIES + 1):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES:
                raise
            time.sleep(REPLACE_RETRY_DELAY_SECONDS)


def _load_descriptions(path: Path, expected_names: set) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Invalid descriptions file: '{path}'.")

    descriptions = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid description entry in '{path}'.")
        name = item.get("name")
        description = item.get("description")
        if name not in expected_names or name in descriptions:
            raise ValueError(f"Unexpected or duplicate database '{name}' in '{path}'.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Invalid description for database '{name}' in '{path}'.")
        descriptions[name] = {
            "name": name,
            "description": description.strip(),
        }
    return descriptions


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


def generate_DB_descriptions(
    file_path: str,
    output_file_name: str,
    replicate_ambiguous_clones: bool = False,
) -> None:
    """Generates descriptions and resumes from an atomic per-output checkpoint."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
        system_prompt = file.read()

    schemas = get_DB_table_att(file_path)
    clone_groups = {}
    if replicate_ambiguous_clones:
        for schema in schemas:
            database_name = schema["database_name"]
            base_name, separator, suffix = database_name.rpartition("_")
            if not separator or not base_name or suffix not in {"1", "2", "3"}:
                raise ValueError(f"Invalid ambiguous database name: '{database_name}'.")
            if suffix in clone_groups.setdefault(base_name, {}):
                raise ValueError(f"Duplicate ambiguous database: '{database_name}'.")
            clone_groups[base_name][suffix] = schema

        for base_name, clones in clone_groups.items():
            if set(clones) != {"1", "2", "3"}:
                raise ValueError(f"Incomplete ambiguous clone group: '{base_name}'.")
            if any(
                clones[suffix]["tables"] != clones["1"]["tables"]
                for suffix in ("2", "3")
            ):
                raise ValueError(
                    f"Different schemas in ambiguous clone group: '{base_name}'."
                )

        schemas_to_generate = [clones["1"] for clones in clone_groups.values()]
    else:
        schemas_to_generate = schemas

    output_path = Path(output_file_name)
    checkpoint_path = output_path.with_name(f"{output_path.stem}.checkpoint.json")
    expected_output_names = {schema["database_name"] for schema in schemas}
    if len(expected_output_names) != len(schemas):
        raise ValueError("Duplicate database names in the schema file.")
    expected_source_names = {
        schema["database_name"] for schema in schemas_to_generate
    }

    if output_path.exists():
        completed = _load_descriptions(output_path, expected_output_names)
        if set(completed) != expected_output_names:
            raise ValueError(f"Incomplete existing output file: '{output_path}'.")
        checkpoint_path.unlink(missing_ok=True)
        print(f"Output already complete: {output_path}")
        return

    descriptions = (
        _load_descriptions(checkpoint_path, expected_source_names)
        if checkpoint_path.exists()
        else {}
    )
    if replicate_ambiguous_clones:
        for name, item in descriptions.items():
            if name not in item["description"]:
                raise ValueError(
                    f"Checkpoint description does not mention database '{name}'."
                )
    if descriptions:
        print(
            f"Resuming from checkpoint: {len(descriptions)}/"
            f"{len(schemas_to_generate)} descriptions."
        )

    for database_schema in schemas_to_generate:
        database_name = database_schema["database_name"]
        if database_name in descriptions:
            continue

        for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
            try:
                response = query_bedrock(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(database_schema, indent=2),
                        },
                    ]
                )
                description = _parse_llm_response(response, database_name)
                if (
                    replicate_ambiguous_clones
                    and description["name"] not in description["description"]
                ):
                    raise ValueError(
                        f"Description does not mention database "
                        f"'{description['name']}'."
                    )
                break
            except Exception as error:
                print(
                    f"[{database_name}] attempt {attempt}/{DEFAULT_MAX_ATTEMPTS} "
                    f"failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt == DEFAULT_MAX_ATTEMPTS:
                    raise

        descriptions[database_name] = description
        checkpoint = [
            descriptions[schema["database_name"]]
            for schema in schemas_to_generate
            if schema["database_name"] in descriptions
        ]
        _write_json_atomic(checkpoint_path, checkpoint)

    if replicate_ambiguous_clones:
        replicated = {}
        for base_name in clone_groups:
            source_name = f"{base_name}_1"
            source_description = descriptions[source_name]["description"]
            for suffix in ("1", "2", "3"):
                target_name = f"{base_name}_{suffix}"
                replicated[target_name] = {
                    "name": target_name,
                    "description": source_description.replace(source_name, target_name),
                }
        output = [replicated[schema["database_name"]] for schema in schemas]
    else:
        output = [descriptions[schema["database_name"]] for schema in schemas]

    _write_json_atomic(output_path, output)
    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    file_path = "../" + paths.DATASETS.SQALE3_ambiguous.value + "dev_tables.json"
    output_file_name = "./DB_descriptions/SQALE3_DB_descriptions_ambiguous.json"
    print(f"{GREEN}Start generation DBs descriptions...{RESET} ({file_path})")
    generate_DB_descriptions(
        file_path,
        output_file_name,
        replicate_ambiguous_clones=True,   # Set 'True' only for ambiguous benchmark
    )
    print(f"{GREEN}DBs descriptions generated!{RESET} ({file_path})")
