"""Generate dev_tables.json from dev_databases/<db_id>/<db_id>.sqlite.

Uses only SQLite metadata (SQLite >= 3.37) and the standard library. Original names are copied
verbatim; friendly names replace underscores with spaces. Types are the declared types in
lowercase (empty if absent).
The eight fields follow BIRD's layout, including grouped composite primary keys.
SQLite internal/shadow tables and views are excluded; generated columns are kept.
No annotations are inferred from row values.

Database IDs are sorted; tables follow sqlite_schema.rowid and columns their cid.
Primary-key indices follow column order; implicit foreign keys use the actual
primary-key constraint order. Foreign keys follow SQLite's (id, seq) order, with
duplicate column pairs removed. Unresolved foreign-key constraints fail by
default; callers handling known noisy metadata may skip them and receive a
deterministic summary. Existing output files are never overwritten.

Edit DATASET_PATH and OUTPUT_PATH below, then run from the src directory:
    python generate_dev_tables.py
Relative paths from paths.DATASETS are resolved against the working directory.
"""

from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import paths



DATASET_PATH = paths.DATASETS.BEAVER.value
OUTPUT_PATH = None  # Default: Path(DATASET_PATH) / "dev_tables.json"; or set another Path.


def identifier_key(name):
    # SQLite identifiers are case-insensitive for ASCII only.
    return name.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))


def read_schema(database_path, skip_unresolved_foreign_keys=False,
                unresolved_foreign_keys=None):
    """Read one database without writing to it or consulting an existing JSON."""
    database_path = Path(database_path).resolve()
    with closing(sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)) as connection:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_schema "
            "WHERE type = 'table' AND name NOT GLOB 'sqlite_*' "
            "AND name NOT IN (SELECT name FROM pragma_table_list "
            "WHERE schema = 'main' AND type = 'shadow') ORDER BY rowid"
        )]
        if not tables:
            raise ValueError(f"No user tables in {database_path}")
        schema = {
            "db_id": database_path.stem,
            "table_names_original": tables,
            "table_names": [name.replace("_", " ") for name in tables],
            "column_names_original": [[-1, "*"]],
            "column_names": [[-1, "*"]],
            "column_types": ["text"],
            "primary_keys": [],
            "foreign_keys": [],
        }
        positions, primary_columns = {}, {}
        for table_index, table in enumerate(tables):
            columns = list(connection.execute(
                'SELECT name, type, pk FROM pragma_table_xinfo(?) WHERE hidden != 1 ORDER BY cid',
                (table,),
            ))
            primary_columns[identifier_key(table)] = [
                name for name, _, pk in sorted(columns, key=lambda column: column[2]) if pk
            ]
            keys = []
            for name, declared_type, pk in columns:
                position = len(schema["column_names_original"])
                positions[(identifier_key(table), identifier_key(name))] = position
                schema["column_names_original"].append([table_index, name])
                schema["column_names"].append([table_index, name.replace("_", " ")])
                schema["column_types"].append(declared_type.lower())
                if pk:
                    keys.append(position)
            if keys:
                schema["primary_keys"].append(keys[0] if len(keys) == 1 else keys)

        for table in tables:
            foreign_keys = list(connection.execute(
                'SELECT id, seq, "table", "from", "to" FROM pragma_foreign_key_list(?) ORDER BY id, seq',
                (table,),
            ))
            constraints = {}
            for foreign_key in foreign_keys:
                constraints.setdefault(foreign_key[0], []).append(foreign_key)

            for constraint in constraints.values():
                pairs = []
                error = None
                for _, sequence, target_table, source, target in constraint:
                    source_position = (identifier_key(table), identifier_key(source))
                    if source_position not in positions:
                        error = "missing source column"
                        break

                    target_table_key = identifier_key(target_table)
                    if target_table_key not in primary_columns:
                        error = "missing target table"
                        break

                    if target is None:
                        primary = primary_columns[target_table_key]
                        if len(primary) != len(constraint) or sequence >= len(primary):
                            error = "invalid implicit primary-key reference"
                            break
                        target = primary[sequence]

                    target_position = (target_table_key, identifier_key(target))
                    if target_position not in positions:
                        error = "missing target column"
                        break
                    pairs.append([positions[source_position], positions[target_position]])

                if error is not None:
                    if skip_unresolved_foreign_keys:
                        if unresolved_foreign_keys is not None:
                            unresolved_foreign_keys[error] += 1
                        continue
                    raise ValueError(
                        f"{database_path}: unresolved foreign key {table}.{source} -> "
                        f"{target_table}.{target}: {error}"
                    )

                for pair in pairs:
                    if pair not in schema["foreign_keys"]:
                        schema["foreign_keys"].append(pair)
        return schema


def generate_dev_tables(dataset_path, output_path=None,
                        skip_unresolved_foreign_keys=False):
    """Create the JSON for every DB folder, optionally skipping unresolved FKs."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path) if output_path is not None else dataset_path / "dev_tables.json"
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}. Choose a different output path.")
    directories = sorted(
        (path for path in (dataset_path / "dev_databases").iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not directories:
        raise ValueError(f"No database directories in {dataset_path / 'dev_databases'}")
    schemas = []
    unresolved_foreign_keys = Counter()
    affected_databases = 0
    for directory in directories:
        database_stats = Counter()
        schemas.append(read_schema(
            directory / f"{directory.name}.sqlite",
            skip_unresolved_foreign_keys=skip_unresolved_foreign_keys,
            unresolved_foreign_keys=database_stats,
        ))
        if database_stats:
            affected_databases += 1
            unresolved_foreign_keys.update(database_stats)

    payload = json.dumps(schemas, ensure_ascii=False, indent=2) + "\n"
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
    if skip_unresolved_foreign_keys:
        print("Foreign-key validation:")
        print(f"  unresolved constraints skipped: {sum(unresolved_foreign_keys.values())}")
        print(f"  affected databases: {affected_databases}")
        for reason in (
            "missing target table",
            "missing target column",
            "missing source column",
            "invalid implicit primary-key reference",
        ):
            print(f"  {reason}: {unresolved_foreign_keys[reason]}")
    return output_path


def main():
    try:
        output = generate_dev_tables(DATASET_PATH, OUTPUT_PATH)
    except (OSError, sqlite3.Error, ValueError) as error:
        raise SystemExit(str(error)) from None
    print(f"Created {output}")


if __name__ == "__main__":
    main()
