"""Builds datasets/BEAVER/ from the BEAVER benchmark, in this project's layout.

Produces the standard files:

    dev.json                         question_id, db_id, question, evidence, SQL, difficulty
    dev_tables.json                  Spider-style schema, from the dump's own DDL
    dev_databases/<db>/<db>.sqlite   the three databases, fully populated

doc.json is deliberately not produced. Generate it afterwards like the other
datasets:

    create_benchmark_doc('../datasets/BEAVER/')
    add_bm25_text_to_benchmark_doc('../datasets/BEAVER/')

Both source repositories are gated on HuggingFace: run `hf auth login` once
with an account that has been granted access. Everything downloaded is removed
once the dataset is built.
"""

import json
import re
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "datasets" / "BEAVER"
WORK_DIR = ROOT / "datasets" / "_beaver_work"
DATABASES = ("dw", "neutron", "nova")

# MySQL base type -> (SQLite declared type, dev_tables.json type)
TYPES = {
    "tinyint": ("INTEGER", "integer"), "smallint": ("INTEGER", "integer"),
    "mediumint": ("INTEGER", "integer"), "int": ("INTEGER", "integer"),
    "integer": ("INTEGER", "integer"), "bigint": ("INTEGER", "integer"),
    "decimal": ("REAL", "real"), "numeric": ("REAL", "real"),
    "float": ("REAL", "real"), "double": ("REAL", "real"),
    "date": ("TEXT", "date"), "datetime": ("TEXT", "datetime"),
    "timestamp": ("TEXT", "datetime"),
}
TEXT_TYPE = ("TEXT", "text")

CREATE_RE = re.compile(r"^CREATE TABLE `([^`]+)`")
INSERT_RE = re.compile(r"^INSERT INTO `([^`]+)` VALUES ")
COLUMN_RE = re.compile(r"^`([^`]+)`\s+(\w+)")
PRIMARY_RE = re.compile(r"^PRIMARY KEY \((.+?)\)")
FOREIGN_RE = re.compile(r"FOREIGN KEY \((.+?)\) REFERENCES `([^`]+)` \((.+?)\)")
BACKTICK_RE = re.compile(r"`([^`]+)`")

# A quoted string (backslash escapes included), a delimiter, or a bare literal.
VALUE_RE = re.compile(r"'((?:[^'\\]|\\.)*)'|([(),])|([^,()']+)")
ESCAPE_RE = re.compile(r"\\(.)")
ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}


def parse_values(payload):
    """Parse a mysqldump `VALUES (...),(...)` payload into a list of row tuples.

    Quoting decides the type: quoted literals stay strings, bare ones are NULL
    or numbers. Values are returned as Python objects and bound as parameters,
    so no SQL escaping is ever re-emitted.
    """
    rows, row = [], None

    for match in VALUE_RE.finditer(payload):
        string, delimiter, bare = match.group(1), match.group(2), match.group(3)

        if string is not None:
            row.append(ESCAPE_RE.sub(lambda m: ESCAPES.get(m.group(1), m.group(1)), string))
        elif delimiter == "(":
            row = []
        elif delimiter == ")":
            rows.append(row)
            row = None
        elif delimiter != "," and row is not None:
            literal = bare.strip()
            if not literal:
                continue
            if literal.upper() == "NULL":
                row.append(None)
                continue
            try:
                row.append(int(literal))
            except ValueError:
                try:
                    row.append(float(literal))
                except ValueError:
                    row.append(literal)

    return rows


def create_table(connection, table):
    definitions = [
        f'"{name}" {TYPES.get(kind, TEXT_TYPE)[0]}'
        for name, kind in table["columns"]
    ]
    if table["primary_key"]:
        keys = ", ".join(f'"{column}"' for column in table["primary_key"])
        definitions.append(f"PRIMARY KEY ({keys})")

    connection.execute(f'CREATE TABLE "{table["name"]}" ({", ".join(definitions)})')


def read_dump(archive, name, connection):
    """Stream one MySQL dump: create the tables, insert the rows, return the DDL.

    Identifiers are upper-cased so that they match the gold SQL and the table
    annotations BEAVER publishes (`dw` is dumped upper-case, the other two
    lower-case).
    """
    tables, columns_of, pending, inserted = [], {}, None, 0

    with zipfile.ZipFile(archive).open(f"beaver_db/{name}.sql") as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8", "replace")

            if pending is None:
                match = CREATE_RE.match(line)
                if match:
                    pending = {"name": match.group(1).upper(), "columns": [],
                               "primary_key": [], "foreign_keys": []}
                    continue

                match = INSERT_RE.match(line)
                if match:
                    table_name = match.group(1).upper()
                    payload = line[line.index(" VALUES ") + 8:].rstrip().rstrip(";")
                    rows = parse_values(payload)
                    marks = ", ".join("?" * len(columns_of[table_name]))
                    connection.executemany(
                        f'INSERT INTO "{table_name}" VALUES ({marks})', rows
                    )
                    inserted += len(rows)
                continue

            entry = line.strip().rstrip(",")

            if entry.startswith(")"):
                create_table(connection, pending)
                columns_of[pending["name"]] = pending["columns"]
                tables.append(pending)
                pending = None
                continue

            match = COLUMN_RE.match(entry)
            if match:
                pending["columns"].append((match.group(1).upper(), match.group(2).lower()))
                continue

            match = PRIMARY_RE.match(entry)
            if match:
                pending["primary_key"] = [
                    column.upper() for column in BACKTICK_RE.findall(match.group(1))
                ]
                continue

            match = FOREIGN_RE.search(entry)
            if match:
                pending["foreign_keys"].append((
                    [column.upper() for column in BACKTICK_RE.findall(match.group(1))],
                    match.group(2).upper(),
                    [column.upper() for column in BACKTICK_RE.findall(match.group(3))],
                ))

    connection.commit()
    print(f"  {name}: {len(tables)} tables, {inserted} rows")
    return tables


def build_schema(name, tables):
    """Assemble one dev_tables.json entry.

    table_names / column_names keep the original identifiers: BEAVER publishes
    no natural-language names, and normalize_and_stem_text already splits on
    underscores, so deriving them would add nothing.
    """
    schema = {
        "db_id": name,
        "table_names_original": [table["name"] for table in tables],
        "table_names": [table["name"] for table in tables],
        "column_names_original": [[-1, "*"]],
        "column_names": [[-1, "*"]],
        "column_types": ["text"],
        "primary_keys": [],
        "foreign_keys": [],
    }

    position_of = {}
    for table_index, table in enumerate(tables):
        for column, kind in table["columns"]:
            position_of[(table["name"], column)] = len(schema["column_names_original"])
            schema["column_names_original"].append([table_index, column])
            schema["column_names"].append([table_index, column])
            schema["column_types"].append(TYPES.get(kind, TEXT_TYPE)[1])

    for table in tables:
        keys = [position_of[(table["name"], column)] for column in table["primary_key"]]
        if keys:
            schema["primary_keys"].append(keys[0] if len(keys) == 1 else keys)

        for sources, target_table, targets in table["foreign_keys"]:
            for source, target in zip(sources, targets):
                pair = [position_of.get((table["name"], source)),
                        position_of.get((target_table, target))]
                if None not in pair and pair not in schema["foreign_keys"]:
                    schema["foreign_keys"].append(pair)

    return schema


def build_dev(query_dir):
    samples = []

    for path in sorted(Path(query_dir).rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            samples.append({
                "question_id": len(samples),
                "db_id": row["db"],
                "question": row["question"],
                "evidence": " ".join(json.loads(row["domain_knowledge"])),
                "SQL": row["sql"],
                "difficulty": row["category"],
            })

    return samples


def write(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  {path.relative_to(ROOT)} ({len(payload)} entries)")


def main():
    print("downloading sources...")
    query_dir = snapshot_download(
        "beaverbench/beaver-query", repo_type="dataset",
        local_dir=WORK_DIR / "beaver-query", allow_patterns=["*.parquet"],
    )
    archive = hf_hub_download(
        "beaverbench/beaver-table", "beaver_db.zip",
        repo_type="dataset", local_dir=WORK_DIR,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("writing dataset...")
    write(OUTPUT_DIR / "dev.json", build_dev(query_dir))

    schemas = []
    for name in DATABASES:
        directory = OUTPUT_DIR / "dev_databases" / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.sqlite"
        path.unlink(missing_ok=True)

        connection = sqlite3.connect(path)
        try:
            schemas.append(build_schema(name, read_dump(archive, name, connection)))
        finally:
            connection.close()

    write(OUTPUT_DIR / "dev_tables.json", schemas)

    shutil.rmtree(WORK_DIR)
    print(f"removed {WORK_DIR.relative_to(ROOT)}")
    print("\nnext: create_benchmark_doc('../datasets/BEAVER/') "
          "then add_bm25_text_to_benchmark_doc('../datasets/BEAVER/')")


if __name__ == "__main__":
    main()
