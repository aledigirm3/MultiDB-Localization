"""Build ``datasets/SQALE3`` in the native layout used by this project.

The source datasets are downloaded from pinned Hugging Face revisions. The
builder performs the following deterministic pipeline:

1. keep query rows with ``execution_match == 1`` and, when ``VERIFIED_ONLY`` is
   enabled, ``pair_verified is True``;
2. create every referenced SQLite database from its raw ``Full schema`` DDL;
3. load ``Schema content`` without changing that DDL. Content columns are
   matched case-insensitively, after removing one balanced quote pair, or by a
   unique separator-insensitive match. A final conservative rule recovers
   unquoted multi-word columns that SQLite split between the column name and
   its declared type. Unmatched content fields are counted and ignored;
4. insert compatible scalar rows with ``INSERT OR IGNORE``. The source numeric
   sentinel ``NaN`` and positive overflow are clamped to Python's maximum
   finite float; negative overflow is clamped symmetrically. A source field
   absent from one row is passed as ``NULL`` rather than replaced by a DDL
   default. Rows containing structured values or producing insertion errors
   are counted and skipped. Bulk insertion is retried row by row only after a
   bulk failure;
5. verify ``PRAGMA integrity_check`` and validate every query against its
   supplied ``execution_result``. Top-level ``ORDER BY`` preserves row order;
   otherwise rows are compared as a multiset. A supplied unordered result that
   reaches the 50-row source cap is checked as a multiset subset of the full
   local result, avoiding a comparison between two arbitrary first pages.
   Floats use a fixed tolerance. Result mismatches, SQLite errors and timeouts
   are excluded from ``dev.json``;
6. call ``remove_unused_databases`` and generate ``dev_tables.json`` from the
   final SQLite metadata. Foreign-key constraints whose target cannot be
   resolved are skipped atomically and reported; valid constraints are kept.
   ``doc.json`` and LLM descriptions remain separate.

The source DDL is authoritative: the builder never creates a column merely
because it occurs in ``Schema content``. A disagreement in the declared table
count is reported but is not allowed to override the DDL. Source revisions,
comparison limits and numeric tolerances are constants below for reproducible
runs. Existing output and work directories are never overwritten. The final
directory move is retried briefly when Windows reports a transient file lock;
after a persistent failure, the completed build remains in the work directory.

Run from the ``src`` directory:
    python sqale3_build.py
"""

from collections import Counter, defaultdict
from contextlib import closing
import json
import math
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
import sqlglot
from sqlglot.errors import ParseError

from data_manipulation import remove_unused_databases
from generate_dev_tables import generate_dev_tables


VERIFIED_ONLY = True

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "datasets" / "SQALE3"
WORK_DIR = ROOT / "datasets" / "_sqale3_work"
BUILD_DIR = WORK_DIR / "benchmark"

QUERIES_REPOSITORY = "cwolff/queries_new3"
QUERIES_REVISION = "7039ddd42ca37dfee6388d02ddcb65fdf120541d"
SCHEMAS_REPOSITORY = "cwolff/schemas_new3"
SCHEMAS_REVISION = "25ec7203c9035b675cd97de5d06484f0b7bde58a"

EXECUTION_RESULT_LIMIT = 50
QUERY_TIMEOUT_SECONDS = 5.0
FLOAT_TOLERANCE = 1e-9
SQLITE_MIN_INTEGER = -(2**63)
SQLITE_MAX_INTEGER = 2**63 - 1

QUESTION_ID_RE = re.compile(r"^q_(\d+)$")
SCHEMA_ID_RE = re.compile(r"^schema_\d+$")
ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)
NON_ALPHANUMERIC_RE = re.compile(r"[^0-9a-z]+")
SQL_TYPE_RE = re.compile(
    r"^(?:"
    r"int(?:eger)?|tinyint|smallint|mediumint|bigint|unsigned\s+big\s+int|"
    r"int2|int8|character|varchar|varying\s+character|nchar|"
    r"native\s+character|nvarchar|char|text|clob|blob|"
    r"real|double(?:\s+precision)?|float|numeric|decimal|"
    r"boolean|date|datetime|timestamp|time|json"
    r")(?:\s*\([^)]*\))?$",
    re.IGNORECASE,
)
JOINED_TYPE_NAMES = (
    "doubleprecision", "unsignedbigint", "varyingcharacter", "nativecharacter",
    "mediumint", "smallint", "tinyint", "timestamp", "datetime", "character",
    "nvarchar", "varchar", "integer", "decimal", "numeric", "boolean",
    "bigint", "float", "double", "nchar", "int8", "int2", "real", "text",
    "clob", "blob", "date", "time", "json", "char", "int",
)


def write_json(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parquet_rows(directory, columns, batch_size=256):
    files = sorted(Path(directory).rglob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files found in {directory}")
    for path in files:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
                yield from batch.to_pylist()


def json_object(value, field, question_id):
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {field} JSON for {question_id}: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected an object in {field} for {question_id}")
    return parsed


def build_dev(query_directory):
    columns = [
        "questions", "evidence", "question id", "sql statament", "difficulty",
        "schema id", "execution_result", "execution_match", "pair_verified",
    ]
    samples = []
    expected_results = {}
    question_ids = set()
    database_ids = set()
    rows_read = 0
    non_matching = 0
    non_verified = 0

    for row in parquet_rows(query_directory, columns):
        rows_read += 1
        if row["execution_match"] != 1:
            non_matching += 1
            continue
        if VERIFIED_ONLY and row["pair_verified"] is not True:
            non_verified += 1
            continue

        source_question_id = row["question id"]
        match = QUESTION_ID_RE.fullmatch(source_question_id or "")
        if not match:
            raise ValueError(f"Invalid question id: {source_question_id!r}")
        question_id = int(match.group(1))
        if question_id in question_ids:
            raise ValueError(f"Duplicate question id: {source_question_id}")
        question_ids.add(question_id)

        db_id = row["schema id"]
        if not isinstance(db_id, str) or not SCHEMA_ID_RE.fullmatch(db_id):
            raise ValueError(f"Invalid schema id for {source_question_id}: {db_id!r}")

        questions = json_object(row["questions"], "questions", source_question_id)
        question = questions.get("bird_aligned")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing bird_aligned question for {source_question_id}")

        evidence_values = json_object(row["evidence"], "evidence", source_question_id)
        evidence = evidence_values.get("bird_aligned", "")
        if not isinstance(evidence, str):
            raise ValueError(f"Invalid bird_aligned evidence for {source_question_id}")

        sql = row["sql statament"]
        difficulty = row["difficulty"]
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"Missing SQL for {source_question_id}")
        if not isinstance(difficulty, str) or not difficulty.strip():
            raise ValueError(f"Missing difficulty for {source_question_id}")
        if not isinstance(row["execution_result"], str):
            raise ValueError(f"Missing execution_result for {source_question_id}")

        samples.append({
            "question_id": question_id,
            "db_id": db_id,
            "question": question.strip(),
            "evidence": evidence.strip(),
            "SQL": sql.strip(),
            "difficulty": difficulty,
        })
        expected_results[question_id] = row["execution_result"]
        database_ids.add(db_id)

    if not samples:
        raise ValueError("No query passed the selected filters")

    return samples, expected_results, database_ids, {
        "rows_read": rows_read,
        "execution_mismatch_skipped": non_matching,
        "not_verified_skipped": non_verified,
    }


def quote_identifier(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def identifier_key(identifier):
    """Return the ASCII case-insensitive key used by SQLite identifiers."""
    return str(identifier).translate(
        str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
    )


def strip_identifier_quotes(identifier):
    identifier = str(identifier)
    pairs = {'"': '"', "'": "'", "`": "`", "[": "]"}
    if len(identifier) >= 2 and pairs.get(identifier[0]) == identifier[-1]:
        return identifier[1:-1]
    return identifier


def normalized_identifier(identifier):
    return NON_ALPHANUMERIC_RE.sub("", identifier_key(identifier))


def split_declared_type(declared_type):
    """Yield prefixes accidentally parsed as part of an unquoted column type."""
    declared_type = str(declared_type or "").strip()
    for match in re.finditer(r"\s+", declared_type):
        prefix = declared_type[:match.start()].strip()
        sql_type = declared_type[match.end():].strip()
        if prefix and SQL_TYPE_RE.fullmatch(sql_type):
            yield prefix


def joined_type_alias(column_name, declared_type):
    """Recover ``name`` from a malformed DDL token such as ``nameBigInt``."""
    if str(declared_type or "").strip():
        return None
    compact_key = normalized_identifier(column_name)
    for sql_type in JOINED_TYPE_NAMES:
        if compact_key.endswith(sql_type) and len(compact_key) > len(sql_type):
            return compact_key[:-len(sql_type)]
    return None


def build_column_mapping(source_columns, actual_columns, stats):
    """Map source fields to real DDL columns, accepting only unique matches."""
    exact = {identifier_key(column[0]): column[0] for column in actual_columns}
    normalized = defaultdict(list)
    recovered = defaultdict(list)
    for name, declared_type, *_ in actual_columns:
        normalized[normalized_identifier(name)].append(name)
        for prefix in split_declared_type(declared_type):
            recovered[normalized_identifier(f"{name}{prefix}")].append(name)
        alias = joined_type_alias(name, declared_type)
        if alias:
            recovered[normalized_identifier(alias)].append(name)

    mapping = {}
    for source in source_columns:
        unquoted = strip_identifier_quotes(source)
        direct = exact.get(identifier_key(source))
        if direct is not None:
            mapping[source] = direct
            stats["columns_mapped_exactly"] += 1
            continue
        direct = exact.get(identifier_key(unquoted))
        if direct is not None:
            mapping[source] = direct
            stats["columns_mapped_after_unquoting"] += 1
            continue
        candidates = normalized.get(normalized_identifier(unquoted), [])
        if len(candidates) == 1:
            mapping[source] = candidates[0]
            stats["columns_mapped_after_normalization"] += 1
            continue
        candidates = list(dict.fromkeys(recovered.get(normalized_identifier(unquoted), [])))
        if len(candidates) == 1:
            mapping[source] = candidates[0]
            stats["columns_recovered_from_malformed_ddl"] += 1
            continue
        mapping[source] = None
        stats["unmapped_content_columns"] += 1
    return mapping


def sqlite_value(value):
    """Return a bindable, bounded SQLite scalar or raise for structured data."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return min(max(value, SQLITE_MIN_INTEGER), SQLITE_MAX_INTEGER)
    if isinstance(value, float):
        if math.isnan(value):
            return sys.float_info.max
        if math.isinf(value):
            return math.copysign(sys.float_info.max, value)
        return min(max(value, -sys.float_info.max), sys.float_info.max)
    raise ValueError(f"unsupported SQLite value: {type(value).__name__}")


def insert_group(connection, table, columns, rows, stats):
    statement = (
        f"INSERT OR IGNORE INTO {quote_identifier(table)} "
        f"({', '.join(quote_identifier(column) for column in columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})"
    )
    connection.execute("SAVEPOINT sqale3_bulk_insert")
    changes_before = connection.total_changes
    try:
        connection.executemany(statement, rows)
    except (sqlite3.Error, OverflowError):
        connection.execute("ROLLBACK TO sqale3_bulk_insert")
        connection.execute("RELEASE sqale3_bulk_insert")
    else:
        inserted = connection.total_changes - changes_before
        connection.execute("RELEASE sqale3_bulk_insert")
        stats["inserted_rows"] += inserted
        stats["constraint_ignored_rows"] += len(rows) - inserted
        return

    for values in rows:
        changes_before = connection.total_changes
        try:
            connection.execute(statement, values)
        except (sqlite3.Error, OverflowError):
            stats["insertion_error_rows"] += 1
            continue
        inserted = connection.total_changes - changes_before
        stats["inserted_rows"] += inserted
        stats["constraint_ignored_rows"] += 1 - inserted


def materialize_database(row, database_path):
    db_id = row["schema id"]
    ddl = row["Full schema"]
    try:
        contents = json.loads(row["Schema content"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Schema content for {db_id}: {error}") from error
    if not isinstance(ddl, str) or not ddl.strip():
        raise ValueError(f"Missing Full schema for {db_id}")
    if not isinstance(contents, dict):
        raise ValueError(f"Schema content must be an object for {db_id}")

    temporary_path = database_path.with_suffix(".sqlite.tmp")
    stats = Counter()
    connection = sqlite3.connect(temporary_path)
    try:
        connection.executescript("BEGIN;\n" + ddl)
        if not connection.in_transaction:
            connection.execute("BEGIN")
        table_names = [
            result[0]
            for result in connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'table' AND name NOT GLOB 'sqlite_*' "
                "AND name NOT IN (SELECT name FROM pragma_table_list "
                "WHERE schema = 'main' AND type = 'shadow')"
            )
        ]
        if row["number of tables"] != len(table_names):
            stats["declared_table_count_mismatches"] += 1

        table_lookup = {identifier_key(name): name for name in table_names}
        if len(table_lookup) != len(table_names):
            raise ValueError(f"Case-insensitive duplicate table names in {db_id}")

        for source_table, rows in contents.items():
            if not isinstance(rows, list):
                raise ValueError(f"Rows must be a list in {db_id}.{source_table}")
            stats["source_rows"] += len(rows)
            table = table_lookup.get(identifier_key(source_table))
            if table is None:
                stats["unknown_content_tables"] += 1
                stats["unknown_content_table_rows"] += len(rows)
                continue

            actual_columns = [
                (result[1], result[2], result[3], result[4], result[5])
                for result in connection.execute(
                    "SELECT * FROM pragma_table_xinfo(?) WHERE hidden = 0 ORDER BY cid",
                    (table,),
                )
            ]
            actual_positions = {
                identifier_key(column[0]): position
                for position, column in enumerate(actual_columns)
            }
            source_columns = list(dict.fromkeys(
                key for item in rows if isinstance(item, dict) for key in item
            ))
            column_mapping = build_column_mapping(source_columns, actual_columns, stats)
            source_by_actual = {}
            for source_column in source_columns:
                actual_column = column_mapping[source_column]
                if actual_column is None:
                    continue
                key = identifier_key(actual_column)
                if key in source_by_actual:
                    stats["duplicate_content_columns"] += 1
                    continue
                source_by_actual[key] = source_column

            keys = tuple(sorted(source_by_actual, key=actual_positions.__getitem__))
            columns = tuple(actual_columns[actual_positions[key]][0] for key in keys)
            if not columns:
                stats["no_mapped_column_rows"] += len(rows)
                continue
            prepared_rows = []

            for item in rows:
                if not isinstance(item, dict):
                    stats["malformed_source_rows"] += 1
                    continue
                try:
                    values = tuple(
                        sqlite_value(item.get(source_by_actual[key]))
                        for key in keys
                    )
                except ValueError:
                    stats["unsupported_value_rows"] += 1
                    continue
                prepared_rows.append(values)

            if prepared_rows:
                insert_group(connection, table, columns, prepared_rows, stats)

        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise ValueError(f"SQLite integrity check failed for {db_id}: {integrity}")
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()
        temporary_path.replace(database_path)
        return stats


def materialize_schemas(schema_directory, required_database_ids, databases_directory):
    columns = ["schema id", "Full schema", "Schema content", "number of tables"]
    inventory = []
    seen = set()
    materialized = 0
    stats = Counter()
    for row in parquet_rows(schema_directory, columns, batch_size=1):
        db_id = row["schema id"]
        if not isinstance(db_id, str) or not SCHEMA_ID_RE.fullmatch(db_id):
            raise ValueError(f"Invalid source schema id: {db_id!r}")
        if db_id in seen:
            raise ValueError(f"Duplicate source schema id: {db_id}")
        seen.add(db_id)
        inventory.append({"db_id": db_id})
        if db_id not in required_database_ids:
            continue

        directory = databases_directory / db_id
        directory.mkdir()
        stats.update(materialize_database(row, directory / f"{db_id}.sqlite"))
        materialized += 1
        if materialized % 100 == 0:
            print(f"  materialized {materialized}/{len(required_database_ids)} databases")

    missing = sorted(required_database_ids - seen)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing {len(missing)} referenced schemas: {preview}")
    return inventory, materialized, stats


def parse_expected_result(value, question_id):
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid execution_result JSON for q_{question_id:07d}: {error}"
        ) from error
    if not isinstance(result, list) or any(not isinstance(row, list) for row in result):
        raise ValueError(f"Invalid execution_result rows for q_{question_id:07d}")
    if len(result) > EXECUTION_RESULT_LIMIT:
        raise ValueError(
            f"execution_result exceeds {EXECUTION_RESULT_LIMIT} rows for q_{question_id:07d}"
        )
    return result


def has_top_level_order_by(sql):
    """Return (ordered, used_fallback) for result comparison."""
    try:
        expression = sqlglot.parse_one(sql, read="sqlite")
    except ParseError:
        return bool(ORDER_BY_RE.search(sql)), True
    return expression.args.get("order") is not None, False


def values_equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        return math.isclose(
            float(left), float(right),
            rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE,
        )
    return left == right


def rows_equal(left, right):
    return len(left) == len(right) and all(
        values_equal(left_value, right_value)
        for left_value, right_value in zip(left, right)
    )


def results_equal(expected, actual, ordered):
    if len(expected) != len(actual):
        return False
    if ordered:
        return all(rows_equal(left, right) for left, right in zip(expected, actual))
    unmatched = list(actual)
    for expected_row in expected:
        for position, actual_row in enumerate(unmatched):
            if rows_equal(expected_row, actual_row):
                unmatched.pop(position)
                break
        else:
            return False
    return not unmatched


def query_result_matches(connection, sql, expected, ordered):
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS

    def interrupt_if_expired():
        return int(time.monotonic() >= deadline)

    connection.set_progress_handler(interrupt_if_expired, 1000)
    try:
        cursor = connection.execute(sql)
        if not ordered and len(expected) == EXECUTION_RESULT_LIMIT:
            unmatched = list(expected)
            while unmatched:
                rows = cursor.fetchmany(256)
                if not rows:
                    return False
                for actual_row in rows:
                    for position, expected_row in enumerate(unmatched):
                        if rows_equal(expected_row, actual_row):
                            unmatched.pop(position)
                            break
                    if not unmatched:
                        return True

        actual = [list(row) for row in cursor.fetchmany(EXECUTION_RESULT_LIMIT)]
        return results_equal(expected, actual, ordered)
    finally:
        connection.set_progress_handler(None, 0)


def validate_queries(samples, expected_results, databases_directory):
    by_database = defaultdict(list)
    for sample in samples:
        by_database[sample["db_id"]].append(sample)
    accepted_ids = set()
    rejected = []
    stats = Counter()
    checked = 0

    for db_id in sorted(by_database):
        database_path = databases_directory / db_id / f"{db_id}.sqlite"
        uri = database_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            for sample in by_database[db_id]:
                question_id = sample["question_id"]
                expected = parse_expected_result(expected_results[question_id], question_id)
                ordered, fallback = has_top_level_order_by(sample["SQL"])
                stats["order_detection_fallbacks"] += int(fallback)
                try:
                    matches = query_result_matches(
                        connection,
                        sample["SQL"],
                        expected,
                        ordered,
                    )
                except sqlite3.Error as error:
                    reason = "timeout" if "interrupted" in str(error).lower() else "execution_error"
                    stats[reason] += 1
                    rejected.append((question_id, db_id, reason, str(error)))
                else:
                    if matches:
                        accepted_ids.add(question_id)
                        stats["matches"] += 1
                    else:
                        stats["result_mismatches"] += 1
                        rejected.append((question_id, db_id, "result_mismatch", ""))
                checked += 1
                if checked % 5000 == 0:
                    print(f"  validated {checked}/{len(samples)} queries")

    accepted = [sample for sample in samples if sample["question_id"] in accepted_ids]
    stats["rejected_queries"] = len(rejected)
    return accepted, rejected, stats


def main():
    if OUTPUT_DIR.exists():
        raise SystemExit(f"Output already exists: {OUTPUT_DIR}")
    if WORK_DIR.exists():
        raise SystemExit(f"Work directory already exists: {WORK_DIR}")

    WORK_DIR.mkdir(parents=True)
    BUILD_DIR.mkdir()
    databases_directory = BUILD_DIR / "dev_databases"
    databases_directory.mkdir()

    print("Downloading pinned SQaLe3 sources...")
    query_directory = snapshot_download(
        QUERIES_REPOSITORY,
        revision=QUERIES_REVISION,
        repo_type="dataset",
        local_dir=WORK_DIR / "queries",
        allow_patterns=["data/*.parquet"],
    )
    schema_directory = snapshot_download(
        SCHEMAS_REPOSITORY,
        revision=SCHEMAS_REVISION,
        repo_type="dataset",
        local_dir=WORK_DIR / "schemas",
        allow_patterns=["data/*.parquet"],
    )

    print("Filtering candidate queries...")
    samples, expected_results, database_ids, query_stats = build_dev(query_directory)
    print(
        f"  {query_stats['rows_read']} rows read, {len(samples)} candidates, "
        f"{len(database_ids)} databases"
    )

    print("Materializing referenced SQLite databases from Full schema...")
    inventory, materialized, materialization_stats = materialize_schemas(
        schema_directory, database_ids, databases_directory,
    )

    print("Validating queries against the supplied execution_result...")
    samples, rejected, validation_stats = validate_queries(
        samples, expected_results, databases_directory,
    )
    dev_path = BUILD_DIR / "dev.json"
    write_json(dev_path, samples)
    if rejected:
        preview = ", ".join(str(item[0]) for item in rejected[:20])
        suffix = " ..." if len(rejected) > 20 else ""
        print(f"  rejected question_ids: {preview}{suffix}")

    provisional_tables_path = BUILD_DIR / "dev_tables.json"
    write_json(provisional_tables_path, inventory)
    print("Filtering schemas and SQLite folders with remove_unused_databases...")
    remove_unused_databases(dev_path, provisional_tables_path)
    with provisional_tables_path.open("r", encoding="utf-8") as file:
        retained_inventory = json.load(file)
    retained_ids = {item["db_id"] for item in retained_inventory}
    final_database_ids = {sample["db_id"] for sample in samples}
    if retained_ids != final_database_ids:
        raise ValueError("Filtered schema inventory does not match dev.json")
    provisional_tables_path.unlink()

    print("Generating dev_tables.json from SQLite metadata...")
    generate_dev_tables(BUILD_DIR, skip_unresolved_foreign_keys=True)

    for attempt in range(1, 11):
        try:
            BUILD_DIR.replace(OUTPUT_DIR)
            break
        except PermissionError:
            if attempt == 10:
                raise
            print(f"  final move blocked; retrying in 3 seconds ({attempt}/9)")
            time.sleep(3)
    shutil.rmtree(WORK_DIR)
    print(
        f"Build completed: {OUTPUT_DIR}\n"
        f"  samples: {len(samples)}\n"
        f"  databases materialized: {materialized}\n"
        f"  databases retained: {len(final_database_ids)}\n"
        f"  rows skipped by execution_match: {query_stats['execution_mismatch_skipped']}\n"
        f"  rows skipped by pair_verified: {query_stats['not_verified_skipped']}\n"
        f"  validation mismatches: {validation_stats['result_mismatches']}\n"
        f"  validation execution errors: {validation_stats['execution_error']}\n"
        f"  validation timeouts: {validation_stats['timeout']}\n"
        f"  ORDER BY parser fallbacks: {validation_stats['order_detection_fallbacks']}\n"
        f"  source rows: {materialization_stats['source_rows']}\n"
        f"  inserted rows: {materialization_stats['inserted_rows']}\n"
        f"  rows ignored by SQLite constraints: {materialization_stats['constraint_ignored_rows']}\n"
        f"  rows skipped for unsupported values: {materialization_stats['unsupported_value_rows']}\n"
        f"  rows skipped after insertion errors: {materialization_stats['insertion_error_rows']}\n"
        f"  rows with no mapped column: {materialization_stats['no_mapped_column_rows']}\n"
        f"  unmapped content columns: {materialization_stats['unmapped_content_columns']}\n"
        f"  duplicate content columns: {materialization_stats['duplicate_content_columns']}\n"
        f"  malformed source rows: {materialization_stats['malformed_source_rows']}\n"
        f"  unknown content tables: {materialization_stats['unknown_content_tables']}\n"
        f"  rows from unknown content tables: "
        f"{materialization_stats['unknown_content_table_rows']}\n"
        f"  declared table-count mismatches: {materialization_stats['declared_table_count_mismatches']}"
    )


if __name__ == "__main__":
    main()
