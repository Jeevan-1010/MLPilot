
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import psycopg
from psycopg import sql
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQLITE_PATH = PROJECT_ROOT / "mlflow.db"

load_dotenv(PROJECT_ROOT / ".env", override=True)


def get_sqlite_tables(conn):
    return [
        row[0]
        for row in conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
    ]


def get_sqlite_columns(conn, table):
    return [
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{table}")'
        )
    ]


def get_neon_columns(conn, table):
    rows = conn.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position
    """, (table,)).fetchall()

    return {
        name: {
            "type": data_type,
            "nullable": nullable,
        }
        for name, data_type, nullable in rows
    }


def get_insertion_order(conn, tables):
    parents = defaultdict(set)

    for table in tables:
        rows = conn.execute(
            f'PRAGMA foreign_key_list("{table}")'
        ).fetchall()

        for row in rows:
            parents[table].add(row[2])

    remaining = set(tables)
    ordered = []

    while remaining:
        ready = sorted(
            table for table in remaining
            if not (parents[table] & remaining)
        )

        if not ready:
            raise RuntimeError(
                f"Circular dependencies: {sorted(remaining)}"
            )

        ordered.extend(ready)
        remaining.difference_update(ready)

    return ordered


def convert_value(value, postgres_type):
    if value is None:
        return None

    if postgres_type == "boolean":
        return bool(value)

    # SQLite BLOB values are returned as bytes.
    # psycopg automatically adapts bytes to PostgreSQL bytea.

    # JSON strings are cast explicitly in the INSERT statement.
    return value


def migrate():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is missing.")

    if not SQLITE_PATH.exists():
        raise FileNotFoundError(SQLITE_PATH)

    source = sqlite3.connect(
        f"file:{SQLITE_PATH.as_posix()}?mode=ro",
        uri=True,
    )

    target = psycopg.connect(
        tracking_uri,
        connect_timeout=15,
        autocommit=True,
    )

    try:
        tables = get_sqlite_tables(source)

        neon_tables = {
            row[0]
            for row in target.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
            """).fetchall()
        }

        if set(tables) != neon_tables:
            raise RuntimeError(
                "SQLite and Neon table sets do not match."
            )

        order = get_insertion_order(source, tables)

        print("MLFLOW METADATA MIGRATION")
        print("=" * 55)
        print("Source: local SQLite")
        print("Destination: Neon migration branch")
        print("Tables:", len(tables))
        print("Mode: TRANSACTIONAL MIGRATION")
        print()

        # All destination checks and writes occur in one transaction.
        with target.transaction():

            # Refuse to merge into a non-empty destination.
            for table in tables:
                count = target.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(table)
                    )
                ).fetchone()[0]

                if count != 0:
                    raise RuntimeError(
                        f"Destination table {table} is not empty "
                        f"({count} rows). Migration cancelled."
                    )

            total_rows = 0

            for table in order:
                source_columns = get_sqlite_columns(source, table)
                target_columns = get_neon_columns(target, table)

                missing = set(source_columns) - set(target_columns)

                if missing:
                    raise RuntimeError(
                        f"{table}: missing Neon columns {sorted(missing)}"
                    )

                # Neon has a nullable duration_ns column in spans.
                # It is deliberately omitted so PostgreSQL stores NULL.
                columns = source_columns

                postgres_types = [
                    target_columns[column]["type"]
                    for column in columns
                ]

                select_query = (
                    f'SELECT {", ".join(chr(34) + c + chr(34) for c in columns)} '
                    f'FROM "{table}"'
                )

                rows = source.execute(select_query).fetchall()

                if not rows:
                    print(f"{table}: 0 rows")
                    continue

                placeholders = []

                for postgres_type in postgres_types:
                    if postgres_type in ("json", "jsonb"):
                        placeholders.append("%s::" + postgres_type)
                    else:
                        placeholders.append("%s")

                insert_query = sql.SQL(
                    "INSERT INTO {} ({}) VALUES ({})"
                ).format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(
                        sql.Identifier(column) for column in columns
                    ),
                    sql.SQL(", ").join(
                        sql.SQL(value) for value in placeholders
                    ),
                )

                converted_rows = [
                    tuple(
                        convert_value(value, postgres_type)
                        for value, postgres_type in zip(row, postgres_types)
                    )
                    for row in rows
                ]

                with target.cursor() as cursor:
                    cursor.executemany(insert_query, converted_rows)

                inserted = target.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(table)
                    )
                ).fetchone()[0]

                if inserted != len(rows):
                    raise RuntimeError(
                        f"{table}: expected {len(rows)}, got {inserted}"
                    )

                total_rows += inserted
                print(f"{table}: {inserted} rows migrated")

            # Synchronize PostgreSQL sequences after preserving IDs.
            sequence_rows = target.execute("""
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND (
                      column_default LIKE 'nextval(%'
                      OR is_identity = 'YES'
                  )
                ORDER BY table_name, ordinal_position
            """).fetchall()

            for table, column in sequence_rows:
                sequence = target.execute(
                    "SELECT pg_get_serial_sequence(%s, %s)",
                    (f"public.{table}", column),
                ).fetchone()[0]

                if not sequence:
                    continue

                maximum = target.execute(
                    sql.SQL("SELECT MAX({}) FROM {}").format(
                        sql.Identifier(column),
                        sql.Identifier(table),
                    )
                ).fetchone()[0]

                if maximum is not None:
                    target.execute(
                        "SELECT setval(%s, %s, true)",
                        (sequence, maximum),
                    )

            # Verify every table before allowing the transaction to commit.
            print("\nVERIFYING ROW COUNTS")

            for table in tables:
                expected = source.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]

                actual = target.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(table)
                    )
                ).fetchone()[0]

                if expected != actual:
                    raise RuntimeError(
                        f"Verification failed for {table}: "
                        f"SQLite={expected}, Neon={actual}"
                    )

        print("\n" + "=" * 55)
        print("METADATA MIGRATION COMMITTED")
        print("Total rows migrated:", total_rows)
        print("Tables verified:", len(tables))
        print("Transaction: COMMITTED")

    finally:
        source.close()
        target.close()


if __name__ == "__main__":
    try:
        migrate()
    except Exception as exc:
        print("\nMIGRATION FAILED — TRANSACTION ROLLED BACK")
        print(type(exc).__name__ + ":", exc)
        sys.exit(1)
