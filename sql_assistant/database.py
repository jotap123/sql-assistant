import sqlite3
import pandas as pd
from pathlib import Path

from sql_assistant.query import QueryResult


class DatabaseConnection:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_schema(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            schema_parts = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                schema_parts.append(f"Table: {table_name}")
                schema_parts.extend(f"- {col[1]} ({col[2]})" for col in columns)
                schema_parts.append("")

            return "\n".join(schema_parts)
    
    def execute_query(self, query: str) -> QueryResult:
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                return df
        except Exception as e:
            print(f"{e}")
            return pd.DataFrame()
