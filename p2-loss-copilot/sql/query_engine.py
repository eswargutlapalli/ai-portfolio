"""
query_engine.py
Translates natural language to SQL using Claude, runs it against losses.db.
Author: Eswar Gutlapalli
"""

import os
import sqlite3 
import anthropic
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SCHEMA = """
Table: losses
Columns:
  id              INTEGER  -- primary key
  region          TEXT     -- Midwest, Northeast, South, West
  product         TEXT     -- Auto Loan, Credit Card, Mortgage, Commercial RE
  quarter         TEXT     -- e.g. Q3-2023
  loss_amount     INTEGER  -- total loss in dollars
  delinquency_rate REAL    -- 0.0–1.0
  severity        TEXT     -- Low, Medium, High
"""

def _get_client() -> anthropic.Anthropic:
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key)

def nl_to_sql(question: str) -> str:
    """Ask Claude to produce a SQL SELECT statement for the given question."""
    client = _get_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0,
        system=f"""You are a SQL expert. Given a schema and a question, return ONLY a valid SQLite SELECT statement.
        No explanation. No markdown. No backticks. Just the raw SQL.

        Schema:
        {SCHEMA}""",
        messages=[{"role": "user", "content": question}]
    )
    print(message)
    return message.content[0].text.strip()

def run_sql(sql: str, db_path: str="data/losses.db") -> pd.DataFrame:
    # Execute SQL and return results as DataFrame
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df

def query(question: str, db_path: str="data/losses.db") -> dict:
    # Full pipeline: question → SQL → DataFrame → dict with both
    sql = nl_to_sql(question)
    df = run_sql(sql, db_path)
    return {
        "sql" : sql,
        "results" : df,
        "summary" : df.to_string(index=False) if not df.empty else "No results found."
    }

if __name__ == "__main__":
    result = query("Which region had the highest total loss amount?")
    print("SQL:", result["sql"])
    print("Results:\n", result["results"])