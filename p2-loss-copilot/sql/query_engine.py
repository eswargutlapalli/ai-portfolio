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
import pathlib
from dotenv import load_dotenv

load_dotenv()

SCHEMA = """
Table: losses
Columns:
  id              INTEGER  -- primary key
  region          TEXT     -- Midwest, Northeast, South, West
  product         TEXT     -- Auto Loan, Credit Card, Mortgage, Commercial RE,
                           --  Personal Loan, Small Business Loan
  quarter         TEXT     -- Q1-2023 to Q4-2024
  loss_amount     INTEGER  -- total loss in dollars
  delinquency_rate REAL    -- 0.0–1.0 (e.g. 0.12 = 12%)
  severity        TEXT     -- Low, Medium, High
"""

SQL_RULES = """
SQLite rules you must follow:
1. Only use columns that exist in the schema above — never invent column names.
2. Always include GROUP BY when using aggregate functions (SUM, AVG, COUNT, MIN, MAX).
3. Use simple column references only — no window functions, no CTEs, no subqueries unless essential.
4. When comparing product or region values use exact strings from the schema comments.
5. Return ONLY the raw SQL — no explanation, no markdown, no backticks.
6. Never use SELECT * — always name the columns you need.
7. LIMIT 20 unless the question asks for all rows.
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
        max_tokens=512,
        temperature=0,
        system=f"""You are a SQL expert. Given a schema and a question, return ONLY a valid SQLite SELECT statement.
        No explanation. No markdown. No backticks. Just the raw SQL.

        Schema:
        {SCHEMA}

        {SQL_RULES}""",
        messages=[{"role": "user", "content": question}]
    )
    return message.content[0].text.strip()

def run_sql(sql: str, db_path: str=None) -> pd.DataFrame:
    if db_path is None:
        db_path = str(pathlib.Path(__file__).parent.parent / "data/losses.db")
    # Execute SQL and return results as DataFrame
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df

def query(question: str, db_path: str="data/losses.db") -> dict:
    """Full pipeline: question → SQL → DataFrame → dict with summary.
    
    Returns a clean error dict instead of raising — so the agentic loop
    can handle failures gracefully without crashing the UI or eval runner.
    """
    try:
        sql = nl_to_sql(question)
        df = run_sql(sql, db_path)
        return {
            "sql" : sql,
            "results" : df,
            "summary" : df.to_string(index=False) if not df.empty else "No results found."
        }
    except Exception as e:
        fallback_sql = "-- SQL generation or execution failed"
        return {
            "sql": fallback_sql,
            "results": pd.DataFrame(),
            "summary": f"Query failed {str(e)}"
        }

if __name__ == "__main__":
    result = query("Which region had the highest total loss amount?")
    print("SQL:", result["sql"])
    print("Results:\n", result["results"])