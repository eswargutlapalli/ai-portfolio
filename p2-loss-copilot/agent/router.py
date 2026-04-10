"""
router.py
Claude reads the user's question and picks which tools to use.
Author: Eswar Gutlapalli
"""

import os
import anthropic
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROUTING_PROMPT = """You are a routing agent for a credit risk analytics system.

Given a user question, decide which tools are needed:
- "sql"  — question asks for numbers, aggregations, filters, rankings from structured data
- "rag"  — question asks about narrative context, causes, trends from documents  
- "both" — question needs both quantitative data AND narrative context

Respond with ONLY one word: sql, rag, or both.

Examples:
"Which region had the highest losses?" → sql
"What caused the Midwest delinquencies?" → rag
"Summarize the top loss drivers and their amounts" → both"""

def route(question: str) -> str:
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        temperature=0,
        system=ROUTING_PROMPT,
        messages=[{"role": "user", "content": question}]
    )
    decision = message.content[0].text.strip().lower()
    return decision if decision in ("sql", "rag", "both") else "both"

if __name__ == "__main__":
    tests = [
        "Which region had the highest loss?",
        "What caused the Midwest defaults?",
        "Give me the top loss regions and explain why",
    ]
    for q in tests:
        print(f"Q: {q}\n-> {route(q)}\n")
