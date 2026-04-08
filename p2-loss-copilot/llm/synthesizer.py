"""
Synthesizer - combines RAG chunks with Claude to generate insights
Author: Eswar Gutlapalli
"""

import os
import time
import anthropic
import streamlit as st
from dotenv import load_dotenv 

load_dotenv()

def synthesize(query: str, chunks: list) -> str:
    context = "\n\n".join([chunk.page_content for chunk in chunks])

    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY") 
    except Exception:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model = "claude-sonnet-4-6",
                max_tokens = 1024,
                temperature=0.1,
                system="""You are a credit risk analyst. Write a concise 3-sentence executive summary.
                Use only the context provided. Do not infer facts not present.
                If information is unavailable, state 'insufficient context'.""",
                messages=[
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )

            return message.content[0].text
        
        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                wait = 2 ** attempt     #1s, 2s, 4s
                print(f"Claude overloaded. Retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise       # re-raise if it's a different error

    return "Claude is currently overloaded. Please try again in a moment."

if __name__ == "__main__":
    from rag.embedder import build_index
    from rag.retriever import get_relevant_chunks

    sample_docs = [
        "Midwest region showed 12% increase in auto loan defaults in Q3.",
        "Commercial real estate losses concentrated in urban markets.",
        "Credit card delinquencies rose among 25-34 age group."
    ]

    print("Building index...")
    index = build_index(sample_docs)

    print("Retrieving chunks...")
    chunks = get_relevant_chunks("What are the key loss drivers?", index)

    print("Synthesizing with Claude...")
    response = synthesize("What are the key loss drivers?", chunks)
    print("\nClaude's response:")
    print(response)