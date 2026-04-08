"""
Synthesizer - combines RAG chunks with Claude to generate insights
Author: Eswar Gutlapalli
"""

import os
import anthropic
import streamlit as st
from dotenv import load_dotenv 

load_dotenv()

# def synthesize(query: str, chunks: list):
#     context = "\n\n".join([chunk.page_content for chunk in chunks])

#     api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
#     client = anthropic.Anthropic(api_key=api_key)

#     message = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=1024,
#         temperature=0.1,
#         system="""You are a credit risk analyst. Write a concise 3-sentence executive summary.
#         Use only the context provided. Do not infer facts not present.
#         If information is unavailable, state 'insufficient context'.""",
#         messages=[
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
#         ]
#     )

#     return message.content[0].text

def synthesize(query: str, chunks: list) -> str:
    context = "\n\n".join([chunk.page_content for chunk in chunks])

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=0.1,
        system="""You are a credit risk analyst. Write a concise 3-sentence executive summary.
        Use only the context provided. Do not infer facts not present.
        If information is unavailable, state 'insufficient context'.""",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return message.content[0].text

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