"""
Synthesizer - combines RAG chunks with Claude to generate insights
Author: Eswar Gutlapalli
"""

import anthropic
import streamlit as st

def synthesize(query: str, chunks: list):
    context = "\n\n".join([chunk.page_content for chunk in chunks])

    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

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