"""
app.py
Streamlit UI connecting all 3 RAG layers.
Accepts data input, question and returns Claude-powered insights.
"""

import streamlit as st
from llm.synthesizer import synthesize
from rag.embedder import build_index
from rag.retriever import get_relevant_chunks

st.set_page_config(page_title="Loss Intelligence Copilot", layout="wide")

st.title("Loss Intelligence Copilot")
st.caption("Powered by Claude · Upload loss documents and get instant insights")

# get imputs
uploaded_file = st.file_uploader("Upload financial txt (.txt)", type=["txt"])
query_input = st.text_input("Enter your question: ")

if st.button("Generate Insights"):
    if uploaded_file and query_input:
        with st.spinner("Analyzing with Claude..."):
            # read uploaded file as plain text, split into lines
            content = uploaded_file.read().decode("utf-8")
            documents = [line for line in content.split("\n") if line.strip()]

            # RAG pipeline
            index = build_index(documents)
            chunks = get_relevant_chunks(query_input, index)
            response = synthesize(query_input, chunks)

            st.subheader("Insights")
            st.write(response)