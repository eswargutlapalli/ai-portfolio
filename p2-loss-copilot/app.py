"""
app.py — P3 Autonomous Risk Analyst Agent
Streamlit UI with RAG + SQL + agent routing.
Author: Eswar Gutlapalli
"""

import streamlit as st
from llm.synthesizer import synthesize
from rag.embedder import build_index
from rag.retriever import get_relevant_chunks
from agent.router import route
from sql.query_engine import query as sql_query

st.set_page_config(page_title="Autonomous Risk Analyst Agent", layout="wide")

st.title("Autonomous Risk Analyst Agent")
st.caption("Powered by Claude · Multi-source: RAG + SQL · Agent routing")

# Sidebar: Document upload
with st.sidebar:
    st.subheader("Data Sources")
    uploaded_file = st.file_uploader("Upload loss document (.txt)", type=["txt"])
    st.caption("losses.db is pre-loaded from data/")

# Main question
question = st.text_input("Ask question about credit losses:")

if st.button("Analyze") and question:
    decision = route(question)
    st.info(f"Agent Decision: **{decision.upper()}**")

    rag_context = ""
    sql_context = ""

    col1, col2 = st.columns(2)

    # RAG path
    if decision in ("rag", "both"):
        with col1:
            st.subheader("RAG Results:")
            if uploaded_file:
                content = uploaded_file.read().decode("utf-8")
                docs = [line for line in content.split("\n") if line.strip()]
                index = build_index(docs)
                chunks = get_relevant_chunks(question, index)
                rag_context = "\n\n".join([c.page_content for c in chunks])
                st.write(rag_context)
            else:
                st.write("No document uploaded - RAG skipped.")

    # SQL path
    if decision in ("sql", "both"):
        with col2:
            st.subheader("SQL Results:")
            result = sql_query(question)
            sql_context = result["summary"]
            st.code(result["sql"], language="sql")
            st.dataframe(result["results"])

    # Final synthesis
    st.divider()
    st.subheader("Synthesized Insight:")

    #Re-use P2's Synthesizer Multi
    from llm.synthesizer import synthesize_multi
    response = synthesize_multi(question, rag_context, sql_context)
    st.write(response)