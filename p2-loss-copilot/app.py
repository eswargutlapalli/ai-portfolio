"""
app.py — P4 Native Tool Use Agent
Streamlit UI with RAG + SQL + agent routing.
Author: Eswar Gutlapalli
"""

import streamlit as st
from rag.embedder import build_index
from rag.retriever import get_relevant_chunks
from agent.tool_agent import run_agent
from sql.query_engine import query as sql_query

st.set_page_config(page_title="Native Tool Use Agent", layout="wide")

st.title("Native Tool Use Agent")
st.caption("Powered by Claude · Native tool use · Agentic loop")

# Sidebar: Document upload
with st.sidebar:
    st.subheader("Data Sources")
    uploaded_file = st.file_uploader("Upload loss document (.txt)", type=["txt"])
    st.caption("losses.db is pre-loaded from data/")

# Build RAG index once per session
index = None
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    docs = [line for line in content.split("\n") if line.strip()]
    index = build_index(docs)

# Main question
question = st.text_input("Ask question about credit losses:")

if st.button("Analyze") and question:
    
    def tool_executor(tool_name: str, tool_input: dict) -> str:
        """Executes whichever tool Claude requests."""
        if tool_name == "search_documents":
            if index is None:
                return "No document loaded. RAG search unavailable."
            chunks = get_relevant_chunks(tool_input["query"], index)
            return "\n\n".join([c.page_content for c in chunks])
        
        elif tool_name == "query_database":
            result = sql_query(tool_input["question"])
            return result["summary"]
        
        return f"Unknown tool: {tool_name}"
    
    with st.spinner("Agent thinking..."):
        output = run_agent(question, tool_executor)

    # Show what tools Claude actually called
    if output["tool_calls"]:
        st.subheader("Tools Claude used")
        for call in output["tool_calls"]:
            with st.expander(f"🔧 {call['name']} — {list(call['input'].values())[0][:60]}"):
                st.json(call["input"])
                st.text(call["result"][:500])

    st.divider()
    st.subheader("Agent response")
    st.write(output["answer"])
