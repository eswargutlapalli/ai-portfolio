"""
tool_agent.py
Native tool-use agent. Claude decides which tools to call and when.
Replaces the manual router from P3.
Author: Eswar Gutlapalli
"""

import os
import streamlit as st
import anthropic
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

# Tool schemas — Claude reads these to understand its capabilities

TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search unstructured loss documents using semantic similarity. "
            "Use this for narrative context, causes, trends or qualitative explanations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "semantic search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "query_database",
        "description": (
            "Run a natural language query against the structured losses database. "
            "Use this for exact numbers, aggregations, rankings or filters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "question about structured loss data"}
            },
        "required": ["question"]
        }
    }
]

SYSTEM = """You are a credit risk analyst agent with two tools:
- search_documents - for narrative, causes, qualitative trends
- query_database - for exact numbers, aggregations, rankings

Use whichever tools are need to answer the question fully.
You may call both the tools if the answer requires quantitative and qualitative analysis.
After receiving tool results, write a concise executive summary (3-5 sentence)
Never use $ sign - write amount as 'USD X.XM'."""

def _getclient():
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key)

# to catch incorrect keys typed in the dict: for ex, toll_calls in place of tool_calls
class AgentOutput(TypedDict):
    answer: str
    tool_calls: list

def run_agent(question: str, tool_executor) -> AgentOutput:
    """
    Agentic loop. Runs until Claude stops requesting tools.

    Args: 
        question: user's question
        tool_executor: callable(tool_name, tool_input) -> str

    Returns:
        dict with keys:
            - answer: Claude's final text response
            - tool_calls: list of {name, input, results} dicts 
    """

    client = _getclient()
    messages = [{"role": "user", "content": question}]
    tool_calls_log = []

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages
        )

        # Append Claude's response to conversation history
        messages.append({"role": "assistant", "content": response.content})

        # Done — Claude wrote a final answer
        if response.stop_reason == "end_turn":
            answer = next(
                (block.text for block in response.content if hasattr(block, "text")),
                ""
            )
            return {"answer": answer, "tool_calls": tool_calls_log}
        
        # Claude wants to call tools
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                # Execute the requested tool
                result_str = tool_executor(block.name, block.input)
                tool_calls_log.append({
                    "name": block.name,
                    "input": block.input,
                    "result": result_str
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str
                })

            # Feed all results back in one user turn
            messages.append({"role": "user", "content": tool_results})
            # Loop — Claude will now process results and either call more tools or write answer

        # Safety exit — unexpected stop reason
        else:
            return {
                "answer": f"Unexpected stop reason: {response.stop_reason}",
                "tool_calls": tool_calls_log
            }