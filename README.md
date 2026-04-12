# AI Portfolio — Credit Risk Intelligence
> A progressive series of AI systems built on the Anthropic Claude API, demonstrating the evolution from simple LLM calls to production-grade agentic architectures — applied to credit risk analytics.

**Author:** Eswar Gutlapalli
**Stack:** Python 3.11 · Anthropic Claude API · LangChain · FAISS · HuggingFace Embeddings · Streamlit · SQLite · pandas

---

## Portfolio Progression

| Project | What It Is | Key Concept |
|---|---|---|
| [P1 — Financial Insight Generator](#p1--financial-insight-generator) | LLM over structured CSV data | Prompt engineering, API basics |
| [P2 — Loss Intelligence Copilot](#p2--loss-intelligence-copilot) | RAG over unstructured documents | Embeddings, vector search, retrieval |
| [P3 — Autonomous Risk Analyst Agent](#p3--autonomous-risk-analyst-agent) | Multi-source agent: RAG + SQL | Agent routing, nl_to_sql, synthesis |
| [P4 — Native Tool Use Agent](#p4--native-tool-use-agent) | Production agentic loop | Native tool use, multi-step reasoning |

### P1 — Financial Insight Generator [`v1.0p1`]
Reads structured loan portfolio CSV, computes loss rates by segment,
generates board-ready executive narratives using Claude.
**Key concepts:** Anthropic API basics, prompt engineering, pandas

### P2 — Loss Intelligence Copilot [`v1.0p2`]
RAG system over unstructured loss documents. Semantic search with
FAISS + HuggingFace embeddings, synthesized by Claude.
**Key concepts:** RAG, embeddings, vector search, LangChain

### P3 — Autonomous Risk Analyst Agent [`v1.0p3`]
Multi-source agent combining RAG + SQL. Claude routes queries to the
right data source, synthesizes structured and unstructured results.
**Key concepts:** Agent routing, nl_to_sql, multi-source synthesis

### P4 — Native Tool Use Agent [`v1.0p4`]
Production-grade agentic loop using Claude's native tool use API.
Claude decides which tools to call, chains results across turns,
handles parallel tool calls.
**Key concepts:** Native tool use, agentic loop, dependency injection

### How to Run It
```bash
cd p2-loss-copilot
pip install -r requirements.txt
cp ../.env.example .env          # add your ANTHROPIC_API_KEY
python data/create_db.py         # seed losses.db (only needed once)
streamlit run app.py
```
Upload `data/sample_losses.txt` when prompted. Then try these queries in order:

```
1. "Which region had the highest total loss?"
   → Claude calls: query_database

2. "What caused the Midwest delinquencies?"
   → Claude calls: search_documents + query_database

3. "Summarize the top loss regions and explain why"
   → Claude calls: query_database + search_documents

4. "Which product has the worst delinquency rate and what does the narrative say about it?"
   → Claude calls: query_database → search_documents → search_documents
   (genuine multi-step: Claude queries DB, reads result, decides it needs
   two separate document searches for the specific product found)
```

Query 4 demonstrates true agentic behavior — Claude's second and third tool calls are determined by the result of the first, not by any logic in your code.

### Sample Output
```
Query: "Which product has the worst delinquency rate and what does
        the narrative say about it?"

Tools Claude used:
  🔧 query_database — Which product has the worst delinquency rate?
  🔧 search_documents — credit card delinquency causes
  🔧 search_documents — credit card default Northeast Q3

Agent response: "Credit Cards carried the highest average delinquency
rate at 16.5%, concentrated in the Northeast where Q3 charge-offs
accelerated sharply. Narrative context indicates the primary driver
was a combination of post-pandemic revolving balance normalization
and rising minimum payment burdens as rates increased. The South
showed secondary pressure at 15%, attributed to subprime origination
vintages from 2021-2022 now entering peak default windows. Immediate
tightening of underwriting standards for revolving credit is warranted."
```

---

## Repository Structure

```
ai-portfolio/
├── p1-financial-insight/
│   ├── app.py                  Streamlit UI
│   ├── insight_generator.py    Core pipeline
│   ├── data/loan_data.csv      Sample portfolio data
│   └── requirements.txt
│
├── p2-loss-copilot/            P2, P3, and P4 live here
│   ├── app.py                  Streamlit UI (updated each project)
│   ├── agent/
│   │   ├── router.py           P3 — manual classifier router
│   │   └── tool_agent.py       P4 — native tool use agentic loop
│   ├── rag/
│   │   ├── embedder.py         Chunk + embed + FAISS index
│   │   └── retriever.py        Semantic search → top-k chunks
│   ├── llm/
│   │   └── synthesizer.py      synthesize() P2 | synthesize_multi() P3
│   ├── sql/
│   │   └── query_engine.py     nl_to_sql + run_sql + query()
│   ├── data/
│   │   ├── create_db.py        Seeds losses.db
│   │   └── sample_losses.txt   Unstructured loss document for RAG
│   └── requirements.txt
│
├── .env.example                API key template
└── .gitignore                  Excludes *.db, .env, venv
```

---

## Setup

```bash
git clone https://github.com/eswargutlapalli/ai-portfolio.git
cd ai-portfolio
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here
```

Each project has its own `requirements.txt`. Install inside a virtual environment per project.

---

## Tags

| Tag | Project | Description |
|---|---|---|
| `v1.0p1` | P1 | Financial Insight Generator |
| `v1.0p2` | P2 | Loss Intelligence Copilot — RAG |
| `v1.0p3` | P3 | Autonomous Risk Analyst — RAG + SQL routing |
| `v1.0p4` | P4 | Native Tool Use Agent — agentic loop |