# AI Portfolio — Credit Risk Intelligence
> A progressive series of AI systems built on the Anthropic Claude API, demonstrating the evolution from simple LLM calls to production-grade agentic architectures — applied to credit risk analytics.

**Author:** Eswar Gutlapalli | **Goal:** AI Career Transition → Anthropic / OpenAI / FAANG  
**Stack:** Python 3.11 · Anthropic Claude API · LangChain · FAISS · HuggingFace Embeddings · Streamlit · SQLite · pandas

---

## Portfolio Progression

| Project | What It Is | Key Concept |
|---|---|---|
| [P1 — Financial Insight Generator](#p1--financial-insight-generator) | LLM over structured CSV data | Prompt engineering, API basics |
| [P2 — Loss Intelligence Copilot](#p2--loss-intelligence-copilot) | RAG over unstructured documents | Embeddings, vector search, retrieval |
| [P3 — Autonomous Risk Analyst Agent](#p3--autonomous-risk-analyst-agent) | Multi-source agent: RAG + SQL | Agent routing, nl_to_sql, synthesis |
| [P4 — Native Tool Use Agent](#p4--native-tool-use-agent) | Production agentic loop | Native tool use, multi-step reasoning |

---

## P1 — Financial Insight Generator

### What It Does
Reads a structured loan portfolio CSV, computes loss rates by segment and quarter using pandas, builds a context-aware prompt, and generates a board-ready executive risk narrative using the Anthropic Claude API. No UI — pure Python pipeline from data to insight.

### Why I Built It
To establish the foundational pattern: domain data → structured prompt → LLM synthesis. Credit risk analysts spend hours writing portfolio commentary. This proves a Claude-powered pipeline can produce the same output in seconds, with the right prompt engineering.

### Key Technical Decisions
- **Pandas for computation, Claude for narrative** — separation of concerns. Claude does not compute loss rates; pandas does. Claude only narrates what pandas computed. This prevents hallucinated numbers.
- **Prompt includes computed context, not raw CSV** — feeding raw CSV to Claude is noisy and expensive. The prompt includes only the aggregated metrics Claude needs to write the summary.
- **No dollar signs in output** — Streamlit renders `$...$` as LaTeX math. System prompt instructs Claude to write `USD X.XM` instead. Prompt engineering fix, not a code fix.

### Architecture
```
loan_data.csv
↓
pandas — compute loss rates by segment and quarter
↓
prompt builder — format metrics into Claude context
↓
Claude API — generate executive narrative
↓
console / Streamlit output
```

### How to Run It
```bash
cd p1-financial-insight
pip install -r requirements.txt
cp ../.env.example .env          # add your ANTHROPIC_API_KEY
python insight_generator.py      # console output
streamlit run app.py             # Streamlit UI
```

### Sample Output
```
Given loan portfolio data across 4 segments and 4 quarters, Claude generated:

"The USD 28,655M loan portfolio presents elevated and accelerating risk
in the unsecured lending segments. Personal Loans and Credit Cards carry
annual average loss rates of 3.4% and 3.38% respectively, with Q4 rates
climbing to 3.80% and 3.70% — signaling deteriorating credit quality
entering the new year. Immediate portfolio review is recommended for
unsecured consumer exposure, particularly in regions with rising
unemployment correlation."
```

---

## P2 — Loss Intelligence Copilot

### What It Does
A Retrieval-Augmented Generation (RAG) system over unstructured loss documents. Users upload a `.txt` loss report, ask natural language questions, and Claude synthesizes answers grounded in the document — not from its training data. Built with FAISS vector search and HuggingFace embeddings.

### Why I Built It
P1 worked on structured data. Real credit risk analysis relies heavily on unstructured data — narrative loss memos, examiner reports, board commentary. P2 solves the problem of making Claude answer questions about *your specific documents*, not generic knowledge.

### Key Technical Decisions
- **FAISS over managed vector DBs** — chose FAISS (local, in-memory) over Pinecone or Weaviate to keep the stack simple and free. For a portfolio demonstrator, operational complexity of a managed DB adds no value.
- **HuggingFace embeddings, not OpenAI** — `sentence-transformers/all-MiniLM-L6-v2` runs locally, no additional API cost, and is fast enough for small document sets.
- **Chunk retrieval, not full document** — feeding the entire document into Claude's context is expensive and dilutes relevance. Retrieval returns only the top-k most semantically similar chunks.
- **Synthesizer system prompt role** — system prompt frames Claude as "an analyst reading documents." This grounds Claude's persona to the retrieval context and reduces hallucination on out-of-document questions.

### Architecture
```
User uploads .txt document
↓
rag/embedder.py — chunk + embed + build FAISS index
↓
User asks question
↓
rag/retriever.py — semantic search → top-k chunks
↓
llm/synthesizer.py — synthesize() — Claude generates answer from chunks
↓
Streamlit UI
```

### How to Run It
```bash
cd p2-loss-copilot
pip install -r requirements.txt
cp ../.env.example .env          # add your ANTHROPIC_API_KEY
streamlit run app.py
```
Upload `data/sample_losses.txt` to test immediately.

### Sample Output
```
Query: "What caused the Midwest delinquencies?"

Claude synthesized: "The Midwest delinquency spike was driven primarily
by auto loan defaults concentrated in manufacturing-dependent counties,
where unemployment rose 2.1 percentage points following plant closures
in Q2. Secondary pressure came from adjustable-rate mortgage resets
affecting borrowers who had not refinanced during the low-rate window.
The combination of income shock and payment shock created a compounding
delinquency effect not observed in other regions."
```
---

## P3 — Autonomous Risk Analyst Agent

### What It Does
Combines RAG (unstructured documents) and SQL (structured database) into a single multi-source agent. Claude acts as a router — reading the user's question and deciding whether to use RAG, SQL, or both — then a synthesizer combines both results into a unified executive insight.

### Why I Built It
P2 answered qualitative questions well but couldn't give exact figures. Real risk analysis requires both: "Northeast had USD 9.8M in Commercial RE losses" (SQL) AND "driven by rising cap rates and tightening credit conditions" (RAG). P3 builds the bridge.

### Key Technical Decisions
- **Separate synthesizer functions for RAG vs multi-source** — `synthesize()` handles RAG-only (Document objects). `synthesize_multi()` takes separate `rag_context` and `sql_context` string parameters with a system prompt that explicitly distinguishes them. Jamming SQL results into a Document object causes Claude to hallucinate on numbers — it doesn't know it's looking at structured data.
- **Router at temperature=0** — the router Claude call uses `temperature=0` for deterministic classification. LLMs are probabilistic, not guaranteed. Always validate router output before using it as a control signal.
- **Defensive default — else "both"** — if Claude returns anything other than "sql", "rag", or "both", default to "both". Safest option when validation fails.
- **nl_to_sql with schema-only system prompt** — the SQL generation prompt contains only the schema. No examples, no explanation. Returns raw SQL, no markdown, no backticks. `temperature=0` for deterministic output.

### Architecture
```
User Query
↓
agent/router.py — Claude classifies: sql / rag / both
├── RAG path: rag/retriever.py → semantic search → chunks
└── SQL path: sql/query_engine.py → nl_to_sql → run_sql → DataFrame
↓
llm/synthesizer.py — synthesize_multi() — combines both sources
↓
Streamlit UI — shows routing decision + both results + synthesis
```

### How to Run It
```bash
cd p2-loss-copilot
pip install -r requirements.txt
cp ../.env.example .env          # add your ANTHROPIC_API_KEY
python data/create_db.py         # seed losses.db (only needed once)
streamlit run app.py
```

### Sample Output
```
Query: "Give me the top loss regions and explain why"

Agent decision: BOTH

SQL result:
  region        total_loss    avg_delinquency
  Northeast     12,500,000    0.13
  Midwest        7,500,000    0.08
  South          7,100,000    0.12
  West           6,500,000    0.055

Synthesized insight: "Northeast carried the highest total loss at
USD 12.5M, driven by Commercial Real Estate exposure in urban markets
facing post-pandemic occupancy pressure. Midwest showed emerging
deterioration — delinquency rates not yet fully reflected in loss
figures — suggesting forward risk concentration in auto loan portfolios
tied to manufacturing employment. South and West remain within
acceptable ranges."
```

---

## P4 — Native Tool Use Agent

### What It Does
Replaces the manual router from P3 with Claude's native `tools=[]` parameter. Claude is given tool schemas and decides at runtime which tools to call, in what order, and whether to chain results across multiple turns. The system runs an agentic loop — not a single API call — until Claude signals it is done reasoning.

### Why I Built It
P3's router was a classifier: one Claude call returns a label, your code acts on the label. That is not agentic. P4 is the production pattern used at Anthropic and OpenAI: Claude is the orchestrator, tools are its capabilities, and multi-step reasoning emerges from the loop — not from your routing logic.

### Key Technical Decisions

**Dependency injection for tool_executor** — `tool_agent.py` never imports from `app.py`. Instead, `tool_executor` is passed as a callable into `run_agent`. This breaks the circular import that would result from mutual imports, makes the agent reusable across contexts, and makes it fully testable in isolation by swapping in a mock executor.

**messages list as agent memory** — Claude has no memory between API calls. The growing `messages` list is the memory. Every tool call, every result, every Claude response is appended before the next API call. If you reset messages each loop, Claude loses context of what it already called and what came back.

**tool_use_id for result matching** — every `tool_use` block has a unique ID. Tool results reference that ID via `tool_use_id`. This is how Claude matches results to requests, especially critical for parallel tool calls where multiple results arrive for multiple simultaneous requests.

**messages.append outside the for loop** — tool results are collected across the entire for loop, then appended once as a single user turn. Appending inside the loop duplicates earlier results on each iteration, producing malformed conversation history.

**TypedDict return type** — `AgentOutput` TypedDict catches key typos at development time rather than at runtime. A typo like `toll_calls` instead of `tool_calls` produces a valid dict that crashes only when the caller reads the wrong key — TypedDict surfaces this in the IDE immediately.

**Safety exit for unexpected stop_reason** — the while loop has an explicit `else` branch that returns a structured response for any stop reason that is not `end_turn` or `tool_use`. Without this, an unexpected stop reason causes the loop to exit without returning, producing `None`, and crashing the caller with a `KeyError`.

### Architecture
```
User Query
↓
agent/tool_agent.py — run_agent()
│
│  while True:
│  ├── Claude API call (messages + tools)
│  │
│  ├── stop_reason = "tool_use"
│  │   ├── for each tool_use block:
│  │   │   └── tool_executor(name, input) → result string
│  │   └── append tool_results to messages
│  │
│  └── stop_reason = "end_turn"
│      └── return {answer, tool_calls}
│
tool_executor (app.py) — executes whatever Claude requested:
├── search_documents → rag/retriever.py
└── query_database  → sql/query_engine.py
↓
Streamlit UI — shows tool calls + inputs + results + final answer
```

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