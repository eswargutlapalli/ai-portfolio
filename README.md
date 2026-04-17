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
| [P4 — Native Tool Use Agent + Eval Framework](#p4--native-tool-use-agent--eval-framework) | Production agentic loop + validation | Native tool use, eval design, LLM-as-judge |

### P1 — Financial Insight Generator [`v1.0p1`]
Reads structured loan portfolio CSV, computes loss rates by segment,
generates executive-ready narratives using Claude.
**Key concepts:** Anthropic API basics, prompt engineering, pandas

### P2 — Loss Intelligence Copilot [`v1.0p2`]
RAG system over unstructured loss documents. Semantic search with
FAISS + HuggingFace embeddings, synthesized by Claude.
**Key concepts:** RAG, embeddings, vector search, LangChain

### P3 — Autonomous Risk Analyst Agent [`v1.0p3`]
Multi-source agent combining RAG + SQL. Claude routes queries to the
right data source, synthesizes structured and unstructured results.
**Key concepts:** Agent routing, nl_to_sql, multi-source synthesis

### P4 — Native Tool Use Agent + Eval Framework [`v1.0p4-eval`]
Production-grade agentic loop using Claude's native tool use API,
closed with a custom evaluation framework. Claude decides which tools
to call, chains results across turns, handles parallel tool calls.
Validated across 30 golden set queries with variance-aware tool behavior
scoring and LLM-as-judge answer correctness.
**Key concepts:** Native tool use, agentic loop, eval design, LLM-as-judge, variance-aware scoring

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

## P4 Eval Framework

### Why Custom?

Existing frameworks don't address native tool-use agent validation:
HELM scores completion quality. OpenAI Evals uses input/output pairs.
LangSmith and AgentBench don't handle variance-aware tool sequence scoring
for agentic loops. Custom schema was the right call — and demonstrates
production thinking that off-the-shelf tools can't show.

### Architecture

```
eval/
├── golden_set.json      # 30 queries with expected tool behavior + answer criteria
├── eval_runner.py       # Component 1: runs agent, times execution, scores tool behavior
├── answer_scorer.py     # Component 2: LLM-as-judge scores answer correctness
├── report_generator.py  # Component 3: generates 7-section markdown report
├── run_eval.py          # Orchestrator: chains all three, handles CLI args
└── results/             # Runtime outputs — gitignored
```

### How to Run the Eval

```bash
# Always run from eval/ directory
cd p2-loss-copilot/eval

# Full run — all 30 queries
python run_eval.py

# Subset run
python run_eval.py --ids Q001 Q006 Q014

# Skip answer scoring (faster, no judge API calls)
python run_eval.py --no-score
```

Output lands in `eval/results/run_<timestamp>.json` and `eval/results/run_<timestamp>_report.md`.

### Final Scores — Run 4 (reference: `run_20260416_020313`)

| Metric             | Score |
|--------------------|-------|
| Tool behavior      | 0.967 |
| Answer correctness | 0.833 |
| **Combined**       | **0.900** |

**By query type:**

| Type     | Tool  | Answer |
|----------|-------|--------|
| sql_only | 1.000 | 0.950  |
| rag_only | 0.900 | 1.000  |
| hybrid   | 0.967 | 0.700  |

**By difficulty:**

| Difficulty | Tool  | Answer |
|------------|-------|--------|
| easy       | 0.964 | 0.964  |
| medium     | 0.950 | 0.800  |
| hard       | 1.000 | 0.583  |

**System metrics:** p50 latency 13.3s · p95 27.0s · 92,765 tokens total · mean 2.43 loop iterations/query

### Key Findings

**Four-run iteration: 0.183 → 0.900**

Each run surfaced a specific bug — the framework worked as designed.

| Run | Combined | Root Cause |
|-----|----------|------------|
| 1   | 0.183    | Path resolution failures — 22/30 queries hitting wrong db location |
| 2   | 0.725    | Path fixed; golden set labels not yet calibrated to observed behavior |
| 3   | 0.883    | Labels updated; loop guard not yet enforced |
| 4   | 0.900    | Clean baseline |

**The silent failure problem**

`query_engine.py` has graceful error handling — it catches DB exceptions and
returns a clean error dict. Production-correct behavior. But during eval, the
agent received the error summary as a tool result and wrote a polished
"database unavailable" response. The tool behavior scorer gave it 1.0 — the
tool was called. Three runs before the failure was isolated, by reading
`final_answer` fields in the raw JSON output.

Lesson: eval frameworks need visibility into tool *results*, not just tool call *presence*.

**Hybrid queries are the hardest**

Answer score drops to 0.700 for hybrid (RAG + SQL) queries vs. 0.950/1.000
for single-source queries. Multi-source synthesis requires more loop iterations
and is most sensitive to the MAX_ITERATIONS cap.

**Variance-aware scoring**

Temperature=0 reduces but doesn't eliminate variance on complex multi-step
queries. A three-tier system (none/low/high variance) correctly handles
non-determinism without penalizing legitimate investigative tool use on
open-ended queries, while still enforcing strict sequence expectations on
deterministic ones.

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
│   │   ├── expand_data.py      Expands to 100-row dataset (Session A)
│   │   └── sample_losses.txt   20 narrative entries for RAG
│   ├── eval/
│   │   ├── golden_set.json     30 queries — tool behavior + answer criteria
│   │   ├── eval_runner.py      Component 1 — agent runner + tool scorer
│   │   ├── answer_scorer.py    Component 2 — LLM-as-judge answer scorer
│   │   ├── report_generator.py Component 3 — 7-section markdown report
│   │   └── run_eval.py         Orchestrator — CLI entry point
│   └── requirements.txt
│
├── .env.example                API key template
└── .gitignore                  Excludes *.db, .env, venv, eval/results/
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
| `v1.0p4` | P4 | Native Tool Use Agent |
| `v1.0p4-eval` | P4 | Native Tool Use Agent + Eval Framework |