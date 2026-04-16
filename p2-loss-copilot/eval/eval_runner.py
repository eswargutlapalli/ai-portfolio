"""
eval_runner.py

Loops all 30 golden set entries. For each entry:
  1. Calls run_agent() from agent/tool_agent.py
  2. Captures observed tool sequence
  3. Scores tool behavior using score_tool_behavior()
  4. Collects system metrics (latency, tokens, loop iterations)
 
Writes results to eval/results/run_<timestamp>.json

Author: Eswar Gutlapalli
"""

import json
import time
import sys
import pathlib
import statistics 
from datetime import datetime

# ── constants ─-
# NOTE: Always run scripts from the project root:
#   cd p2-loss-copilot
#   python eval/eval_runner.py

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from agent.tool_agent import run_agent

GOLDEN_SET_PATH = pathlib.Path("../eval/golden_set.json")
RESULTS_DIR = pathlib.Path("../eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#  Tool behavior scorer  (from Session A spec)
def score_tool_behavior(observed: list[str], entry: dict) -> dict:
    behavior = entry["tool_call_behavior"]
    expected_types = set(entry["expected_tools"])
    observed_types = set(observed)
    variance = behavior["variance"]

    # Check 1 — minimum tool calls made
    if len(observed_types) < behavior["min_required"]:
        return {"score": 0.0, "reason": f"below min_required ({len(observed_types) < behavior['min_required']}) "}

    # Check 2 — all required tool types present
    if not expected_types.issubset(observed_types):
        missing = expected_types - observed_types
        return {"score": 0.5, "reason": f"missing tool types: {sorted(missing)}"}
    
    # Check 3 — call count within variance-aware bounds
    lo, hi = behavior["observed_range"]

    if len(observed) <= hi:
        return {"score": 1.0, "reason": f"within observed range [{lo}, {hi}]"}
    
    # Above upper bound — score depends on variance class
    if variance == "high":
        return {"score": 1.0, "reason": f"high variance - extra calls acceptable"}
    if variance == "low":
        return {"score": 0.5, "reason": f"right tools, count above observed range (got {len(observed)}, max {hi})"}
    return {"score": 0.0, "reason": f"none-variance query exceeded range (got {len(observed)}, max{hi})"}
    
#  Single query runner
def run_single(entry: dict) -> dict:
    question = entry["question"]
    qid = entry["id"]

    print(f"\n{'-'*60}")
    print(f"  {qid}  |  {question[:70]}")

    t_start = time.perf_counter()

    try:
        agent_output = run_agent(question)
        success = True
        error_msg = None
    except Exception as exc:
        agent_output = {}
        success = False
        error_msg = str(exc)
        print(f"  ⚠  Agent raise exception: {exc}")

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    tool_calls_raw = agent_output.get("tool_calls", [])
    observed_tools = [t["name"] for t in tool_calls_raw]
    final_answer = agent_output.get("answer", "")
    usage = agent_output.get("usage", {})
    loop_iterations = len(observed_tools)

    tool_score = score_tool_behavior(observed_tools, entry)

    print(f"  Tools observed : {observed_tools}")
    print(f"  Tool score : {tool_score['score']} - {tool_score['reason']}")
    print(f"  Latency : {elapsed_ms:.0f} ms")

    return {
        "id"              : qid,
        "question"        : question,
        "query_type"      : entry["query_type"],
        "difficulty"      : entry["difficulty"],
        "reviewer_audience": entry["reviewer_audience"],
        "sequence"        : entry["sequence"],
        "variance"        : entry["tool_call_behavior"]["variance"],
        "success"         : success,
        "error"           : error_msg,
        "observed_tools"  : observed_tools,
        "expected_tools"  : entry["expected_tools"],
        "tool_score"      : tool_score["score"],
        "tool_score_reason": tool_score["reason"],
        "final_answer"    : final_answer,
        "ground_truth"    : entry["ground_truth"],
        "latency_ms"      : elapsed_ms,
        "input_tokens"    : usage.get("input_tokens", 0),
        "output_tokens"   : usage.get("output_tokens", 0),
        "total_tokens"    : usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "loop_iterations" : loop_iterations,
        "notes"           : entry.get("notes", ""),
        "verify"          : entry.get("verify", False),
    }

#  System metrics aggregator
def compute_system_metrics(results: list[dict]) -> dict:
    successful = [r for r in results if r["success"]]
    latencies = [r["latency_ms"] for r in successful]
    total_tokens = [r["total_tokens"] for r in successful]
    loop_iters = [r["loop_iterations"] for r in successful]

    def pct(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p/100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    return {
        "total_queries"  : len(results),
        "successful_runs": len(successful),
        "failed_runs"    : len(results) - len(successful),
        "latency_ms": {
            "min" : round(min(latencies, default=0), 1),
            "max" : round(max(latencies, default=0), 1),
            "mean": round(statistics.mean(latencies), 1) if latencies else 0,
            "p50" : round(pct(latencies, 50), 1),
            "p95" : round(pct(latencies, 95), 1),
        },
        "tokens": {
            "total_consumed" : sum(total_tokens),
            "mean_per_query" : round(statistics.mean(total_tokens), 1) if total_tokens else 0,
            "max_single_query": max(total_tokens, default=0),
        },
        "loop_iterations": {
            "mean": round(statistics.mean(loop_iters), 2) if loop_iters else 0,
            "max" : max(loop_iters, default=0),
            "distribution": {
                str(i): loop_iters.count(i)
                for i in sorted(set(loop_iters))
            }
        },
    }

#  Main eval loop
def run_eval(golden_set_path: pathlib.Path = GOLDEN_SET_PATH, 
             query_ids: list[str] | None = None) -> pathlib.Path:
    
    golden = json.loads(golden_set_path.read_text())

    if query_ids:
        golden = [e for e in golden if e["id"] in query_ids]
        print(f"\n▶ Running subset: {[e['id'] for e in golden]}")
    else:
        print(f"\n▶ Running full golden set - {len(golden)} queries")

    results = []
    t_total = time.perf_counter()

    for entry in golden:
        result = run_single(entry)
        results.append(result)

    total_elapsed = (time.perf_counter() - t_total) * 1000
    system_metrics = compute_system_metrics(results)

    print(f"\n{'='*60}")
    print(f"  Eval complete - {len(results)} queries in {total_elapsed/1000:.1f}s")

    output = {
        "run_id"          : datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        "timestamp"       : datetime.now().isoformat(),
        "total_elapsed_ms": round(total_elapsed, 1),
        "system_metrics"  : system_metrics,
        "results"         : results,
    }

    out_path = RESULTS_DIR / f"{output['run_id']}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results → {out_path}")
    return out_path

# CLI entry point
if __name__ == "__main__":
    ids = sys.argv[1:] or None
    run_eval(query_ids=ids)