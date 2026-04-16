"""
report_generator.py
Agent Validation Framework 
 
Consumes a scored results JSON (output of eval_runner + answer_scorer)
and produces the full 7-section eval report — to stdout and as a .md file.
 
Report sections:
  1. Overall tool behavior score across 30 queries
  2. Accuracy by query type   (sql_only / rag_only / hybrid)
  3. Accuracy by difficulty   (easy / medium / hard)
  4. Accuracy by reviewer audience (anthropic / credit_risk)
  5. Non-determinism report   (high-variance queries)
  6. Failed queries with reason codes
  7. System metrics summary   (latency p50/p95, tokens, loop iterations)
 
Usage:
    python eval/report_generator.py eval/results/run_<timestamp>.json
 
Or import and call:
    from eval.report_generator import generate_report
    md = generate_report(results_path)

Author: Eswar Gutlapalli
"""

import json
import pathlib
import sys
from collections import defaultdict

def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0

def pct_str(num: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{num}/{denom} ({100*num//denom}%)"

def score_bar(score: float) -> str:
    filled = int(score * 20)
    return f"[{'█' * filled}{'░' * (20 - filled)}] {score:.3f}"

def section_1_overall(results: list[dict]) -> str:
    tool_scores   = [r["tool_score"]                for r in results]
    answer_scores = [r["answer_verdict"]["score"]   for r in results if "answer_verdict" in r]

    tool_avg   = avg(tool_scores)
    answer_avg = avg(answer_scores)
    combined   = avg([tool_avg, answer_avg])

    perfect_tool   = sum(1 for s in tool_scores   if s == 1.0)
    perfect_answer = sum(1 for s in answer_scores if s == 1.0)

    lines = [
        "## 1. Overall Scores",
        "",
        f"| Metric             | Score | Bar                          |",
        f"|--------------------| ------|------------------------------|",
        f"| Tool behavior      | {tool_avg:.3f} | {score_bar(tool_avg)} |",
        f"| Answer correctness | {answer_avg:.3f} | {score_bar(answer_avg)} |",
        f"| Combined           | {combined:.3f} | {score_bar(combined)} |",
        "",
        f"- Tool behavior:      {pct_str(perfect_tool,   len(tool_scores))} scored 1.0",
        f"- Answer correctness: {pct_str(perfect_answer, len(answer_scores))} scored 1.0",
    ]
    return "\n".join(lines)


def section_2_by_query_type(results: list[dict]) -> str:
    tool_by_type   = defaultdict(list)
    answer_by_type = defaultdict(list)

    for r in results:
        qt = r["query_type"]
        tool_by_type[qt].append(r["tool_score"])
        if "answer_verdict" in r:
            answer_by_type[qt].append(r["answer_verdict"]["score"])

    lines = [
        "## 2. Accuracy by Query Type",
        "",
        f"| Query Type    | Tool Score | Answer Score |",
        f"|---------------|------------|--------------|",
    ]
    for qt in sorted(tool_by_type):
        ts  = avg(tool_by_type[qt])
        as_ = avg(answer_by_type[qt])
        lines.append(f"| {qt:<13} | {ts:.3f}      | {as_:.3f}        |")
    return "\n".join(lines)

def section_3_by_difficulty(results: list[dict]) -> str:
    order          = ["easy", "medium", "hard"]
    tool_by_diff   = defaultdict(list)
    answer_by_diff = defaultdict(list)
    count_by_diff  = defaultdict(int)

    for r in results:
        d = r["difficulty"]
        tool_by_diff[d].append(r["tool_score"])
        count_by_diff[d] += 1
        if "answer_verdict" in r:
            answer_by_diff[d].append(r["answer_verdict"]["score"])

    lines = [
        "## 3. Accuracy by Difficulty",
        "",
        f"| Difficulty | Count | Tool Score | Answer Score |",
        f"|------------|-------|------------|--------------|",
    ]
    for d in order:
        ts  = avg(tool_by_diff[d])
        as_ = avg(answer_by_diff[d])
        n   = count_by_diff[d]
        lines.append(f"| {d:<10} | {n:<5} | {ts:.3f}      | {as_:.3f}        |")
    return "\n".join(lines)


def section_4_by_audience(results: list[dict]) -> str:
    audiences      = ["anthropic", "credit_risk"]
    tool_by_aud    = defaultdict(list)
    answer_by_aud  = defaultdict(list)

    for r in results:
        for aud in r.get("reviewer_audience", []):
            tool_by_aud[aud].append(r["tool_score"])
            if "answer_verdict" in r:
                answer_by_aud[aud].append(r["answer_verdict"]["score"])

    lines = [
        "## 4. Accuracy by Reviewer Audience",
        "",
        f"| Audience     | Tool Score | Answer Score | Queries |",
        f"|--------------|------------|--------------|---------|",
    ]
    for aud in audiences:
        ts  = avg(tool_by_aud[aud])
        as_ = avg(answer_by_aud[aud])
        n   = len(tool_by_aud[aud])
        lines.append(f"| {aud:<12} | {ts:.3f}      | {as_:.3f}        | {n:<7} |")

    lines += [
        "",
        "> **anthropic lens:** tool selection, hallucination, grounding, multi-hop reasoning  ",
        "> **credit_risk lens:** financial figure accuracy, narrative coherence, domain validity",
    ]
    return "\n".join(lines)

def section_5_nondeterminism(results: list[dict]) -> str:
    high_var = [r for r in results if r.get("variance") == "high"]

    lines = [
        "## 5. Non-Determinism Report",
        "",
        f"High-variance queries: {len(high_var)} of {len(results)}",
        "",
    ]

    if not high_var:
        lines.append("No high-variance queries in this run.")
        return "\n".join(lines)

    lines += [
        f"| ID    | Tools Observed         | Tool Score | Answer Score |",
        f"|-------|------------------------|------------|--------------|",
    ]
    for r in high_var:
        observed_str = ", ".join(r.get("observed_tools", []))
        ts           = r["tool_score"]
        as_          = r.get("answer_verdict", {}).get("score", 0.0)
        lines.append(f"| {r['id']:<5} | {observed_str:<22} | {ts:.1f}        | {as_:.1f}          |")

    lines += [
        "",
        "> Tool behavior score is independent of exact call count for high-variance queries.",
        "> Eval scores tool coverage and answer correctness — not sequence determinism.",
    ]
    return "\n".join(lines)


def section_6_failures(results: list[dict]) -> str:
    agent_errors = [r for r in results if not r.get("success")]
    tool_fails   = [r for r in results if r["tool_score"] < 1.0]
    answer_fails = [r for r in results
                    if "answer_verdict" in r and r["answer_verdict"]["score"] < 1.0]

    lines = ["## 6. Failed Queries", ""]

    if agent_errors:
        lines += ["### Agent Errors (exception raised)", ""]
        for r in agent_errors:
            lines.append(f"- **{r['id']}** — {r.get('error', 'unknown error')}")
        lines.append("")

    if tool_fails:
        lines += ["### Tool Behavior Failures", ""]
        lines += [
            f"| ID    | Score | Reason                                                |",
            f"|-------|-------|-------------------------------------------------------|",
        ]
        for r in tool_fails:
            lines.append(
                f"| {r['id']:<5} | {r['tool_score']:.1f}   | {r['tool_score_reason'][:55]} |"
            )
        lines.append("")

    if answer_fails:
        lines += ["### Answer Correctness Failures", ""]
        lines += [
            f"| ID    | Score | Reason Code           | Summary                              |",
            f"|-------|-------|-----------------------|--------------------------------------|",
        ]
        for r in answer_fails:
            v = r["answer_verdict"]
            lines.append(
                f"| {r['id']:<5} | {v['score']:.1f}   | {v['reason_code']:<21} | {v['summary'][:38]} |"
            )
        lines.append("")

    if not agent_errors and not tool_fails and not answer_fails:
        lines.append("✅ No failures detected in this run.")

    return "\n".join(lines)

def section_7_system_metrics(metrics: dict) -> str:
    lat  = metrics["latency_ms"]
    tok  = metrics["tokens"]
    loop = metrics["loop_iterations"]

    dist_str = "  ".join(
        f"{k} calls × {v} queries"
        for k, v in loop["distribution"].items()
    )

    lines = [
        "## 7. System Metrics",
        "",
        "### Latency",
        f"| Metric | Value    |",
        f"|--------|----------|",
        f"| Min    | {lat['min']:.0f} ms |",
        f"| Mean   | {lat['mean']:.0f} ms |",
        f"| p50    | {lat['p50']:.0f} ms |",
        f"| p95    | {lat['p95']:.0f} ms |",
        f"| Max    | {lat['max']:.0f} ms |",
        "",
        "### Token Consumption",
        f"| Metric           | Value |",
        f"|------------------|-------|",
        f"| Total consumed   | {tok['total_consumed']:,} |",
        f"| Mean per query   | {tok['mean_per_query']:,.0f} |",
        f"| Max single query | {tok['max_single_query']:,} |",
        "",
        "### Loop Iterations",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean   | {loop['mean']:.2f} |",
        f"| Max    | {loop['max']} |",
        "",
        f"Distribution: {dist_str}",
        "",
        "> p95 latency is the production SLA indicator.",
        "> High p95 with low p50 = tail latency from multi-hop flexible queries — expected behavior.",
    ]
    return "\n".join(lines)


def generate_report(results_path: pathlib.Path) -> str:
    data    = json.loads(pathlib.Path(results_path).read_text())
    results = data["results"]
    metrics = data["system_metrics"]
    run_id  = data["run_id"]
    ts      = data["timestamp"]

    header = "\n".join([
        "# Agent Validation Framework — Eval Report",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Timestamp:** {ts}  ",
        f"**Queries:** {metrics['total_queries']} total "
        f"({metrics['successful_runs']} successful, {metrics['failed_runs']} failed)  ",
        f"**Wall time:** {data['total_elapsed_ms']/1000:.1f}s  ",
        "",
        "---",
        "",
    ])

    sections = [
        section_1_overall(results),
        section_2_by_query_type(results),
        section_3_by_difficulty(results),
        section_4_by_audience(results),
        section_5_nondeterminism(results),
        section_6_failures(results),
        section_7_system_metrics(metrics),
    ]

    report = header + "\n\n---\n\n".join(sections)

    report_path = pathlib.Path(results_path).with_name(
        pathlib.Path(results_path).stem + "_report.md"
    )
    report_path.write_text(report)
    print(f"\n📄 Report → {report_path}")
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        results_dir = pathlib.Path("eval/results")
        files       = sorted(results_dir.glob("run_*.json"))
        if not files:
            print("No results files found. Run eval_runner.py first.")
            sys.exit(1)
        results_path = files[-1]
        print(f"Using latest results: {results_path}")
    else:
        results_path = pathlib.Path(sys.argv[1])

    md = generate_report(results_path)
    print("\n" + md)