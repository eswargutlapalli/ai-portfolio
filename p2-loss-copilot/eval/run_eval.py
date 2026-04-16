"""
run_eval.py
Agent Validation Framework 
 
Single entry point that runs the full eval pipeline:
  1. eval_runner     → calls agent for all 30 queries, scores tool behavior
  2. answer_scorer   → LLM judge scores answer correctness
  3. report_generator → produces 7-section markdown report
 
Usage:
    # Full run — all 30 queries
    cd p2-loss-copilot
    python eval/run_eval.py
 
    # Subset run — specific query IDs
    python eval/run_eval.py --ids Q001 Q006 Q014
 
    # Skip answer scoring (tool behavior only — faster, no API calls for judge)
    python eval/run_eval.py --no-score
 
Output:
    eval/results/run_<timestamp>.json       ← full scored results
    eval/results/run_<timestamp>_report.md  ← human-readable report

Author: Eswar Gutlapalli
"""

import argparse
import json
import pathlib

from eval_runner import run_eval, GOLDEN_SET_PATH
from answer_scorer import score_all
from report_generator import generate_report

def main():
    parser = argparse.ArgumentParser(
        description="Run the P4 Agent Validation Framework eval pipeline."
    )
    parser.add_argument(
        "--ids", nargs="*", default=None,
        help="Run only these query IDs (e.g. --ids Q001 Q006). Default: all 30."
    )
    parser.add_argument(
        "--no-score", action="store_true",
        help="Skip answer correctness scoring. Tool behavior only."
    )
    parser.add_argument(
        "--golden", default=str(GOLDEN_SET_PATH),
        help="Path to golden_set.json (default: eval/golden_set.json)"
    )
    args = parser.parse_args()

    print("═" * 60)
    print("  P4 Agent Validation Framework — Session B")
    print("═" * 60)

    # ── Component 1: eval runner ───────────────────────────────────────────
    results_path = run_eval(
        golden_set_path = pathlib.Path(args.golden),
        query_ids       = args.ids or None,
    )

    # ── Load results for scoring ───────────────────────────────────────────
    data    = json.loads(results_path.read_text())
    results = data["results"]

    # ── Component 2: answer correctness scorer ─────────────────────────────
    if not args.no_score:
        results = score_all(results)
        data["results"] = results
        results_path.write_text(json.dumps(data, indent=2))
        print(f"\n  Scored results saved → {results_path}")
    else:
        print("\n  ⚠  Answer scoring skipped (--no-score flag)")

    # ── Component 3: report generator ─────────────────────────────────────
    print("\n" + "─" * 60)
    generate_report(results_path)

    print("\n" + "═" * 60)
    print("  Eval pipeline complete.")
    print(f"  Results : {results_path}")
    print(f"  Report  : {results_path.with_name(results_path.stem + '_report.md')}")
    print("═" * 60)


if __name__ == "__main__":
    main()