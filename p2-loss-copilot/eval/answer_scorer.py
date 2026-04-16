"""
answer_scorer.py
Agent Validation Framework 

LLM judge that scores each agent answer against the golden set's
ground_truth.must_mention and ground_truth.forbidden fields.
 
Design decisions:
  - Uses claude-sonnet-4-20250514 (same model as the agent) as judge.
    Same-model judging is acceptable here because the judge receives
    structured criteria — it is not grading its own free-form reasoning.
  - Returns a structured JSON verdict, not a prose opinion.
  - Scoring is independent of tool behavior scoring (Session A design).
  - Scores are 0.0, 0.5, or 1.0 with a machine-readable reason code.
 
Score semantics:
  1.0  — all must_mention satisfied, no forbidden content present
  0.5  — all must_mention satisfied but a forbidden item was detected
  0.0  — one or more must_mention items missing (forbidden may also fire)
 
Usage:
    from eval.answer_scorer import score_answer, score_all
 
    # Score one result dict (as produced by eval_runner)
    verdict = score_answer(result)
 
    # Score an entire results list in-place (adds "answer_score" key)
    scored_results = score_all(results)

Author: Eswar Gutlapalli
"""

import json
import time
import anthropic

# Anthropic client
client = anthropic.Anthropic()

JUDGE_MODEL  = "claude-sonnet-4-20250514"
JUDGE_TOKENS = 512

# Judge prompt
JUDGE_SYSTEM = """You are an answer quality judge for an AI agent that answers
credit risk questions using SQL and RAG tools. You receive:

1. The question asked
2. The agent's final answer
3. A list of facts that MUST be mentioned (must_mention)
4. A list of statements that must NOT appear (forbidden)

Your job: return ONLY a JSON object with these exact fields — no preamble,
no markdown, no explanation outside the JSON.

{
  "must_mention_results": [
    {"fact": "<exact fact string>", "present": true|false, "evidence": "<short quote or 'not found'>"}
  ],
  "forbidden_results": [
    {"item": "<exact forbidden string>", "detected": true|false, "evidence": "<short quote or 'not found'>"}
  ],
  "score": 0.0 | 0.5 | 1.0,
  "reason_code": "PASS" | "FORBIDDEN_DETECTED" | "MUST_MENTION_MISSING" | "BOTH_FAIL",
  "summary": "<one sentence>"
}

Scoring rules:
  - score 1.0 + PASS               → all must_mention present AND no forbidden detected
  - score 0.5 + FORBIDDEN_DETECTED → all must_mention present BUT a forbidden item was found
  - score 0.0 + MUST_MENTION_MISSING → one or more must_mention items absent
  - score 0.0 + BOTH_FAIL          → must_mention items missing AND forbidden detected

For must_mention: a fact is present if the answer conveys the same information,
even if worded differently. Do not penalize paraphrasing.
For forbidden: flag only if the exact claim or its clear equivalent appears."""

JUDGE_USER_TEMPLATE = """Question: {question}

Agent answer:
{answer}

Must mention (all required):
{must_mention}

Forbidden (none allowed):
{forbidden}

Return your JSON verdict now."""

def score_answer(result: dict, retry: int = 1) -> dict:
    qid          = result["id"]
    question     = result["question"]
    answer       = result.get("final_answer", "")
    ground_truth = result.get("ground_truth", {})
    must_mention = ground_truth.get("must_mention", [])
    forbidden    = ground_truth.get("forbidden", [])

    # Edge case — no criteria means nothing to judge
    if not must_mention and not forbidden:
        return {
            "id"                  : qid,
            "score"               : 1.0,
            "reason_code"         : "NO_CRITERIA",
            "summary"             : "No must_mention or forbidden criteria defined.",
            "must_mention_results": [],
            "forbidden_results"   : [],
            "judge_error"         : None,
        }

    user_msg = JUDGE_USER_TEMPLATE.format(
        question     = question,
        answer       = answer or "(no answer returned)",
        must_mention = json.dumps(must_mention, indent=2),
        forbidden    = json.dumps(forbidden, indent=2),
    )

    for attempt in range(retry + 1):
        try:
            response = client.messages.create(
                model      = JUDGE_MODEL,
                max_tokens = JUDGE_TOKENS,
                system     = JUDGE_SYSTEM,
                messages   = [{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            verdict          = json.loads(raw)
            verdict["id"]    = qid
            verdict["judge_error"] = None
            return verdict

        except json.JSONDecodeError as exc:
            if attempt < retry:
                time.sleep(1)
                continue
            return {
                "id"                  : qid,
                "score"               : 0.0,
                "reason_code"         : "JUDGE_PARSE_ERROR",
                "summary"             : f"Judge returned non-JSON: {str(exc)[:80]}",
                "must_mention_results": [],
                "forbidden_results"   : [],
                "judge_error"         : str(exc),
            }
        except Exception as exc:
            return {
                "id"                  : qid,
                "score"               : 0.0,
                "reason_code"         : "JUDGE_API_ERROR",
                "summary"             : f"API error: {str(exc)[:80]}",
                "must_mention_results": [],
                "forbidden_results"   : [],
                "judge_error"         : str(exc),
            }
        
def score_all(results: list[dict], delay_s: float = 0.3) -> list[dict]:
    print(f"\n▶  Scoring answers — {len(results)} queries")

    for i, result in enumerate(results):
        verdict = score_answer(result)
        result["answer_verdict"] = verdict

        score  = verdict["score"]
        reason = verdict["reason_code"]
        icon   = "✅" if score == 1.0 else ("⚠️ " if score == 0.5 else "❌")
        print(f"  {icon}  {result['id']:5s}  answer={score:.1f}  {reason}")

        if i < len(results) - 1:
            time.sleep(delay_s)

    passed  = sum(1 for r in results if r["answer_verdict"]["score"] == 1.0)
    partial = sum(1 for r in results if r["answer_verdict"]["score"] == 0.5)
    failed  = sum(1 for r in results if r["answer_verdict"]["score"] == 0.0)
    print(f"\n  Answer scoring complete — ✅ {passed}  ⚠️  {partial}  ❌ {failed}")
    return results

