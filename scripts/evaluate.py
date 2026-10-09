"""End-to-end evaluation (plan T7.1): run the full pipeline on N patients and
record how accurately the Escalation Agent flags them.

The expected flag comes from each synthetic patient's generator `scenario`
(never shown to the agents):

    routine -> none   single_miss -> watch
    non_responder / warning_symptom / missed_critical -> escalate

    python scripts/evaluate.py --count 20            # real LLM calls (Ollama / Gemini free tier)
    python scripts/evaluate.py --count 20 --dry-run  # rules only, no API calls

Writes docs/test_report.md. Note: runs append to each patient's long-term
memory, like real runs do.
"""
import argparse
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import agents  # noqa: E402
from src.memory import all_responses, load_memory  # noqa: E402
from src.storage import load_patients  # noqa: E402

EXPECTED = {
    "routine": "none",
    "single_miss": "watch",
    "non_responder": "escalate",
    "warning_symptom": "escalate",
    "missed_critical": "escalate",
}
REPORT = os.path.join(os.path.dirname(__file__), "..", "docs", "test_report.md")


def pick(patients: list[dict], count: int) -> list[dict]:
    """Round-robin across scenarios so the sample covers every edge case."""
    buckets = {s: [p for p in patients if p["scenario"] == s] for s in EXPECTED}
    sample = []
    while len(sample) < count and any(buckets.values()):
        for s in EXPECTED:
            if buckets[s] and len(sample) < count:
                sample.append(buckets[s].pop(0))
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="evaluate the deterministic rules only")
    args = parser.parse_args()

    sample = pick(sorted(load_patients(), key=lambda p: p["patient_id"]), args.count)
    rows = []
    if args.dry_run:
        for p in sample:
            flag = agents.escalation_rules(p, all_responses(p, load_memory(p["patient_id"]))).risk_flag
            rows.append((p, flag, flag, []))
    else:
        from src.rag import load_index
        from src.workflow import build_workflow, run_followup

        workflow = build_workflow(load_index())
        for p in sample:
            state = run_followup(workflow, p["patient_id"])
            esc = state["escalation"]
            rows.append((p, esc["risk_flag"], esc["rule_flag"], state["errors"]))
            print(f"{p['patient_id']} {p['scenario']:<16} expected={EXPECTED[p['scenario']]:<8} got={esc['risk_flag']}")

    correct = sum(flag == EXPECTED[p["scenario"]] for p, flag, _, _ in rows)
    missed_escalations = sum(EXPECTED[p["scenario"]] == "escalate" and flag != "escalate" for p, flag, _, _ in rows)
    by_scenario = Counter(p["scenario"] for p, *_ in rows)
    ok_by_scenario = Counter(p["scenario"] for p, flag, _, _ in rows if flag == EXPECTED[p["scenario"]])
    fallbacks = sum(bool(errs) for *_, errs in rows)

    lines = [
        "# Test report — end-to-end escalation accuracy",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='minutes')}",
        f"- Mode: {'rules only (dry run)' if args.dry_run else 'full pipeline (RAG + 3 agents)'}",
        f"- Patients: {len(rows)}",
        f"- **Flag accuracy: {correct}/{len(rows)} ({correct / max(len(rows), 1):.0%})**",
        f"- Missed escalations (expected escalate, got lower): {missed_escalations}",
        f"- Runs that used a fallback after an LLM error: {fallbacks}",
        "",
        "| Scenario | Expected | Correct |",
        "|---|---|---|",
        *[f"| {s} | {EXPECTED[s]} | {ok_by_scenario[s]}/{by_scenario[s]} |" for s in EXPECTED if by_scenario[s]],
        "",
        "| Patient | Scenario | Expected | Rules | Final | Errors |",
        "|---|---|---|---|---|---|",
        *[f"| {p['patient_id']} | {p['scenario']} | {EXPECTED[p['scenario']]} | {rule} | {flag} | "
          f"{'; '.join(errs) or '—'} |" for p, flag, rule, errs in rows],
        "",
    ]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nAccuracy {correct}/{len(rows)} -- report written to {os.path.relpath(REPORT)}")


if __name__ == "__main__":
    main()
