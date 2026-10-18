"""Ask-the-assistant: free-text questions from care staff, answered from the records.

    "summary for P0007"                     -> answered from P0007's records (RAG)
    "what medications changed for P0012?"   -> same
    "run the follow-up for P0005"           -> runs the full 3-agent pipeline
    "which patients need escalation?"       -> answered from the cohort overview

Patient IDs are pulled from the question itself. A question without an ID
is answered from an overview of every patient, plus the records of the
patient discussed in the previous turn, so follow-ups like "and their
medications?" keep working. Answers are grounded only in retrieved
records; every Q&A is logged to storage like the agent outputs.
"""
import re
import uuid
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from . import agents
from .llm import LLMError, ask_structured
from .memory import all_responses, empty_memory, now_iso
from .rag import PatientIndex
from .storage import get_storage

PATIENT_ID = re.compile(r"\bp\s?-?(\d{4})\b", re.IGNORECASE)
MAX_PATIENTS_PER_QUESTION = 3

SYSTEM = (
    "You are the question-answering assistant of a hospital post-discharge "
    "follow-up system, used by care staff. All patient data is synthetic. "
    "Answer only from the records provided; if the answer is not in them, say "
    "so plainly. Always name the patient ID(s) you are talking about. Do not "
    "diagnose or recommend treatment changes -- for clinical decisions, refer "
    "to the clinician."
)


class Answer(BaseModel):
    action: Literal["answer", "run_followup"] = Field(
        description="run_followup only if the user explicitly asks to run / generate / send a new "
                    "follow-up, reminder or escalation check for a patient; otherwise answer"
    )
    answer: str = Field(description="markdown answer for care staff")
    sources: list[str] = Field(description="patient IDs and record sections the answer relies on")


def _shorten(block: str, max_lines: int = 25) -> str:
    lines = block.splitlines()
    if len(lines) <= max_lines:
        return block
    return "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"])


def extract_patient_ids(text: str) -> list[str]:
    ids = [f"P{m}" for m in PATIENT_ID.findall(text)]
    return list(dict.fromkeys(ids))


def patient_context(index: PatientIndex, patient: dict, memory: dict, question: str) -> str:
    pid = patient["patient_id"]
    docs = index.retrieve(pid, question)
    sections = "\n\n".join(f"[{pid} / {d.metadata['section']}]\n{d.page_content}" for d in docs)
    rules = agents.escalation_rules(patient, all_responses(patient, memory))
    latest = memory["escalations"][-1] if memory["escalations"] else None
    status = (f"[{pid} / current status]\nRule-based risk flag right now: {rules.risk_flag}"
              f" ({'; '.join(rules.reasons) or 'no concerns'}).")
    if latest:
        status += f"\nLast follow-up run {latest['created_at']}: flag {latest['risk_flag']}. {latest['clinician_alert']}"
    if memory["reminders"]:
        r = memory["reminders"][-1]
        status += f"\nLast reminder sent {r['created_at']} via {r['channel']}: {r['text']}"
    return f"{sections}\n\n{status}"


def cohort_context(patients: list[dict], memories: dict[str, dict]) -> str:
    lines = ["Overview of all patients (current rule-based flag; last run flag if any):"]
    for p in patients:
        memory = memories.get(p["patient_id"]) or empty_memory(p["patient_id"])
        rules = agents.escalation_rules(p, all_responses(p, memory))
        last = memory["escalations"][-1]["risk_flag"] if memory["escalations"] else "not run"
        upcoming = [a for a in p["appointments"] if a["status"] == "scheduled"]
        lines.append(
            f"- {p['patient_id']} {p['demographics']['name']}: {p['discharge']['diagnosis']}, "
            f"discharged {p['discharge']['date']}; flag now={rules.risk_flag}"
            f"{' (' + '; '.join(rules.reasons) + ')' if rules.reasons else ''}; last run={last}; "
            f"next visit={upcoming[0]['type'] + ' ' + upcoming[0]['date'] if upcoming else 'none'}"
        )
    return "\n".join(lines)


def answer_question(
    index: PatientIndex,
    question: str,
    patients: dict[str, dict],
    memories: dict[str, dict],
    history: list[dict] | None = None,
    previous_patient_ids: list[str] | None = None,
) -> dict:
    """Returns {"patient_ids", "unknown_ids", "action", "answer", "sources", "error"}."""
    history = history or []
    asked = extract_patient_ids(question)
    unknown = [pid for pid in asked if pid not in patients]
    pids = [pid for pid in asked if pid in patients][:MAX_PATIENTS_PER_QUESTION]

    blocks = []
    if not asked:
        blocks.append(cohort_context(list(patients.values()), memories))
        pids = [pid for pid in (previous_patient_ids or []) if pid in patients]
    for pid in pids:
        memory = memories.get(pid) or empty_memory(pid)
        blocks.append(patient_context(index, patients[pid], memory, question))

    if unknown and not pids and asked:
        return {"patient_ids": [], "unknown_ids": unknown, "action": "answer", "sources": [], "error": None,
                "answer": f"I couldn't find {', '.join(unknown)}. Patient IDs run from "
                          f"{min(patients)} to {max(patients)}."}

    conversation = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:]) or "(none)"
    prompt = f"""Records:
<records>
{chr(10).join(blocks)}
</records>

Earlier in this conversation:
{conversation}

Today is {date.today().isoformat()}.
Question from care staff: {question}
{f"(Unknown patient IDs in the question: {', '.join(unknown)})" if unknown else ""}"""

    result = {"patient_ids": pids, "unknown_ids": unknown, "error": None}
    try:
        reply = ask_structured(prompt, Answer, system=SYSTEM, effort="medium")
        result.update(action=reply.action, answer=reply.answer, sources=reply.sources)
    except LLMError as exc:
        result.update(action="answer", sources=[], error=str(exc),
                      answer="The LLM is unavailable right now, so here are the matching records:\n\n"
                             + "\n\n".join(f"```\n{_shorten(b)}\n```" for b in blocks))
    if result["action"] == "run_followup" and len(pids) != 1:
        result["action"] = "answer"  # the pipeline runs for exactly one patient

    get_storage().write_json(
        f"logs/{date.today().isoformat()}/qa/{uuid.uuid4().hex[:12]}.json",
        {"logged_at": now_iso(), "question": question, **result},
    )
    return result
