"""The three follow-up agents (plan T3.1-T3.3).

Each agent is a plain function: inputs in, a structured (Pydantic) result
out. workflow.py wires them into the sequential LangGraph pipeline and owns
memory, logging and error handling.

  1. Patient Summary Agent -- RAG context -> FollowUpSummary
  2. Reminder Agent        -- summary + reminder history -> Reminder
  3. Escalation Agent      -- summary + reminder history + responses -> Escalation
                              (deterministic rules + LLM risk flag)

Guardrails: synthetic data only; reminders never contain clinical advice
(checked after generation); escalations are alerts for a human clinician,
never automated actions.
"""
import re
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from .llm import ask_structured

RiskFlag = Literal["none", "watch", "escalate"]
FLAG_ORDER = {"none": 0, "watch": 1, "escalate": 2}

SYSTEM = (
    "You are part of a hospital post-discharge follow-up assistant. All patient "
    "data is synthetic. Use only facts present in the provided records; if "
    "something is not in the records, leave it out rather than guessing."
)


# --- Output schemas --------------------------------------------------------------


class Medication(BaseModel):
    name: str
    instructions: str = Field(description="dose and frequency exactly as written in the record")
    change: Literal["new", "changed", "unchanged", "stopped"]


class DueVisit(BaseModel):
    appointment_id: str
    type: str
    date: str
    critical: bool
    status: Literal["scheduled", "completed", "missed"]


class FollowUpSummary(BaseModel):
    diagnosis: str
    discharge_date: str
    medications: list[Medication]
    due_visits: list[DueVisit]
    open_tasks: list[str] = Field(description="concrete follow-up actions still outstanding")
    narrative: str = Field(description="clinician-facing summary, under 100 words")


class Reminder(BaseModel):
    appointment_id: str | None = Field(description="the visit this reminder is about, or null if none")
    channel: Literal["sms", "email", "phone"]
    send_at: str = Field(description="ISO 8601 date-time to send, in the future")
    text: str
    skipped_duplicate: bool = Field(
        default=False, description="true if a near-identical reminder was already sent recently"
    )


class Escalation(BaseModel):
    risk_flag: RiskFlag
    reasons: list[str]
    clinician_alert: str = Field(description="short note for the care team; empty if risk_flag is none")


# --- 1. Patient Summary Agent -----------------------------------------------------


def summary_agent(patient_id: str, context: str) -> FollowUpSummary:
    prompt = f"""You are the Patient Summary Agent.

Retrieved records for patient {patient_id}:
<records>
{context}
</records>

Produce a follow-up summary: diagnosis, discharge date, every discharge
medication with its change status, every appointment (due visits) with its
status, and the open follow-up tasks (e.g. upcoming visits to attend,
medication changes to confirm, missed visits to rebook). Today is {date.today().isoformat()}."""
    return ask_structured(prompt, FollowUpSummary, system=SYSTEM, effort="medium")


def fallback_summary(patient: dict) -> FollowUpSummary:
    """Built straight from the structured record when the LLM call fails."""
    return FollowUpSummary(
        diagnosis=patient["discharge"]["diagnosis"],
        discharge_date=patient["discharge"]["date"],
        medications=[Medication(name=m["name"], instructions=f"{m['dose']}, {m['frequency']}", change=m["change"])
                     for m in patient["medications"]],
        due_visits=[DueVisit(**{k: a[k] for k in DueVisit.model_fields}) for a in patient["appointments"]],
        open_tasks=[f"Attend {a['type']} on {a['date']}" for a in patient["appointments"] if a["status"] == "scheduled"]
        + [f"Rebook missed {a['type']} ({a['date']})" for a in patient["appointments"] if a["status"] == "missed"],
        narrative="(LLM unavailable -- summary generated directly from the structured record.)",
    )


# --- 2. Reminder Agent ------------------------------------------------------------

# Dose instructions or "start/stop/increase" style directions = clinical advice.
_CLINICAL_ADVICE = re.compile(
    r"\b\d+(\.\d+)?\s?(mg|mcg|g|ml|units?)\b|\b(increase|decrease|stop taking|start taking|skip (a |your )?dose)\b",
    re.IGNORECASE,
)


def contains_clinical_advice(text: str) -> bool:
    return bool(_CLINICAL_ADVICE.search(text))


def reminder_agent(patient: dict, summary: FollowUpSummary, past_reminders: list[dict],
                   responses: list[dict]) -> Reminder:
    d = patient["demographics"]
    history = "\n".join(
        f"- {r['created_at']} via {r['channel']} about {r.get('appointment_id')}: {r['text']}"
        for r in past_reminders[-5:]
    ) or "- none"
    replies = "\n".join(f"- {r['date']} via {r['channel']}: {r['status']} {r['message']}".rstrip()
                        for r in responses[-5:]) or "- none"
    prompt = f"""You are the Reminder Agent.

Patient first name: {d['name'].split()[0]}
Preferred channel: {d['preferred_channel']}
Now: {datetime.now().isoformat(timespec='minutes')}

Follow-up summary:
{summary.model_dump_json(indent=2)}

Reminders already sent by this system (most recent last):
{history}

Recent patient responses (most recent last):
{replies}

Write ONE personalized reminder for the most pressing open task (normally the
next scheduled visit, or rebooking a missed one).
- Channel: the preferred channel, unless the patient has repeatedly not
  responded on it -- then pick another channel.
- Send time: a sensible local daytime slot in the future, before the visit.
- If an equivalent reminder for the same visit was sent in the last 2 days,
  set skipped_duplicate=true and still return the text you would have sent.
- Tone: warm and brief (SMS under 300 characters; email may be longer;
  phone = a short call script).
- NO clinical advice: no doses, no telling them to start/stop/change any
  medicine, no interpreting symptoms. For medication questions or if they
  feel unwell, tell them to contact their care team."""
    reminder = ask_structured(prompt, Reminder, system=SYSTEM, effort="low")
    if contains_clinical_advice(reminder.text):
        reminder.text = safe_reminder_text(patient, summary)
    return reminder


def _next_visit(summary: FollowUpSummary) -> DueVisit | None:
    missed = [v for v in summary.due_visits if v.status == "missed"]
    upcoming = sorted((v for v in summary.due_visits if v.status == "scheduled"), key=lambda v: v.date)
    return (missed or upcoming or [None])[0]


def safe_reminder_text(patient: dict, summary: FollowUpSummary) -> str:
    name = patient["demographics"]["name"].split()[0]
    visit = _next_visit(summary)
    if visit is None:
        return f"Hi {name}, this is your care team checking in after your discharge. Reply or call us if you need anything."
    if visit.status == "missed":
        return f"Hi {name}, we missed you at your {visit.type} on {visit.date}. Please call us to rebook."
    return (f"Hi {name}, a reminder of your {visit.type} on {visit.date}. "
            "Questions about your medicines? Please contact your care team.")


def fallback_reminder(patient: dict, summary: FollowUpSummary) -> Reminder:
    visit = _next_visit(summary)
    send_at = (datetime.now() + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
    return Reminder(
        appointment_id=visit.appointment_id if visit else None,
        channel=patient["demographics"]["preferred_channel"],
        send_at=send_at.isoformat(timespec="minutes"),
        text=safe_reminder_text(patient, summary),
    )


# --- 3. Escalation Agent ------------------------------------------------------------


def trailing_no_responses(responses: list[dict]) -> int:
    count = 0
    for r in reversed(responses):
        if r["status"] != "no_response":
            break
        count += 1
    return count


def reported_warning_signs(patient: dict, responses: list[dict]) -> list[str]:
    signs = patient["discharge"]["warning_signs"]
    found = []
    for r in responses:
        text = r["message"].lower()
        found += [s for s in signs if s.lower() in text and s not in found]
    return found


def escalation_rules(patient: dict, responses: list[dict]) -> Escalation:
    """Deterministic rules from the architecture doc. The LLM can raise this
    flag but never lower it."""
    escalate, watch = [], []

    misses = trailing_no_responses(responses)
    if misses >= 2:
        escalate.append(f"missed {misses} reminders in a row")
    elif misses == 1:
        watch.append("did not respond to the latest reminder")

    for sign in reported_warning_signs(patient, responses):
        escalate.append(f"reported warning symptom: {sign}")

    for a in patient["appointments"]:
        if a["status"] == "missed":
            (escalate if a["critical"] else watch).append(
                f"missed {'critical ' if a['critical'] else ''}{a['type']} on {a['date']}"
            )

    if escalate:
        return Escalation(risk_flag="escalate", reasons=escalate + watch, clinician_alert="")
    if watch:
        return Escalation(risk_flag="watch", reasons=watch, clinician_alert="")
    return Escalation(risk_flag="none", reasons=[], clinician_alert="")


def escalation_agent(patient: dict, summary: FollowUpSummary, past_reminders: list[dict],
                     responses: list[dict], rules: Escalation) -> Escalation:
    reminders = "\n".join(f"- {r['created_at']} via {r['channel']}: {r['text']}" for r in past_reminders[-5:]) or "- none"
    replies = "\n".join(f"- {r['date']} via {r['channel']}: {r['status']} {r['message']}".rstrip()
                        for r in responses) or "- none"
    prompt = f"""You are the Escalation Agent. Decide the patient's follow-up risk flag.

Follow-up summary:
{summary.model_dump_json(indent=2)}

Warning signs for this condition: {', '.join(patient['discharge']['warning_signs'])}

Reminders already sent:
{reminders}

Patient responses (oldest first):
{replies}

Hospital escalation rules already evaluated: flag={rules.risk_flag}; reasons={rules.reasons or 'none'}

Flags: none = on track; watch = some concern, keep monitoring;
escalate = a clinician must contact the patient (e.g. 2+ missed reminders in a
row, any warning symptom in a reply, a missed critical appointment, or another
clear safety concern in the responses).
You may raise the flag above the rules' result, never lower it. If the flag is
watch or escalate, write a short clinician_alert (who, what happened, what to
check); otherwise leave it empty. Do not diagnose or recommend treatment."""
    llm = ask_structured(prompt, Escalation, system=SYSTEM, effort="medium")
    return combine(rules, llm, patient)


def combine(rules: Escalation, llm: Escalation | None, patient: dict) -> Escalation:
    """Final flag = the higher of rules and LLM; reasons merged; alert always present when flagged."""
    if llm is None or FLAG_ORDER[rules.risk_flag] > FLAG_ORDER[llm.risk_flag]:
        flag = rules.risk_flag
    else:
        flag = llm.risk_flag
    reasons = list(dict.fromkeys(rules.reasons + (llm.reasons if llm else [])))
    alert = (llm.clinician_alert if llm else "").strip()
    if flag != "none" and not alert:
        d = patient["demographics"]
        alert = (f"{d['name']} ({patient['patient_id']}, {patient['discharge']['diagnosis']}) flagged "
                 f"{flag.upper()}: {'; '.join(reasons)}. Please review and contact the patient via {d['phone']}.")
    if flag == "none":
        alert = ""
    return Escalation(risk_flag=flag, reasons=reasons, clinician_alert=alert)
