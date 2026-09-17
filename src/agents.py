"""Agentic workflow for patient follow-up, with memory.

Structured directly after ollama-FAPI/routerbasedagenticworkflowhealthcare.py:
a LangGraph StateGraph where a router node classifies the incoming request
with an LLM call, then conditional edges forward it to the specialist node
that handles it -- each specialist also an LLM call. Swapped
ChatGoogleGenerativeAI for Claude (see src/llm.py -- official `anthropic`
Python SDK, model defaults to claude-opus-5).

  * **Router Agent** is the single entry point. It reads the user's free-form
    query and classifies it into SUMMARY / REMINDER / ESCALATION / FULL, then
    forwards to the one agent that handles it -- callers don't need to know
    which agent does what, they just ask a question. FULL (a general/unclear
    request, e.g. "run followup for P0001") walks the whole pipeline in order:
      Patient Summary Agent -> Reminder Agent -> Escalation Agent

The graph is compiled with a checkpointer, so state persists per patient
(thread_id = patient_id) across separate calls -- e.g. a summary requested on
day 1 is still on record when a reminder is requested on day 3.
"""
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .llm import ask


class FollowUpState(TypedDict, total=False):
    patient: dict
    query: str
    model: str  # which Claude model this run should use; None = src/llm.py's DEFAULT_MODEL
    intent: str
    summary: str
    reminder: str
    escalation_needed: bool
    escalation: str


# --- Router Agent -----------------------------------------------------------


def router_agent(state: FollowUpState) -> dict:
    """Classifies the user's query and decides which agent should handle it."""
    query = state.get("query") or ""

    prompt = f"""
    You are the Router Agent for a hospital patient follow-up system.

    Categorize the user's request into one of the following categories:
    - SUMMARY
    - REMINDER
    - ESCALATION
    - FULL

    SUMMARY: the user wants a clinical follow-up summary only.
    REMINDER: the user wants a reminder message drafted/sent for the patient.
    ESCALATION: the user wants to know if/why the case should be escalated to
      the care team.
    FULL: the request is general, unclear, or explicitly asks for everything
      (e.g. "run followup for this patient").

    User request: {query!r}

    Return only the category name."""

    category = ask(prompt, max_tokens=20, effort="low", model=state.get("model")).strip().upper()

    if "SUMMARY" in category:
        intent = "summary"
    elif "REMINDER" in category:
        intent = "reminder"
    elif "ESCALATION" in category:
        intent = "escalation"
    else:
        intent = "full"

    return {"intent": intent}


def _route_from_router(state: FollowUpState) -> str:
    return state["intent"]


def _continue_if_full(state: FollowUpState) -> str:
    return "continue" if state["intent"] == "full" else "stop"


# --- Patient Summary Agent ---------------------------------------------------


def patient_summary_agent(state: FollowUpState) -> dict:
    """Turns raw patient/discharge data into a concise clinical follow-up summary."""
    patient = state["patient"]

    prompt = f"""
    You are the Patient Summary Agent for a hospital follow-up system.

    Patient record:
    {patient}

    Write a concise clinical follow-up summary covering:
    - Why the patient needs follow-up (diagnosis, discharge context)
    - What follow-up action is due and by when
    - Any risk factors (risk level, missed prior appointments)

    Keep it under 120 words. Do not invent facts not in the record.
    """

    return {"summary": ask(prompt, model=state.get("model"))}


# --- Reminder Agent -----------------------------------------------------------


def reminder_agent(state: FollowUpState) -> dict:
    """Drafts a personalized reminder message for the patient's preferred channel."""
    patient = state["patient"]
    summary = state.get("summary", "(no summary generated yet)")

    prompt = f"""
    You are the Reminder Agent for a hospital follow-up system.

    Patient follow-up summary:
    {summary}

    Preferred contact method: {patient['preferred_contact_method']}
    Follow-up type: {patient['follow_up_type']}
    Follow-up due date: {patient['follow_up_due_date']}

    Write a short, warm, personalized reminder message to the patient
    (addressed to them by name: {patient['name']}) about their upcoming
    follow-up. Match the tone to the contact method (e.g. brief for SMS,
    slightly more detailed for email). Do not include medical advice
    beyond what's in the summary.
    """

    return {"reminder": ask(prompt, model=state.get("model"))}


# --- Escalation Agent ---------------------------------------------------------


def escalation_agent(state: FollowUpState) -> dict:
    """Decides whether a missed/high-risk follow-up needs care-team escalation."""
    patient = state["patient"]
    summary = state.get("summary", "(no summary generated yet)")

    prompt = f"""
    You are the Escalation Agent for a hospital follow-up system.

    Patient follow-up summary:
    {summary}

    Risk level: {patient['risk_level']}
    Missed prior appointments: {patient['missed_appointments']}

    Decide if this case should be escalated to the care team (escalate when
    risk level is high, or 2+ appointments have been missed, or the summary
    indicates urgent concern).

    Respond in this exact format:
    ESCALATE: yes|no
    REASON: <one sentence reason>
    MESSAGE: <if yes, a short escalation note to the care team; if no, write "N/A">
    """

    response = ask(prompt, model=state.get("model"))
    escalate = "escalate: yes" in response.lower()

    return {"escalation_needed": escalate, "escalation": response}


# --- Graph wiring ---------------------------------------------------------
#
#                              +---> patient_summary_agent --(full?)--> reminder_agent --(full?)--> escalation_agent --> END
#                              |            (stop -> END)                    (stop -> END)
#  router_agent ---(intent)----+---> reminder_agent --> END
#                              +---> escalation_agent --> END

graph = StateGraph(FollowUpState)

graph.add_node("router_agent", router_agent)
graph.add_node("patient_summary_agent", patient_summary_agent)
graph.add_node("reminder_agent", reminder_agent)
graph.add_node("escalation_agent", escalation_agent)

graph.set_entry_point("router_agent")

graph.add_conditional_edges(
    "router_agent",
    _route_from_router,
    {
        "summary": "patient_summary_agent",
        "full": "patient_summary_agent",
        "reminder": "reminder_agent",
        "escalation": "escalation_agent",
    },
)

graph.add_conditional_edges(
    "patient_summary_agent",
    _continue_if_full,
    {"continue": "reminder_agent", "stop": END},
)

graph.add_conditional_edges(
    "reminder_agent",
    _continue_if_full,
    {"continue": "escalation_agent", "stop": END},
)

graph.add_edge("escalation_agent", END)

_memory = MemorySaver()
followup_workflow = graph.compile(checkpointer=_memory)


def run_query(patient: dict, query: str = "", model: str | None = None) -> dict:
    """Routes one query for one patient through the graph; keyed by patient_id for memory.

    `model` (e.g. "claude-sonnet-5") overrides the default Claude model for
    every agent call made during this run only -- no server restart, no env
    var change. Omit it to use src/llm.py's DEFAULT_MODEL.
    """
    config = {"configurable": {"thread_id": patient["patient_id"]}}
    return followup_workflow.invoke(
        {"patient": patient, "query": query, "model": model}, config=config
    )


def run_followup(patient: dict) -> dict:
    """Convenience wrapper: always runs the full sequential pipeline for one patient."""
    return run_query(patient, query="")
