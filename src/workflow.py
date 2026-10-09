"""Sequential LangGraph pipeline with short- and long-term memory (plan T4.1-T4.3).

    load_context -> summary_agent -> reminder_agent -> escalation_agent -> persist_memory

* Short-term memory: `FollowUpState`, passed between the nodes of one run, so
  each agent sees everything earlier agents wrote.
* Long-term memory: memory/<patient_id>.json in storage (see memory.py),
  loaded in `load_context` and appended to in `persist_memory`, then re-indexed
  into the vector store so the next run's retrieval sees it.
* Every agent output is logged to logs/<date>/<run_id>/<step>.json.
* A failed LLM call never crashes the run: the agent falls back to a
  deterministic result and the error is recorded in `state["errors"]`.
"""
import logging
import operator
import uuid
from datetime import date
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from . import agents
from .llm import LLMError
from .memory import all_responses, load_memory, now_iso, save_memory
from .rag import PatientIndex
from .storage import get_storage, load_patient

log = logging.getLogger(__name__)

RETRIEVAL_QUERY = "discharge diagnosis, medications and changes, follow-up appointments, reminder responses, history"


class FollowUpState(TypedDict, total=False):
    run_id: str
    patient_id: str
    patient: dict          # structured record (rules + demographics)
    context: str           # RAG-retrieved text that grounds the LLM calls
    memory: dict           # long-term memory loaded at the start of the run
    summary: dict
    reminder: dict
    escalation: dict
    errors: Annotated[list[str], operator.add]


def _log_output(state: FollowUpState, step: str, output: dict) -> None:
    get_storage().write_json(
        f"logs/{date.today().isoformat()}/{state['run_id']}/{step}.json",
        {"run_id": state["run_id"], "patient_id": state["patient_id"], "step": step,
         "logged_at": now_iso(), "output": output},
    )


def build_workflow(index: PatientIndex):
    def load_context(state: FollowUpState) -> dict:
        patient = load_patient(state["patient_id"])
        if patient is None:
            raise KeyError(f"unknown patient_id {state['patient_id']}")
        docs = index.retrieve(state["patient_id"], RETRIEVAL_QUERY)
        context = "\n\n".join(f"[{d.metadata['section']}]\n{d.page_content}" for d in docs)
        _log_output(state, "retrieval", {"sections": [d.metadata["section"] for d in docs]})
        return {"patient": patient, "context": context, "memory": load_memory(state["patient_id"])}

    def summary_node(state: FollowUpState) -> dict:
        errors = []
        try:
            summary = agents.summary_agent(state["patient_id"], state["context"])
        except LLMError as exc:
            errors.append(f"summary_agent: {exc}")
            summary = agents.fallback_summary(state["patient"])
        out = summary.model_dump()
        _log_output(state, "summary_agent", out)
        return {"summary": out, "errors": errors}

    def reminder_node(state: FollowUpState) -> dict:
        errors = []
        summary = agents.FollowUpSummary.model_validate(state["summary"])
        responses = all_responses(state["patient"], state["memory"])
        try:
            reminder = agents.reminder_agent(state["patient"], summary, state["memory"]["reminders"], responses)
        except LLMError as exc:
            errors.append(f"reminder_agent: {exc}")
            reminder = agents.fallback_reminder(state["patient"], summary)
        out = reminder.model_dump()
        _log_output(state, "reminder_agent", out)
        return {"reminder": out, "errors": errors}

    def escalation_node(state: FollowUpState) -> dict:
        errors = []
        summary = agents.FollowUpSummary.model_validate(state["summary"])
        responses = all_responses(state["patient"], state["memory"])
        rules = agents.escalation_rules(state["patient"], responses)
        try:
            escalation = agents.escalation_agent(
                state["patient"], summary, state["memory"]["reminders"], responses, rules
            )
        except LLMError as exc:
            errors.append(f"escalation_agent: {exc}")
            escalation = agents.combine(rules, None, state["patient"])
        out = {**escalation.model_dump(), "rule_flag": rules.risk_flag}
        _log_output(state, "escalation_agent", out)
        return {"escalation": out, "errors": errors}

    def persist_memory(state: FollowUpState) -> dict:
        memory = state["memory"]
        created = now_iso()
        reminder = state["reminder"]
        if not reminder.get("skipped_duplicate"):
            memory["reminders"].append({"run_id": state["run_id"], "created_at": created, **{
                k: reminder[k] for k in ("appointment_id", "channel", "send_at", "text")}})
        esc = state["escalation"]
        memory["escalations"].append({"run_id": state["run_id"], "created_at": created,
                                      "risk_flag": esc["risk_flag"], "reasons": esc["reasons"],
                                      "clinician_alert": esc["clinician_alert"]})
        save_memory(memory)
        try:
            index.upsert_history(memory)
        except Exception as exc:  # index refresh is best-effort; memory is already saved
            log.warning("history re-index failed for %s: %s", state["patient_id"], exc)
            return {"errors": [f"index: {exc}"]}
        return {}

    graph = StateGraph(FollowUpState)
    graph.add_node("load_context", load_context)
    graph.add_node("summary_agent", summary_node)
    graph.add_node("reminder_agent", reminder_node)
    graph.add_node("escalation_agent", escalation_node)
    graph.add_node("persist_memory", persist_memory)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "summary_agent")
    graph.add_edge("summary_agent", "reminder_agent")
    graph.add_edge("reminder_agent", "escalation_agent")
    graph.add_edge("escalation_agent", "persist_memory")
    graph.add_edge("persist_memory", END)
    return graph.compile()


def run_followup(workflow, patient_id: str) -> FollowUpState:
    run_id = f"{patient_id}-{uuid.uuid4().hex[:8]}"
    state = workflow.invoke({"run_id": run_id, "patient_id": patient_id, "errors": []})
    if state["errors"]:
        log.warning("run %s completed with fallbacks: %s", run_id, state["errors"])
    return state
