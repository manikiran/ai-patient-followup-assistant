"""RAG retrieval and the end-to-end LangGraph run, with the LLM faked."""
import pytest

from src import agents
from src.llm import LLMError
from src.memory import load_memory
from src.rag import build_index, load_index
from src.workflow import build_workflow, run_followup


@pytest.fixture
def index(patients, storage):
    return build_index(storage=storage)


def test_retriever_only_returns_the_requested_patient(index, patients):
    for p in patients[:10]:
        docs = index.retrieve(p["patient_id"], "medications and appointments")
        assert docs
        assert {d.metadata["patient_id"] for d in docs} == {p["patient_id"]}


def test_index_is_snapshotted_and_restorable(index, storage, tmp_path):
    assert "vectorstore/index.tar.gz" in storage.list_keys("vectorstore/")
    restored = load_index(storage=storage, directory=str(tmp_path / "restored"))
    pid = "P0001"
    assert {d.metadata["patient_id"] for d in restored.retrieve(pid, "discharge")} == {pid}


def fake_llm(prompt, schema, **_):
    if schema is agents.FollowUpSummary:
        raise LLMError("simulated outage")  # exercise the fallback path
    if schema is agents.Reminder:
        return agents.Reminder(appointment_id=None, channel="sms", send_at="2099-01-01T10:00",
                               text="Increase your dose to 80 mg")  # must be caught by the guardrail
    return agents.Escalation(risk_flag="none", reasons=[], clinician_alert="")


def test_end_to_end_run_with_memory(index, patients, monkeypatch):
    monkeypatch.setattr(agents, "ask_structured", fake_llm)
    workflow = build_workflow(index)
    p = next(p for p in patients if p["scenario"] == "non_responder")

    first = run_followup(workflow, p["patient_id"])
    assert first["errors"] == ["summary_agent: simulated outage"]          # failed call didn't crash the run
    assert first["summary"]["diagnosis"] == p["discharge"]["diagnosis"]    # deterministic fallback summary
    assert not agents.contains_clinical_advice(first["reminder"]["text"])  # guardrail replaced the text
    assert first["escalation"]["risk_flag"] == "escalate"                  # rules win over the LLM's "none"
    assert first["escalation"]["clinician_alert"]

    memory = load_memory(p["patient_id"])
    assert len(memory["reminders"]) == 1 and len(memory["escalations"]) == 1

    # Second run sees the first run's history, via memory and the vector index.
    second = run_followup(workflow, p["patient_id"])
    assert "[history]" in second["context"]
    assert len(load_memory(p["patient_id"])["escalations"]) == 2
