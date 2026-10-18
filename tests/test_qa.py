"""Ask-the-assistant: patient ID handling, context selection and fallbacks (LLM faked)."""
import pytest

from src import qa
from src.llm import LLMError
from src.rag import build_index


@pytest.fixture
def setup(patients, storage):
    return build_index(storage=storage), {p["patient_id"]: p for p in patients}


def capture_prompt(monkeypatch, action="answer"):
    seen = {}

    def fake(prompt, schema, **_):
        seen["prompt"] = prompt
        return qa.Answer(action=action, answer="ok", sources=["records"])

    monkeypatch.setattr(qa, "ask_structured", fake)
    return seen


def test_extract_patient_ids():
    assert qa.extract_patient_ids("summary for p0007 and P-0012, also P 0003") == ["P0007", "P0012", "P0003"]
    assert qa.extract_patient_ids("which patients are escalated?") == []


def test_patient_question_uses_only_that_patients_records(setup, monkeypatch):
    index, patients = setup
    seen = capture_prompt(monkeypatch)
    result = qa.answer_question(index, "what medications changed for P0004?", patients, {})
    assert result["patient_ids"] == ["P0004"]
    assert "[P0004 / medications]" in seen["prompt"]
    assert "P0005 /" not in seen["prompt"] and "Overview of all patients" not in seen["prompt"]


def test_open_question_uses_cohort_overview_and_previous_patient(setup, monkeypatch):
    index, patients = setup
    seen = capture_prompt(monkeypatch)
    qa.answer_question(index, "which patients need escalation?", patients, {}, previous_patient_ids=["P0002"])
    assert "Overview of all patients" in seen["prompt"]
    assert all(pid in seen["prompt"] for pid in patients)
    assert "[P0002 / current status]" in seen["prompt"]


def test_unknown_patient_is_reported_without_an_llm_call(setup, monkeypatch):
    index, patients = setup
    monkeypatch.setattr(qa, "ask_structured", lambda *a, **k: pytest.fail("LLM should not be called"))
    result = qa.answer_question(index, "summary for P9999", patients, {})
    assert "couldn't find P9999" in result["answer"]


def test_run_followup_only_for_a_single_patient(setup, monkeypatch):
    index, patients = setup
    capture_prompt(monkeypatch, action="run_followup")
    assert qa.answer_question(index, "run the follow-up for P0001", patients, {})["action"] == "run_followup"
    assert qa.answer_question(index, "run follow-ups for P0001 and P0002", patients, {})["action"] == "answer"


def test_llm_failure_falls_back_to_records(setup, monkeypatch, storage):
    index, patients = setup

    def boom(*_, **__):
        raise LLMError("simulated outage")

    monkeypatch.setattr(qa, "ask_structured", boom)
    result = qa.answer_question(index, "summary for P0003", patients, {})
    assert result["error"] == "simulated outage"
    assert "P0003" in result["answer"]
    assert storage.list_keys("logs/")  # every Q&A is logged
