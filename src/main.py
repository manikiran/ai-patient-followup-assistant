"""FastAPI service exposing the patient follow-up agent workflow.

Two ways in:
  * A plain REST API (/patients, /followup/run) for direct integration.
  * An OpenAI-compatible /v1/chat/completions + /v1/models pair, so Open WebUI
    can be pointed at this service as an "OpenAI API" connection and drive the
    agents straight from chat (e.g. "run followup for P0001").
"""
import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agents import run_query
from .llm import AVAILABLE_MODELS, resolve_model
from .synthetic_data import generate_patients, load_patients, save_patients

DATA_PATH = os.getenv(
    "PATIENT_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_patients.json"),
)

app = FastAPI(title="AI Patient Follow-Up Assistant")


def _get_patients() -> list[dict]:
    patients = load_patients(DATA_PATH)
    if not patients:
        patients = generate_patients(20)
        save_patients(patients, DATA_PATH)
    return patients


def _find_patient(patient_id: str) -> dict:
    for patient in _get_patients():
        if patient["patient_id"].lower() == patient_id.lower():
            return patient
    raise HTTPException(status_code=404, detail=f"Unknown patient_id '{patient_id}'")


class GenerateRequest(BaseModel):
    count: int = 20


class FollowUpRequest(BaseModel):
    patient_id: str
    query: str = ""  # optional: routed through the Router Agent, e.g. "just the reminder"
    model: str = ""  # optional: e.g. "claude-sonnet-5" -- see GET /v1/models for choices


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/patients")
def list_patients():
    return _get_patients()


@app.post("/patients/generate")
def generate(req: GenerateRequest):
    patients = generate_patients(req.count)
    save_patients(patients, DATA_PATH)
    return {"count": len(patients), "patients": patients}


@app.post("/followup/run")
def followup_run(req: FollowUpRequest):
    patient = _find_patient(req.patient_id)
    try:
        result = run_query(patient, req.query, model=req.model or None)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc
    return {
        "patient_id": patient["patient_id"],
        "model": resolve_model(req.model or None),
        "intent": result.get("intent"),
        "summary": result.get("summary"),
        "reminder": result.get("reminder"),
        "escalation_needed": result.get("escalation_needed"),
        "escalation": result.get("escalation"),
    }


# --- OpenAI-compatible surface, for Open WebUI ---------------------------


@app.get("/v1/models")
def list_models():
    # Each entry is a real Claude model id -- picking one in Open WebUI's model
    # dropdown sends it back as `model` on /v1/chat/completions, which we use
    # to override which Claude model backs that request. No restart needed.
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": "anthropic"}
            for model_id in AVAILABLE_MODELS
        ],
    }


def _extract_patient_id(user_message: str) -> str | None:
    tokens = user_message.replace(",", " ").split()
    for token in tokens:
        candidate = token.strip().upper()
        if candidate.startswith("P") and candidate[1:].isdigit():
            return candidate
    return None


def _format_result(patient: dict, result: dict) -> str:
    """Renders only the sections the Router Agent actually dispatched to."""
    parts = [f"**Patient: {patient['name']} ({patient['patient_id']})**"]

    if "summary" in result:
        parts.append(f"**Summary:**\n{result['summary']}")
    if "reminder" in result:
        parts.append(f"**Reminder message:**\n{result['reminder']}")
    if "escalation" in result:
        parts.append(
            f"**Escalation:** {'YES' if result['escalation_needed'] else 'no'}\n"
            f"{result['escalation']}"
        )

    return "\n\n".join(parts)


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    messages = body.get("messages", [])
    user_message = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    # Open WebUI sends whichever model the user picked from its dropdown here.
    requested_model = resolve_model(body.get("model"))

    patient_id = _extract_patient_id(user_message)
    if not patient_id:
        reply = (
            "Tell me which patient and what you need -- e.g. \"summary for P0001\", "
            "\"send a reminder to P0002\", \"does P0003 need escalation\", or "
            "\"run followup for P0004\" for the full pipeline. "
            "Call GET /patients to see available IDs."
        )
    else:
        try:
            patient = _find_patient(patient_id)
            result = run_query(patient, user_message, model=requested_model)
            reply = _format_result(patient, result)
        except HTTPException as exc:
            reply = str(exc.detail)
        except Exception as exc:
            reply = f"Sorry, the LLM provider returned an error: {exc}"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
    }
