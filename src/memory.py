"""Long-term memory: what happened in earlier runs, per patient.

Stored at memory/<patient_id>.json (S3 or local) and re-indexed into the
vector store after every run, so the next run can avoid duplicate reminders
and spot repeated non-response. Short-term memory is the LangGraph state
(see workflow.py) and lives only for one run.

    {
      "patient_id": "P0001",
      "reminders":   [{"run_id", "created_at", "appointment_id", "channel", "send_at", "text"}],
      "responses":   [{"date", "channel", "status", "message", "source"}],
      "escalations": [{"run_id", "created_at", "risk_flag", "reasons", "clinician_alert"}]
    }
"""
from datetime import date, datetime, timezone

from .storage import get_storage


def _key(patient_id: str) -> str:
    return f"memory/{patient_id}.json"


def empty_memory(patient_id: str) -> dict:
    return {"patient_id": patient_id, "reminders": [], "responses": [], "escalations": []}


def load_memory(patient_id: str, storage=None) -> dict:
    storage = storage or get_storage()
    return storage.read_json(_key(patient_id)) or empty_memory(patient_id)


def save_memory(memory: dict, storage=None) -> None:
    storage = storage or get_storage()
    storage.write_json(_key(memory["patient_id"]), memory)


def list_memories(storage=None) -> list[dict]:
    storage = storage or get_storage()
    return [storage.read_json(key) for key in storage.list_keys("memory/") if key.endswith(".json")]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_response(patient_id: str, status: str, message: str, channel: str, storage=None) -> dict:
    """Appends a (simulated) patient response; it affects the next run's flag."""
    memory = load_memory(patient_id, storage)
    memory["responses"].append({
        "date": date.today().isoformat(),
        "channel": channel,
        "status": status,
        "message": message,
        "source": "simulated",
    })
    save_memory(memory, storage)
    return memory


def all_responses(patient: dict, memory: dict) -> list[dict]:
    """Responses from the original record plus any recorded since, oldest first."""
    return sorted(patient.get("responses", []) + memory.get("responses", []), key=lambda r: r["date"])
