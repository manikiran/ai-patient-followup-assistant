# Architecture — AI Patient Follow-Up Assistant

Implements *Architecture One-Pager v1.0.0* (Sep 24, 2026). Where the
one-pager offers a choice of tools, the choice made here is in **bold**.
Everything except the LLM is free (open source or AWS free tier). Claude runs
on your own API key.

```
 Streamlit UI ──► LangGraph orchestrator (sequential, shared state)
 (app.py)            │
                     ├─ load_context ── RAG: Chroma index, filter patient_id ◄── patient records + history (S3)
                     ├─ 1. Patient Summary Agent ── LLM ──► summary
                     ├─ 2. Reminder Agent ────────── LLM ──► reminder text / channel / send time
                     ├─ 3. Escalation Agent ─ rules + LLM ─► none / watch / escalate + clinician alert
                     └─ persist_memory ──► memory/<id>.json (S3) ──► re-indexed as "history"
                                         └► logs/<date>/<run>/<step>.json (S3), every agent output
```

## Agents

| Step | Agent | Input | Output written to state / memory |
|---|---|---|---|
| 1 | Patient Summary (`agents.summary_agent`) | Patient record retrieved via RAG | `FollowUpSummary`: diagnosis, meds (with change status), due visits, open tasks, narrative |
| 2 | Reminder (`agents.reminder_agent`) | Summary + reminder history + responses | `Reminder`: text, channel, send time, target visit, duplicate flag |
| 3 | Escalation (`agents.escalation_agent`) | Summary + reminder history + responses | `Escalation`: risk flag, reasons, clinician alert |

All three use structured outputs (Pydantic schemas). **Escalation
rules** are deterministic (`agents.escalation_rules`) and the LLM can only
*raise* the flag:

- **escalate**: 2+ reminders unanswered in a row; any discharge warning sign
  in a patient reply; a missed critical appointment.
- **watch**: latest reminder unanswered; a missed non-critical appointment.

## Memory

- **Short-term**: `workflow.FollowUpState`, the LangGraph state passed between
  nodes in one run.
- **Long-term**: `memory/<patient_id>.json` holds past reminders, responses
  (including simulated ones) and escalations. It is loaded at the start of a run,
  appended to at the end, and re-indexed into the vector store as
  `section="history"`. That lets the next run skip duplicate reminders and
  notice repeated non-response.

## Tech stack

| Layer | One-pager suggestion | Used |
|---|---|---|
| Agent framework | LangGraph / CrewAI | **LangGraph** `StateGraph` |
| LLM | OpenAI / Gemini | **Claude** (`claude-opus-5`, official `anthropic` SDK, your API key) with server-side refusal fallbacks. Free alternatives: **Ollama** (local) or **Gemini free tier**, via LangChain. Set with `LLM_PROVIDER` |
| RAG | LangChain | **LangChain** loader (`PatientRecordLoader`), `RecursiveCharacterTextSplitter`, vector-store retriever |
| Vector DB | ChromaDB / FAISS | **ChromaDB** (FAISS via `VECTOR_STORE=faiss`); local MiniLM embeddings |
| Frontend | Streamlit | **Streamlit** (`app.py`) |
| Deployment | AWS EC2 | **EC2 t3.small** (free-tier eligible) + Docker via `deploy/ec2/deploy.sh`. The doc's t3.medium isn't free-tier eligible, and Claude runs at Anthropic, so a small instance is enough |
| Storage | AWS S3 | **S3** (`S3_BUCKET`); local `data/` folder when unset |
| Containerization | Docker (optional) | **Docker**: one image, built on the EC2 instance at deploy time (GitHub Actions also builds it to GHCR on push) |

## Guardrails

- Synthetic data only: `src/synthetic_data.py` generates every record (Faker +
  templates) and validates it against `schema/patient.schema.json`.
- Reminders contain no clinical advice. The prompt forbids it, and
  `agents.contains_clinical_advice` then checks the text. Doses or
  start/stop/change instructions get replaced with a safe template.
- Escalations are alerts on the dashboard's **Alerts** tab for a human
  clinician. The system takes no automated clinical action.
- Every agent output (and retrieval trace) is logged to storage for review.
- Retrieval is always filtered by `patient_id`, plus a post-filter, so one
  patient's records can never ground another patient's prompt.
- Secrets: the Claude key comes from env, fed from SSM Parameter Store
  (standard tier, free) on EC2. It never goes into code, the image or logs.

## Failure handling

Each LLM call is retried with exponential backoff (the Anthropic SDK retries
429/5xx/connection errors 4 times). If it still fails,
is refused, or returns output that doesn't match the schema, the agent uses a deterministic fallback (summary
built from the record, template reminder, rules-only flag), so the run still
completes. The error is recorded in `state["errors"]` and shown in the UI.
Setting `LLM_CACHE=true` caches responses by prompt hash, which cuts API
cost while iterating.
