# AI Patient Follow-Up Assistant

**Problem:** Hospitals miss follow-up communication with discharged patients —
reminders don't go out, at-risk patients fall through the cracks, and nobody
escalates the ones who need a human to step in.

This project prototypes a fix: a small sequential multi-agent pipeline that
turns a patient's discharge record into (1) a clinical follow-up summary,
(2) a personalized reminder message, and (3) an escalation decision for the
care team — with memory, so the pipeline recalls a patient's history across
runs instead of treating every check-in as the first.

## Architecture

```
                                              +--> Patient Summary Agent --(full?)--> Reminder Agent --(full?)--> Escalation Agent
                                              |         (else END)                        (else END)
Open WebUI --(OpenAI-compatible API)--> agent-app --> Router Agent
                                              +--> Reminder Agent --> END
                                              +--> Escalation Agent --> END
                                                    |
                                              (each node calls Claude)
```

- **Synthetic data** (`src/synthetic_data.py`) — fabricates patient discharge
  records (diagnosis, follow-up type/due date, risk level, missed
  appointments, preferred contact method) since no real hospital data is used.
- **Agents** (`src/agents.py`) — a LangGraph `StateGraph`, built directly on
  the pattern in `ollama-FAPI/routerbasedagenticworkflowhealthcare.py`: a
  router node that classifies the request with an LLM call, and conditional
  edges that forward to the specialist node that handles it (also an LLM
  call). The LLM is Claude (`src/llm.py`, official `anthropic` Python SDK,
  model `claude-opus-5` by default) instead of that reference script's Gemini:
  - **Router Agent** — the single entry point. Reads the user's free-form
    query and asks Claude to classify it (summary / reminder / escalation /
    unclear), then forwards to the one agent that handles it — callers just
    ask a question, they don't need to know which agent does what. An
    unclear/general request ("run followup for P0001") falls through to the
    full sequential pipeline.
  - **Patient Summary Agent** — reads the raw record, writes a concise
    clinical follow-up summary.
  - **Reminder Agent** — drafts a personalized reminder, matched to the
    patient's preferred contact channel.
  - **Escalation Agent** — decides whether risk level / missed appointments
    warrant flagging the case to the care team.
  - The graph is compiled with a `MemorySaver` checkpointer keyed by
    `patient_id`, so state persists per patient across separate invocations —
    e.g. ask for just a reminder later and the earlier summary is still on
    record for that patient.
  - Requires `ANTHROPIC_API_KEY` (see **Configuration** below). LLM failures
    (rate limits, quota, network) are caught and surfaced as a readable
    message instead of crashing the request.
- **Backend** (`src/main.py`) — FastAPI service exposing:
  - `GET /patients` — synthetic patient roster (auto-generated on first call)
  - `POST /patients/generate` — regenerate the synthetic roster
  - `POST /followup/run` — run the full agent pipeline for one `patient_id`
  - `GET /v1/models`, `POST /v1/chat/completions` — an OpenAI-compatible
    surface so Open WebUI can drive the pipeline directly from chat.
- **Frontend** — [Open WebUI](https://github.com/open-webui/open-webui)
  (`docker pull openwebui/open-webui`), configured to talk to the agent-app's
  OpenAI-compatible endpoint. No custom frontend code — just wiring. Open
  WebUI doesn't know or care that there's no LLM behind the "model" it's
  talking to — the agent-app's `/v1/chat/completions` response is exactly
  what it expects.
- **Deployment** — Kubernetes manifests in `k8s/`, following the same
  Deployment/Service/PVC pattern as `ollama-FAPI/k8s`, covering the agent-app
  and Open WebUI. See `k8s/AWS_DEPLOY.md` for the EKS-specific steps (ECR
  push, node sizing, LoadBalancer service for public access).

## Repo layout

```
.
├── src/
│   ├── synthetic_data.py   # synthetic patient generator
│   ├── llm.py               # Claude client config (official anthropic SDK)
│   ├── agents.py           # LangGraph router + agents + memory
│   └── main.py             # FastAPI app (REST + OpenAI-compatible API)
├── data/                   # generated synthetic_patients.json lands here
├── k8s/                    # Deployment/Service/PVC manifests + AWS_DEPLOY.md
├── Dockerfile              # agent-app image
└── requirements.txt
```

## Configuration

The agents call the Anthropic (Claude) API, so an API key is required:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

`CLAUDE_MODEL` in `.env` sets the *default* model (`claude-opus-5` if unset).
**Never commit `.env`** — it's already git-ignored. On Kubernetes, the key is
injected via a Secret instead (see `k8s/AWS_DEPLOY.md`).

### Switching models dynamically (no restart)

The model isn't hardcoded at startup — every agent call in `src/agents.py`
reads `state["model"]` and passes it through to `src/llm.py`'s `ask()`,
which resolves it per request (falling back to `CLAUDE_MODEL` if unset or
not in `AVAILABLE_MODELS`). Two ways to use this:

- **Open WebUI** — `GET /v1/models` advertises `claude-opus-5`,
  `claude-sonnet-5`, and `claude-haiku-4-5`. Pick one from Open WebUI's model
  dropdown and it's sent back as `model` on every `/v1/chat/completions`
  request from then on — switching models is just switching the dropdown.
- **REST API** — pass `"model": "claude-sonnet-5"` in the `/followup/run`
  body.

To add another model, add its id to `AVAILABLE_MODELS` in `src/llm.py` — no
other code changes needed. Note memory doesn't carry the model choice
forward: each call's `model` is independent, only the generated content
(`summary`/`reminder`/`escalation`) persists across calls for a patient.

## Running locally

### The agent API

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn src.main:app --reload
```

Generate synthetic data and try the pipeline:

```bash
python -m src.synthetic_data
curl -X POST localhost:8000/followup/run -H 'Content-Type: application/json' \
  -d '{"patient_id": "P0001"}'
```

### Open WebUI (frontend)

```bash
docker run -d --name open-webui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8000/v1" \
  -e OPENAI_API_KEY="not-needed" \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

Open http://localhost:3000, pick a model from the dropdown (`claude-opus-5`,
`claude-sonnet-5`, or `claude-haiku-4-5` — see **Switching models
dynamically** above), and try any of: `summary for P0001`,
`send a reminder to P0002`, `does P0004 need escalation`, or
`run followup for P0001` for the full pipeline.

## Kubernetes / AWS

```bash
# local minikube smoke test, same as ollama-FAPI
eval $(minikube docker-env) && docker build -t agent-app:latest .
kubectl apply -f k8s/
```

For a real deployment to AWS EKS (ECR image push, node sizing, public access
via LoadBalancer), see [`k8s/AWS_DEPLOY.md`](k8s/AWS_DEPLOY.md).

## Notes / limitations

- This is a learning/prototyping project: synthetic data only, no real PHI,
  no auth on the API. Do not point this at real patient data as-is.
- The escalation and reminder agents are advisory text generation, not a
  substitute for clinical judgment — they explicitly avoid diagnosing or
  prescribing.
- Every router/agent call is a real, billed Anthropic API request on
  `claude-opus-5` by default. A "full" pipeline run costs up to 4 calls
  (router + 3 agents). Keep an eye on usage at console.anthropic.com if
  you're generating followups in bulk, or set `CLAUDE_MODEL=claude-sonnet-5`
  / `claude-haiku-4-5` in `.env` for a cheaper model.
