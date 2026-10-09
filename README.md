# AI Patient Follow-Up Assistant

Hospitals miss follow-up communication after discharge. That leads to skipped
appointments, missed medication changes and avoidable readmissions. This app
closes the gap with three agents that run in sequence for each patient:

1. **Patient Summary Agent** summarizes the patient's case, grounded in their
   records through RAG.
2. **Reminder Agent** writes a personalized reminder and picks the channel and
   send time.
3. **Escalation Agent** sets a risk flag (none / watch / escalate) and writes
   a clinician alert.

Care staff run follow-ups, review results and handle alerts in a Streamlit
dashboard. The app is deployed as one Docker container on AWS EC2, with data,
memory and logs in S3. **All patient data is synthetic.**

**Costs:** the agents use **Claude** with your own Anthropic API key, the
only paid component. Everything else is free: local MiniLM embeddings,
open-source ChromaDB, and an AWS deployment that uses free-tier services only.
For zero LLM cost too, switch `LLM_PROVIDER` to the free
[Ollama](https://ollama.com) (local) or Gemini free tier.

See [docs/architecture.md](docs/architecture.md) for the design. The project
plan and architecture one-pager (`.docx`) are the source specs.

## Repo layout

```
app.py                    Streamlit dashboard (Patients / Alerts / Simulate response)
src/
  config.py               settings from env / .env
  synthetic_data.py       generates + validates synthetic patients
  storage.py              S3 or local-folder JSON storage
  rag.py                  LangChain loader, chunking, Chroma/FAISS index, patient-filtered retriever
  llm.py                  Claude (default) / Ollama / Gemini: structured outputs, retries, fallbacks, cache
  agents.py               the three agents + escalation rules + guardrails
  memory.py               long-term memory per patient
  workflow.py             sequential LangGraph pipeline
schema/patient.schema.json
data/patients/            100 generated patients (local mode)
scripts/evaluate.py       end-to-end flag-accuracy test -> docs/test_report.md
tests/                    offline pytest suite (no API key / AWS needed)
deploy/ec2/               one-command free-tier EC2 deploy (deploy.sh / teardown.sh)
docker-compose.yml        local stack (optional Ollama profile)
Dockerfile                one image: app + agents + embedding model
```

## Quick start (local, no AWS)

Requires Python 3.11–3.12 (ChromaDB doesn't support 3.14 yet).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set ANTHROPIC_API_KEY (LLM_PROVIDER=anthropic is the default)

python -m src.synthetic_data --count 100   # already committed; regenerates data/patients/
python -m src.rag build                    # builds ./vectorstore (downloads an 80 MB embedding model once)
streamlit run app.py                       # http://localhost:8501
```

Free LLM alternatives: install [Ollama](https://ollama.com), run
`ollama pull llama3.2:3b` and set `LLM_PROVIDER=ollama`. Or get a free Gemini
key at https://aistudio.google.com/apikey and set `LLM_PROVIDER=gemini` and
`GOOGLE_API_KEY`.

With `S3_BUCKET` empty, everything lives under `data/`. Patients are under
`data/patients/` and are committed. Memory, logs and the index snapshot are
generated at runtime and git-ignored.

### Using the dashboard

- **Patients**: select one or more patients and click **Run follow-ups**. Each
  result shows the summary, the reminder and the escalation flag (with what the
  rules alone decided).
- **Alerts**: the latest *watch* and *escalate* flags, for clinician review.
- **Simulate response**: record a fake reply ("no response", or a message like
  "I've had chest pain"). Run that patient again and the flag changes. For
  example, two no-responses in a row escalate.

### Docker

```bash
docker compose up -d                       # app, LLM from .env (Claude)
docker compose --profile ollama up -d      # + local Ollama (set LLM_PROVIDER=ollama)
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` (Claude, your key), `ollama` (local, free) or `gemini` (free tier) |
| `ANTHROPIC_API_KEY` / `CLAUDE_MODEL` | — / `claude-opus-5` | Claude key and model |
| `CLAUDE_FALLBACKS` | `default` | Server-side refusal fallback; `off` to disable |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | `llama3.2:3b` / `http://localhost:11434` | Any Ollama model works, e.g. `qwen2.5:7b` if you have the RAM |
| `GEMINI_MODEL` / `GOOGLE_API_KEY` | `gemini-2.5-flash` / — | Gemini free tier |
| `LLM_CACHE` | `false` | Cache LLM responses on disk (cuts API cost and rate-limit hits while iterating) |
| `S3_BUCKET` / `S3_PREFIX` / `AWS_REGION` | empty / `patient-followup` / `us-east-1` | S3 storage; empty bucket = local `data/` |
| `VECTOR_STORE` | `chroma` | `chroma` or `faiss` (`pip install langchain-community faiss-cpu`) |
| `RETRIEVER_K` | `8` | Chunks retrieved per patient |

## Testing

```bash
pytest -q                                       # offline: schema, rules, guardrail, retriever filter, end-to-end with a fake LLM
python scripts/evaluate.py --count 20 --dry-run # rules-only flag accuracy, no API calls
python scripts/evaluate.py --count 20           # full pipeline on 20 patients (real LLM calls)
```

`evaluate.py` picks patients across every scenario (routine, single miss,
non-responder, warning symptom, missed critical appointment). It compares each
final flag with the expected one and writes `docs/test_report.md`.

## Deployment

```bash
PYTHON=.venv/bin/python bash deploy/ec2/deploy.sh   # prints http://<ip>:8501
bash deploy/ec2/teardown.sh                          # remove it again
```

The script creates a private S3 bucket and stores the Claude key in SSM
Parameter Store (free). It also creates an IAM role, a security group that
opens 8501 to your IP only, and a free-tier t3.small that builds and runs the
image. See [deploy/ec2/README.md](deploy/ec2/README.md), including the IAM
permissions the deploying user needs.

## Guardrails and limitations

- Synthetic data only. Do not load real patient information. The dashboard has
  no authentication, so restrict access with the security group.
- Reminders avoid clinical advice: the prompt forbids it and a post-check
  replaces any text that includes doses or medication instructions.
  Escalations always go to a human clinician.
- Each follow-up makes 3 LLM calls, billed to your Anthropic account on
  `claude-opus-5`. For bulk runs, set `LLM_CACHE=true`, or a cheaper
  `CLAUDE_MODEL` such as `claude-sonnet-5`.
- The free alternatives cost nothing but are weaker (small local models) or
  rate-limited (Gemini). The escalation rules are deterministic, so a weaker
  model can't lower a flag.
