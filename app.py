"""Streamlit dashboard for care staff (plan T5.1-T5.3).

    streamlit run app.py

* Ask       -- free-text questions, with or without a patient ID
* Patients  -- patient list, run follow-ups, see summary / reminder / escalation
* Alerts    -- latest watch/escalate flag per patient, for clinician review
* Simulate  -- record a fake patient response to test escalation on the next run
"""
import pandas as pd
import streamlit as st

from src import config
from src.llm import model_name
from src.memory import list_memories, load_memory, record_response
from src.qa import answer_question
from src.rag import build_index, load_index
from src.storage import get_storage, load_patients
from src.workflow import build_workflow, run_followup

st.set_page_config(page_title="AI Patient Follow-Up Assistant", layout="wide")

FLAG_BADGE = {"none": "🟢 none", "watch": "🟡 watch", "escalate": "🔴 escalate"}


@st.cache_resource(show_spinner="Loading vector index...")
def get_index():
    return load_index()


@st.cache_resource
def get_workflow(_index):
    return build_workflow(_index)


@st.cache_data(ttl=60)
def get_patients() -> list[dict]:
    return sorted(load_patients(), key=lambda p: p["patient_id"])


def latest_flags() -> dict[str, dict]:
    return {m["patient_id"]: m["escalations"][-1] for m in list_memories() if m["escalations"]}


def show_result(state: dict) -> None:
    summary, reminder, esc = state["summary"], state["reminder"], state["escalation"]
    if state.get("errors"):
        st.warning("Completed with fallbacks: " + "; ".join(state["errors"]))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1 · Summary")
        st.markdown(f"**{summary['diagnosis']}** — discharged {summary['discharge_date']}")
        st.write(summary["narrative"])
        st.markdown("**Medications**")
        st.dataframe(pd.DataFrame(summary["medications"]), hide_index=True, width="stretch")
        st.markdown("**Due visits**")
        st.dataframe(pd.DataFrame(summary["due_visits"]), hide_index=True, width="stretch")
        st.markdown("**Open tasks**")
        st.markdown("\n".join(f"- {t}" for t in summary["open_tasks"]) or "_none_")
    with col2:
        st.subheader("2 · Reminder")
        st.markdown(f"**Channel:** {reminder['channel']}  \n**Send at:** {reminder['send_at']}  \n"
                    f"**Visit:** {reminder['appointment_id'] or '—'}")
        if reminder.get("skipped_duplicate"):
            st.info("Not sent: an equivalent reminder went out recently.")
        st.text_area("Reminder text", reminder["text"], height=180, disabled=True)
    with col3:
        st.subheader("3 · Escalation")
        st.markdown(f"### {FLAG_BADGE[esc['risk_flag']]}")
        st.caption(f"Rules alone: {esc['rule_flag']}")
        st.markdown("\n".join(f"- {r}" for r in esc["reasons"]) or "_no concerns_")
        if esc["clinician_alert"]:
            st.error(esc["clinician_alert"])


# --- Sidebar ---------------------------------------------------------------------

with st.sidebar:
    st.title("Follow-Up Assistant")
    st.caption(f"Storage: `{get_storage().describe()}`  \nLLM: `{config.LLM_PROVIDER}:{model_name()}`  \n"
               f"Vector store: `{config.VECTOR_STORE}`")
    if st.button("Rebuild vector index"):
        with st.spinner("Re-indexing patient records and history..."):
            get_index.clear()
            get_workflow.clear()
            build_index()
        st.success("Index rebuilt.")
    st.caption("Synthetic data only. Reminders contain no clinical advice; "
               "every escalation goes to a human clinician.")

patients = get_patients()
if not patients:
    st.error("No patients found. Run `python -m src.synthetic_data` (add `--upload` for S3) first.")
    st.stop()

by_id = {p["patient_id"]: p for p in patients}
workflow = get_workflow(get_index())
tab_ask, tab_patients, tab_alerts, tab_simulate = st.tabs(["Ask", "Patients", "Alerts", "Simulate response"])

# --- Ask -------------------------------------------------------------------------------

EXAMPLES = [
    "Give me the follow-up summary for P0007",
    "What medications changed for P0012 and are there open tasks?",
    "Does P0005 need escalation? Why?",
    "Which patients need escalation right now?",
    "Run the follow-up for P0003",
]

with tab_ask:
    st.subheader("Ask about a patient")
    st.caption("Include a patient ID (e.g. P0007) for a specific patient, or ask an open question "
               "about all patients. Say \"run the follow-up for P0003\" to run the full agent pipeline.")
    chat = st.session_state.setdefault("chat", [])

    picked = None
    cols = st.columns(len(EXAMPLES) + 1)
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, key=f"ex-{example}", width="stretch"):
            picked = example
    if cols[-1].button("Clear chat", width="stretch"):
        st.session_state["chat"] = chat = []

    for turn in chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("followup"):
                show_result(turn["followup"])
            if turn.get("caption"):
                st.caption(turn["caption"])

    question = st.chat_input("Ask a question, e.g. 'summary for P0007' or 'which patients missed visits?'") or picked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        previous = next((t["patient_ids"] for t in reversed(chat) if t.get("patient_ids")), [])
        with st.chat_message("assistant"), st.spinner("Looking through the records..."):
            result = answer_question(
                get_index(), question, by_id,
                {m["patient_id"]: m for m in list_memories()},
                history=[{"role": t["role"], "content": t["content"]} for t in chat],
                previous_patient_ids=previous,
            )
            followup = None
            if result["action"] == "run_followup":
                followup = run_followup(workflow, result["patient_ids"][0])
            st.markdown(result["answer"])
            if followup:
                show_result(followup)
            caption = ("Sources: " + ", ".join(result["sources"])) if result["sources"] else ""
            if result["error"]:
                st.warning(f"LLM error: {result['error']}")
            if caption:
                st.caption(caption)
        chat.append({"role": "user", "content": question})
        chat.append({"role": "assistant", "content": result["answer"], "followup": followup,
                     "caption": caption, "patient_ids": result["patient_ids"]})

# --- Patients ----------------------------------------------------------------------

with tab_patients:
    flags = latest_flags()
    table = pd.DataFrame([{
        "patient_id": p["patient_id"],
        "name": p["demographics"]["name"],
        "diagnosis": p["discharge"]["diagnosis"],
        "discharged": p["discharge"]["date"],
        "channel": p["demographics"]["preferred_channel"],
        "last flag": FLAG_BADGE.get(flags.get(p["patient_id"], {}).get("risk_flag"), "— not run"),
    } for p in patients])
    st.dataframe(table, hide_index=True, width="stretch", height=280)

    left, right = st.columns([3, 1])
    selected = left.multiselect("Patients to follow up", list(by_id), default=list(by_id)[:1],
                                format_func=lambda pid: f"{pid} — {by_id[pid]['demographics']['name']}")
    right.write("")
    run = right.button("▶ Run follow-ups", type="primary", disabled=not selected)

    if run:
        results = {}
        progress = st.progress(0.0)
        for i, pid in enumerate(selected):
            with st.spinner(f"Running Summary → Reminder → Escalation for {pid}..."):
                results[pid] = run_followup(workflow, pid)
            progress.progress((i + 1) / len(selected))
        st.session_state["results"] = results

    for pid, state in st.session_state.get("results", {}).items():
        with st.expander(f"{pid} — {by_id[pid]['demographics']['name']} · "
                         f"{FLAG_BADGE[state['escalation']['risk_flag']]}", expanded=len(selected) == 1):
            show_result(state)

# --- Alerts ------------------------------------------------------------------------

with tab_alerts:
    st.subheader("Patients needing clinician attention")
    alerts = [(pid, e) for pid, e in latest_flags().items() if e["risk_flag"] != "none"]
    alerts.sort(key=lambda a: (a[1]["risk_flag"] != "escalate", a[0]))
    if not alerts:
        st.info("No open alerts. Run follow-ups from the Patients tab.")
    for pid, e in alerts:
        name = by_id[pid]["demographics"]["name"] if pid in by_id else pid
        with st.container(border=True):
            st.markdown(f"**{FLAG_BADGE[e['risk_flag']]} · {pid} — {name}** · flagged {e['created_at']}")
            st.write(e["clinician_alert"])
            st.caption("Reasons: " + "; ".join(e["reasons"]))

# --- Simulate response -------------------------------------------------------------------

with tab_simulate:
    st.subheader("Simulate a patient response")
    st.caption("Recorded in long-term memory; the next follow-up run for this patient takes it into account.")
    with st.form("simulate"):
        pid = st.selectbox("Patient", list(by_id),
                           format_func=lambda p: f"{p} — {by_id[p]['demographics']['name']}")
        status = st.radio("Response", ["no_response", "responded"], horizontal=True)
        message = st.text_input("Message (if responded)", placeholder="e.g. I've had chest pain since yesterday")
        if st.form_submit_button("Record response"):
            channel = by_id[pid]["demographics"]["preferred_channel"]
            record_response(pid, status, message if status == "responded" else "", channel)
            st.success(f"Recorded for {pid}. Run a follow-up for {pid} to see the updated flag.")
    if pid:
        memory = load_memory(pid)
        if memory["responses"]:
            st.markdown("**Recorded responses**")
            st.dataframe(pd.DataFrame(memory["responses"]), hide_index=True, width="stretch")
