from src import agents
from src.memory import all_responses, load_memory, record_response
from src.synthetic_data import generate_patients, validate_patients

EXPECTED = {"routine": "none", "single_miss": "watch", "non_responder": "escalate",
            "warning_symptom": "escalate", "missed_critical": "escalate"}


def test_generated_patients_match_schema_and_cover_edge_cases():
    patients = generate_patients(100, seed=42)
    validate_patients(patients)
    assert len({p["patient_id"] for p in patients}) == 100
    assert {p["scenario"] for p in patients} == set(EXPECTED)


def test_generation_is_reproducible():
    assert generate_patients(5, seed=1) == generate_patients(5, seed=1)


def test_rules_flag_every_scenario_correctly():
    for p in generate_patients(100, seed=42):
        flag = agents.escalation_rules(p, p["responses"]).risk_flag
        assert flag == EXPECTED[p["scenario"]], (p["patient_id"], p["scenario"], flag)


def test_simulated_responses_change_the_next_flag(patients, storage):
    p = next(p for p in patients if p["scenario"] == "routine")
    assert agents.escalation_rules(p, all_responses(p, load_memory(p["patient_id"]))).risk_flag == "none"

    record_response(p["patient_id"], "no_response", "", "sms")
    assert agents.escalation_rules(p, all_responses(p, load_memory(p["patient_id"]))).risk_flag == "watch"

    record_response(p["patient_id"], "no_response", "", "sms")
    assert agents.escalation_rules(p, all_responses(p, load_memory(p["patient_id"]))).risk_flag == "escalate"


def test_warning_symptom_in_reply_escalates(patients):
    p = next(p for p in patients if p["scenario"] == "routine")
    sign = p["discharge"]["warning_signs"][0]
    responses = p["responses"] + [{"date": "2099-01-01", "channel": "sms", "status": "responded",
                                   "message": f"I have {sign.upper()} today"}]
    result = agents.escalation_rules(p, responses)
    assert result.risk_flag == "escalate"
    assert any(sign in r for r in result.reasons)


def test_llm_can_raise_but_not_lower_the_rule_flag(patients):
    p = patients[0]
    rules = agents.Escalation(risk_flag="escalate", reasons=["missed 2 reminders in a row"], clinician_alert="")
    llm_low = agents.Escalation(risk_flag="none", reasons=[], clinician_alert="")
    combined = agents.combine(rules, llm_low, p)
    assert combined.risk_flag == "escalate"
    assert combined.clinician_alert  # an alert is always written for the clinician

    llm_high = agents.Escalation(risk_flag="watch", reasons=["sounds unsure"], clinician_alert="check in")
    assert agents.combine(agents.Escalation(risk_flag="none", reasons=[], clinician_alert=""),
                          llm_high, p).risk_flag == "watch"


def test_clinical_advice_guardrail():
    assert agents.contains_clinical_advice("Take 40 mg of furosemide tonight")
    assert agents.contains_clinical_advice("Please stop taking your aspirin")
    assert not agents.contains_clinical_advice("Hi Ana, reminder of your clinic visit on 2026-10-09.")
