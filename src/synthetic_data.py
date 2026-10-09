"""Generates synthetic patient follow-up data (plan tasks T1.3-T1.5).

Every patient is fictional (Faker + fixed clinical templates). Each one is
built around a `scenario` so the dataset deliberately includes the edge cases
the Escalation Agent must catch -- non-responders, warning symptoms, missed
critical appointments -- alongside routine patients.

    python -m src.synthetic_data --count 100            # write + validate
    python -m src.synthetic_data --count 100 --upload   # same, into S3_BUCKET
"""
import argparse
import json
import os
import random
from datetime import date, timedelta

import jsonschema
from faker import Faker

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema", "patient.schema.json")

# diagnosis -> (discharge summary template, warning signs, medications)
CONDITIONS = {
    "Heart failure exacerbation": (
        "Admitted with shortness of breath and leg swelling; diuresed and discharged on adjusted diuretics.",
        ["chest pain", "shortness of breath", "weight gain over 2 kg in 2 days"],
        [("Furosemide", "40 mg", "once daily", "changed"), ("Lisinopril", "10 mg", "once daily", "unchanged"),
         ("Metoprolol succinate", "25 mg", "once daily", "new")],
    ),
    "Community-acquired pneumonia": (
        "Treated with IV antibiotics, switched to oral therapy; oxygen weaned before discharge.",
        ["fever above 38.5 C", "shortness of breath", "coughing blood"],
        [("Amoxicillin-clavulanate", "875 mg", "twice daily for 5 days", "new"),
         ("Paracetamol", "500 mg", "as needed", "unchanged")],
    ),
    "Type 2 diabetes with hyperglycemia": (
        "Admitted with blood glucose above 400 mg/dL; insulin started and diabetes education provided.",
        ["confusion", "blood sugar above 300", "fainting"],
        [("Insulin glargine", "18 units", "at bedtime", "new"), ("Metformin", "1000 mg", "twice daily", "unchanged")],
    ),
    "Total knee replacement": (
        "Elective right knee replacement; mobilising with a frame, wound clean and dry at discharge.",
        ["wound redness or discharge", "calf swelling", "fever above 38.5 C"],
        [("Apixaban", "2.5 mg", "twice daily for 14 days", "new"), ("Oxycodone", "5 mg", "as needed", "new")],
    ),
    "COPD exacerbation": (
        "Treated with nebulisers and steroids; inhaler technique reviewed before discharge.",
        ["shortness of breath", "lips turning blue", "confusion"],
        [("Prednisolone", "30 mg", "once daily for 5 days", "new"), ("Tiotropium", "18 mcg", "once daily", "unchanged")],
    ),
    "Ischaemic stroke (minor)": (
        "Minor stroke with resolving left arm weakness; started on secondary prevention.",
        ["new weakness or numbness", "slurred speech", "severe headache"],
        [("Aspirin", "75 mg", "once daily", "new"), ("Atorvastatin", "80 mg", "at night", "new")],
    ),
    "Acute kidney injury": (
        "Dehydration-related kidney injury, creatinine improving; nephrotoxic medicines paused.",
        ["passing very little urine", "swelling of legs", "confusion"],
        [("Ibuprofen", "400 mg", "as needed", "stopped"), ("Amlodipine", "5 mg", "once daily", "unchanged")],
    ),
    "Laparoscopic appendectomy": (
        "Uncomplicated appendectomy; tolerating diet and pain controlled with oral analgesia.",
        ["fever above 38.5 C", "worsening abdominal pain", "wound redness or discharge"],
        [("Paracetamol", "1 g", "four times daily as needed", "new")],
    ),
}

FOLLOW_UPS = [
    ("Primary care visit", False),
    ("Lab test (bloodwork)", False),
    ("Physical therapy session", False),
    ("Medication review", False),
    ("Specialist clinic visit", True),
    ("Post-op wound check", True),
]

# Scenario mix for a 100-patient set: mostly routine, with enough edge cases
# to exercise every escalation rule.
SCENARIO_WEIGHTS = {
    "routine": 0.50,
    "single_miss": 0.15,
    "non_responder": 0.15,
    "warning_symptom": 0.10,
    "missed_critical": 0.10,
}

OK_REPLIES = [
    "Thanks, I'll be there.",
    "Got it, confirmed.",
    "Feeling better, see you then.",
    "Yes, my daughter will drive me.",
    "Picked up the new prescription, thanks.",
]


def _iso(d: date) -> str:
    return d.isoformat()


def generate_patient(patient_id: str, scenario: str, fake: Faker, rng: random.Random, today: date) -> dict:
    diagnosis = rng.choice(list(CONDITIONS))
    summary, warning_signs, meds = CONDITIONS[diagnosis]
    discharge_date = today - timedelta(days=rng.randint(5, 20))
    channel = rng.choice(["sms", "email", "phone"])

    appointments = []
    for i, (appt_type, critical) in enumerate(rng.sample(FOLLOW_UPS, k=rng.randint(1, 3))):
        appointments.append({
            "appointment_id": f"{patient_id}-A{i + 1}",
            "type": appt_type,
            "date": _iso(today + timedelta(days=rng.randint(2, 21))),
            "critical": critical,
            "status": "scheduled",
        })

    responses = []
    reminder_day = discharge_date + timedelta(days=2)
    for _ in range(rng.randint(0, 2)):
        responses.append({"date": _iso(reminder_day), "channel": channel, "status": "responded",
                          "message": rng.choice(OK_REPLIES)})
        reminder_day += timedelta(days=2)

    if scenario == "single_miss":
        # One unanswered reminder, or a missed non-critical visit -> "watch".
        if rng.random() < 0.5:
            responses.append({"date": _iso(reminder_day), "channel": channel, "status": "no_response", "message": ""})
        else:
            appointments.append({"appointment_id": f"{patient_id}-A9", "type": "Lab test (bloodwork)",
                                 "date": _iso(today - timedelta(days=2)), "critical": False, "status": "missed"})
    elif scenario == "non_responder":
        for _ in range(rng.randint(2, 3)):
            responses.append({"date": _iso(reminder_day), "channel": channel, "status": "no_response", "message": ""})
            reminder_day += timedelta(days=2)
    elif scenario == "warning_symptom":
        symptom = rng.choice(warning_signs)
        responses.append({"date": _iso(reminder_day), "channel": channel, "status": "responded",
                          "message": f"I'll try to come, but I've had {symptom} since yesterday."})
    elif scenario == "missed_critical":
        appointments.append({"appointment_id": f"{patient_id}-A9", "type": "Specialist clinic visit",
                             "date": _iso(today - timedelta(days=rng.randint(1, 4))), "critical": True,
                             "status": "missed"})

    return {
        "patient_id": patient_id,
        "scenario": scenario,
        "demographics": {
            "name": fake.name(),
            "age": rng.randint(25, 92),
            "gender": rng.choice(["female", "male", "non-binary"]),
            "phone": fake.phone_number(),
            "email": fake.free_email(),
            "preferred_channel": channel,
            "language": "English",
        },
        "discharge": {
            "date": _iso(discharge_date),
            "diagnosis": diagnosis,
            "summary": summary,
            "warning_signs": list(warning_signs),
        },
        "medications": [
            {"name": n, "dose": d, "frequency": f, "change": c} for n, d, f, c in meds
        ],
        "appointments": sorted(appointments, key=lambda a: a["date"]),
        "responses": responses,
    }


def generate_patients(count: int = 100, seed: int = 42, today: date | None = None) -> list[dict]:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)
    today = today or date.today()
    scenarios = rng.choices(list(SCENARIO_WEIGHTS), weights=list(SCENARIO_WEIGHTS.values()), k=count)
    return [
        generate_patient(f"P{i + 1:04d}", scenario, fake, rng, today)
        for i, scenario in enumerate(scenarios)
    ]


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_patients(patients: list[dict]) -> None:
    """Raises jsonschema.ValidationError on the first invalid record."""
    validator = jsonschema.Draft202012Validator(load_schema())
    for patient in patients:
        validator.validate(patient)


def main() -> None:
    from .storage import LocalStorage, get_storage, save_patients
    from . import config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true",
                        help="write to S3_BUCKET (default: local data/ folder)")
    args = parser.parse_args()

    patients = generate_patients(args.count, args.seed)
    validate_patients(patients)

    if args.upload and not config.S3_BUCKET:
        parser.error("--upload needs S3_BUCKET set")
    storage = get_storage() if args.upload else LocalStorage(config.LOCAL_DATA_DIR)
    save_patients(patients, storage)

    counts = {s: sum(p["scenario"] == s for p in patients) for s in SCENARIO_WEIGHTS}
    print(f"Wrote {len(patients)} validated patients to {storage.describe()}/patients/ -- {counts}")


if __name__ == "__main__":
    main()
