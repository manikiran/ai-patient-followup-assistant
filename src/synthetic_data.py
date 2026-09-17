"""Generates synthetic patient follow-up data.

Hospitals miss follow-up communication because there's no clean, realistic
dataset to prototype against. This module fabricates that dataset: patients
discharged with a diagnosis, a required follow-up action, and enough signal
(missed appointments, risk level) for the agents downstream to reason about.
"""
import json
import os
import random
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()

DIAGNOSES = [
    "Type 2 Diabetes",
    "Hypertension",
    "Post-Cardiac Surgery",
    "Pneumonia",
    "Fractured Femur (post-op)",
    "Chronic Kidney Disease",
    "Asthma Exacerbation",
    "Appendectomy (post-op)",
    "Stroke Recovery",
    "COPD",
]

FOLLOW_UP_TYPES = [
    "Phone check-in",
    "In-person clinic visit",
    "Lab test (bloodwork)",
    "Physical therapy session",
    "Medication review",
    "Wound check",
]

CONTACT_METHODS = ["phone", "sms", "email"]

RISK_LEVELS = ["low", "medium", "high"]


def _random_recent_date(days_back_max: int) -> str:
    days_back = random.randint(1, days_back_max)
    return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")


def _random_future_date(days_ahead_max: int) -> str:
    days_ahead = random.randint(1, days_ahead_max)
    return (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


def generate_patient(patient_id: str) -> dict:
    """Builds one synthetic patient follow-up record."""
    risk_level = random.choices(RISK_LEVELS, weights=[0.5, 0.3, 0.2])[0]
    missed_appointments = random.choices([0, 1, 2, 3], weights=[0.6, 0.2, 0.12, 0.08])[0]

    return {
        "patient_id": patient_id,
        "name": fake.name(),
        "age": random.randint(19, 89),
        "gender": random.choice(["female", "male", "non-binary"]),
        "diagnosis": random.choice(DIAGNOSES),
        "discharge_date": _random_recent_date(14),
        "follow_up_type": random.choice(FOLLOW_UP_TYPES),
        "follow_up_due_date": _random_future_date(10),
        "preferred_contact_method": random.choice(CONTACT_METHODS),
        "phone": fake.phone_number(),
        "email": fake.email(),
        "risk_level": risk_level,
        "missed_appointments": missed_appointments,
        "last_contact_date": _random_recent_date(7),
        "notes": fake.sentence(nb_words=12),
    }


def generate_patients(n: int = 20) -> list[dict]:
    return [generate_patient(f"P{str(i + 1).zfill(4)}") for i in range(n)]


def save_patients(patients: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(patients, f, indent=2)


def load_patients(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_patients.json")
    patients = generate_patients(20)
    save_patients(patients, default_path)
    print(f"Wrote {len(patients)} synthetic patients to {default_path}")
