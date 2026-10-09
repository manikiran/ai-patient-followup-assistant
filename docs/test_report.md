# Test report — end-to-end escalation accuracy

- Generated: 2026-10-05T23:13
- Mode: rules only (dry run)
- Patients: 20
- **Flag accuracy: 20/20 (100%)**
- Missed escalations (expected escalate, got lower): 0
- Runs that used a fallback after an LLM error: 0

| Scenario | Expected | Correct |
|---|---|---|
| routine | none | 4/4 |
| single_miss | watch | 4/4 |
| non_responder | escalate | 4/4 |
| warning_symptom | escalate | 4/4 |
| missed_critical | escalate | 4/4 |

| Patient | Scenario | Expected | Rules | Final | Errors |
|---|---|---|---|---|---|
| P0002 | routine | none | none | none | — |
| P0001 | single_miss | watch | watch | watch | — |
| P0005 | non_responder | escalate | escalate | escalate | — |
| P0007 | warning_symptom | escalate | escalate | escalate | — |
| P0025 | missed_critical | escalate | escalate | escalate | — |
| P0003 | routine | none | none | none | — |
| P0012 | single_miss | watch | watch | watch | — |
| P0006 | non_responder | escalate | escalate | escalate | — |
| P0019 | warning_symptom | escalate | escalate | escalate | — |
| P0034 | missed_critical | escalate | escalate | escalate | — |
| P0004 | routine | none | none | none | — |
| P0015 | single_miss | watch | watch | watch | — |
| P0022 | non_responder | escalate | escalate | escalate | — |
| P0021 | warning_symptom | escalate | escalate | escalate | — |
| P0054 | missed_critical | escalate | escalate | escalate | — |
| P0008 | routine | none | none | none | — |
| P0016 | single_miss | watch | watch | watch | — |
| P0032 | non_responder | escalate | escalate | escalate | — |
| P0029 | warning_symptom | escalate | escalate | escalate | — |
| P0061 | missed_critical | escalate | escalate | escalate | — |
