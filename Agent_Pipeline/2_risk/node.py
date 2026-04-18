import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from state import GridAdvisorState


def risk_analysis_node(state: GridAdvisorState) -> dict:
    profile = state["hourly_profile"]

    if not profile:
        return {
            "risk_summary": {"variability_score": 0.0, "low_hours": [], "ramp_events": []},
            "risk_flags": ["No hourly profile available — risk analysis skipped."],
        }

    _max_cf = max(profile)
    peak_cf = _max_cf if _max_cf > 0 else 1.0
    ramp_rates = [abs(profile[h] - profile[h - 1]) for h in range(1, 24)]
    variability_score = round(sum(ramp_rates) / (23 * peak_cf), 4)

    low_threshold = _max_cf * 0.10
    low_hours = [h for h, cf in enumerate(profile) if cf < low_threshold]
    ramp_events = [h for h in range(1, 24) if abs(profile[h] - profile[h - 1]) > 0.12]

    risk_flags = []
    if ramp_events:
        risk_flags.append(f"High variability window: hours {ramp_events[0]}-{ramp_events[-1]}")
    if low_hours:
        risk_flags.append(f"Minimal generation risk: {len(low_hours)} hours below 5% capacity")
    if variability_score < 0.05:
        risk_flags.append("Low variability day - stable but low output expected")

    return {
        "risk_summary": {
            "variability_score": variability_score,
            "low_hours": low_hours,
            "ramp_events": ramp_events,
        },
        "risk_flags": risk_flags,
    }
