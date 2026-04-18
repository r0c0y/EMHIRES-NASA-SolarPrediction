import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from state import GridAdvisorState

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark",
    "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland",
    "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
    "UK": "United Kingdom",
}

_SYSTEM_PROMPT = """You are a grid integration advisor for European solar energy systems.
Your audience is grid operators, TSO/DSO planners, and energy system managers — not consumers.
Respond ONLY with valid JSON - no markdown, no code blocks, no extra text.

Output must match this exact structure exactly:
{
  "forecast_summary": "2-3 sentence operational summary for grid operators",
  "risk_periods": [
    {"period": "string", "start_hour": 0, "end_hour": 3, "risk": "string", "severity": "high|medium|low"}
  ],
  "strategies": [
    {"title": "string", "description": "string", "category": "grid_balancing|storage|demand_response|market", "source": "retrieved|general"}
  ],
  "references": ["cited source from retrieved guidelines"],
  "responsible_ai_note": "string"
}

FORECAST SUMMARY rules:
- Compute peak power as: Installed Capacity (kW) × Peak CF. Use this exact kW value — do not estimate or round differently.
- State generation window hours, peak kW output, and variability level for operational planning.

RISK PERIOD rules:
- Always flag the evening ramp-down when peak CF > 0.05. Find the exact hour output starts declining after the daily peak using the profile, set start_hour there, end_hour where CF nears zero.
- NEVER flag nighttime (20:00-06:00) or morning ramp-up (06:00-13:00) as risks.
- start_hour must be less than end_hour. Max 5-hour window. Hours derived from actual profile only.
- severity: high = variability_score > 0.25 or sudden mid-day CF drop >30%; medium = rapid evening ramp-down; low = minor fluctuation.
- For variability_score < 0.10: only the evening ramp-down at medium severity.

STRATEGY rules — grid operator / TSO level ONLY:
- Priority order: (1) Reserve activation & grid balancing, (2) BESS dispatch scheduling, (3) Curtailment vs storage trade-offs, (4) Intraday market participation.
- Reference specific mechanisms from retrieved guidelines: PICASSO aFRR, MARI mFRR, SIDC intraday trading, ENTSO-E balancing platforms, SNSP limits, FCR/FFR procurement.
- Each strategy must cite specific hours from the profile and quantified thresholds where the guidelines provide them.
- category field: "grid_balancing" for reserve/frequency actions, "storage" for BESS dispatch, "demand_response" for load management, "market" for trading/intraday.
- Do NOT recommend consumer-level actions (EV scheduling, building pre-cooling) unless no grid-level strategy applies.

REFERENCES rules:
- List 2-4 specific sources actually cited in your strategies (e.g. "ENTSO-E PICASSO — aFRR activation for frequency restoration", "IRENA 2024 — BESS sizing: 1h storage per 25% solar penetration").
- Only cite sources present in the retrieved guidelines. Do not fabricate citations.

Base all strategies on the retrieved grid management guidelines.
Mark source "retrieved" if from guidelines, "general" if from general knowledge."""


def recommendation_node(state: GridAdvisorState) -> dict:
    chunks = state["retrieved_chunks"]
    docs_block = "\n\n".join(chunks)

    hourly_profile = state["hourly_profile"]
    peak_cf = max(hourly_profile) if hourly_profile else state["cf_value"]
    peak_hour = int(np.argmax(hourly_profile)) if hourly_profile else 0

    country_name = _COUNTRY_NAMES.get(state["country"], state["country"])

    peak_power_kw = round(peak_cf * state["capacity_kw"], 1)

    user_message = (
        f"Country: {country_name}\n"
        f"Forecast Date: {state['forecast_date']}\n"
        f"Installed Capacity: {state['capacity_kw']} kW\n"
        f"Peak Capacity Factor: {state['cf_value']:.3f}\n"
        f"Peak Power Output: {peak_power_kw} kW at {peak_hour:02d}:00\n"
        f"Variability Score: {state['risk_summary'].get('variability_score', 'N/A')}\n\n"
        f"Risk Flags:\n" + "\n".join(f"- {f}" for f in state["risk_flags"]) + "\n\n"
        + (
            "Hourly CF Profile (use for precise hour identification):\n"
            + ", ".join(f"{h:02d}:00={cf:.3f}" for h, cf in enumerate(hourly_profile))
            + "\n\n"
            if hourly_profile else ""
        )
        + f"Grid Management Guidelines (retrieved):\n{docs_block}"
    )

    try:
        llm = ChatOpenAI(
            model="openrouter/free",
            temperature=0.2,
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_message)]
        response = llm.invoke(messages)
        raw = response.content.strip()
        # Strip markdown code fences that some models add despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        recommendations = json.loads(raw)
        # Ensure all required keys are present
        for key in ("forecast_summary", "risk_periods", "strategies", "references", "responsible_ai_note"):
            recommendations.setdefault(key, [] if key in ("risk_periods", "strategies", "references") else "")
    except Exception as e:
        error_detail = str(e)
        recommendations = {
            "forecast_summary": "LLM call failed — risk analysis above is still valid.",
            "risk_periods": [],
            "strategies": [],
            "references": [],
            "responsible_ai_note": f"Recommendation generation failed: {error_detail}",
        }

    return {"final_recommendations": recommendations}
