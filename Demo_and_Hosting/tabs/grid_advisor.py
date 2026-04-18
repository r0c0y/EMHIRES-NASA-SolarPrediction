import streamlit as st
import numpy as np
import os
import sys as _sys
import importlib.util as _ilu
import datetime as _dt
import threading
import plotly.graph_objects as go
from utils import COUNTRIES, PLOT_LAYOUT, AMBER

# Module-level setup — runs once when the module is imported, not on every rerender
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_AP = os.path.join(_REPO_ROOT, "Agent_Pipeline")
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
if _AP not in _sys.path:
    _sys.path.insert(0, _AP)


@st.cache_resource
def get_agent_graph():
    _store_path = os.path.join(_AP, "3_rag", "store.py")
    _store_spec = _ilu.spec_from_file_location("store", _store_path)
    _store_mod = _ilu.module_from_spec(_store_spec)
    _sys.modules["store"] = _store_mod
    _store_spec.loader.exec_module(_store_mod)
    _store_mod.build_vector_store(os.path.join(_AP, "knowledge_base"))
    from graph import build_graph
    return build_graph()


# Pre-warm the graph in background so it's ready when the user clicks Run
threading.Thread(target=get_agent_graph, daemon=True).start()


def render():
    st.subheader("Grid Advisor : Agentic AI Optimization")

    st.markdown("#### Settings")
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        adv_country = st.selectbox(
            "Country", list(COUNTRIES.keys()),
            index=list(COUNTRIES.keys()).index("ES"),
            format_func=lambda x: f"{x} — {COUNTRIES[x]}",
            key="adv_country",
        )
    with adv_col2:
        adv_capacity = st.number_input(
            "System Size (kW)", value=100.0, min_value=1.0, step=10.0, key="adv_capacity"
        )
    with adv_col3:
        _today = _dt.date.today()
        adv_date = st.date_input(
            "Forecast Date",
            value=_today + _dt.timedelta(days=1),
            min_value=_today + _dt.timedelta(days=1),
            max_value=_today + _dt.timedelta(days=15),
            key="adv_date",
        )

    run_btn = st.button("Run Grid Advisor", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Analyzing solar forecast and generating recommendations…"):
            graph = get_agent_graph()
            try:
                agent_input = {
                    "country": adv_country,
                    "capacity_kw": adv_capacity,
                    "model_name": "Random Forest",
                    "forecast_date": str(adv_date),
                    "weather_forecast": [],
                    "cf_value": 0.0,
                    "hourly_profile": [],
                    "monthly_profile": [],
                    "risk_summary": {},
                    "risk_flags": [],
                    "retrieved_chunks": [],
                    "final_recommendations": {},
                    "error": None,
                }
                result = graph.invoke(agent_input)
                st.session_state["advisor_result"] = result
                st.session_state["advisor_country"] = adv_country
                st.session_state["advisor_capacity"] = adv_capacity
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

    if "advisor_result" in st.session_state:
        result = st.session_state["advisor_result"]
        # Always use capacity from the run that produced this result — not the current widget value
        run_capacity = st.session_state.get("advisor_capacity", adv_capacity)
        country_label = COUNTRIES.get(st.session_state.get("advisor_country", ""), "")
        rec = result.get("final_recommendations", {})
        risk = result.get("risk_summary", {})
        hourly = result.get("hourly_profile", [])

        daily_kwh = round(sum(hourly) * run_capacity, 1) if hourly else 0.0
        active_hours = 24 - len(risk.get("low_hours", []))
        peak_power_kw = round(result.get("cf_value", 0) * run_capacity, 1)

        st.markdown("---")

        if rec.get("forecast_summary"):
            st.info(f"**Forecast Summary:** {rec['forecast_summary']}")
            st.markdown("")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Peak Power", f"{peak_power_kw} kW")
            st.caption("Maximum power output at the best hour of the day")
        with mc2:
            st.metric("Est. Daily Output", f"{daily_kwh} kWh")
            st.caption("Total energy this system is expected to generate today")
        with mc3:
            st.metric("Active Solar Hours", str(active_hours))
            st.caption("Hours of meaningful solar generation today")
        with mc4:
            st.metric("Variability Score", f"{risk.get('variability_score', 0):.4f}")
            st.caption("How smoothly output changes between hours (lower = more stable)")

        st.markdown("---")

        if hourly:
            st.markdown("#### Hourly Generation Profile")
            fig_adv = go.Figure()
            fig_adv.add_vrect(x0=0, x1=6, fillcolor="rgba(30,30,30,0.4)", line_width=0,
                              annotation_text="Night", annotation_position="top left")
            fig_adv.add_vrect(x0=18, x1=24, fillcolor="rgba(30,30,30,0.4)", line_width=0,
                              annotation_text="Night", annotation_position="top right")
            fig_adv.add_trace(go.Scatter(
                x=list(range(24)), y=[cf * run_capacity for cf in hourly],
                mode="lines", line=dict(color=AMBER, width=3, shape="spline"),
                fill="tozeroy", fillcolor="rgba(217,119,6,0.12)", name="Output (kW)",
            ))
            peak_cf_val = max(hourly)
            peak_h = int(np.argmax(hourly))
            fig_adv.add_trace(go.Scatter(
                x=[peak_h], y=[peak_cf_val * run_capacity],
                mode="markers+text",
                marker=dict(color="#22C55E", size=12, symbol="star"),
                text=[f"Peak {peak_cf_val * run_capacity:.1f} kW"],
                textposition="top center",
                textfont=dict(color="#22C55E", size=11),
                name="Peak Hour", showlegend=False,
            ))
            sev_chart_colors = {"high": "rgba(239,68,68,0.15)", "medium": "rgba(245,158,11,0.15)", "low": "rgba(34,197,94,0.10)"}
            for rp in rec.get("risk_periods", []):
                sh = rp.get("start_hour")
                eh = rp.get("end_hour")
                if sh is not None and eh is not None and isinstance(sh, int) and isinstance(eh, int) and sh < eh:
                    sev = rp.get("severity", "low").lower()
                    fig_adv.add_vrect(
                        x0=sh, x1=eh,
                        fillcolor=sev_chart_colors.get(sev, "rgba(120,113,108,0.10)"),
                        line_width=0,
                        annotation_text=rp.get("period", "").split("(")[0].strip(),
                        annotation_position="top left",
                        annotation_font_size=10,
                    )
            fig_adv.update_layout(
                xaxis=dict(title="Hour (UTC)", dtick=2),
                yaxis_title="Output (kW)", height=300,
                legend=dict(orientation="h", y=1.12), **PLOT_LAYOUT,
            )
            st.plotly_chart(fig_adv, use_container_width=True)
            st.caption(f"Real Open-Meteo forecast for {country_label} on {result.get('forecast_date', 'N/A')} — {run_capacity} kW system")

        st.markdown("---")

        risk_periods = rec.get("risk_periods", [])
        if risk_periods:
            st.markdown("#### Risk Periods")
            sev_colors = {"high": "#EF4444", "medium": "#F59E0B", "low": "#22C55E"}
            for rp in risk_periods:
                sev = rp.get("severity", "low").lower()
                color = sev_colors.get(sev, "#78716C")
                st.markdown(
                    f"<div style='border-left:4px solid {color}; padding:8px 14px; margin-bottom:8px; "
                    f"background:#1C1917; border-radius:4px'>"
                    f"<strong style='color:{color}'>[{sev.upper()}]</strong> "
                    f"<strong>{rp.get('period','')}</strong> — {rp.get('risk','')}</div>",
                    unsafe_allow_html=True,
                )

        strategies = rec.get("strategies", [])
        if strategies:
            st.markdown("#### Grid Optimization Strategies")
            cat_colors = {"grid_balancing": "#3B82F6", "storage": "#8B5CF6", "demand_response": "#F59E0B", "market": "#22C55E"}
            for s in strategies:
                cat = s.get("category", "general")
                cat_color = cat_colors.get(cat, "#6B7280")
                cat_label = cat.replace("_", " ").title()
                st.markdown(
                    f"<div style='border:1px solid #44403C; padding:10px 14px; margin-bottom:8px; "
                    f"background:#1C1917; border-radius:6px'>"
                    f"<strong>{s.get('title','')}</strong> "
                    f"<span style='font-size:11px; background:{cat_color}; color:white; padding:1px 7px; border-radius:3px; margin-left:6px'>{cat_label}</span><br>"
                    f"<span style='color:#A8A29E; font-size:13px'>{s.get('description','')}</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### Knowledge Base Sources")
        st.caption(
            "Recommendations are grounded in the following research documents ingested into the RAG knowledge base:"
        )
        _KB_SOURCES = [
            ("Solar Variability & Grid Stability", "Ramp rates, frequency regulation, spinning reserve requirements, voltage fluctuation management for high solar penetration."),
            ("Battery Energy Storage Systems (BESS)", "Optimal sizing relative to solar capacity, charge/discharge strategies, peak shaving, round-trip efficiency benchmarks, European deployment examples."),
            ("Demand-Side Management & Load Shifting", "Time-of-use tariff design, industrial demand response, smart EV charging alignment, residential load shifting, European DSM policy."),
            ("Curtailment, Balancing & Cross-Border Grid Management", "Curtailment vs. storage trade-offs, cross-border interconnection, intraday market mechanisms, ENTSO-E balancing, technical minimum generation constraints."),
        ]
        for title, desc in _KB_SOURCES:
            st.markdown(
                f"<div style='border:1px solid #44403C; padding:8px 14px; margin-bottom:6px; background:#1C1917; border-radius:6px'>"
                f"<strong style='color:#A8A29E'>{title}</strong><br>"
                f"<span style='color:#78716C; font-size:12px'>{desc}</span></div>",
                unsafe_allow_html=True,
            )
