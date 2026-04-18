"""SolarIntel — Energy Generation Analytics"""

import os
import streamlit as st
from dotenv import load_dotenv
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

load_dotenv()
import tabs.prediction_dashboard as prediction_dashboard
import tabs.country_comparison as country_comparison
import tabs.grid_advisor as grid_advisor

st.set_page_config(page_title="SolarIntel", page_icon="☀️", layout="wide")

st.title("☀️ SolarIntel")
st.caption("Solar Energy Generation Forecasting — EMHIRES × NASA")

tab_forecast, tab_compare, tab_advisor = st.tabs(
    ["Prediction Dashboard", "Country Comparison", "Grid Advisor"]
)

with tab_forecast:
    prediction_dashboard.render()

with tab_compare:
    country_comparison.render()

with tab_advisor:
    grid_advisor.render()
