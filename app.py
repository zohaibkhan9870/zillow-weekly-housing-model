import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="US Real Estate Price Outlook", layout="wide")
st.title("🏡 US Real Estate Price Outlook Dashboard")
st.write("Zillow + FRED + ML → Client-ready housing signals")
st.markdown("---")


# =================================================
# STATE MAP (FULL NAMES)
# =================================================
STATE_MAP = {
    "AL": "Alabama","AK": "Alaska","AZ": "Arizona","AR": "Arkansas","CA": "California",
    "CO": "Colorado","CT": "Connecticut","DE": "Delaware","FL": "Florida","GA": "Georgia",
    "HI": "Hawaii","ID": "Idaho","IL": "Illinois","IN": "Indiana","IA": "Iowa",
    "KS": "Kansas","KY": "Kentucky","LA": "Louisiana","ME": "Maine","MD": "Maryland",
    "MA": "Massachusetts","MI": "Michigan","MN": "Minnesota","MS": "Mississippi",
    "MO": "Missouri","MT": "Montana","NE": "Nebraska","NV": "Nevada","NH": "New Hampshire",
    "NJ": "New Jersey","NM": "New Mexico","NY": "New York","NC": "North Carolina",
    "ND": "North Dakota","OH": "Ohio","OK": "Oklahoma","OR": "Oregon","PA": "Pennsylvania",
    "RI": "Rhode Island","SC": "South Carolina","SD": "South Dakota","TN": "Tennessee",
    "TX": "Texas","UT": "Utah","VT": "Vermont","VA": "Virginia","WA": "Washington",
    "WV": "West Virginia","WI": "Wisconsin","WY": "Wyoming"
}


# =================================================
# HELPERS
# =================================================
def friendly_label(p):
    if p >= 0.65:
        return "🟢 Supportive"
    elif p <= 0.45:
        return "🔴 Risky"
    return "🟡 Unclear"


def regime_from_prob(p):
    if p >= 0.65:
        return "Supportive"
    elif p <= 0.45:
        return "Risky"
    return "Unclear"


def deal_score(p):
    return int(np.clip(round(p * 100), 0, 100))


def suggested_action(prob):
    if prob >= 0.65:
        return "Supportive conditions. This is generally a favorable time to move forward."
    elif prob <= 0.45:
        return "Be careful — risk is high. Waiting or demanding strong value may be prudent."
    else:
        return "Market conditions are unclear. Staying flexible and monitoring trends is advised."


def proxy_up_probability(price_series):
    pct = price_series.pct_change(13).dropna()
    if pct.empty:
        return None
    raw = float(pct.tail(1).values[0])
    prob = 0.50 + np.clip(raw, -0.10, 0.10) * 2.0
    return float(np.clip(prob, 0.05, 0.95))


# =================================================
# FRED LOADER
# =================================================
def load_fred(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": st.secrets["FRED_API_KEY"],
        "file_type": "json"
    }
    r = requests.get(url, params=params, timeout=30)
    df = pd.DataFrame(r.json()["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")[["value"]]


# =================================================
# FILE UPLOAD
# =================================================
st.subheader("📤 Upload Zillow Files")
c1, c2 = st.columns(2)
with c1:
    price_file = st.file_uploader("Weekly Median Sale Price CSV", type="csv")
with c2:
    value_file = st.file_uploader("Monthly ZHVI CSV", type="csv")

if not price_file or not value_file:
    st.info("Upload both Zillow files to continue.")
    st.stop()

price_df = pd.read_csv(price_file)
value_df = pd.read_csv(value_file)


# =================================================
# LOCATION SELECTION (FINAL, FIXED)
# =================================================
st.subheader("🌍 Select Location")

metro_list = sorted(
    set(price_df["RegionName"]).intersection(
        set(value_df["RegionName"])
    )
)

# Build structured metro table
metro_records = []
for m in metro_list:
    city, abbr = m.rsplit(",", 1)
    abbr = abbr.strip()
    if abbr in STATE_MAP:
        metro_records.append({
            "metro_raw": m,
            "metro_display": f"{city.strip()}, {STATE_MAP[abbr]}",
            "state_full": STATE_MAP[abbr]
        })

metro_df = pd.DataFrame(metro_records)

search = st.text_input("🔍 Search metro (optional)", "").strip()

auto_state = None
auto_metro = None

if search:
    matches = metro_df[metro_df["metro_display"].str.lower().str.contains(search.lower())]
    if not matches.empty:
        auto_state = matches.iloc[0]["state_full"]
        auto_metro = matches.iloc[0]["metro_raw"]

states = sorted(metro_df["state_full"].unique())
state_index = states.index(auto_state) if auto_state in states else 0
selected_state = st.selectbox("Choose State", states, index=state_index)

state_metros_df = metro_df[metro_df["state_full"] == selected_state]

metro_index = (
    state_metros_df["metro_raw"].tolist().index(auto_metro)
    if auto_metro in state_metros_df["metro_raw"].tolist()
    else 0
)

selected_metro = st.selectbox(
    "Choose Metro",
    state_metros_df["metro_raw"].tolist(),
    index=metro_index
)

run = st.button("✅ Run Forecast")
if not run:
    st.stop()
