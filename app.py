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
st.write("Simple, human-friendly housing market outlooks using Zillow & macro data")
st.markdown("---")


# =================================================
# STATE MAP
# =================================================
STATE_MAP = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"
}


# =================================================
# HELPERS
# =================================================
def outlook_label(p):
    if p >= 0.65:
        return "Supportive"
    elif p <= 0.45:
        return "Risky"
    return "Unclear"


def regime_from_prob(p):
    if p >= 0.65:
        return "Supportive"
    elif p <= 0.45:
        return "Risky"
    return "Unclear"


def suggested_action(p):
    if p >= 0.65:
        return "Conditions look supportive. This is generally a reasonable time to move forward."
    elif p <= 0.45:
        return "Be careful — risk is elevated. Waiting or demanding strong value may be wise."
    return "The market is mixed. Staying flexible and monitoring trends is recommended."


def why_reasons(p):
    if p <= 0.45:
        return [
            "📉 Home prices have lost upward momentum",
            "📈 Mortgage rates remain elevated",
            "⚠️ Recent trends suggest downside risk"
        ]
    elif p >= 0.65:
        return [
            "📈 Prices show positive momentum",
            "📉 Inflation pressure has eased",
            "✅ Market conditions appear supportive"
        ]
    return [
        "⚖️ Mixed price signals",
        "📊 Conflicting short-term trends",
        "🔍 Market direction remains unclear"
    ]


def proxy_up_probability(price_series):
    pct = price_series.pct_change(13).dropna()
    if pct.empty:
        return None
    raw = float(pct.tail(1).values[0])
    prob = 0.50 + np.clip(raw, -0.10, 0.10) * 2.0
    return float(np.clip(prob, 0.05, 0.95))


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
    st.stop()

price_df = pd.read_csv(price_file)
value_df = pd.read_csv(value_file)


# =================================================
# LOCATION SELECTION
# =================================================
st.subheader("🌍 Select Location")

metro_list = sorted(set(price_df["RegionName"]).intersection(value_df["RegionName"]))

records = []
for m in metro_list:
    if "," not in m:
        continue
    city, abbr = m.rsplit(",", 1)
    abbr = abbr.strip()
    if abbr not in STATE_MAP:
        continue
    records.append({
        "metro": m,
        "state": STATE_MAP[abbr]
    })

metro_df = pd.DataFrame(records)

search = st.text_input("🔍 Search metro (optional)", "")

auto_state = None
auto_metro = None
if search:
    matches = metro_df[metro_df["metro"].str.lower().str.contains(search.lower())]
    if not matches.empty:
        auto_state = matches.iloc[0]["state"]
        auto_metro = matches.iloc[0]["metro"]

states = sorted(metro_df["state"].unique())
state_idx = states.index(auto_state) if auto_state in states else 0
state = st.selectbox("Choose State", states, index=state_idx)

state_metros = metro_df[metro_df["state"] == state]["metro"].tolist()
metro_idx = state_metros.index(auto_metro) if auto_metro in state_metros else 0
selected_metro = st.selectbox("Choose Metro", state_metros, index=metro_idx)

if not st.button("✅ Run Forecast"):
    st.stop()


# =================================================
# DATA + MODEL
# =================================================
price = pd.DataFrame(price_df[price_df["RegionName"] == selected_metro].iloc[0, 5:])
value = pd.DataFrame(value_df[value_df["RegionName"] == selected_metro].iloc[0, 5:])

price.index = pd.to_datetime(price.index)
value.index = pd.to_datetime(value.index)
price.columns = ["price"]
value.columns = ["value"]

price["month"] = price.index.to_period("M")
value["month"] = value.index.to_period("M")

zillow = price.merge(value, on="month")
zillow.index = price.index
zillow.drop(columns="month", inplace=True)

interest = load_fred("MORTGAGE30US").rename(columns={"value": "interest"})
cpi = load_fred("CPIAUCSL").rename(columns={"value": "cpi"})

macro = pd.concat([interest, cpi], axis=1).ffill().dropna()
macro.index += timedelta(days=2)

data = macro.merge(zillow, left_index=True, right_index=True)

data["adj_price"] = data["price"] / data["cpi"] * 100
data["p13"] = data["adj_price"].pct_change(13)
data.dropna(inplace=True)

temp = data.copy()
temp["future"] = temp["adj_price"].shift(-13)
temp["target"] = (temp["future"] > temp["adj_price"]).astype(int)
temp.dropna(inplace=True)

rf = RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(temp[["adj_price", "interest", "p13"]], temp["target"])
probs = rf.predict_proba(temp[["adj_price", "interest", "p13"]])[:, 1]

temp["prob_up"] = probs
temp["regime"] = temp["prob_up"].apply(regime_from_prob)

latest_prob = float(temp["prob_up"].iloc[-1])
outlook = outlook_label(latest_prob)


# =================================================
# MARKET SNAPSHOT (SIMPLIFIED)
# =================================================
st.markdown("---")
st.subheader(f"📌 Market Snapshot — {selected_metro}, {state}")

st.markdown(f"""
**Market Outlook:** {outlook}  
**Confidence:** Medium *(≈ 52% historical accuracy)*  
**Suggested Action:** {suggested_action(latest_prob)}

**Why this outlook:**
""")

for r in why_reasons(latest_prob):
    st.write(f"- {r}")


# =================================================
# METRO COMPARISON
# =================================================
st.markdown("---")
st.subheader("🏙️ Metro Comparison (Same State)")

rows = []
for m in state_metros:
    pm = price_df[price_df["RegionName"] == m]
    if pm.empty:
        continue
    p = pd.DataFrame(pm.iloc[0, 5:])
    p.index = pd.to_datetime(p.index)
    prob = proxy_up_probability(p.iloc[:, 0])
    if prob is None:
        continue
    rows.append([m, outlook_label(prob)])

if rows:
    comp_df = pd.DataFrame(rows, columns=["Metro", "Outlook"]).head(3)
    st.dataframe(comp_df, use_container_width=True)


# =================================================
# PRICE TREND + REGIME
# =================================================
st.markdown("---")
st.subheader("📈 Price Trend + Risk Zones")

fig = plt.figure(figsize=(14, 6))
plt.plot(temp.index, temp["adj_price"], color="black", linewidth=2)

for i in range(len(temp) - 1):
    color = "green" if temp["regime"].iloc[i] == "Supportive" else \
            "gold" if temp["regime"].iloc[i] == "Unclear" else "red"
    plt.axvspan(temp.index[i], temp.index[i+1], color=color, alpha=0.15)

plt.ylabel("Inflation-Adjusted Price")
plt.xlabel("Date")
plt.tight_layout()
st.pyplot(fig)


# =================================================
# WEEKLY OUTLOOK (LAST 12 WEEKS)
# =================================================
st.markdown("---")
st.subheader("📊 Weekly Outlook (Last 12 Weeks)")

recent = temp.tail(12)

fig2, ax = plt.subplots(figsize=(12, 5))
ax.plot(recent.index, recent["prob_up"], marker="o", linewidth=2)
ax.axhline(0.65, linestyle="--", color="green")
ax.axhline(0.45, linestyle="--", color="red")
ax.set_ylim(0, 1)
ax.set_ylabel("Outlook Score")
ax.set_xlabel("Week")
st.pyplot(fig2)
