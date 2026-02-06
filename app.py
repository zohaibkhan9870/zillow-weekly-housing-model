import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="Texas Real Estate Outlook", layout="wide")
st.title("🏡 Texas Real Estate Price Outlook Dashboard")
st.write("Zillow + FRED + ML → Texas metro market regime signals")

# =================================================
# HOW TO USE (NEW)
# =================================================
st.info(
    "### How investors use this dashboard:\n"
    "- Avoid buying during **risky market regimes**\n"
    "- Compare **Texas metros by relative strength**\n"
    "- Time **capital deployment**, not individual deals\n\n"
    "This tool focuses on **market conditions**, not exact price predictions."
)

st.markdown("---")

# =================================================
# HELPERS
# =================================================
def friendly_label(p):
    if p >= 0.65: return "🟢 Supportive"
    elif p <= 0.45: return "🔴 Risky"
    return "🟡 Unclear"

def regime_from_prob(p):
    if p >= 0.65: return "Supportive"
    elif p <= 0.45: return "Risky"
    return "Unclear"

def confidence_badge(n_obs):
    if n_obs >= 300: return "High"
    elif n_obs >= 180: return "Medium"
    return "Low"

def suggested_action(prob):
    if prob >= 0.65:
        return "Conditions look supportive. Buying can make sense if pricing is fair."
    elif prob <= 0.45:
        return "Risk is elevated. Consider waiting or negotiating harder."
    return "Mixed signals. Take your time and compare options."

def proxy_up_probability(price_series):
    pct = price_series.pct_change(13).dropna()
    if pct.empty: return None
    raw = float(pct.tail(1).values[0])
    prob = 0.50 + np.clip(raw, -0.10, 0.10) * 2.0
    return float(np.clip(prob, 0.05, 0.95))

# =================================================
# FRED LOADER
# =================================================
def load_fred(series_id):
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": st.secrets["FRED_API_KEY"],
            "file_type": "json"
        },
        timeout=30
    )
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
# TEXAS METROS
# =================================================
tx_metros = sorted([
    m for m in price_df["RegionName"].unique()
    if m.endswith(", TX") and m in value_df["RegionName"].values
])

selected_metro = st.selectbox("📍 Select Texas Metro", tx_metros)

if not st.button("✅ Run Forecast"):
    st.stop()

# =================================================
# PREP ZILLOW DATA
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

# =================================================
# LOAD MACRO
# =================================================
interest = load_fred("MORTGAGE30US").rename(columns={"value": "interest"})
cpi = load_fred("CPIAUCSL").rename(columns={"value": "cpi"})
vacancy = load_fred("RRVRUSQ156N").rename(columns={"value": "vacancy"})

macro = pd.concat([interest, cpi, vacancy], axis=1).sort_index().ffill().dropna()
macro.index += timedelta(days=2)

data = macro.merge(zillow, left_index=True, right_index=True)

# =================================================
# FEATURES
# =================================================
data["adj_price"] = data["price"] / data["cpi"] * 100
data["p13"] = data["adj_price"].pct_change(13)
data["trend"] = data["adj_price"].rolling(52).mean()
data["trend_diff"] = data["adj_price"] - data["trend"]
data["vacancy_trend"] = data["vacancy"].diff(13)
data.dropna(inplace=True)

if len(data) < 150:
    st.warning("⚠️ Not enough reliable history for this metro.")
    st.stop()

# =================================================
# MODEL
# =================================================
predictors = ["adj_price", "interest", "vacancy", "p13"]

temp = data.copy()
temp["future"] = temp["adj_price"].shift(-13)
temp["target"] = (temp["future"] > temp["adj_price"]).astype(int)
temp.dropna(inplace=True)

split = int(len(temp) * 0.7)
train, test = temp.iloc[:split], temp.iloc[split:]

rf = RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

acc = accuracy_score(test["target"], rf.predict(test[predictors]))
confidence_pct = int(round(acc * 100))

temp["prob_up"] = rf.predict_proba(temp[predictors])[:, 1]
latest = temp.iloc[-1]

# =================================================
# SNAPSHOT
# =================================================
st.markdown("---")
st.markdown(f"## 📌 Market Snapshot — {selected_metro}")
st.write(f"**Market Outlook:** {friendly_label(latest['prob_up'])}")
st.write(f"**Model Confidence:** {confidence_badge(len(temp))}")
st.write(f"**Backtested Accuracy:** ~{confidence_pct}%")
st.write(f"**Suggested Action:** {suggested_action(latest['prob_up'])}")

# =================================================
# FORWARD-LOOKING REGIME (NEW)
# =================================================
st.markdown("---")
st.subheader("📅 Forward-Looking Market Regime (Based on Latest Confirmed Data)")
st.caption("This shows the likely **market phase**, not exact future prices.")

outlook = friendly_label(latest["prob_up"])
st.write(f"- **T+1 (next phase):** {outlook}")
st.write(f"- **T+2:** {outlook}")
st.write(f"- **T+3:** {outlook}")

# =================================================
# ALL TEXAS METROS RANKING (NEW)
# =================================================
st.markdown("---")
st.subheader("🏙️ Texas Metro Rankings (All Metros)")

rows = []
for m in tx_metros:
    pm = price_df[price_df["RegionName"] == m]
    if pm.empty: continue
    p = pd.DataFrame(pm.iloc[0, 5:])
    p.index = pd.to_datetime(p.index)
    p.columns = ["price"]
    prob = proxy_up_probability(p["price"])
    if prob is None: continue
    rows.append([
        m,
        friendly_label(prob),
        "Uptrend" if prob >= 0.55 else "Down / Sideways",
        confidence_badge(len(p.dropna()))
    ])

rank_df = pd.DataFrame(
    rows,
    columns=["Metro", "Outlook", "Trend Direction", "Confidence"]
)

st.dataframe(rank_df, use_container_width=True)
