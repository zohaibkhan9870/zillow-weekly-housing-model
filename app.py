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
st.set_page_config(page_title="Texas Real Estate Price Outlook", layout="wide")
st.title("🏡 Texas Real Estate Price Outlook Dashboard")
st.write("Zillow + FRED + ML → Texas metro housing market signals")

# =================================================
# HOW TO USE
# =================================================
st.info(
    "### How to use this dashboard\n"
    "- Avoid buying when risk is high and prices are likely to struggle\n"
    "- Compare Texas cities to see which markets look stronger or weaker right now\n"
    "- Decide when to act — buy, wait, or negotiate — based on market conditions\n\n"
    "This tool helps with market risk and timing, not exact future prices."
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
    if n_obs >= 300: return "🟢 High"
    elif n_obs >= 180: return "🟡 Medium"
    return "🔴 Low"

def suggested_action(prob, trend_diff, vol, vacancy_trend):
    if prob >= 0.65:
        return "Market has strength. Buying can make sense if value is fair."
    elif prob <= 0.45:
        return "Risk is elevated. Consider waiting or negotiating harder."
    return "Balanced market. Compare options carefully."

def action_for_table(prob):
    if prob >= 0.65: return "Favorable — consider buying"
    elif prob <= 0.45: return "Risky — be cautious"
    return "Mixed — take your time"

def proxy_up_probability(price_series):
    pct = price_series.pct_change(13).dropna()
    if pct.empty: return None
    raw = float(pct.tail(1).values[0])
    prob = 0.50 + np.clip(raw, -0.10, 0.10) * 2.0
    return float(np.clip(prob, 0.05, 0.95))

def simple_reasons(row, prob):
    reasons = []
    reasons.append("📉 Prices are below normal levels" if row["trend_diff"] < 0 else "📈 Prices are holding above normal levels")
    reasons.append("↘️ Prices have been slowing recently" if row["p13"] < 0 else "↗️ Prices are still moving upward")
    reasons.append("🏘️ More homes are coming onto the market" if row["vacancy_trend"] > 0 else "🏠 Limited supply is supporting prices")
    return reasons

# =================================================
# EARLY MARKET SIGNAL
# =================================================
def early_market_signal(row, prev_row):
    if row["p13"] > prev_row["p13"]:
        return "🟡 Prices are still going down, but the drop has started to ease."
    else:
        return "⚪ Prices are still going down, and the drop has not eased yet."

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
# LOAD MACRO DATA
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
data["vol"] = data["p13"].rolling(26).std()
data["vacancy_trend"] = data["vacancy"].diff(13)

data.dropna(inplace=True)

if len(data) < 150:
    st.warning("⚠️ Not enough reliable historical data.")
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
temp["regime"] = temp["prob_up"].apply(regime_from_prob)

latest = temp.iloc[-1]
previous = temp.iloc[-2]
early_signal = early_market_signal(latest, previous)

# =================================================
# MARKET SNAPSHOT
# =================================================
st.markdown("---")
st.markdown(f"## 📌 Market Snapshot — {selected_metro}")
st.write(f"**Market Outlook:** {friendly_label(latest['prob_up'])}")
st.write(f"**Backtested Accuracy:** ~{confidence_pct}%")
st.write(f"**Data Confidence:** {confidence_badge(len(temp))}")
st.write(f"**Suggested Action:** {suggested_action(latest['prob_up'], latest['trend_diff'], latest['vol'], latest['vacancy_trend'])}")

st.markdown("### Early market signal:")
st.write(early_signal)

st.markdown("### Why this outlook:")
for r in simple_reasons(latest, latest["prob_up"]):
    st.write(f"- {r}")

# =================================================
# NEW: FUTURE MARKET OUTLOOK TABLE
# =================================================
st.markdown("---")
st.subheader("🗓️ Future Market Outlook")

support_now = int(round(latest["prob_up"] * 100))

future_table = pd.DataFrame([
    ["1 month", max(support_now - 10, 20), "Market conditions are likely to remain weak", "Avoid rushing"],
    ["2 months", max(support_now - 7, 25), "Market conditions are likely to remain weak", "Be very selective"],
    ["3 months", max(support_now - 5, 30), "Market conditions are likely to remain weak", "Focus on discounts"],
    ["6 months", min(max(support_now, 40), 55), "Market conditions may start to balance", "Start monitoring"],
    ["1 year", min(max(support_now + 5, 45), 60), "Market conditions may improve", "Plan ahead"],
], columns=["Time Ahead", "Market affecting prices", "What this means", "Investor approach"])

future_table["Market Support Level"] = future_table["Market Support Level"].astype(str) + "%"

st.dataframe(future_table, use_container_width=True)

# =================================================
# METRO COMPARISON — TOP 3
# =================================================
st.markdown("---")
st.subheader("🏙️ Texas Metro Comparison — Top 3")

rows = []
for m in tx_metros:
    pm = price_df[price_df["RegionName"] == m]
    if pm.empty: continue
    p = pd.DataFrame(pm.iloc[0, 5:])
    p.index = pd.to_datetime(p.index)
    p.columns = ["price"]
    prob = proxy_up_probability(p["price"])
    if prob is None: continue
    rows.append([m, f"{prob*100:.0f}%", friendly_label(prob), action_for_table(prob)])

if rows:
    comp_df = pd.DataFrame(rows, columns=["Metro", "Price Up Chance", "Outlook", "What to Do"])
    st.dataframe(comp_df.sort_values("Price Up Chance", ascending=False).head(3), use_container_width=True)

# =================================================
# ALL METROS RANKING
# =================================================
st.markdown("---")
st.subheader("🏙️ Texas Metro Rankings (All Metros)")

rank_rows = []
for m in tx_metros:
    pm = price_df[price_df["RegionName"] == m]
    if pm.empty: continue
    p = pd.DataFrame(pm.iloc[0, 5:])
    p.index = pd.to_datetime(p.index)
    p.columns = ["price"]
    prob = proxy_up_probability(p["price"])
    if prob is None: continue
    trend = "Uptrend" if prob >= 0.55 else "Down / Sideways"
    rank_rows.append([m, friendly_label(prob), trend, confidence_badge(len(p.dropna()))])

rank_df = pd.DataFrame(rank_rows, columns=["Metro", "Outlook", "Trend Direction", "Confidence"])
st.dataframe(rank_df, use_container_width=True)

# =================================================
# HISTORICAL PRICE & REGIMES
# =================================================
st.markdown("---")
st.subheader("📈 Historical Price Trend & Risk Regimes")

fig = plt.figure(figsize=(14,6))
plt.plot(temp.index, temp["adj_price"], color="black", linewidth=2)

for i in range(len(temp)-1):
    color = "green" if temp["regime"].iloc[i] == "Supportive" else "red"
    plt.axvspan(temp.index[i], temp.index[i+1], color=color, alpha=0.15)

st.pyplot(fig)

# =================================================
# WEEKLY OUTLOOK
# =================================================
st.markdown("---")
st.subheader("📊 Weekly Outlook (Last 12 Weeks)")

recent = temp.tail(12)
fig2, ax = plt.subplots(figsize=(12,5))
ax.plot(recent.index, recent["prob_up"], marker="o", linewidth=2, color="black")
ax.axhline(0.65, linestyle="--", color="green", alpha=0.6)
ax.axhline(0.45, linestyle="--", color="red", alpha=0.6)
ax.set_ylim(0,1)

st.pyplot(fig2)
st.caption("Above 0.65 = supportive • Below 0.45 = risky")
