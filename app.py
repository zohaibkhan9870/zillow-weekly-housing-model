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
st.set_page_config(page_title="Texas Housing Outlook", layout="wide")
st.title("🏡 Texas Real Estate Price Outlook Dashboard")
st.write("Zillow + FRED + ML → Texas metro housing market signals")
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

def suggested_action(prob, trend_diff, vol, vacancy_trend):
    if prob >= 0.65:
        return "Market has strength. Buying can make sense if value is fair."
    elif prob <= 0.45:
        return "Risk is elevated. Consider waiting."
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
    reasons.append("📈 Prices above long-term trend" if row["trend_diff"] > 0 else "📉 Prices below long-term trend")
    reasons.append("↗️ Momentum improving" if row["p13"] > 0 else "↘️ Momentum slowing")
    reasons.append("🏘️ Inventory rising" if row["vacancy_trend"] > 0 else "🏠 Inventory tight")
    reasons.append("✅ Model signals supportive conditions" if prob >= 0.65
                   else "⚠️ Model shows mixed or risky conditions")
    return reasons

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
# TEXAS METROS ONLY
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
    st.warning("⚠️ Not enough historical data for this metro.")
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
weekly_label = friendly_label(latest["prob_up"])
monthly_regime = temp.resample("M")["regime"].agg(lambda x: x.value_counts().index[0]).iloc[-1]

# =================================================
# SNAPSHOT
# =================================================
st.markdown("---")
st.markdown(f"## 📌 Market Snapshot — {selected_metro}")
st.write(f"**Market Outlook:** {weekly_label}")
st.write(f"**Confidence:** ~{confidence_pct}% backtested accuracy")
st.write(f"**Suggested Action:** {suggested_action(latest['prob_up'], latest['trend_diff'], latest['vol'], latest['vacancy_trend'])}")

st.markdown("### Why this outlook:")
for r in simple_reasons(latest, latest["prob_up"]):
    st.write(f"- {r}")

# =================================================
# WEEKLY + MONTHLY LABELS
# =================================================
st.markdown("---")
st.subheader("📌 Weekly Prediction")
st.info(f"Weekly Outlook: {weekly_label}")

st.subheader("📌 Monthly Prediction")
st.info(f"Monthly Trend: {monthly_regime}")

# =================================================
# METRO COMPARISON
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
    st.dataframe(comp_df.sort_values("Price Up Chance", ascending=False).head(3))

# =================================================
# HISTORICAL PRICE + REGIME GRAPH
# =================================================
st.markdown("---")
st.subheader("📈 Historical Price Trend & Risk Regimes")

fig = plt.figure(figsize=(14,6))
plt.plot(temp.index, temp["adj_price"], color="black")

for i in range(len(temp)-1):
    color = "green" if temp["regime"].iloc[i] == "Supportive" else "red"
    plt.axvspan(temp.index[i], temp.index[i+1], color=color, alpha=0.15)

st.pyplot(fig)

# =================================================
# WEEKLY PROBABILITY GRAPH
# =================================================
st.markdown("---")
st.subheader("📊 Weekly Outlook (Last 12 Weeks)")

recent = temp.tail(12)
fig2, ax = plt.subplots(figsize=(12,5))
ax.plot(recent.index, recent["prob_up"], marker="o", linewidth=2, color="black")
ax.axhline(0.65, linestyle="--", color="green", alpha=0.6)
ax.axhline(0.45, linestyle="--", color="red", alpha=0.6)
ax.set_ylim(0, 1)

st.pyplot(fig2)
st.caption("Above 0.65 = supportive • Below 0.45 = risky")

# =================================================
# MONTHLY FUTURE PROJECTION GRAPH (NEW)
# =================================================
st.markdown("---")
st.subheader("📅 Monthly Outlook — Forward Projection (Next 3 Months)")
st.caption("Dashed line shows a model-based projection assuming current conditions persist.")

last_date = temp.index[-1]
last_price = temp["adj_price"].iloc[-1]
last_prob = temp["prob_up"].iloc[-1]

future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthEnd(1),
    periods=3,
    freq="M"
)

if last_prob >= 0.65:
    monthly_change = 0.01
elif last_prob <= 0.45:
    monthly_change = -0.01
else:
    monthly_change = 0.0

future_prices = [
    last_price * ((1 + monthly_change) ** (i + 1))
    for i in range(len(future_dates))
]

past_monthly = temp["adj_price"].resample("M").last().tail(12)

fig3, ax3 = plt.subplots(figsize=(12,5))
ax3.plot(past_monthly.index, past_monthly.values, color="black", linewidth=2, label="Historical")
ax3.plot(
    future_dates,
    future_prices,
    linestyle="--",
    marker="o",
    color="green" if last_prob >= 0.65 else "red" if last_prob <= 0.45 else "gold",
    label="Projected Outlook"
)

ax3.axvline(last_date, linestyle=":", color="gray", alpha=0.7)
ax3.set_ylabel("Inflation-Adjusted Price Index")
ax3.legend()

st.pyplot(fig3)
