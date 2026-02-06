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
st.title("🏡 Texas Real Estate Price Outlook")
st.write("Zillow + FRED + ML → Texas metro housing signals")
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
        return "Market conditions favor buyers if pricing is reasonable."
    elif prob <= 0.45:
        return "Risk is elevated. Consider waiting."
    return "Mixed signals. Take a cautious approach."

def proxy_up_probability(price_series):
    pct = price_series.pct_change(13).dropna()
    if pct.empty: return None
    raw = float(pct.tail(1).values[0])
    prob = 0.50 + np.clip(raw, -0.10, 0.10) * 2.0
    return float(np.clip(prob, 0.05, 0.95))

def simple_reasons(latest_row, prob):
    r=[]
    r.append("📈 Above trend" if latest_row["trend_diff"] > 0 else "📉 Below trend")
    r.append("↗️ Momentum improving" if latest_row["p13"] > 0 else "↘️ Momentum slowing")
    r.append("🏘️ Inventory rising" if latest_row["vacancy_trend"] > 0 else "🏠 Inventory tight")
    r.append("✅ Supportive model signal" if prob >= 0.65 else "⚠️ Mixed or risky signal")
    return r

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
# TEXAS METROS ONLY
# =================================================
tx_metros = sorted([
    m for m in price_df["RegionName"].unique()
    if m.endswith(", TX") and m in value_df["RegionName"].values
])

selected_metro = st.selectbox("📍 Choose Texas Metro", tx_metros)

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
# LOAD FRED
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
train = temp.iloc[:split]
test = temp.iloc[split:]

rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=10,
    random_state=42
)
rf.fit(train[predictors], train["target"])

acc = accuracy_score(test["target"], rf.predict(test[predictors]))
confidence_pct = int(round(acc * 100))

temp["prob_up"] = rf.predict_proba(temp[predictors])[:, 1]
latest = temp.iloc[-1]

# =================================================
# OUTPUT
# =================================================
st.markdown("---")
st.markdown(f"## 📌 Market Snapshot — {selected_metro}")
st.write(f"**Market Outlook:** {friendly_label(latest['prob_up'])}")
st.write(f"**Confidence:** ~{confidence_pct}% backtested accuracy")
st.write(f"**Suggested Action:** {suggested_action(latest['prob_up'], latest['trend_diff'], latest['vol'], latest['vacancy_trend'])}")

st.markdown("### Why this outlook:")
for r in simple_reasons(latest, latest["prob_up"]):
    st.write(f"- {r}")

# =================================================
# CHART
# =================================================
st.markdown("---")
st.subheader("📈 Price Trend & Regime")

fig = plt.figure(figsize=(14,6))
plt.plot(temp.index, temp["adj_price"], color="black")

for i in range(len(temp)-1):
    color = "green" if temp["prob_up"].iloc[i] >= 0.65 else "red"
    plt.axvspan(temp.index[i], temp.index[i+1], color=color, alpha=0.12)

st.pyplot(fig)
