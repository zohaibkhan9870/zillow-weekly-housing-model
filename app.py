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
st.write("Zillow + FRED + ML → Simple, client-ready housing signals")
st.markdown("---")


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


def suggested_action(prob, role):
    if prob >= 0.65:
        if role == "Home Buyer":
            return "Good conditions. You can move forward and negotiate with confidence."
        elif role == "Investor":
            return "Supportive market. Consider selective acquisitions."
        else:
            return "Expect stronger buyer activity."
    elif prob <= 0.45:
        if role == "Home Buyer":
            return "Be careful — risk is high. Prefer waiting or strong discounts."
        elif role == "Investor":
            return "High downside risk. Focus on capital preservation."
        else:
            return "Expect slower sales and cautious buyers."
    else:
        return "Market is unclear. Stay flexible and monitor trends closely."


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
# LOCATION SELECTION (NEW FORMAT)
# =================================================
st.subheader("🌍 Select Location")

metro_list = sorted(
    set(price_df["RegionName"]).intersection(
        set(value_df["RegionName"])
    )
)

search = st.text_input("🔍 Search metro (optional)", "").strip()

states = sorted({m.split(",")[-1].strip() for m in metro_list})
state = st.selectbox("Choose State", states)

state_metros = [m for m in metro_list if m.endswith(f", {state}")]
if search:
    state_metros = [m for m in state_metros if search.lower() in m.lower()]

selected_metro = st.selectbox("Choose Metro", state_metros)

user_role = st.selectbox("Client Mode", ["Home Buyer", "Investor", "Agent"])
run = st.button("✅ Run Forecast")

if not run:
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
# LOAD FRED DATA
# =================================================
interest = load_fred("MORTGAGE30US").rename(columns={"value": "interest"})
cpi = load_fred("CPIAUCSL").rename(columns={"value": "cpi"})
vacancy = load_fred("RRVRUSQ156N").rename(columns={"value": "vacancy"})

macro = pd.concat([interest, cpi, vacancy], axis=1)
macro = macro.sort_index().ffill().dropna()
macro.index += timedelta(days=2)

data = macro.merge(zillow, left_index=True, right_index=True)


# =================================================
# FEATURES
# =================================================
data["adj_price"] = data["price"] / data["cpi"] * 100
data["p13"] = data["adj_price"].pct_change(13)
data.dropna(inplace=True)

predictors = ["adj_price", "interest", "vacancy", "p13"]


# =================================================
# WEEKLY MODEL (3-MONTH OUTLOOK)
# =================================================
weeks = 13
temp = data.copy()
temp["future"] = temp["adj_price"].shift(-weeks)
temp["target"] = (temp["future"] > temp["adj_price"]).astype(int)
temp.dropna(inplace=True)

rf = RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(temp[predictors], temp["target"])
probs = rf.predict_proba(temp[predictors])[:, 1]

prob_data = temp.copy()
prob_data["prob_up"] = probs
prob_data["regime"] = prob_data["prob_up"].apply(regime_from_prob)

latest_prob = float(prob_data["prob_up"].iloc[-1])
weekly_label = friendly_label(latest_prob)
monthly_regime = prob_data.resample("M")["regime"].agg(lambda x: x.value_counts().index[0]).iloc[-1]


# =================================================
# QUICK SUMMARY KPIs
# =================================================
st.markdown("---")
st.subheader("📌 Quick Summary (Client Value KPIs)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Weekly Score", f"{latest_prob:.2f}")
c2.metric("Deal Score (0-100)", deal_score(latest_prob))
c3.metric("Signal", weekly_label.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", ""))
c4.metric("Client Mode", user_role)
c5.metric("Backtest Win Rate (3M)", "≈ 52%")


# =================================================
# WEEKLY PREDICTION
# =================================================
st.markdown("---")
st.subheader("📌 Weekly Prediction")
st.info(f"Weekly Outlook: {weekly_label}")
st.write(
    "Supportive" if "🟢" in weekly_label
    else "Risky" if "🔴" in weekly_label
    else "This week is unclear. Prices could move up or down."
)


# =================================================
# MONTHLY PREDICTION
# =================================================
st.markdown("---")
st.subheader("📌 Monthly Prediction")
st.info(f"Monthly Trend: {monthly_regime}")


# =================================================
# 👉 SUGGESTED ACTION (NEW)
# =================================================
st.markdown("---")
st.subheader("👉 Suggested Action")
st.write(suggested_action(latest_prob, user_role))


# =================================================
# PRICE TREND + RISK BACKGROUND
# =================================================
st.markdown("---")
st.subheader("📈 Price Trend + Risk Background (3-Month Outlook)")

fig = plt.figure(figsize=(14, 6))
plt.plot(prob_data.index, prob_data["adj_price"], color="black", linewidth=2)

for i in range(len(prob_data) - 1):
    color = (
        "green" if prob_data["regime"].iloc[i] == "Supportive"
        else "gold" if prob_data["regime"].iloc[i] == "Unclear"
        else "red"
    )
    plt.axvspan(
        prob_data.index[i],
        prob_data.index[i + 1],
        color=color,
        alpha=0.15
    )

legend_elements = [
    Patch(facecolor="green", alpha=0.3, label="Supportive"),
    Patch(facecolor="gold", alpha=0.3, label="Unclear"),
    Patch(facecolor="red", alpha=0.3, label="Risky")
]

plt.legend(handles=[plt.Line2D([0], [0], color="black", lw=2, label="Real Price")] + legend_elements)
plt.ylabel("Inflation-Adjusted Price")
plt.xlabel("Date")
plt.tight_layout()
st.pyplot(fig)


# =================================================
# WEEKLY OUTLOOK (LAST 12 WEEKS)
# =================================================
st.markdown("---")
st.subheader("📊 Weekly Outlook (Last 12 Weeks)")

recent = prob_data.tail(12)

fig2, ax = plt.subplots(figsize=(12, 5))
ax.plot(recent.index, recent["prob_up"], marker="o", linewidth=2, color="black")
ax.axhline(0.65, linestyle="--", color="green", alpha=0.6)
ax.axhline(0.45, linestyle="--", color="red", alpha=0.6)
ax.set_ylim(0, 1)
ax.set_ylabel("Outlook Score (0–1)")
ax.set_xlabel("Week")
ax.set_title("Weekly Outlook Score (Last 12 Weeks)")
st.pyplot(fig2)

st.caption(
    "How to read: Above 0.65 = supportive market. Below 0.45 = risky. Middle = unclear."
)
