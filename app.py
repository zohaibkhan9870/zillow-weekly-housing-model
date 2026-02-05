import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="US Real Estate Price Outlook",
    layout="wide"
)

st.title("🏡 US Real Estate Price Outlook Dashboard")
st.write("Zillow + FRED + Machine Learning → Simple, explainable housing signals")

st.markdown("---")


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def friendly_label(p):
    if p >= 0.65:
        return "🟢 Good time"
    elif p <= 0.45:
        return "🔴 Risky"
    return "🟡 Unclear"


def regime_from_prob(p):
    if p >= 0.65:
        return "Bull"
    elif p <= 0.45:
        return "Risk"
    return "Neutral"


def deal_score(p):
    return int(np.clip(round(p * 100), 0, 100))


def role_action(label, role):
    if role == "Buyer":
        return "Buy / negotiate" if "🟢" in label else "Wait / be selective"
    if role == "Investor":
        return "Deploy capital" if "🟢" in label else "Capital preservation"
    return "Guide clients carefully"


def expected_return(prob, weeks):
    factor = np.sqrt(weeks / 13)
    exp = (prob - 0.5) * 8 * factor
    band = 4 * factor
    return exp, exp - band, exp + band


# -------------------------------------------------
# FRED loader
# -------------------------------------------------
def load_fred(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": st.secrets["FRED_API_KEY"],
        "file_type": "json"
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")[["value"]]


# -------------------------------------------------
# Upload
# -------------------------------------------------
st.subheader("📤 Upload Zillow Files")

col1, col2 = st.columns(2)
with col1:
    price_file = st.file_uploader("Weekly Median Sale Price", type="csv")
with col2:
    value_file = st.file_uploader("Monthly ZHVI", type="csv")

if not price_file or not value_file:
    st.info("Upload both Zillow files to continue.")
    st.stop()

price_df = pd.read_csv(price_file)
value_df = pd.read_csv(value_file)


# -------------------------------------------------
# Location selection
# -------------------------------------------------
st.subheader("🌍 Select Location")

metros = sorted(
    set(price_df["RegionName"]).intersection(
        set(value_df["RegionName"])
    )
)

selected_metro = st.selectbox("Choose Metro", metros)

role = st.selectbox("Client Type", ["Buyer", "Investor", "Agent"])
run = st.button("✅ Run Forecast")

if not run:
    st.stop()


# -------------------------------------------------
# Data prep
# -------------------------------------------------
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


# -------------------------------------------------
# FRED
# -------------------------------------------------
interest = load_fred("MORTGAGE30US").rename(columns={"value": "interest"})
cpi = load_fred("CPIAUCSL").rename(columns={"value": "cpi"})
vacancy = load_fred("RRVRUSQ156N").rename(columns={"value": "vacancy"})

macro = pd.concat([interest, cpi, vacancy], axis=1)
macro = macro.sort_index().ffill().dropna()
macro.index += timedelta(days=2)

data = macro.merge(zillow, left_index=True, right_index=True)


# -------------------------------------------------
# Features
# -------------------------------------------------
data["adj_price"] = data["price"] / data["cpi"] * 100
data["adj_value"] = data["value"] / data["cpi"] * 100
data["p13"] = data["adj_price"].pct_change(13)
data["v52"] = data["adj_value"].pct_change(52)
data.dropna(inplace=True)

predictors = ["adj_price", "adj_value", "interest", "vacancy", "p13", "v52"]


# -------------------------------------------------
# Forecast horizons
# -------------------------------------------------
horizons = {
    "1 Month": 4,
    "3 Months": 13,
    "6 Months": 26,
    "1 Year": 52
}

results = []

for name, weeks in horizons.items():
    temp = data.copy()
    temp["future"] = temp["adj_price"].shift(-weeks)
    temp["target"] = (temp["future"] > temp["adj_price"]).astype(int)
    temp.dropna(inplace=True)

    rf = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(temp[predictors], temp["target"])
    prob = rf.predict_proba(temp[predictors].tail(1))[:, 1][0]

    label = friendly_label(prob)
    action = role_action(label, role)
    exp, lo, hi = expected_return(prob, weeks)

    results.append([
        name,
        f"{prob*100:.0f}%",
        label,
        deal_score(prob),
        action,
        f"{exp:+.1f}%",
        f"[{lo:+.1f}%, {hi:+.1f}%]"
    ])

out = pd.DataFrame(
    results,
    columns=[
        "Horizon",
        "Up Probability",
        "Outlook",
        "Deal Score",
        "Suggested Action",
        "Expected Change",
        "Expected Range"
    ]
)

# -------------------------------------------------
# Output
# -------------------------------------------------
st.markdown("---")
st.subheader("📊 Forecast Results")
st.dataframe(out, use_container_width=True)

latest_prob = float(out.iloc[1]["Up Probability"].replace("%", "")) / 100
st.metric("Weekly Signal", friendly_label(latest_prob))
st.metric("Deal Score", deal_score(latest_prob))


# -------------------------------------------------
# Download
# -------------------------------------------------
csv = out.to_csv(index=False).encode()
st.download_button("⬇️ Download CSV", csv, f"{selected_metro}_forecast.csv")

