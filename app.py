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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io


# ----------------------------
# Streamlit Setup
# ----------------------------
st.set_page_config(page_title="US Real Estate Price Outlook Dashboard", layout="wide")

st.title("🏡 US Real Estate Price Outlook Dashboard")
st.write("Upload Zillow files → select State & Metro → get price outlook for multiple time horizons.")

st.markdown("---")
if st.button("🔄 Reset / Clear All"):
    st.session_state.clear()
    st.rerun()


# ----------------------------
# Zillow CSV Setup
# ----------------------------
st.subheader("📂 Zillow CSV Setup")

file_status = st.selectbox(
    "Do you already have the Zillow CSV files downloaded?",
    ["Select an option...", "✅ Yes, I already have them", "⬇️ No, I need to download them"],
    index=0
)

if file_status == "Select an option...":
    st.stop()

if file_status == "⬇️ No, I need to download them":
    st.link_button("🌐 Open Zillow Data Page", "https://www.zillow.com/research/data/")
    confirm_download = st.checkbox("✅ I downloaded both Zillow CSV files")
    if not confirm_download:
        st.stop()
else:
    confirm_download = True


# ----------------------------
# Upload
# ----------------------------
st.subheader("📤 Upload Zillow Files")

price_file = st.file_uploader("Upload Weekly Median Sale Price CSV", type=["csv"])
value_file = st.file_uploader("Upload ZHVI CSV", type=["csv"])

if not price_file or not value_file:
    st.stop()

zillow_price = pd.read_csv(price_file)
zillow_value = pd.read_csv(value_file)
# ----------------------------
# FRED Loader
# ----------------------------
def load_fred_series(series_id):
    api_key = st.secrets["FRED_API_KEY"]

    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json"
        },
        timeout=30
    )

    df = pd.DataFrame(r.json()["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.set_index("date", inplace=True)

    return df[["value"]]


# ----------------------------
# Helper Functions
# ----------------------------
def friendly_label(prob_up):
    if prob_up >= 0.65:
        return "🟢 Good time"
    elif prob_up <= 0.45:
        return "🔴 Risky"
    else:
        return "🟡 Unclear"


def regime_from_prob(p):
    if p >= 0.65:
        return "Bull"
    elif p <= 0.45:
        return "Risk"
    else:
        return "Neutral"


def deal_score(prob_up):
    return int(np.clip(round(prob_up * 100), 0, 100))


def role_based_action(label, user_role):
    if "🟢" in label:
        return "Supportive market"
    elif "🔴" in label:
        return "Risky market"
    return "Mixed market"
# ----------------------------
# Location Selection
# ----------------------------
st.subheader("🌎 Select Location")

metro_list = sorted(
    set(zillow_price["RegionName"].dropna().unique()).intersection(
        set(zillow_value["RegionName"].dropna().unique())
    )
)

selected_metro = st.selectbox("Choose Metro", metro_list)

user_role = st.selectbox(
    "Client Type",
    ["🏠 Home Buyer", "💼 Investor", "🧑‍💼 Agent"]
)

run_button = st.button("✅ Run Forecast")
# ----------------------------
# Run Model
# ----------------------------
if run_button:

    price = pd.DataFrame(
        zillow_price[zillow_price["RegionName"] == selected_metro].iloc[0, 5:]
    )

    value = pd.DataFrame(
        zillow_value[zillow_value["RegionName"] == selected_metro].iloc[0, 5:]
    )

    interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
    cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})
    vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})

    fed_data = pd.concat([interest, cpi, vacancy], axis=1).ffill().dropna()

    price.index = pd.to_datetime(price.index)
    value.index = pd.to_datetime(value.index)

    price["month"] = price.index.to_period("M")
    value["month"] = value.index.to_period("M")

    pv = price.merge(value, on="month")
    pv.columns = ["price", "month", "value"]
    pv.set_index(price.index, inplace=True)

    data = fed_data.merge(pv, left_index=True, right_index=True)

    data["adj_price"] = data["price"] / data["cpi"] * 100
    data["adj_value"] = data["value"] / data["cpi"] * 100

    data["price_13w_change"] = data["adj_price"].pct_change(13)
    data["value_52w_change"] = data["adj_value"].pct_change(52)

    data.dropna(inplace=True)

    predictors = [
        "adj_price",
        "adj_value",
        "interest",
        "vacancy",
        "price_13w_change",
        "value_52w_change",
    ]

    START = 104
    STEP = 26

    # -------- STEP 5 FIXED --------
    horizon_weeks = 13
    temp3 = data.copy()

    temp3["future_price"] = temp3["adj_price"].shift(-horizon_weeks)
    temp3["target"] = (
        temp3["future_price"] > temp3["adj_price"]
    ).astype(int)

    temp3.dropna(inplace=True)

    def predict_proba_3(train, test):
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(train[predictors], train["target"])
        return rf.predict_proba(test[predictors])[:, 1]

    all_probs_3 = []

    for i in range(START, temp3.shape[0], STEP):
        train = temp3.iloc[:i]
        test = temp3.iloc[i:i + STEP]

        if len(test) == 0 or len(train) < 20:
            continue

        try:
            p = predict_proba_3(train, test)
            if len(p) > 0:
                all_probs_3.append(p)
        except:
            continue

    if len(all_probs_3) == 0:
        st.error("Not enough historical data.")
        st.stop()

    probs3 = np.concatenate(all_probs_3)

    prob_data = temp3.iloc[START:].copy()

    min_len = min(len(prob_data), len(probs3))
    prob_data = prob_data.iloc[:min_len]

    prob_data["prob_up"] = probs3[:min_len]
    prob_data["regime"] = prob_data["prob_up"].apply(regime_from_prob)

    # Monthly signals
    monthly = prob_data.copy()
    monthly["month"] = monthly.index.to_period("M")

    monthly_signal = monthly.groupby("month").agg({
        "prob_up": "mean",
        "regime": lambda x: x.value_counts().index[0]
    })

    latest_prob = float(prob_data["prob_up"].tail(1))

    st.metric("Deal Score", deal_score(latest_prob))
    st.write("Weekly Outlook:", friendly_label(latest_prob))
    # ----------------------------
    # Alerts
    # ----------------------------
    st.markdown("---")
    st.subheader("🔔 Alert Status")

    if latest_prob >= 0.65:
        st.success("✅ Market looks supportive")
    elif latest_prob <= 0.45:
        st.error("⚠️ Market looks risky")
    else:
        st.info("Market is mixed")


    # ----------------------------
    # Simple Metro Ranking
    # ----------------------------
    st.markdown("---")
    st.subheader("🏆 Metro Ranking (Simple Proxy)")

    ranking_rows = []

    for m in metro_list[:15]:
        pm = zillow_price[zillow_price["RegionName"] == m]

        if len(pm) == 0:
            continue

        p = pd.DataFrame(pm.iloc[0, 5:])
        p.index = pd.to_datetime(p.index)

        pct = p.iloc[:, 0].pct_change(13).dropna()

        if pct.empty:
            continue

        proxy = 0.5 + np.clip(float(pct.tail(1)), -0.1, 0.1) * 2

        ranking_rows.append([
            m,
            f"{proxy*100:.0f}%",
            deal_score(proxy)
        ])

    if ranking_rows:
        rank_df = pd.DataFrame(
            ranking_rows,
            columns=["Metro", "Up Chance", "Deal Score"]
        )

        rank_df = rank_df.sort_values(
            "Deal Score",
            ascending=False
        ).head(10)

        st.dataframe(rank_df, use_container_width=True)


    # ----------------------------
    # Feature Importance
    # ----------------------------
    st.markdown("---")
    st.subheader("🧠 Feature Importance")

    rf = RandomForestClassifier()
    rf.fit(temp3[predictors], temp3["target"])

    fi = pd.Series(
        rf.feature_importances_,
        index=predictors
    ).sort_values()

    st.bar_chart(fi)


    # ----------------------------
    # Download CSV
    # ----------------------------
    st.markdown("---")

    out_df = prob_data[["prob_up"]].copy()

    csv = out_df.to_csv().encode("utf-8")

    st.download_button(
        "⬇️ Download Forecast CSV",
        csv,
        file_name="forecast.csv"
    )
