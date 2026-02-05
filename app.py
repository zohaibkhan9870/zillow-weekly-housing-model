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

# PDF export
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io


# ----------------------------
# Streamlit Setup
# ----------------------------
st.set_page_config(page_title="US Real Estate Price Outlook Dashboard", layout="wide")

st.title("🏡 US Real Estate Price Outlook Dashboard")
st.write("Upload Zillow files → select State & Metro → get price outlook for multiple time horizons.")


# ----------------------------
# Reset Button
# ----------------------------
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
    st.info("✅ Please select an option to continue.")
    st.stop()


# ----------------------------
# Download instructions
# ----------------------------
if file_status == "⬇️ No, I need to download them":
    st.markdown("### ✅ Step 1: Open Zillow Research Page")
    st.link_button("🌐 Open Zillow Data Page", "https://www.zillow.com/research/data/")

    st.markdown("### ✅ Step 2: Download these 2 CSV files")

    st.markdown("#### 🏠 Home Values")
    st.markdown("**Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv**")

    st.markdown("#### 💰 Sales")
    st.markdown("**Metro_median_sale_price_uc_sfrcondo_sm_week.csv**")

    confirm_download = st.checkbox("✅ I downloaded both Zillow CSV files")
    if not confirm_download:
        st.stop()

if file_status == "✅ Yes, I already have them":
    confirm_download = True


# ----------------------------
# File Validation
# ----------------------------
EXPECTED_PRICE_FILENAME = "Metro_median_sale_price_uc_sfrcondo_sm_week.csv"
EXPECTED_VALUE_FILENAME = "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


def safe_median_date_diff_days(parsed_dates):
    if parsed_dates is None or len(parsed_dates) < 2:
        return None
    sorted_dates = pd.Series(parsed_dates).sort_values()
    diffs = sorted_dates.diff().dropna()
    if diffs.empty:
        return None
    return diffs.dt.total_seconds().median() / 86400


def validate_zillow_csv(uploaded_file, expected_type):
    df = pd.read_csv(uploaded_file)
    if "RegionName" not in df.columns:
        return False, "Missing RegionName column", None

    date_cols = df.columns[5:]
    parsed_dates = pd.to_datetime(date_cols, errors="coerce").dropna()
    median_days = safe_median_date_diff_days(parsed_dates)

    if expected_type == "weekly_price" and median_days > 15:
        return False, "Not weekly data", None
    if expected_type == "monthly_value" and median_days < 20:
        return False, "Not monthly data", None

    return True, "OK", df


# ----------------------------
# Upload
# ----------------------------
st.subheader("📤 Upload Zillow Files")

price_file = st.file_uploader("Weekly Sale Price CSV", type=["csv"])
value_file = st.file_uploader("Monthly ZHVI CSV", type=["csv"])

if not price_file or not value_file:
    st.stop()

ok1, _, zillow_price = validate_zillow_csv(price_file, "weekly_price")
ok2, _, zillow_value = validate_zillow_csv(value_file, "monthly_value")

if not ok1 or not ok2:
    st.error("Invalid files uploaded")
    st.stop()


# ----------------------------
# FRED Loader
# ----------------------------
def load_fred_series(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": st.secrets["FRED_API_KEY"],
        "file_type": "json"
    }
    r = requests.get(url, params=params)
    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")[["value"]]


# ----------------------------
# Helper functions
# ----------------------------
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


# ----------------------------
# Location selection
# ----------------------------
metro_list = sorted(
    set(zillow_price["RegionName"]).intersection(set(zillow_value["RegionName"]))
)

selected_metro = st.selectbox("Choose Metro", metro_list)

run_button = st.button("✅ Run Forecast")


# ----------------------------
# Run Model
# ----------------------------
if run_button:
    price = pd.DataFrame(
        zillow_price[zillow_price["RegionName"] == selected_metro].iloc[0, 5:]
    )
    price.index = pd.to_datetime(price.index)
    price.columns = ["price"]

    value = pd.DataFrame(
        zillow_value[zillow_value["RegionName"] == selected_metro].iloc[0, 5:]
    )
    value.index = pd.to_datetime(value.index)
    value.columns = ["value"]

    interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
    cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})
    vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})

    fed = pd.concat([interest, cpi, vacancy], axis=1).ffill().dropna()
    fed.index += timedelta(days=2)

    price["month"] = price.index.to_period("M")
    value["month"] = value.index.to_period("M")
    pv = price.merge(value, on="month").drop(columns="month")

    data = fed.merge(pv, left_index=True, right_index=True)

    data["adj_price"] = data["price"] / data["cpi"] * 100
    data["price_13w"] = data["adj_price"].pct_change(13)
    data.dropna(inplace=True)

    predictors = ["adj_price", "interest", "vacancy", "price_13w"]

    START, STEP = 104, 26

    temp = data.copy()
    temp["future"] = temp["adj_price"].shift(-13)
    temp["target"] = (temp["future"] > temp["adj_price"]).astype(int)
    temp.dropna(inplace=True)

    all_probs_3 = []

    for i in range(START, temp.shape[0], STEP):
        train = temp.iloc[:i]
        test = temp.iloc[i:i + STEP]
        if len(test) == 0:
            continue
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(train[predictors], train["target"])
        all_probs_3.append(rf.predict_proba(test[predictors])[:, 1])

    # ✅ FIX — THIS IS THE CRASH FIX
    if len(all_probs_3) == 0:
        st.error(
            "❌ Not enough historical data to compute weekly signals for this metro. "
            "Please try another metro."
        )
        st.stop()

    probs3 = np.concatenate(all_probs_3)

    prob_data = temp.iloc[START:].copy()
    prob_data["prob_up"] = probs3
    prob_data["regime"] = prob_data["prob_up"].apply(regime_from_prob)

    latest_prob = prob_data["prob_up"].iloc[-1]
    label = friendly_label(latest_prob)

    st.subheader("📌 Weekly Outlook")
    st.metric("Up Probability", f"{latest_prob:.2f}")
    st.success(label)
