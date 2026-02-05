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

    st.markdown("#### 🏠 Home Values Section")
    st.markdown("Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")

    st.markdown("#### 💰 Sales Section")
    st.markdown("Metro_median_sale_price_uc_sfrcondo_sm_week.csv")

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


def safe_median_date_diff_days(parsed_dates: pd.DatetimeIndex):
    if parsed_dates is None or len(parsed_dates) < 2:
        return None

    sorted_dates = pd.Series(parsed_dates).sort_values().reset_index(drop=True)
    diffs = sorted_dates.diff().dropna()

    if diffs.empty:
        return None

    median_days = diffs.dt.total_seconds().median() / (60 * 60 * 24)
    return median_days


def validate_zillow_csv(uploaded_file, expected_type):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return False, "❌ Could not read this CSV file.", None

    if "RegionName" not in df.columns:
        return False, "❌ Wrong file content: missing required column 'RegionName'.", None

    if df.shape[1] < 10:
        return False, "❌ Wrong file content: not enough columns.", None

    date_cols = list(df.columns[5:])
    parsed_dates = pd.to_datetime(date_cols, errors="coerce").dropna()

    if len(parsed_dates) < 10:
        return False, "❌ Could not detect valid date columns.", None

    median_days = safe_median_date_diff_days(parsed_dates)
    if median_days is None:
        return False, "❌ Could not infer frequency.", None

    if expected_type == "weekly_price" and median_days > 15:
        return False, "❌ Not weekly price data.", None

    if expected_type == "monthly_value" and median_days < 20:
        return False, "❌ Not monthly ZHVI data.", None

    return True, "✅ Correct file uploaded.", df


# ----------------------------
# Upload Section
# ----------------------------
st.subheader("📤 Upload Zillow Files")

price_file = st.file_uploader("Upload Weekly Median Sale Price CSV", type=["csv"])
value_file = st.file_uploader("Upload ZHVI Home Value Index CSV", type=["csv"])

if not price_file or not value_file:
    st.stop()

ok1, _, zillow_price = validate_zillow_csv(price_file, "weekly_price")
ok2, _, zillow_value = validate_zillow_csv(value_file, "monthly_value")

if not ok1 or not ok2:
    st.error("❌ Invalid Zillow files uploaded.")
    st.stop()


# ----------------------------
# FRED API Loader
# ----------------------------
def load_fred_series(series_id):
    api_key = st.secrets["FRED_API_KEY"]
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json"
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise Exception(f"FRED API failed for {series_id}")

    data = r.json()["observations"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.set_index("date", inplace=True)
    return df[["value"]]


# ----------------------------
# Helper functions
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


# ----------------------------
# Location selection
# ----------------------------
metro_list = sorted(
    set(zillow_price["RegionName"].dropna()).intersection(
        set(zillow_value["RegionName"].dropna())
    )
)

selected_metro = st.selectbox("Choose Metro", metro_list)
run_button = st.button("✅ Run Forecast")


# ----------------------------
# Run Model
# ----------------------------
if run_button:
    price_matches = zillow_price[zillow_price["RegionName"] == selected_metro]
    value_matches = zillow_value[zillow_value["RegionName"] == selected_metro]

    price = pd.DataFrame(price_matches.iloc[0, 5:])
    value = pd.DataFrame(value_matches.iloc[0, 5:])

    price.index = pd.to_datetime(price.index)
    value.index = pd.to_datetime(value.index)

    price.columns = ["price"]
    value.columns = ["value"]

    interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
    cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})
    vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})

    fed_data = pd.concat([interest, cpi, vacancy], axis=1).ffill().dropna()
    fed_data.index = fed_data.index + timedelta(days=2)

    price["month"] = price.index.to_period("M")
    value["month"] = value.index.to_period("M")

    price_data = price.merge(value, on="month").drop(columns=["month"])
    data = fed_data.merge(price_data, left_index=True, right_index=True)

    data["adj_price"] = data["price"] / data["cpi"] * 100
    data["price_13w_change"] = data["adj_price"].pct_change(13)
    data.dropna(inplace=True)

    predictors = ["adj_price", "interest", "vacancy", "price_13w_change"]

    START = 104
    STEP = 26

    # -------- Weekly / Monthly signals (FIXED SECTION) --------
    temp3 = data.copy()
    temp3["future_price"] = temp3["adj_price"].shift(-13)
    temp3["target"] = (temp3["future_price"] > temp3["adj_price"]).astype(int)
    temp3.dropna(inplace=True)

    def predict_proba_3(train, test):
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(train[predictors], train["target"])
        return rf.predict_proba(test[predictors])[:, 1]

    all_probs_3 = []
    for i in range(START, temp3.shape[0], STEP):
        train = temp3.iloc[:i]
        test = temp3.iloc[i:i + STEP]
        if len(test) == 0:
            continue
        all_probs_3.append(predict_proba_3(train, test))

    # ✅ FIX: prevent empty concatenate crash
    if len(all_probs_3) == 0:
        st.error(
            "❌ Not enough historical data to compute weekly signals for this metro. "
            "Try another metro or state."
        )
        st.stop()

    probs3 = np.concatenate(all_probs_3)

    prob_data = temp3.iloc[START:].copy()
    prob_data["prob_up"] = probs3
    prob_data["regime"] = prob_data["prob_up"].apply(regime_from_prob)

    latest_week_prob = float(prob_data["prob_up"].tail(1).values[0])
    weekly_label = friendly_label(latest_week_prob)

    st.subheader("📌 Weekly Outlook")
    st.metric("Probability Price Goes Up", f"{latest_week_prob:.2f}")
    st.success(weekly_label)
