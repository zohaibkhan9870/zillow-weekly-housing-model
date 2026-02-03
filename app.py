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


# ----------------------------
# Streamlit Setup
# ----------------------------
st.set_page_config(page_title="Zillow Housing Forecast (Multi-Horizon)", layout="wide")

st.title("🏡 Zillow Housing Forecast (Metro-Based)")
st.write("Upload Zillow files → select metro → run forecast for multiple horizons.")


# ----------------------------
# Sidebar Uploads
# ----------------------------
st.sidebar.header("📂 Upload Zillow Files")

price_file = st.sidebar.file_uploader(
    "Upload Weekly Median Sale Price CSV",
    type=["csv"]
)

value_file = st.sidebar.file_uploader(
    "Upload ZHVI Home Value Index CSV",
    type=["csv"]
)


# ----------------------------
# ✅ FRED API Loader (JSON Method)
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
        raise Exception(f"FRED API request failed for {series_id}. Status: {r.status_code}")

    data = r.json()
    if "observations" not in data:
        raise Exception(f"Invalid API response for {series_id}")

    df = pd.DataFrame(data["observations"])
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


def simple_action(label):
    if "🟢" in label:
        return "Buying/investing looks safer than usual."
    elif "🔴" in label:
        return "Be careful — risk is high."
    else:
        return "Wait and monitor the market."


def regime_from_prob(p):
    if p >= 0.65:
        return "Bull"
    elif p <= 0.45:
        return "Risk"
    else:
        return "Neutral"


# ----------------------------
# ✅ STEP 1: If files uploaded → show dropdown FIRST
# ----------------------------
selected_metro = None

if price_file and value_file:
    zillow_price = pd.read_csv(price_file)
    zillow_value = pd.read_csv(value_file)

    if "RegionName" not in zillow_price.columns or "RegionName" not in zillow_value.columns:
        st.error("❌ Your Zillow file is missing the 'RegionName' column.")
        st.stop()

    metro_list = sorted(
        set(zillow_price["RegionName"].dropna().unique()).intersection(
            set(zillow_value["RegionName"].dropna().unique())
        )
    )

    if len(metro_list) == 0:
        st.error("❌ Could not find matching metros between both files.")
        st.stop()

    st.sidebar.header("🏙️ Select Metro")
    selected_metro = st.sidebar.selectbox("Choose Metro", metro_list)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("✅ Run Forecast")

else:
    st.info("⬅️ Please upload both Zillow CSV files to continue.")
    st.stop()


# ----------------------------
# ✅ STEP 2: Only run model when button pressed
# ----------------------------
if run_button:
    st.success(f"✅ Running forecast for: {selected_metro}")

    # Extract selected metro row
    price_matches = zillow_price[zillow_price["RegionName"] == selected_metro]
    value_matches = zillow_value[zillow_value["RegionName"] == selected_metro]

    if price_matches.empty or value_matches.empty:
        st.error("❌ Selected metro not found in both files.")
        st.stop()

    price = pd.DataFrame(price_matches.iloc[0, 5:])
    value = pd.DataFrame(value_matches.iloc[0, 5:])

    # ----------------------------
    # Load FRED Data
    # ----------------------------
    try:
        interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
        vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})
        cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})

        unemployment = load_fred_series("UNRATE").rename(columns={"value": "unemployment"})
        jobs = load_fred_series("PAYEMS").rename(columns={"value": "jobs"})
        permits = load_fred_series("PERMIT").rename(columns={"value": "permits"})
        stress = load_fred_series("STLFSI4").rename(columns={"value": "stress"})

        fed_data = pd.concat(
            [interest, vacancy, cpi, unemployment, jobs, permits, stress],
            axis=1
        )

        fed_data = fed_data.sort_index().ffill().dropna()
        fed_data.index = fed_data.index + timedelta(days=2)

    except Exception as e:
        st.error("❌ Failed to fetch FRED macro data using API.")
        st.write("Error details:", str(e))
        st.stop()

    # ----------------------------
    # Prepare Zillow price & value
    # ----------------------------
    price.index = pd.to_datetime(price.index)
    value.index = pd.to_datetime(value.index)

    price["month"] = price.index.to_period("M")
    value["month"] = value.index.to_period("M")

    price_data = price.merge(value, on="month")
    price_data.index = price.index
    price_data.drop(columns=["month"], inplace=True)
    price_data.columns = ["price", "value"]

    # ----------------------------
    # Merge FRED + Zillow
    # ----------------------------
    data = fed_data.merge(price_data, left_index=True, right_index=True)

    # ----------------------------
    # Feature Engineering
    # ----------------------------
    data["adj_price"] = data["price"] / data["cpi"] * 100
    data["adj_value"] = data["value"] / data["cpi"] * 100

    data["price_13w_change"] = data["adj_price"].pct_change(13)
    data["value_52w_change"] = data["adj_value"].pct_change(52)

    data["unemployment_13w_change"] = data["unemployment"].pct_change(13)
    data["jobs_13w_change"] = data["jobs"].pct_change(13)
    data["permits_13w_change"] = data["permits"].pct_change(13)
    data["stress_13w_change"] = data["stress"].pct_change(13)

    data.dropna(inplace=True)

    predictors = [
        "adj_price",
        "adj_value",
        "interest",
        "vacancy",
        "price_13w_change",
        "value_52w_change",
        "unemployment",
        "jobs",
        "permits",
        "stress",
        "unemployment_13w_change",
        "jobs_13w_change",
        "permits_13w_change",
        "stress_13w_change"
    ]

    # ✅ smaller start so more metros work
    START = 104  # ~2 years
    STEP = 26    # ~6 months

    horizons = {
        "1 Month Ahead": 4,
        "2 Months Ahead": 8,
        "3 Months Ahead": 13,
        "6 Months Ahead": 26,
        "1 Year Ahead": 52
    }

    results = []

    for horizon_name, weeks_ahead in horizons.items():
        temp = data.copy()
        temp["future_price"] = temp["adj_price"].shift(-weeks_ahead)
        temp["target"] = (temp["future_price"] > temp["adj_price"]).astype(int)
        temp.dropna(inplace=True)

        if temp.shape[0] <= START:
            results.append([horizon_name, None, "Not enough data", "-"])
            continue

        def predict_proba(train, test):
            rf = RandomForestClassifier(min_samples_split=10, random_state=1)
            rf.fit(train[predictors], train["target"])
            return rf.predict_proba(test[predictors])[:, 1]

        all_probs = []
        for i in range(START, temp.shape[0], STEP):
            train = temp.iloc[:i]
            test = temp.iloc[i:i + STEP]
            if len(test) == 0:
                continue
            all_probs.append(predict_proba(train, test))

        probs = np.concatenate(all_probs)
        pred_df = temp.iloc[START:].copy()
        pred_df["prob_up"] = probs

        latest_prob = float(pred_df["prob_up"].tail(1).values[0])
        label = friendly_label(latest_prob)
        action = simple_action(label)

        results.append([horizon_name, round(latest_prob, 2), label, action])

    out_df = pd.DataFrame(results, columns=["Time Horizon", "Prob Price Up", "Outlook", "Suggested Action"])

    st.subheader("✅ Forecast Results (All Time Horizons)")
    st.dataframe(out_df, use_container_width=True)
