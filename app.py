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
st.write("Upload Zillow files → choose State → choose Metro → run forecast for multiple horizons.")


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
# ✅ STEP 1: If files uploaded → show dropdowns FIRST
# ----------------------------
selected_metro = None
selected_state = None

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

    # ----------------------------
    # ✅ State + Metro Dropdowns
    # ----------------------------
    st.sidebar.header("🌎 Select Location")

    states = sorted(list(set([m.split(",")[-1].strip() for m in metro_list if "," in m])))

    selected_state = st.sidebar.selectbox("Choose State", states)

    filtered_metros = [m for m in metro_list if m.endswith(f", {selected_state}")]

    selected_metro = st.sidebar.selectbox("Choose Metro", filtered_metros)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("✅ Run Forecast")

else:
    st.info("⬅️ Please upload both Zillow CSV files to continue.")
    st.stop()


# ----------------------------
# ✅ STEP 2: Only run model when button pressed
# ----------------------------
if run_button:
    with st.spinner(f"⏳ Processing... Running forecast for {selected_metro}"):
        progress = st.progress(0)
        status = st.empty()

        # Step 1
        status.info("Step 1/5: Loading selected metro data...")
        progress.progress(10)

        price_matches = zillow_price[zillow_price["RegionName"] == selected_metro]
        value_matches = zillow_value[zillow_value["RegionName"] == selected_metro]

        if price_matches.empty or value_matches.empty:
            st.error("❌ Selected metro not found in both files.")
            st.stop()

        price = pd.DataFrame(price_matches.iloc[0, 5:])
        value = pd.DataFrame(value_matches.iloc[0, 5:])

        # Step 2
        status.info("Step 2/5: Fetching macroeconomic data (FRED)...")
        progress.progress(30)

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

        # Step 3
        status.info("Step 3/5: Preparing Zillow price/value data...")
        progress.progress(50)

        price.index = pd.to_datetime(price.index)
        value.index = pd.to_datetime(value.index)

        price["month"] = price.index.to_period("M")
        value["month"] = value.index.to_period("M")

        price_data = price.merge(value, on="month")
        price_data.index = price.index
        price_data.drop(columns=["month"], inplace=True)
        price_data.columns = ["price", "value"]

        data = fed_data.merge(price_data, left_index=True, right_index=True)

        # Step 4
        status.info("Step 4/5: Building features + training models...")
        progress.progress(70)

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

        START = 104
        STEP = 26

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

        # Step 5
        status.success("✅ Done! Forecast is ready.")
        progress.progress(100)

    # ----------------------------
    # OUTPUT
    # ----------------------------
    st.subheader("✅ Forecast Results (All Time Horizons)")
    st.dataframe(out_df, use_container_width=True)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_bytes,
        file_name=f"{selected_metro.replace(',', '').replace(' ', '_')}_forecast_results.csv",
        mime="text/csv"
    )
