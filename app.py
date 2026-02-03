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
st.write("Upload Zillow files → choose a metro → get predictions for multiple time horizons.")


# ----------------------------
# Sidebar Uploads + Inputs
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

TARGET_METRO = st.sidebar.text_input("Enter Target Metro (example: Tampa)", "Tampa")
run_button = st.sidebar.button("✅ Run Forecast")


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
# Helper: simple label for users
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


# ----------------------------
# MAIN RUN
# ----------------------------
if run_button:
    if not price_file or not value_file:
        st.error("❌ Please upload BOTH Zillow CSV files first.")
        st.stop()

    st.success("✅ Files uploaded successfully!")

    # ----------------------------
    # Load Zillow Files
    # ----------------------------
    zillow_price = pd.read_csv(price_file)
    zillow_value = pd.read_csv(value_file)

    # ----------------------------
    # Auto detect metro
    # ----------------------------
    price_matches = zillow_price[
        zillow_price["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
    ]

    value_matches = zillow_value[
        zillow_value["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
    ]

    if price_matches.empty or value_matches.empty:
        st.error(f"❌ Metro '{TARGET_METRO}' not found in Zillow files.")
        st.stop()

    if len(price_matches) > 1:
        st.warning("⚠️ Multiple matches found. Please refine TARGET_METRO.")
        st.write(price_matches["RegionName"].values)
        st.stop()

    metro_name = price_matches["RegionName"].values[0]
    st.info(f"✅ Using Zillow metro: {metro_name}")

    # Zillow data starts from column 5 onward
    price = pd.DataFrame(price_matches.iloc[0, 5:])
    value = pd.DataFrame(value_matches.iloc[0, 5:])

    # ----------------------------
    # Load FRED Data (Extra Macro Indicators)
    # ----------------------------
    try:
        interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
        vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})
        cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})

        # ✅ NEW indicators
        unemployment = load_fred_series("UNRATE").rename(columns={"value": "unemployment"})
        jobs = load_fred_series("PAYEMS").rename(columns={"value": "jobs"})
        permits = load_fred_series("PERMIT").rename(columns={"value": "permits"})
        stress = load_fred_series("STLFSI4").rename(columns={"value": "stress"})

        fed_data = pd.concat(
            [interest, vacancy, cpi, unemployment, jobs, permits, stress],
            axis=1
        )

        # Fill missing values
        fed_data = fed_data.sort_index().ffill().dropna()

        # Small alignment fix
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

    # New macro trend features
    data["unemployment_13w_change"] = data["unemployment"].pct_change(13)
    data["jobs_13w_change"] = data["jobs"].pct_change(13)
    data["permits_13w_change"] = data["permits"].pct_change(13)
    data["stress_13w_change"] = data["stress"].pct_change(13)

    data.dropna(inplace=True)

    # ----------------------------
    # Predictors
    # ----------------------------
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

    # ----------------------------
    # Multi-Horizon Forecast Setup
    # ----------------------------
    horizons = {
        "1 Month Ahead": 4,
        "2 Months Ahead": 8,
        "3 Months Ahead": 13,
        "6 Months Ahead": 26,
        "1 Year Ahead": 52
    }

    START = 260
    STEP = 52

    results = []

    for horizon_name, weeks_ahead in horizons.items():
        temp = data.copy()

        # target: is price higher after X weeks?
        temp["future_price"] = temp["adj_price"].shift(-weeks_ahead)
        temp["target"] = (temp["future_price"] > temp["adj_price"]).astype(int)
        temp.dropna(inplace=True)

        if temp.shape[0] <= START:
            results.append([horizon_name, np.nan, "Not enough data", "-"])
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

    # ----------------------------
    # Show Results Table
    # ----------------------------
    st.subheader("✅ Forecast Results (All Time Horizons)")

    out_df = pd.DataFrame(
        results,
        columns=["Time Horizon", "Prob Price Up", "Outlook", "Suggested Action"]
    )

    st.dataframe(out_df, use_container_width=True)

    # ----------------------------
    # Highlight MAIN message (3 months)
    # ----------------------------
    st.subheader("📌 Simple Summary (Main Outlook)")

    main_row = out_df[out_df["Time Horizon"] == "3 Months Ahead"].iloc[0]

    if "🟢" in main_row["Outlook"]:
        st.success(f"3 Months Ahead Outlook: {main_row['Outlook']} (Prob Up = {main_row['Prob Price Up']})")
    elif "🔴" in main_row["Outlook"]:
        st.error(f"3 Months Ahead Outlook: {main_row['Outlook']} (Prob Up = {main_row['Prob Price Up']})")
    else:
        st.warning(f"3 Months Ahead Outlook: {main_row['Outlook']} (Prob Up = {main_row['Prob Price Up']})")

    st.write("👉 Suggested Action:", main_row["Suggested Action"])

    # ----------------------------
    # Optional Charts (keep your charts)
    # ----------------------------
    st.subheader("📈 Price Trend + Risk Background (Based on 3-Month Model)")

    # Build regime for 3-month horizon just for background coloring
    horizon_weeks = 13
    temp3 = data.copy()
    temp3["future_price"] = temp3["adj_price"].shift(-horizon_weeks)
    temp3["target"] = (temp3["future_price"] > temp3["adj_price"]).astype(int)
    temp3.dropna(inplace=True)

    if temp3.shape[0] > START:
        all_probs_3 = []

        def predict_proba_3(train, test):
            rf = RandomForestClassifier(min_samples_split=10, random_state=1)
            rf.fit(train[predictors], train["target"])
            return rf.predict_proba(test[predictors])[:, 1]

        for i in range(START, temp3.shape[0], STEP):
            train = temp3.iloc[:i]
            test = temp3.iloc[i:i + STEP]
            if len(test) == 0:
                continue
            all_probs_3.append(predict_proba_3(train, test))

        probs3 = np.concatenate(all_probs_3)
        prob_data = temp3.iloc[START:].copy()
        prob_data["prob_up"] = probs3

        def label_zone(p):
            if p >= 0.65:
                return "Bull"
            elif p <= 0.45:
                return "Risk"
            else:
                return "Neutral"

        prob_data["regime"] = prob_data["prob_up"].apply(label_zone)

        fig1 = plt.figure(figsize=(14, 6))

        plt.plot(
            prob_data.index,
            prob_data["adj_price"],
            color="black",
            linewidth=2,
            label="Real Home Price"
        )

        for i in range(len(prob_data) - 1):
            regime = prob_data["regime"].iloc[i]
            if regime == "Bull":
                color = "green"
            elif regime == "Neutral":
                color = "gold"
            else:
                color = "red"

            plt.axvspan(prob_data.index[i], prob_data.index[i + 1], color=color, alpha=0.12)

        plt.title(f"{metro_name} Housing Price Trend (With Risk Zones)", fontsize=14, weight="bold")
        plt.ylabel("Inflation-Adjusted Price")
        plt.xlabel("Date")

        legend_elements = [
            Patch(facecolor="green", alpha=0.25, label="Supportive"),
            Patch(facecolor="gold", alpha=0.25, label="Unclear"),
            Patch(facecolor="red", alpha=0.25, label="Risky"),
        ]

        plt.legend(
            handles=[plt.Line2D([0], [0], color="black", lw=2, label="Real Price")] + legend_elements,
            loc="upper left"
        )

        plt.tight_layout()
        st.pyplot(fig1)

    else:
        st.warning("Not enough data to show charts for the 3-month model yet.")
