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
st.set_page_config(
    page_title="Weekly Zillow Housing Model",
    layout="wide"
)

st.title("🏡 Weekly Zillow Housing Model (Metro-Based)")
st.write("Upload the two Zillow CSV files and run the model metro-wise.")


# ----------------------------
# Sidebar Inputs
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

TARGET_METRO = st.sidebar.text_input(
    "Enter Target Metro (example: Tampa)",
    "Tampa"
)

run_button = st.sidebar.button("✅ Run Weekly Model")


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

    data = r.json()

    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df.set_index("date", inplace=True)
    return df[["value"]]


# ----------------------------
# Main Run
# ----------------------------
if run_button:

    if not price_file or not value_file:
        st.error("❌ Please upload BOTH Zillow CSV files.")
        st.stop()

    st.success("✅ Files uploaded successfully")

    # ----------------------------
    # Load Zillow Data
    # ----------------------------
    zillow_price = pd.read_csv(price_file)
    zillow_value = pd.read_csv(value_file)

    price_match = zillow_price[
        zillow_price["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
    ]

    value_match = zillow_value[
        zillow_value["RegionName"].str.contains(TARGET_METRO, case=False, na=False)
    ]

    if price_match.empty or value_match.empty:
        st.error(f"❌ Metro '{TARGET_METRO}' not found.")
        st.stop()

    if len(price_match) > 1:
        st.warning("⚠️ Multiple metro matches found:")
        st.write(price_match["RegionName"].values)
        st.stop()

    metro_name = price_match["RegionName"].values[0]
    st.info(f"✅ Using Zillow Metro: {metro_name}")

    price = pd.DataFrame(price_match.iloc[0, 5:])
    value = pd.DataFrame(value_match.iloc[0, 5:])

    price.index = pd.to_datetime(price.index)
    value.index = pd.to_datetime(value.index)

    price.columns = ["price"]
    value.columns = ["value"]

    # ----------------------------
    # Load FRED Data
    # ----------------------------
    try:
        interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
        vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})
        cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})

        fed = pd.concat([interest, vacancy, cpi], axis=1)
        fed = fed.sort_index().ffill().dropna()

        fed.index = fed.index + timedelta(days=2)

    except Exception as e:
        st.error("❌ Failed to load FRED data")
        st.write(str(e))
        st.stop()

    # ----------------------------
    # Merge Zillow Data
    # ----------------------------
    price["month"] = price.index.to_period("M")
    value["month"] = value.index.to_period("M")

    zillow = price.merge(value, on="month")
    zillow.index = price.index
    zillow.drop(columns="month", inplace=True)

    # ----------------------------
    # Merge All Data
    # ----------------------------
    data = fed.merge(zillow, left_index=True, right_index=True)

    # ----------------------------
    # Feature Engineering
    # ----------------------------
    data["adj_price"] = data["price"] / data["cpi"] * 100
    data["adj_value"] = data["value"] / data["cpi"] * 100

    data["next_q"] = data["adj_price"].shift(-13)
    data["target"] = (data["next_q"] > data["adj_price"]).astype(int)

    data["price_13w_change"] = data["adj_price"].pct_change(13)
    data["value_52w_change"] = data["adj_value"].pct_change(52)

    data.dropna(inplace=True)

    predictors = [
        "adj_price",
        "adj_value",
        "interest",
        "price_13w_change",
        "value_52w_change"
    ]

    # ----------------------------
    # Walk-Forward Model
    # ----------------------------
    START = 260
    STEP = 52

    probs = []

    for i in range(START, len(data), STEP):
        train = data.iloc[:i]
        test = data.iloc[i:i + STEP]

        if len(test) == 0:
            continue

        model = RandomForestClassifier(
            min_samples_split=10,
            random_state=1
        )

        model.fit(train[predictors], train["target"])
        p = model.predict_proba(test[predictors])[:, 1]
        probs.append(p)

    probs = np.concatenate(probs)

    results = data.iloc[START:].copy()
    results["prob_up"] = probs

    # ----------------------------
    # Regime Classification
    # ----------------------------
    def regime(p):
        if p > 0.65:
            return "Bull"
        elif p < 0.45:
            return "Risk"
        return "Neutral"

    results["regime"] = results["prob_up"].apply(regime)

    # ----------------------------
    # Monthly Summary
    # ----------------------------
    monthly = results.copy()
    monthly["month"] = monthly.index.to_period("M")

    monthly_signal = monthly.groupby("month").agg(
        prob_up=("prob_up", "mean"),
        regime=("regime", lambda x: x.value_counts().index[0])
    )

    # ----------------------------
    # Output Tables
    # ----------------------------
    st.subheader("✅ Latest Weekly Signal")
    st.dataframe(results[["prob_up", "regime"]].tail(1))

    st.subheader("✅ Latest Monthly Signal")
    st.dataframe(monthly_signal.tail(1))

    # ----------------------------
    # Chart 1: Price + Regimes
    # ----------------------------
    st.subheader("📈 Price Trend & Risk Regimes")

    fig = plt.figure(figsize=(14, 6))

    plt.plot(
        results.index,
        results["adj_price"],
        color="black",
        linewidth=2
    )

    for i in range(len(results) - 1):
        c = (
            "green" if results["regime"].iloc[i] == "Bull"
            else "gold" if results["regime"].iloc[i] == "Neutral"
            else "red"
        )

        plt.axvspan(
            results.index[i],
            results.index[i + 1],
            color=c,
            alpha=0.12
        )

    plt.title(f"{metro_name} Housing Market Regime")
    plt.ylabel("Inflation-Adjusted Price")

    legend = [
        Patch(facecolor="green", alpha=0.3, label="Bull"),
        Patch(facecolor="gold", alpha=0.3, label="Neutral"),
        Patch(facecolor="red", alpha=0.3, label="Risk")
    ]

    plt.legend(handles=legend)
    st.pyplot(fig)

    # ----------------------------
    # Chart 2: Last 12 Weeks
    # ----------------------------
    st.subheader("📊 Weekly Outlook (Last 12 Weeks)")

    recent = results.tail(12)

    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent.index, recent["prob_up"], marker="o", linewidth=2)

    ax.axhline(0.65, linestyle="--", alpha=0.6)
    ax.axhline(0.45, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability of Price Increase")

    st.pyplot(fig2)

    # ----------------------------
    # Plain English Summary
    # ----------------------------
    st.subheader("📌 Investor Summary")

    latest = results.iloc[-1]

    if latest["regime"] == "Bull":
        st.success(f"🟢 Favorable conditions (prob={latest['prob_up']:.2f})")
    elif latest["regime"] == "Neutral":
        st.warning(f"🟡 Mixed signals (prob={latest['prob_up']:.2f})")
    else:
        st.error(f"🔴 Elevated risk (prob={latest['prob_up']:.2f})")
