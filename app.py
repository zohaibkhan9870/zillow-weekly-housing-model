import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ----------------------------
# Streamlit Setup
# ----------------------------
st.set_page_config(page_title="Weekly Zillow Housing Model", layout="wide")

st.title("🏡 Weekly Zillow Housing Model (Metro-Based)")
st.write("Upload the two Zillow CSV files and run the model metro-wise.")


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
run_button = st.sidebar.button("✅ Run Weekly Model")


# ----------------------------
# FRED Loader (Safe + Cloud Friendly)
# ----------------------------
def load_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    r = requests.get(url, timeout=30)

    if r.status_code != 200:
        raise Exception(f"FRED request failed for {series_id}. Status code: {r.status_code}")

    # FRED can sometimes return HTML instead of CSV (blocked / error page)
    first_line = r.text.splitlines()[0] if len(r.text.splitlines()) > 0 else ""
    if "DATE" not in first_line:
        raise Exception(
            f"FRED did not return CSV for {series_id}. "
            f"Unexpected response (maybe blocked by server)."
        )

    df = pd.read_csv(StringIO(r.text))

    df.columns = [c.strip() for c in df.columns]

    if "DATE" not in df.columns:
        raise Exception(f"Missing DATE column in FRED response for {series_id}")

    df["DATE"] = pd.to_datetime(df["DATE"])
    df.set_index("DATE", inplace=True)

    # Replace missing values
    df.replace(".", np.nan, inplace=True)
    df = df.astype(float)

    return df


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
    # Load FRED Data
    # ----------------------------
    try:
        interest = load_fred_series("MORTGAGE30US")
        vacancy = load_fred_series("RRVRUSQ156N")
        cpi = load_fred_series("CPIAUCSL")

        fed_data = pd.concat([interest, vacancy, cpi], axis=1)
        fed_data.columns = ["interest", "vacancy", "cpi"]
        fed_data = fed_data.sort_index().ffill().dropna()
        fed_data.index = fed_data.index + timedelta(days=2)

    except Exception as e:
        st.error("❌ Failed to fetch FRED data from St. Louis Fed.")
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
    price_data = fed_data.merge(price_data, left_index=True, right_index=True)

    # ----------------------------
    # Feature Engineering
    # ----------------------------
    price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
    price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100

    price_data["next_quarter"] = price_data["adj_price"].shift(-13)
    price_data["change"] = (price_data["next_quarter"] > price_data["adj_price"]).astype(int)

    price_data["price_13w_change"] = price_data["adj_price"].pct_change(13)
    price_data["value_52w_change"] = price_data["adj_value"].pct_change(52)

    price_data.dropna(inplace=True)

    # ----------------------------
    # Walk-forward ML model
    # ----------------------------
    predictors = [
        "adj_price",
        "adj_value",
        "interest",
        "price_13w_change",
        "value_52w_change"
    ]

    target = "change"
    START = 260
    STEP = 52

    def predict_proba(train, test):
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(train[predictors], train[target])
        return rf.predict_proba(test[predictors])[:, 1]

    all_probs = []

    for i in range(START, price_data.shape[0], STEP):
        train = price_data.iloc[:i]
        test = price_data.iloc[i:i + STEP]
        if len(test) == 0:
            continue
        all_probs.append(predict_proba(train, test))

    probs = np.concatenate(all_probs)

    prob_data = price_data.iloc[START:].copy()
    prob_data["prob_up"] = probs

    # ----------------------------
    # Regime Label
    # ----------------------------
    def label_regime(p):
        if p > 0.65:
            return "Bull"
        elif p < 0.45:
            return "Risk"
        else:
            return "Neutral"

    prob_data["regime"] = prob_data["prob_up"].apply(label_regime)

    # ----------------------------
    # Monthly Summary
    # ----------------------------
    monthly = prob_data.copy()
    monthly["month"] = monthly.index.to_period("M")

    monthly_signal = (
        monthly.groupby("month")
        .agg({
            "prob_up": "mean",
            "regime": lambda x: x.value_counts().index[0]
        })
    )

    # ----------------------------
    # OUTPUT TABLES
    # ----------------------------
    st.subheader("✅ Latest Weekly Signal")
    st.dataframe(prob_data[["prob_up", "regime"]].tail(1))

    st.subheader("✅ Latest Monthly Signal")
    st.dataframe(monthly_signal.tail(1))

    # ----------------------------
    # CHART 1: Price Trend + Regime
    # ----------------------------
    st.subheader("📈 Price Trend + Risk Regimes")

    fig1 = plt.figure(figsize=(14, 6))

    plt.plot(
        prob_data.index,
        prob_data["adj_price"],
        color="black",
        linewidth=2,
        label="Real Home Price (Inflation-Adjusted)"
    )

    for i in range(len(prob_data) - 1):
        regime = prob_data["regime"].iloc[i]
        if regime == "Bull":
            color = "green"
        elif regime == "Neutral":
            color = "gold"
        else:
            color = "red"

        plt.axvspan(
            prob_data.index[i],
            prob_data.index[i + 1],
            color=color,
            alpha=0.12
        )

    plt.title(
        f"{metro_name} Housing Market: Price Trend & Risk Signals",
        fontsize=14,
        weight="bold"
    )

    plt.ylabel("Typical Home Price (Inflation-Adjusted)")
    plt.xlabel("Date")

    legend_elements = [
        Patch(facecolor="green", alpha=0.25, label="Supportive Market (Favorable)"),
        Patch(facecolor="gold", alpha=0.25, label="Mixed Signals (Neutral)"),
        Patch(facecolor="red", alpha=0.25, label="High Risk / Caution"),
    ]

    plt.legend(
        handles=[plt.Line2D([0], [0], color="black", lw=2,
                            label="Real Home Price")] + legend_elements,
        loc="upper left"
    )

    plt.tight_layout()
    st.pyplot(fig1)

    # ----------------------------
    # CHART 2: Weekly Signal (Last 12 Weeks)
    # ----------------------------
    st.subheader("📊 Weekly Housing Market Outlook (Last 12 Weeks)")

    recent = prob_data.tail(12)

    fig2, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        recent.index,
        recent["prob_up"],
        marker="o",
        linewidth=2.5,
        color="black"
    )

    ax.axhline(0.65, color="green", linestyle="--", alpha=0.6)
    ax.axhline(0.45, color="red", linestyle="--", alpha=0.6)

    ax.set_title("Weekly Housing Market Outlook (Last 12 Weeks)", fontsize=14, weight="bold")
    ax.set_ylabel("Market Outlook Confidence")
    ax.set_xlabel("Week")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    st.pyplot(fig2)

    # ----------------------------
    # Plain English Message
    # ----------------------------
    st.subheader("📌 Plain-English Investor Message")

    latest = prob_data.tail(1)
    regime_now = latest["regime"].values[0]
    prob_now = latest["prob_up"].values[0]

    if regime_now == "Bull":
        st.success(f"🟢 FAVORABLE HOUSING ENVIRONMENT (prob_up={prob_now:.2f})")
        st.write("The model expects supportive conditions for housing prices over the next quarter.")
    elif regime_now == "Neutral":
        st.warning(f"🟡 MIXED HOUSING ENVIRONMENT (prob_up={prob_now:.2f})")
        st.write("The model sees an unclear outlook for housing prices over the next quarter.")
    else:
        st.error(f"🔴 RISK ENVIRONMENT (prob_up={prob_now:.2f})")
        st.write("The model expects downside pressure in housing prices over the next quarter.")
