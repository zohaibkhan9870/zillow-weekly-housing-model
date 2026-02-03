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
st.set_page_config(page_title="US Real Estate Price Outlook Dashboard", layout="wide")

st.title("🏡 US Real Estate Price Outlook Dashboard")
st.write("Upload Zillow files → select State & Metro → get price outlook for multiple time horizons.")


# ----------------------------
# Sidebar: Zillow Download Help + Uploads
# ----------------------------
st.sidebar.header("📂 Zillow CSV Setup")

file_status = st.sidebar.radio(
    "Do you already have the Zillow CSV files downloaded?",
    ["✅ Yes, I already have them", "⬇️ No, I need to download them"]
)

# If user needs help downloading
if file_status == "⬇️ No, I need to download them":
    st.sidebar.markdown("### ✅ Step 1: Open Zillow Research Page")
    st.sidebar.markdown("👉 Official Zillow Research Website:")
    st.sidebar.markdown("**https://www.zillow.com/research/data/**")

    # clickable link
    st.sidebar.link_button("🌐 Open Zillow Data Page", "https://www.zillow.com/research/data/")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ Step 2: Download these 2 CSV files")

    st.sidebar.markdown("#### 🏠 Home Values Section")
    st.sidebar.markdown("Download **ZHVI (Home Value Index)** CSV:")
    st.sidebar.code("Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")

    st.sidebar.markdown("#### 💰 Sales Section")
    st.sidebar.markdown("Download **Median Sale Price (Weekly)** CSV:")
    st.sidebar.code("Metro_median_sale_price_uc_sfrcondo_sm_week.csv")

    st.sidebar.markdown("---")
    st.sidebar.info("✅ After downloading both files, come back and select 'Yes' above to upload them.")
    st.stop()


# ----------------------------
# Sidebar Uploads (only if YES)
# ----------------------------
st.sidebar.header("📤 Upload Zillow Files")

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

    st.sidebar.header("🌎 Select Location")

    states = sorted(list(set([m.split(",")[-1].strip() for m in metro_list if "," in m])))
    selected_state = st.sidebar.selectbox("Choose State", states)

    filtered_metros = [m for m in metro_list if m.endswith(f", {selected_state}")]
    selected_metro = st.sidebar.selectbox("Choose Metro", filtered_metros)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("✅ Run Forecast")

else:
    st.info("⬅️ Upload both Zillow CSV files to continue.")
    st.stop()


# ----------------------------
# ✅ STEP 2: Only run model when button pressed
# ----------------------------
if run_button:
    with st.spinner(f"⏳ Processing... Running forecast for {selected_metro}"):
        progress = st.progress(0)
        status = st.empty()

        # Step 1
        status.info("Step 1/5: Loading metro data...")
        progress.progress(10)

        price_matches = zillow_price[zillow_price["RegionName"] == selected_metro]
        value_matches = zillow_value[zillow_value["RegionName"] == selected_metro]

        if price_matches.empty or value_matches.empty:
            st.error("❌ Selected metro not found in both files.")
            st.stop()

        price = pd.DataFrame(price_matches.iloc[0, 5:])
        value = pd.DataFrame(value_matches.iloc[0, 5:])

        # Step 2
        status.info("Step 2/5: Fetching macro data from FRED...")
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
                results.append([horizon_name, None, None, "Not enough data", "-"])
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

            up_pct = round(latest_prob * 100, 0)
            down_pct = round((1 - latest_prob) * 100, 0)

            label = friendly_label(latest_prob)
            action = simple_action(label)

            results.append([horizon_name, f"{int(up_pct)}%", f"{int(down_pct)}%", label, action])

        out_df = pd.DataFrame(
            results,
            columns=["Time Horizon", "Price Up Chance (%)", "Price Down Chance (%)", "Outlook", "Suggested Action"]
        )

        # Step 5
        status.info("Step 5/5: Creating charts + weekly summary...")
        progress.progress(90)

        horizon_weeks = 13
        temp3 = data.copy()
        temp3["future_price"] = temp3["adj_price"].shift(-horizon_weeks)
        temp3["target"] = (temp3["future_price"] > temp3["adj_price"]).astype(int)
        temp3.dropna(inplace=True)

        if temp3.shape[0] <= START:
            prob_data = None
            monthly_signal = None
        else:
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

            probs3 = np.concatenate(all_probs_3)

            prob_data = temp3.iloc[START:].copy()
            prob_data["prob_up"] = probs3
            prob_data["regime"] = prob_data["prob_up"].apply(regime_from_prob)

            monthly = prob_data.copy()
            monthly["month"] = monthly.index.to_period("M")

            monthly_signal = (
                monthly.groupby("month")
                .agg({
                    "prob_up": "mean",
                    "regime": lambda x: x.value_counts().index[0]
                })
            )

        status.success("✅ Done! Forecast is ready.")
        progress.progress(100)

    # OUTPUT
    st.success("✅ Done! Forecast is ready.")
    st.subheader("✅ Forecast Results (All Time Horizons)")
    st.dataframe(out_df, use_container_width=True)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_bytes,
        file_name=f"{selected_metro.replace(',', '').replace(' ', '_')}_forecast_results.csv",
        mime="text/csv"
    )

    if prob_data is None or monthly_signal is None:
        st.warning("Not enough data to build weekly graph and monthly trend yet.")
        st.stop()

    # Weekly Prediction
    st.subheader("📌 Weekly Prediction")

    latest_week_prob = float(prob_data["prob_up"].tail(1).values[0])
    weekly_label = friendly_label(latest_week_prob)
    weekly_action = simple_action(weekly_label)

    if "🟢" in weekly_label:
        st.success(f"✅ Weekly Outlook: {weekly_label}")
        st.write("This week looks supportive. Prices are more likely to go up.")
    elif "🔴" in weekly_label:
        st.error(f"✅ Weekly Outlook: {weekly_label}")
        st.write("This week looks risky. Prices may face downward pressure.")
    else:
        st.warning(f"✅ Weekly Outlook: {weekly_label}")
        st.write("This week is unclear. Prices could move up or down.")

    # Monthly Prediction
    st.subheader("📌 Monthly Prediction")

    latest_month_regime = monthly_signal["regime"].tail(1).values[0]

    if latest_month_regime == "Bull":
        st.info("ℹ️ Monthly Trend: 🟢 Growing trend")
        st.write("The bigger monthly trend looks positive.")
    elif latest_month_regime == "Risk":
        st.info("ℹ️ Monthly Trend: 🔴 Weak trend")
        st.write("The bigger monthly trend looks weak or risky.")
    else:
        st.info("ℹ️ Monthly Trend: 🟡 Still unclear")
        st.write("The bigger monthly trend is still unclear.")

    # Suggested Action
    st.subheader("👉 Suggested Action")
    st.write(weekly_action)

    # Chart 1
    st.subheader("📈 Price Trend + Risk Background (3-Month Outlook)")

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

        plt.axvspan(prob_data.index[i], prob_data.index[i + 1], color=color, alpha=0.12)

    plt.title(f"{selected_metro} Housing Price Trend (With Risk Zones)", fontsize=14, weight="bold")
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

    # Chart 2
    st.subheader("📊 Weekly Outlook (Last 12 Weeks)")

    recent = prob_data.tail(12)

    fig2, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        recent.index,
        recent["prob_up"],
        marker="o",
        linewidth=2.5,
        color="black"
    )

    ax.axhline(0.65, color="green", linestyle="--", alpha=0.6)
    ax.axhline(0.45, color="red", linestyle="--", alpha=0.6)

    ax.set_title("Weekly Outlook Score (Last 12 Weeks)", fontsize=14, weight="bold")
    ax.set_ylabel("Outlook Score (0 to 1)")
    ax.set_xlabel("Week")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    explanation_text = (
        "HOW TO READ THIS CHART (Simple)\n"
        "• Black line with dots: Weekly outlook score (higher = better).\n"
        "• Green dotted line (0.65): Above = Good time (supportive market).\n"
        "• Red dotted line (0.45): Below = Risky time (higher downside risk).\n"
        "• X-axis = Weeks (time) | Y-axis = Score from 0 to 1."
    )

    fig2.text(
        0.5,
        0.01,
        explanation_text,
        ha="center",
        va="bottom",
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    st.pyplot(fig2)
