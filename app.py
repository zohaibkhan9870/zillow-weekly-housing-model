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
# ✅ Reset Button
# ----------------------------
st.markdown("---")
if st.button("🔄 Reset / Clear All"):
    st.session_state.clear()
    st.rerun()


# ----------------------------
# ✅ Zillow CSV Setup (NO DEFAULT SELECTION)
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
# If user needs to download
# ----------------------------
if file_status == "⬇️ No, I need to download them":
    st.markdown("### ✅ Step 1: Open Zillow Research Page")
    st.link_button("🌐 Open Zillow Data Page", "https://www.zillow.com/research/data/")

    st.markdown("### ✅ Step 2: Download these 2 CSV files")

    st.markdown("#### 🏠 Home Values Section")
    st.markdown("Download **ZHVI (Home Value Index)** CSV:")
    st.markdown(
        """
        <div style="background-color:#111827; padding:12px; border-radius:10px; font-size:14px;">
            <b>Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### 💰 Sales Section")
    st.markdown("Download **Median Sale Price (Weekly)** CSV:")
    st.markdown(
        """
        <div style="background-color:#111827; padding:12px; border-radius:10px; font-size:14px;">
            <b>Metro_median_sale_price_uc_sfrcondo_sm_week.csv</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    confirm_download = st.checkbox("✅ I downloaded both Zillow CSV files")

    if not confirm_download:
        st.warning("✅ Please confirm you downloaded both files to unlock uploads.")
        st.stop()

# If user already has files (skip checkbox)
if file_status == "✅ Yes, I already have them":
    confirm_download = True


# ----------------------------
# ✅ File Validation
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
        return False, "❌ Wrong file content: not enough columns (missing time series).", None

    sample_regions = df["RegionName"].dropna().astype(str).head(30).tolist()
    if not any("," in x for x in sample_regions):
        return False, "❌ Wrong file content: RegionName does not look like metro format (City, ST).", None

    date_cols = list(df.columns[5:])
    parsed_dates = pd.to_datetime(date_cols, errors="coerce").dropna()

    if len(parsed_dates) < 10:
        return False, "❌ Wrong file content: could not detect enough valid date columns.", None

    median_days = safe_median_date_diff_days(parsed_dates)
    if median_days is None:
        return False, "❌ Wrong file content: could not detect weekly/monthly frequency.", None

    if expected_type == "weekly_price":
        if median_days > 15:
            return False, "❌ This does NOT look like the Weekly Sale Price file.", None

    if expected_type == "monthly_value":
        if median_days < 20:
            return False, "❌ This does NOT look like the Monthly ZHVI file.", None

    return True, "✅ Correct file uploaded.", df


# ----------------------------
# ✅ Upload Section (AUTO-CLOSE AFTER VERIFIED)
# ----------------------------
st.subheader("📤 Upload Zillow Files")

price_ok = False
value_ok = False
zillow_price = None
zillow_value = None

if "files_verified" not in st.session_state:
    st.session_state.files_verified = False

expander_open = not st.session_state.files_verified

with st.expander("📤 Upload Files (Click to expand)", expanded=expander_open):
    price_file = st.file_uploader("Upload Weekly Median Sale Price CSV", type=["csv"])

    if price_file is not None:
        if price_file.name != EXPECTED_PRICE_FILENAME:
            st.warning(f"⚠️ File name is different than expected.\nExpected: {EXPECTED_PRICE_FILENAME}")

        ok, msg, df_temp = validate_zillow_csv(price_file, "weekly_price")
        if ok:
            price_ok = True
            zillow_price = df_temp
        else:
            st.error(msg)

    value_file = st.file_uploader("Upload ZHVI Home Value Index CSV", type=["csv"])

    if value_file is not None:
        if value_file.name != EXPECTED_VALUE_FILENAME:
            st.warning(f"⚠️ File name is different than expected.\nExpected: {EXPECTED_VALUE_FILENAME}")

        ok, msg, df_temp = validate_zillow_csv(value_file, "monthly_value")
        if ok:
            value_ok = True
            zillow_value = df_temp
        else:
            st.error(msg)

# ✅ Once verified -> Save in session state + auto-collapse
if price_ok and value_ok:
    st.session_state.files_verified = True
    st.success("✅ Files Verified! You can now select location and run forecast.")
else:
    st.info("✅ Upload both correct files to continue.")
    st.stop()


# ----------------------------
# ✅ FRED API Loader
# ----------------------------
def load_fred_series(series_id):
    api_key = st.secrets["FRED_API_KEY"]

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}

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
# ✅ Location selection
# ----------------------------
st.subheader("🌎 Select Location")

metro_list = sorted(
    set(zillow_price["RegionName"].dropna().unique()).intersection(
        set(zillow_value["RegionName"].dropna().unique())
    )
)

metro_search = st.text_input("🔍 Search metro (optional)", "").strip()

states = sorted(list(set([m.split(",")[-1].strip() for m in metro_list if "," in m])))

auto_state = None
auto_metro = None
if metro_search:
    matches = [m for m in metro_list if metro_search.lower() in m.lower()]
    if len(matches) > 0:
        auto_metro = matches[0]
        auto_state = auto_metro.split(",")[-1].strip()

default_state_index = states.index(auto_state) if auto_state in states else 0
selected_state = st.selectbox("Choose State", states, index=default_state_index)

filtered_metros = [m for m in metro_list if m.endswith(f", {selected_state}")]
if metro_search:
    filtered_metros = [m for m in filtered_metros if metro_search.lower() in m.lower()]

if len(filtered_metros) == 0:
    st.warning("⚠️ No metros found. Try another search or change state.")
    st.stop()

default_metro_index = filtered_metros.index(auto_metro) if auto_metro in filtered_metros else 0
selected_metro = st.selectbox("Choose Metro", filtered_metros, index=default_metro_index)

run_button = st.button("✅ Run Forecast")


# ----------------------------
# ✅ Run Model + Results + Charts
# ----------------------------
if run_button:
    with st.spinner(f"⏳ Processing... Running forecast for {selected_metro}"):
        progress = st.progress(0)
        status = st.empty()

        status.info("Step 1/5: Loading metro data...")
        progress.progress(10)

        price_matches = zillow_price[zillow_price["RegionName"] == selected_metro]
        value_matches = zillow_value[zillow_value["RegionName"] == selected_metro]

        price = pd.DataFrame(price_matches.iloc[0, 5:])
        value = pd.DataFrame(value_matches.iloc[0, 5:])

        status.info("Step 2/5: Fetching macro data from FRED...")
        progress.progress(30)

        interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
        vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})
        cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})

        unemployment = load_fred_series("UNRATE").rename(columns={"value": "unemployment"})
        jobs = load_fred_series("PAYEMS").rename(columns={"value": "jobs"})
        permits = load_fred_series("PERMIT").rename(columns={"value": "permits"})
        stress = load_fred_series("STLFSI4").rename(columns={"value": "stress"})

        fed_data = pd.concat([interest, vacancy, cpi, unemployment, jobs, permits, stress], axis=1)
        fed_data = fed_data.sort_index().ffill().dropna()
        fed_data.index = fed_data.index + timedelta(days=2)

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
            "adj_price", "adj_value", "interest", "vacancy",
            "price_13w_change", "value_52w_change",
            "unemployment", "jobs", "permits", "stress",
            "unemployment_13w_change", "jobs_13w_change",
            "permits_13w_change", "stress_13w_change"
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

        out_df = pd.DataFrame(results, columns=[
            "Time Horizon", "Price Up Chance (%)", "Price Down Chance (%)", "Outlook", "Suggested Action"
        ])

        # Weekly + Monthly signals (3 month horizon)
        status.info("Step 5/5: Creating charts + weekly summary...")
        progress.progress(90)

        horizon_weeks = 13
        temp3 = data.copy()
        temp3["future_price"] = temp3["adj_price"].shift(-horizon_weeks)
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

        probs3 = np.concatenate(all_probs_3)

        prob_data = temp3.iloc[START:].copy()
        prob_data["prob_up"] = probs3
        prob_data["regime"] = prob_data["prob_up"].apply(regime_from_prob)

        monthly = prob_data.copy()
        monthly["month"] = monthly.index.to_period("M")
        monthly_signal = monthly.groupby("month").agg({
            "prob_up": "mean",
            "regime": lambda x: x.value_counts().index[0]
        })

        status.success("✅ Done! Forecast is ready.")
        progress.progress(100)

    with st.expander("ℹ️ What does this forecast mean? (Simple explanation)"):
        st.write("✅ **Price Up Chance (%)** = chance home prices may rise.")
        st.write("✅ **Price Down Chance (%)** = chance home prices may fall.")
        st.write("✅ **Outlook** shows the simple signal:")
        st.write("• 🟢 Good time = more chance of prices going up")
        st.write("• 🟡 Unclear = mixed signals (could go up or down)")
        st.write("• 🔴 Risky = higher downside risk")

    st.subheader("✅ Forecast Results (All Time Horizons)")
    st.dataframe(out_df, use_container_width=True)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_bytes,
        file_name=f"{selected_metro.replace(',', '').replace(' ', '_')}_forecast_results.csv",
        mime="text/csv"
    )

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

    st.subheader("👉 Suggested Action")
    st.write(weekly_action)

    st.subheader("📈 Price Trend + Risk Background (3-Month Outlook)")
    fig1 = plt.figure(figsize=(14, 6))
    plt.plot(prob_data.index, prob_data["adj_price"], color="black", linewidth=2)

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
    plt.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    st.pyplot(fig1)

    st.subheader("📊 Weekly Outlook (Last 12 Weeks)")
    recent = prob_data.tail(12)

    fig2, ax = plt.subplots(figsize=(12, 7))
    ax.plot(recent.index, recent["prob_up"], marker="o", linewidth=2.5, color="black")
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

    fig2.text(0.5, 0.01, explanation_text, ha="center", va="bottom", fontsize=10)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    st.pyplot(fig2)


