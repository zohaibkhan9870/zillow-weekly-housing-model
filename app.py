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

# ✅ PDF export
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

if file_status == "✅ Yes, I already have them":
    confirm_download = True


# ----------------------------
# ✅ File Validation
# ----------------------------
EXPECTED_PRICE_FILENAME = "Metro_median_sale_price_uc_sfrcondo_sm_week.csv"
EXPECTED_VALUE_FILENAME = "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


def safe_median_date_diff_days(parsed_dates):
    if parsed_dates is None or len(parsed_dates) < 2:
        return None
    sorted_dates = pd.Series(parsed_dates).sort_values().reset_index(drop=True)
    diffs = sorted_dates.diff().dropna()
    if diffs.empty:
        return None
    return diffs.dt.total_seconds().median() / 86400


def validate_zillow_csv(uploaded_file, expected_type):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return False, "❌ Could not read this CSV file.", None

    if "RegionName" not in df.columns:
        return False, "❌ Missing RegionName column.", None

    date_cols = df.columns[5:]
    parsed_dates = pd.to_datetime(date_cols, errors="coerce").dropna()
    median_days = safe_median_date_diff_days(parsed_dates)

    if expected_type == "weekly_price" and median_days > 15:
        return False, "❌ Not a weekly file.", None
    if expected_type == "monthly_value" and median_days < 20:
        return False, "❌ Not a monthly file.", None

    return True, "✅ File verified.", df


# ----------------------------
# ✅ Upload Section
# ----------------------------
st.subheader("📤 Upload Zillow Files")

price_file = st.file_uploader("Upload Weekly Median Sale Price CSV", type=["csv"])
value_file = st.file_uploader("Upload ZHVI Home Value Index CSV", type=["csv"])

if not price_file or not value_file:
    st.stop()

_, _, zillow_price = validate_zillow_csv(price_file, "weekly_price")
_, _, zillow_value = validate_zillow_csv(value_file, "monthly_value")


# ----------------------------
# ✅ FRED API Loader
# ----------------------------
def load_fred_series(series_id):
    api_key = st.secrets["FRED_API_KEY"]
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    r = requests.get(url, params=params, timeout=30)
    df = pd.DataFrame(r.json()["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.set_index("date", inplace=True)
    return df[["value"]]


# ----------------------------
# Helper functions
# ----------------------------
def friendly_label(p):
    return "🟢 Good time" if p >= 0.65 else "🔴 Risky" if p <= 0.45 else "🟡 Unclear"

def regime_from_prob(p):
    return "Bull" if p >= 0.65 else "Risk" if p <= 0.45 else "Neutral"


# ----------------------------
# ✅ Location selection
# ----------------------------
metro_list = sorted(set(zillow_price["RegionName"]).intersection(zillow_value["RegionName"]))
selected_metro = st.selectbox("Choose Metro", metro_list)

if st.button("✅ Run Forecast"):
    price = pd.DataFrame(zillow_price[zillow_price["RegionName"] == selected_metro].iloc[0, 5:])
    value = pd.DataFrame(zillow_value[zillow_value["RegionName"] == selected_metro].iloc[0, 5:])

    price.index = pd.to_datetime(price.index)
    value.index = pd.to_datetime(value.index)

    interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
    cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})
    vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})

    data = pd.concat([interest, cpi, vacancy, price, value], axis=1).dropna()
    data.columns = ["interest", "cpi", "vacancy", "price", "value"]

    data["adj_price"] = data["price"] / data["cpi"] * 100
    data["adj_value"] = data["value"] / data["cpi"] * 100
    data["price_13w_change"] = data["adj_price"].pct_change(13)
    data["value_52w_change"] = data["adj_value"].pct_change(52)
    data.dropna(inplace=True)

    predictors = ["adj_price", "adj_value", "interest", "vacancy", "price_13w_change", "value_52w_change"]

    # ----------------------------
    # ✅ FIXED weekly + monthly logic
    # ----------------------------
    horizon_weeks = 13
    temp3 = data.copy()
    temp3["future_price"] = temp3["adj_price"].shift(-horizon_weeks)
    temp3["target"] = (temp3["future_price"] > temp3["adj_price"]).astype(int)
    temp3.dropna(inplace=True)

    START, STEP = 104, 26
    all_probs_3 = []

    for i in range(START, temp3.shape[0], STEP):
        train = temp3.iloc[:i]
        test = temp3.iloc[i:i + STEP]
        if len(test) == 0:
            continue
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(train[predictors], train["target"])
        all_probs_3.append(rf.predict_proba(test[predictors])[:, 1])

    if len(all_probs_3) == 0:
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(temp3[predictors], temp3["target"])
        prob_data = temp3.tail(1).copy()
        prob_data["prob_up"] = rf.predict_proba(temp3[predictors].tail(1))[:, 1]
    else:
        probs3 = np.concatenate(all_probs_3)
        prob_data = temp3.iloc[START:].copy()
        prob_data["prob_up"] = probs3

    prob_data["regime"] = prob_data["prob_up"].apply(regime_from_prob)

    st.success("✅ Forecast complete")
    st.metric("Weekly Signal", friendly_label(prob_data["prob_up"].iloc[-1]))
