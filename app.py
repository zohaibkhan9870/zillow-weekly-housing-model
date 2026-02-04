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


def regime_from_prob(p):
    if p >= 0.65:
        return "Bull"
    elif p <= 0.45:
        return "Risk"
    else:
        return "Neutral"


def role_based_action(label, user_role):
    if user_role == "🏠 Home Buyer":
        if "🟢" in label:
            return "Buyer Mode: Safer time to buy. Negotiate and consider locking rate options."
        elif "🔴" in label:
            return "Buyer Mode: Risk is high. Prefer waiting or buying only if strongly discounted."
        else:
            return "Buyer Mode: Unclear market. Monitor price cuts and interest rates."
    elif user_role == "💼 Investor":
        if "🟢" in label:
            return "Investor Mode: Supportive market. Consider entry strategies and value-add deals."
        elif "🔴" in label:
            return "Investor Mode: High downside risk. Prefer low leverage and strong cashflow."
        else:
            return "Investor Mode: Mixed signals. Stay selective and demand strong fundamentals."
    else:
        if "🟢" in label:
            return "Agent Mode: Supportive market. Expect stronger buyer activity."
        elif "🔴" in label:
            return "Agent Mode: Risky conditions. Expect slower sales and more reductions."
        else:
            return "Agent Mode: Mixed conditions. Track inventory and guide clients carefully."


def deal_score(prob_up):
    return max(0, min(100, int(round(prob_up * 100, 0))))


def expected_return_range(prob_up, horizon_weeks):
    horizon_factor = np.sqrt(max(horizon_weeks, 1) / 13)
    expected = (prob_up - 0.5) * 8.0 * horizon_factor
    risk_band = 4.0 * horizon_factor
    return float(expected), float(expected - risk_band), float(expected + risk_band)


def generate_pdf_report(metro, out_df, weekly_label, monthly_regime, suggested_action, deal_score_value):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "US Real Estate Price Outlook Report")
    y -= 25

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Metro: {metro}")
    y -= 20

    c.drawString(50, y, f"Deal Score (0-100): {deal_score_value}")
    y -= 20

    c.drawString(50, y, f"Weekly Outlook: {weekly_label}")
    y -= 20

    c.drawString(50, y, f"Monthly Trend: {monthly_regime}")
    y -= 20

    c.drawString(50, y, f"Suggested Action: {suggested_action}")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Forecast Results:")
    y -= 18

    c.setFont("Helvetica", 10)
    for _, row in out_df.iterrows():
        line = f"{row['Time Horizon']}: Up {row['Price Up Chance (%)']} | Down {row['Price Down Chance (%)']} | {row['Outlook']}"
        c.drawString(55, y, line)
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Note: This report is generated automatically using Zillow + FRED data. For informational use only.")
    c.save()

    buffer.seek(0)
    return buffer.read()


def calc_metro_latest_prob(metro_name, zillow_price, zillow_value, fed_data, predictors, START, STEP):
    pm = zillow_price[zillow_price["RegionName"] == metro_name]
    vm = zillow_value[zillow_value["RegionName"] == metro_name]
    if len(pm) == 0 or len(vm) == 0:
        return None

    p = pd.DataFrame(pm.iloc[0, 5:])
    v = pd.DataFrame(vm.iloc[0, 5:])
    p.index = pd.to_datetime(p.index)
    v.index = pd.to_datetime(v.index)

    p["month"] = p.index.to_period("M")
    v["month"] = v.index.to_period("M")

    pv = p.merge(v, on="month")
    pv.index = p.index
    pv.drop(columns=["month"], inplace=True)
    pv.columns = ["price", "value"]

    d2 = fed_data.merge(pv, left_index=True, right_index=True)
    d2["adj_price"] = d2["price"] / d2["cpi"] * 100
    d2["adj_value"] = d2["value"] / d2["cpi"] * 100
    d2["price_13w_change"] = d2["adj_price"].pct_change(13)
    d2["value_52w_change"] = d2["adj_value"].pct_change(52)
    d2.dropna(inplace=True)

    if d2.shape[0] <= START:
        return None

    t2 = d2.copy()
    t2["future_price"] = t2["adj_price"].shift(-13)
    t2["target"] = (t2["future_price"] > t2["adj_price"]).astype(int)
    t2.dropna(inplace=True)

    if t2.shape[0] <= START:
        return None

    def predict_proba_comp(train, test):
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(train[predictors], train["target"])
        return rf.predict_proba(test[predictors])[:, 1]

    all_probs = []
    for i in range(START, t2.shape[0], STEP):
        train = t2.iloc[:i]
        test = t2.iloc[i:i + STEP]
        if len(test) == 0:
            continue
        all_probs.append(predict_proba_comp(train, test))

    if len(all_probs) == 0:
        return None

    probs = np.concatenate(all_probs)
    return float(probs[-1])


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

selected_state = st.selectbox("Choose State", states, index=0)
state_metros = [m for m in metro_list if m.endswith(f", {selected_state}")]

filtered_metros = state_metros
if metro_search:
    filtered_metros = [m for m in state_metros if metro_search.lower() in m.lower()]

if len(filtered_metros) == 0:
    st.warning("⚠️ No metros found. Try another search or change state.")
    st.stop()

selected_metro = st.selectbox("Choose Metro", filtered_metros, index=0)

st.markdown("### 👤 Client Mode")
user_role = st.selectbox("Choose client type", ["🏠 Home Buyer", "💼 Investor", "🧑‍💼 Agent"], index=0)

st.markdown("### 🏙️ Quick Compare (Top 3 Metros in Same State)")
compare_enabled = st.checkbox("✅ Enable Metro Comparison", value=True)

st.markdown("### 🏆 Metro Ranking (Selected State)")
rank_enabled = st.checkbox("✅ Enable Full Metro Ranking", value=True)

if len(state_metros) >= 3:
    max_rank = min(25, len(state_metros))
    rank_count = st.slider("How many metros to rank?", min_value=3, max_value=max_rank, value=min(10, max_rank))
else:
    rank_enabled = False
    rank_count = len(state_metros)

run_button = st.button("✅ Run Forecast")


# ----------------------------
# ✅ Run Forecast
# ----------------------------
if run_button:
    with st.spinner(f"⏳ Running forecast for {selected_metro}..."):

        # ---------- Load FRED ----------
        interest = load_fred_series("MORTGAGE30US").rename(columns={"value": "interest"})
        cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})
        vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})

        fed_data = pd.concat([interest, cpi, vacancy], axis=1)
        fed_data = fed_data.sort_index().ffill().dropna()
        fed_data.index = fed_data.index + timedelta(days=2)

        # ---------- Load metro price/value ----------
        price_matches = zillow_price[zillow_price["RegionName"] == selected_metro]
        value_matches = zillow_value[zillow_value["RegionName"] == selected_metro]

        price = pd.DataFrame(price_matches.iloc[0, 5:])
        value = pd.DataFrame(value_matches.iloc[0, 5:])

        price.index = pd.to_datetime(price.index)
        value.index = pd.to_datetime(value.index)

        price["month"] = price.index.to_period("M")
        value["month"] = value.index.to_period("M")

        price_data = price.merge(value, on="month")
        price_data.index = price.index
        price_data.drop(columns=["month"], inplace=True)
        price_data.columns = ["price", "value"]

        data = fed_data.merge(price_data, left_index=True, right_index=True)

        # ---------- Features ----------
        data["adj_price"] = data["price"] / data["cpi"] * 100
        data["adj_value"] = data["value"] / data["cpi"] * 100
        data["price_13w_change"] = data["adj_price"].pct_change(13)
        data["value_52w_change"] = data["adj_value"].pct_change(52)
        data.dropna(inplace=True)

        predictors = ["adj_price", "adj_value", "interest", "vacancy", "price_13w_change", "value_52w_change"]

        START = 104
        STEP = 26

        horizons = {
            "1 Month Ahead": 4,
            "2 Months Ahead": 8,
            "3 Months Ahead": 13,
            "6 Months Ahead": 26,
            "1 Year Ahead": 52
        }

        # ---------- Build horizon results ----------
        results = []
        for horizon_name, weeks_ahead in horizons.items():
            temp = data.copy()
            temp["future_price"] = temp["adj_price"].shift(-weeks_ahead)
            temp["target"] = (temp["future_price"] > temp["adj_price"]).astype(int)
            temp.dropna(inplace=True)

            if temp.shape[0] <= START:
                results.append([horizon_name, "-", "-", "Not enough data", "-", "-", "-"])
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
            latest_prob = float(probs[-1])

            label = friendly_label(latest_prob)
            action = role_based_action(label, user_role)

            exp_ret, exp_low, exp_high = expected_return_range(latest_prob, weeks_ahead)

            results.append([
                horizon_name,
                f"{latest_prob*100:.0f}%",
                f"{(1-latest_prob)*100:.0f}%",
                label,
                action,
                f"{exp_ret:+.1f}%",
                f"[{exp_low:+.1f}%, {exp_high:+.1f}%]"
            ])

        out_df = pd.DataFrame(results, columns=[
            "Time Horizon",
            "Price Up Chance (%)",
            "Price Down Chance (%)",
            "Outlook",
            "Suggested Action",
            "Expected Change (%)",
            "Expected Range (%)"
        ])

        # ---------- Weekly/Monthly safe ----------
        horizon_weeks = 13
        temp3 = data.copy()
        temp3["future_price"] = temp3["adj_price"].shift(-horizon_weeks)
        temp3["target"] = (temp3["future_price"] > temp3["adj_price"]).astype(int)
        temp3.dropna(inplace=True)

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
            weekly_available = False
            prob_data = temp3.copy()
            prob_data["prob_up"] = np.nan
            prob_data["regime"] = "Neutral"
            monthly_signal = pd.DataFrame({"prob_up": [np.nan], "regime": ["Neutral"]})
            latest_week_prob = 0.50
        else:
            weekly_available = True
            probs3 = np.concatenate(all_probs_3)
            prob_data = temp3.iloc[START:].copy()
            prob_data["prob_up"] = probs3
            prob_data["regime"] = prob_data["prob_up"].apply(regime_from_prob)

            latest_week_prob = float(prob_data["prob_up"].tail(1).values[0])

            monthly = prob_data.copy()
            monthly["month"] = monthly.index.to_period("M")
            monthly_signal = monthly.groupby("month").agg({
                "prob_up": "mean",
                "regime": lambda x: x.value_counts().index[0]
            })

        weekly_label = friendly_label(latest_week_prob)
        weekly_action = role_based_action(weekly_label, user_role)
        deal_score_value = deal_score(latest_week_prob)

    # ✅ ALWAYS SHOW RESULTS BELOW (NO MATTER WHAT)
    st.success("✅ Done! Forecast is ready.")

    st.markdown("---")
    st.subheader("📌 Quick Summary (Client Value KPIs)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weekly Score", f"{latest_week_prob:.2f}")
    c2.metric("Deal Score (0-100)", f"{deal_score_value}")
    c3.metric("Signal", weekly_label.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", ""))
    c4.metric("Client Mode", user_role.replace("🏠 ", "").replace("💼 ", "").replace("🧑‍💼 ", ""))

    # ✅ Metro Comparison RESULTS
    if compare_enabled:
        st.markdown("---")
        st.subheader("🏙️ Metro Comparison (Same State) — Top 3 by Deal Score")

        comp_rows = []
        sample_metros = state_metros[:12]

        for m in sample_metros:
            p = calc_metro_latest_prob(m, zillow_price, zillow_value, fed_data, predictors, START, STEP)
            if p is None:
                continue
            comp_rows.append([m, f"{p*100:.0f}%", friendly_label(p), deal_score(p)])

        if len(comp_rows) > 0:
            comp_df = pd.DataFrame(comp_rows, columns=["Metro", "Up Chance (3M)", "Outlook", "Deal Score"])
            st.dataframe(comp_df.sort_values("Deal Score", ascending=False).head(3), use_container_width=True)
        else:
            st.warning("⚠️ Not enough metro data available for comparison.")

    # ✅ Metro Ranking RESULTS
    if rank_enabled:
        st.markdown("---")
        st.subheader(f"🏆 Metro Ranking — Top {rank_count} in {selected_state}")

        rank_rows = []
        rank_metros = state_metros[:min(len(state_metros), rank_count * 5)]

        for m in rank_metros:
            p = calc_metro_latest_prob(m, zillow_price, zillow_value, fed_data, predictors, START, STEP)
            if p is None:
                continue
            rank_rows.append([m, f"{p*100:.0f}%", friendly_label(p), deal_score(p)])

        if len(rank_rows) > 0:
            rank_df = pd.DataFrame(rank_rows, columns=["Metro", "Up Chance (3M)", "Outlook", "Deal Score"])
            st.dataframe(rank_df.sort_values("Deal Score", ascending=False).head(rank_count), use_container_width=True)
        else:
            st.warning("⚠️ Not enough metro data available for ranking.")

    # ✅ Forecast Table
    st.markdown("---")
    st.subheader("✅ Forecast Results (All Time Horizons)")
    st.dataframe(out_df, use_container_width=True)

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_bytes,
        file_name=f"{selected_metro.replace(',', '').replace(' ', '_')}_forecast_results.csv",
        mime="text/csv"
    )

    pdf_bytes = generate_pdf_report(
        metro=selected_metro,
        out_df=out_df,
        weekly_label=weekly_label,
        monthly_regime=monthly_signal["regime"].tail(1).values[0],
        suggested_action=weekly_action,
        deal_score_value=deal_score_value
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"{selected_metro.replace(',', '').replace(' ', '_')}_forecast_report.pdf",
        mime="application/pdf"
    )

    # ✅ Weekly Prediction (kept)
    st.markdown("---")
    st.subheader("📌 Weekly Prediction")
    if not weekly_available:
        st.warning("⚠️ Weekly outlook could not be calculated (not enough data).")
    else:
        st.info(f"Weekly Outlook: {weekly_label}")
        st.write(weekly_action)

    # ✅ Monthly Prediction (kept)
    st.markdown("---")
    st.subheader("📌 Monthly Prediction")
    latest_month_regime = monthly_signal["regime"].tail(1).values[0]
    st.info(f"Monthly Trend: {latest_month_regime}")

    # ✅ Chart 1
    st.markdown("---")
    st.subheader("📈 Price Trend + Risk Background (3-Month Outlook)")
    fig1 = plt.figure(figsize=(14, 6))
    plt.plot(prob_data.index, prob_data["adj_price"], color="black", linewidth=2)

    for i in range(len(prob_data) - 1):
        reg = prob_data["regime"].iloc[i]
        if reg == "Bull":
            color = "green"
        elif reg == "Neutral":
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

    # ✅ Chart 2
    st.markdown("---")
    st.subheader("📊 Weekly Outlook (Last 12 Weeks)")
    if not weekly_available:
        st.info("⚠️ Weekly outlook chart is unavailable due to insufficient data.")
    else:
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

        st.pyplot(fig2)
