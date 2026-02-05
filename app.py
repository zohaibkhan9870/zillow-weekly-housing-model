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
# ✅ FRED API Loader (ONLY 3 SERIES)
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


def regime_from_prob(p):
    if p >= 0.65:
        return "Bull"
    elif p <= 0.45:
        return "Risk"
    else:
        return "Neutral"


# ✅ Upgrade #9: Role-based actions
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
    else:  # Agent
        if "🟢" in label:
            return "Agent Mode: Supportive market. Expect stronger buyer activity."
        elif "🔴" in label:
            return "Agent Mode: Risky conditions. Expect slower sales and more reductions."
        else:
            return "Agent Mode: Mixed conditions. Track inventory and guide clients carefully."


# ✅ Deal Score (0–100)
def deal_score(prob_up):
    score = int(round(prob_up * 100, 0))
    return max(0, min(100, score))


# ✅ Expected Return Estimates (simple)
def expected_return_range(prob_up, horizon_weeks):
    horizon_factor = np.sqrt(max(horizon_weeks, 1) / 13)
    expected = (prob_up - 0.5) * 8.0 * horizon_factor
    risk_band = 4.0 * horizon_factor
    return float(expected), float(expected - risk_band), float(expected + risk_band)


# ✅ Backtest / Win Rate
def compute_backtest_metrics(temp_df, predictors, start_idx, step, threshold=0.5):
    if temp_df.shape[0] <= start_idx + 10:
        return None

    def predict_proba(train, test):
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(train[predictors], train["target"])
        return rf.predict_proba(test[predictors])[:, 1]

    all_probs = []
    all_true = []

    for i in range(start_idx, temp_df.shape[0], step):
        train = temp_df.iloc[:i]
        test = temp_df.iloc[i:i + step]
        if len(test) == 0:
            continue
        probs = predict_proba(train, test)
        all_probs.append(probs)
        all_true.append(test["target"].values)

    if len(all_probs) == 0:
        return None

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)
    y_pred = (probs >= threshold).astype(int)

    accuracy = float((y_pred == y_true).mean())
    return {"accuracy": accuracy, "win_rate": accuracy, "n_samples": int(len(y_true))}


# ✅ PDF report generator
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


# ✅ Upgrade #8: Feature Importance Plot
def feature_importance_fig(model, feature_names, title="Feature Importance"):
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(fi_df["Feature"], fi_df["Importance"])
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fi_df, fig


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

# ✅ FIX: Always keep full state metros for ranking/compare
state_metros = [m for m in metro_list if m.endswith(f", {selected_state}")]

# ✅ Search affects only dropdown list
filtered_metros = state_metros
if metro_search:
    filtered_metros = [m for m in state_metros if metro_search.lower() in m.lower()]

if len(filtered_metros) == 0:
    st.warning("⚠️ No metros found. Try another search or change state.")
    st.stop()

default_metro_index = filtered_metros.index(auto_metro) if auto_metro in filtered_metros else 0
selected_metro = st.selectbox("Choose Metro", filtered_metros, index=default_metro_index)

# Client mode
st.markdown("### 👤 Client Mode")
user_role = st.selectbox("Choose client type", ["🏠 Home Buyer", "💼 Investor", "🧑‍💼 Agent"], index=0)

# Advanced toggle
show_advanced = st.checkbox("⚙️ Show Advanced Analytics", value=True)

# Metro Comparison
st.markdown("### 🏙️ Quick Compare (Top 3 Metros in Same State)")
compare_enabled = st.checkbox("✅ Enable Metro Comparison", value=True)

# ✅ Full Metro Ranking (fixed using state_metros)
st.markdown("### 🏆 Metro Ranking (Selected State)")
rank_enabled = st.checkbox("✅ Enable Full Metro Ranking", value=True)

total_metros = len(state_metros)

if total_metros < 3:
    st.warning("⚠️ Not enough metros in this state to run ranking (need at least 3).")
    rank_count = total_metros
    rank_enabled = False
else:
    max_rank = min(25, total_metros)
    default_rank = min(10, max_rank)

    rank_count = st.slider(
        "How many metros to rank?",
        min_value=3,
        max_value=max_rank,
        value=default_rank
    )

# Alerts
st.markdown("### 🔔 Alerts (Basic Version)")
alerts_enabled = st.checkbox("✅ Enable Alerts", value=True)
alert_threshold = st.slider("Alert Threshold (Up Chance)", 0.50, 0.80, 0.65, 0.01)

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
        cpi = load_fred_series("CPIAUCSL").rename(columns={"value": "cpi"})
        vacancy = load_fred_series("RRVRUSQ156N").rename(columns={"value": "vacancy"})

        fed_data = pd.concat([interest, cpi, vacancy], axis=1)
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
        data.dropna(inplace=True)

        predictors = [
            "adj_price",
            "adj_value",
            "interest",
            "vacancy",
            "price_13w_change",
            "value_52w_change"
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

        # Backtest metrics (3M)
        temp_bt = data.copy()
        temp_bt["future_price"] = temp_bt["adj_price"].shift(-13)
        temp_bt["target"] = (temp_bt["future_price"] > temp_bt["adj_price"]).astype(int)
        temp_bt.dropna(inplace=True)

        backtest = compute_backtest_metrics(temp_bt, predictors, START, STEP, threshold=0.5)

        results = []
        stored_models = {}

        for horizon_name, weeks_ahead in horizons.items():
            temp = data.copy()
            temp["future_price"] = temp["adj_price"].shift(-weeks_ahead)
            temp["target"] = (temp["future_price"] > temp["adj_price"]).astype(int)
            temp.dropna(inplace=True)

            if temp.shape[0] <= START:
                results.append([horizon_name, None, None, "Not enough data", "-", "-", "-"])
                continue

            def predict_proba(train, test, return_model=False):
                rf = RandomForestClassifier(min_samples_split=10, random_state=1)
                rf.fit(train[predictors], train["target"])
                p = rf.predict_proba(test[predictors])[:, 1]
                if return_model:
                    return p, rf
                return p

            all_probs = []
            last_model = None

            for i in range(START, temp.shape[0], STEP):
                train = temp.iloc[:i]
                test = temp.iloc[i:i + STEP]
                if len(test) == 0:
                    continue
                proba, last_model = predict_proba(train, test, return_model=True)
                all_probs.append(proba)

            probs = np.concatenate(all_probs)
            pred_df = temp.iloc[START:].copy()
            pred_df["prob_up"] = probs
            latest_prob = float(pred_df["prob_up"].tail(1).values[0])

            stored_models[horizon_name] = last_model

            up_pct = round(latest_prob * 100, 0)
            down_pct = round((1 - latest_prob) * 100, 0)

            label = friendly_label(latest_prob)
            action = role_based_action(label, user_role)

            exp_ret, exp_low, exp_high = expected_return_range(latest_prob, weeks_ahead)
            exp_ret_str = f"{exp_ret:+.1f}%"
            exp_range_str = f"[{exp_low:+.1f}%, {exp_high:+.1f}%]"

            results.append([horizon_name, f"{int(up_pct)}%", f"{int(down_pct)}%", label, action, exp_ret_str, exp_range_str])

        out_df = pd.DataFrame(results, columns=[
            "Time Horizon",
            "Price Up Chance (%)",
            "Price Down Chance (%)",
            "Outlook",
            "Suggested Action",
            "Expected Change (%)",
            "Expected Range (%)"
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

    # KPI Summary
    st.markdown("---")
    st.subheader("📌 Quick Summary (Client Value KPIs)")

    latest_week_prob = float(prob_data["prob_up"].tail(1).values[0])
    weekly_label = friendly_label(latest_week_prob)
    weekly_action = role_based_action(weekly_label, user_role)
    deal_score_value = deal_score(latest_week_prob)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Weekly Score", f"{latest_week_prob:.2f}")
    col2.metric("Deal Score (0-100)", f"{deal_score_value}")
    col3.metric("Signal", weekly_label.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", ""))
    col4.metric("Client Mode", user_role.replace("🏠 ", "").replace("💼 ", "").replace("🧑‍💼 ", ""))
    if backtest is not None:
        col5.metric("Backtest Win Rate (3M)", f"{backtest['win_rate']*100:.1f}%")
    else:
        col5.metric("Backtest Win Rate (3M)", "N/A")

    # Alerts
    if alerts_enabled:
        st.markdown("---")
        st.subheader("🔔 Alert Status")
        if latest_week_prob >= alert_threshold:
            st.success(f"✅ ALERT TRIGGERED: Weekly score {latest_week_prob:.2f} ≥ threshold {alert_threshold:.2f}")
        else:
            st.info(f"ℹ️ No alert: Weekly score {latest_week_prob:.2f} < threshold {alert_threshold:.2f}")

    # ✅ Metro Comparison Results (now always uses state_metros)
    if compare_enabled:
        st.markdown("---")
        st.subheader("🏙️ Metro Comparison (Same State) — Top 3 by Deal Score")

        sample_metros = state_metros[:8]
        comp_rows = []

        for m in sample_metros:
            pm = zillow_price[zillow_price["RegionName"] == m]
            if len(pm) == 0:
                continue

            p = pd.DataFrame(pm.iloc[0, 5:])
            p.index = pd.to_datetime(p.index)

            if len(p) < 30:
                continue

            pct = p.iloc[:, 0].pct_change(13).dropna()
            if pct.empty:
                continue

            proxy = 0.50 + np.clip(float(pct.tail(1).values[0]), -0.10, 0.10) * 2.0
            proxy = float(np.clip(proxy, 0.05, 0.95))

            comp_rows.append([m, f"{proxy*100:.0f}%", friendly_label(proxy), deal_score(proxy)])

        if len(comp_rows) > 0:
            comp_df = pd.DataFrame(comp_rows, columns=["Metro", "Up Chance (Fast)", "Outlook", "Deal Score"])
            comp_df = comp_df.sort_values("Deal Score", ascending=False).head(3)
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.info("Comparison needs more data. Try another state or metro.")

    # ✅ Metro Ranking Results (now always uses state_metros)
    if rank_enabled:
        st.markdown("---")
        st.subheader(f"🏆 Metro Ranking — Top {rank_count} (State: {selected_state})")

        ranking_rows = []
        for m in state_metros[:rank_count * 2]:
            pm = zillow_price[zillow_price["RegionName"] == m]
            if len(pm) == 0:
                continue

            p = pd.DataFrame(pm.iloc[0, 5:])
            p.index = pd.to_datetime(p.index)
            if len(p) < 30:
                continue

            pct = p.iloc[:, 0].pct_change(13).dropna()
            if pct.empty:
                continue

            proxy_prob = 0.50 + np.clip(float(pct.tail(1).values[0]), -0.10, 0.10) * 2.0
            proxy_prob = float(np.clip(proxy_prob, 0.05, 0.95))

            ranking_rows.append([
                m,
                f"{proxy_prob*100:.0f}%",
                friendly_label(proxy_prob),
                deal_score(proxy_prob)
            ])

        if len(ranking_rows) > 0:
            ranking_df = pd.DataFrame(ranking_rows, columns=["Metro", "Proxy Up Chance (Fast)", "Outlook", "Deal Score"])
            ranking_df = ranking_df.sort_values("Deal Score", ascending=False).head(rank_count)
            st.dataframe(ranking_df, use_container_width=True)
        else:
            st.info("Not enough ranking data for this state.")

    # Feature importance
    if show_advanced:
        st.markdown("---")
        st.subheader("🧠 Why the Model Thinks This (Feature Importance)")
        chosen_horizon = st.selectbox("Choose horizon for explainability", list(horizons.keys()), index=2)

        model_for_horizon = stored_models.get(chosen_horizon)
        if model_for_horizon is not None:
            fi_df, fi_fig = feature_importance_fig(model_for_horizon, predictors, title=f"Feature Importance ({chosen_horizon})")
            st.pyplot(fi_fig)
            st.dataframe(fi_df.head(6), use_container_width=True)
        else:
            st.info("Explainability not available for this horizon.")

    # Forecast Results Table
    st.subheader("✅ Forecast Results (All Time Horizons)")
    st.dataframe(out_df, use_container_width=True)

    # Downloads
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

    # Weekly Prediction
    st.subheader("📌 Weekly Prediction")
    if "🟢" in weekly_label:
        st.success(f"✅ Weekly Outlook: {weekly_label}")
        st.write("This week looks supportive. Prices are more likely to go up.")
    elif "🔴" in weekly_label:
        st.error(f"✅ Weekly Outlook: {weekly_label}")
        st.write("This week looks risky. Prices may face downward pressure.")
    else:
        st.warning(f"✅ Weekly Outlook: {weekly_label}")
        st.write("This week is unclear. Prices could move up or down.")
        # ----------------------------
# 📈 Weekly Probability Chart
# ----------------------------
st.markdown("---")
st.subheader("📈 Weekly Price Direction Probability")

fig_w, ax_w = plt.subplots(figsize=(12, 4))

ax_w.plot(
    prob_data.index,
    prob_data["prob_up"],
    linewidth=2,
    label="Weekly Up Probability"
)

ax_w.axhline(0.65, linestyle="--", color="green", alpha=0.6, label="Bull Threshold")
ax_w.axhline(0.45, linestyle="--", color="red", alpha=0.6, label="Risk Threshold")

ax_w.set_ylim(0, 1)
ax_w.set_ylabel("Probability")
ax_w.set_title("Weekly Probability of Price Increase")
ax_w.legend()
ax_w.grid(alpha=0.3)

st.pyplot(fig_w)


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
        # ----------------------------
# 📆 Monthly Trend Chart
# ----------------------------
st.markdown("---")
st.subheader("📆 Monthly Trend (Smoothed Signal)")

fig_m, ax_m = plt.subplots(figsize=(12, 4))

ax_m.plot(
    monthly_signal.index.astype(str),
    monthly_signal["prob_up"],
    marker="o",
    linewidth=2
)

ax_m.axhline(0.65, linestyle="--", color="green", alpha=0.6)
ax_m.axhline(0.45, linestyle="--", color="red", alpha=0.6)

ax_m.set_ylim(0, 1)
ax_m.set_ylabel("Avg Probability")
ax_m.set_xlabel("Month")
ax_m.set_title("Monthly Average Probability of Price Increase")
ax_m.grid(alpha=0.3)

st.pyplot(fig_m)




