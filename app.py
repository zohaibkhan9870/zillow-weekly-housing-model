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


# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="US Real Estate Price Outlook", layout="wide")
st.title("🏡 US Real Estate Price Outlook Dashboard")
st.write("Zillow + FRED + ML → Forecasts, rankings, and explainability")
st.markdown("---")


# =================================================
# HELPERS
# =================================================
def friendly_label(p):
    if p >= 0.65:
        return "🟢 Good time"
    elif p <= 0.45:
        return "🔴 Risky"
    return "🟡 Unclear"


def deal_score(p):
    return int(np.clip(round(p * 100), 0, 100))


def role_action(label, role):
    if role == "Buyer":
        return "Safer time to buy" if "🟢" in label else "Wait or negotiate"
    if role == "Investor":
        return "Deploy capital" if "🟢" in label else "Preserve capital"
    return "Guide clients cautiously"


def expected_return(prob, weeks):
    factor = np.sqrt(max(weeks, 1) / 13)
    exp = (prob - 0.5) * 8 * factor
    band = 4 * factor
    return exp, exp - band, exp + band


def proxy_up_probability(price_series):
    pct = price_series.pct_change(13).dropna()
    if pct.empty:
        return None
    raw = float(pct.tail(1).values[0])
    prob = 0.50 + np.clip(raw, -0.10, 0.10) * 2.0
    return float(np.clip(prob, 0.05, 0.95))


# =================================================
# FRED LOADER
# =================================================
def load_fred(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": st.secrets["FRED_API_KEY"],
        "file_type": "json"
    }
    r = requests.get(url, params=params, timeout=30)
    df = pd.DataFrame(r.json()["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")[["value"]]


# =================================================
# FILE UPLOAD
# =================================================
st.subheader("📤 Upload Zillow Files")
c1, c2 = st.columns(2)
with c1:
    price_file = st.file_uploader("Weekly Median Sale Price CSV", type="csv")
with c2:
    value_file = st.file_uploader("Monthly ZHVI CSV", type="csv")

if not price_file or not value_file:
    st.info("Upload both Zillow files to continue.")
    st.stop()

price_df = pd.read_csv(price_file)
value_df = pd.read_csv(value_file)


# =================================================
# LOCATION
# =================================================
st.subheader("🌍 Location")
metros = sorted(set(price_df["RegionName"]).intersection(set(value_df["RegionName"])))
selected_metro = st.selectbox("Choose Metro", metros)
user_role = st.selectbox("Client Type", ["Buyer", "Investor", "Agent"])
run = st.button("✅ Run Forecast")

if not run:
    st.stop()


# =================================================
# PREP ZILLOW DATA
# =================================================
price = pd.DataFrame(price_df[price_df["RegionName"] == selected_metro].iloc[0, 5:])
value = pd.DataFrame(value_df[value_df["RegionName"] == selected_metro].iloc[0, 5:])

price.index = pd.to_datetime(price.index)
value.index = pd.to_datetime(value.index)
price.columns = ["price"]
value.columns = ["value"]

price["month"] = price.index.to_period("M")
value["month"] = value.index.to_period("M")

zillow = price.merge(value, on="month")
zillow.index = price.index
zillow.drop(columns="month", inplace=True)


# =================================================
# LOAD FRED
# =================================================
interest = load_fred("MORTGAGE30US").rename(columns={"value": "interest"})
cpi = load_fred("CPIAUCSL").rename(columns={"value": "cpi"})
vacancy = load_fred("RRVRUSQ156N").rename(columns={"value": "vacancy"})

macro = pd.concat([interest, cpi, vacancy], axis=1)
macro = macro.sort_index().ffill().dropna()
macro.index += timedelta(days=2)

data = macro.merge(zillow, left_index=True, right_index=True)


# =================================================
# FEATURES
# =================================================
data["adj_price"] = data["price"] / data["cpi"] * 100
data["adj_value"] = data["value"] / data["cpi"] * 100
data["p13"] = data["adj_price"].pct_change(13)
data["v52"] = data["adj_value"].pct_change(52)
data.dropna(inplace=True)

predictors = ["adj_price", "adj_value", "interest", "vacancy", "p13", "v52"]


# =================================================
# MULTI-HORIZON FORECAST
# =================================================
horizons = {
    "1 Month": 4,
    "3 Months": 13,
    "6 Months": 26,
    "1 Year": 52
}

results = []
stored_models = {}

for name, weeks in horizons.items():
    temp = data.copy()
    temp["future"] = temp["adj_price"].shift(-weeks)
    temp["target"] = (temp["future"] > temp["adj_price"]).astype(int)
    temp.dropna(inplace=True)

    rf = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(temp[predictors], temp["target"])
    prob = rf.predict_proba(temp[predictors].tail(1))[:, 1][0]

    stored_models[name] = rf

    label = friendly_label(prob)
    action = role_action(label, user_role)
    exp, lo, hi = expected_return(prob, weeks)

    results.append([
        name,
        f"{prob*100:.0f}%",
        label,
        deal_score(prob),
        action,
        f"{exp:+.1f}%",
        f"[{lo:+.1f}%, {hi:+.1f}%]"
    ])

out_df = pd.DataFrame(
    results,
    columns=[
        "Horizon",
        "Up Probability",
        "Outlook",
        "Deal Score",
        "Suggested Action",
        "Expected Change",
        "Expected Range"
    ]
)

st.markdown("---")
st.subheader("📊 Forecast Results")
st.dataframe(out_df, use_container_width=True)


# =================================================
# CHART 1: PRICE + REGIME
# =================================================
st.subheader("📈 Inflation-Adjusted Price Trend")

fig1 = plt.figure(figsize=(14, 5))
plt.plot(data.index, data["adj_price"], color="black", linewidth=2)

plt.title(f"{selected_metro} – Real Home Price")
plt.ylabel("Real Price (Adj)")
plt.xlabel("Date")
plt.tight_layout()
st.pyplot(fig1)


# =================================================
# CHART 2: WEEKLY PROBABILITY (3M HORIZON)
# =================================================
st.subheader("📊 Weekly Price-Up Probability (3 Months)")

weeks = 13
temp = data.copy()
temp["future"] = temp["adj_price"].shift(-weeks)
temp["target"] = (temp["future"] > temp["adj_price"]).astype(int)
temp.dropna(inplace=True)

rf_weekly = RandomForestClassifier(min_samples_split=10, random_state=1)
rf_weekly.fit(temp[predictors], temp["target"])
probs = rf_weekly.predict_proba(temp[predictors])[:, 1]

fig2, ax = plt.subplots(figsize=(14, 4))
ax.plot(temp.index, probs, linewidth=2)
ax.axhline(0.65, linestyle="--", alpha=0.6)
ax.axhline(0.45, linestyle="--", alpha=0.6)
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("Weekly Price-Up Probability")
st.pyplot(fig2)


# =================================================
# FEATURE IMPORTANCE
# =================================================
st.markdown("---")
st.subheader("🧠 Feature Importance")

chosen_horizon = st.selectbox("Choose horizon to explain", list(horizons.keys()), index=1)
model = stored_models[chosen_horizon]

fi = pd.DataFrame({
    "Feature": predictors,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig3, ax = plt.subplots(figsize=(10, 4))
ax.bar(fi["Feature"], fi["Importance"])
ax.set_title(f"Feature Importance – {chosen_horizon}")
ax.set_ylabel("Importance")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()

st.pyplot(fig3)
st.dataframe(fi, use_container_width=True)


# =================================================
# METRO RANKING
# =================================================
st.markdown("---")
st.subheader("🏆 Metro Ranking (Same State)")

rank_count = st.slider("Number of metros to rank", 5, 25, 10)

if "," in selected_metro:
    state = selected_metro.split(",")[-1].strip()
    state_metros = [m for m in metros if m.endswith(f", {state}")]

    rows = []
    for m in state_metros:
        pm = price_df[price_df["RegionName"] == m]
        if pm.empty:
            continue
        p = pd.DataFrame(pm.iloc[0, 5:])
        p.index = pd.to_datetime(p.index)
        p.columns = ["price"]

        prob = proxy_up_probability(p["price"])
        if prob is None:
            continue

        rows.append([m, f"{prob*100:.0f}%", friendly_label(prob), deal_score(prob)])

    if rows:
        rank_df = pd.DataFrame(
            rows,
            columns=["Metro", "Up Probability (Fast)", "Outlook", "Deal Score"]
        ).sort_values("Deal Score", ascending=False).head(rank_count)

        rank_df.index = np.arange(1, len(rank_df) + 1)
        rank_df.index.name = "Rank"
        st.dataframe(rank_df, use_container_width=True)
