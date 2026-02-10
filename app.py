import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="Texas Housing Market Outlook", layout="wide")
st.title("🏡 Texas Housing Market Outlook")
st.write("Zillow + FRED + ML → Texas metro housing market signals")

# =================================================
# HOW TO USE
# =================================================
st.info(
    "### How to use this dashboard\n"
    "- Avoid buying when risk is high and prices are likely to struggle\n"
    "- Compare Texas cities to see which markets look stronger or weaker right now\n"
    "- Decide when to act — buy, wait, or negotiate — based on market conditions\n\n"
    "This tool helps with market risk and timing, not exact future prices."
)

st.markdown("---")

# =================================================
# HELPERS
# =================================================
def friendly_label(p):
    if p >= 0.65:
        return "🟢 Supportive"
    elif p <= 0.45:
        return "🔴 Risky"
    return "🟡 Balanced"

def market_situation(score):
    if score < 0.45: return "Risky"
    elif score < 0.65: return "Balanced"
    return "Supportive"

def market_effect(score):
    if score < 0.45: return "Against prices"
    elif score < 0.65: return "Neither helping nor hurting"
    return "Helping prices"

def regime_from_prob(p):
    if p >= 0.65: return "Supportive"
    elif p <= 0.45: return "Risky"
    return "Unclear"

def confidence_badge(n_obs):
    if n_obs >= 300: return "🟢 High"
    elif n_obs >= 180: return "🟡 Medium"
    return "🔴 Low"

def suggested_action(prob, *_):
    if prob >= 0.65:
        return "Market has strength. Buying can make sense if value is fair."
    elif prob <= 0.45:
        return "Risk is elevated. Consider waiting or negotiating harder."
    return "Balanced market. Compare options carefully."

def action_for_table(prob):
    if prob >= 0.65:
        return "Favorable — consider buying"
    elif prob <= 0.45:
        return "Risky — be cautious"
    return "No rush — wait for better setup"

def proxy_up_probability(price_series):
    pct = price_series.pct_change(13).dropna()
    if pct.empty: return None
    raw = float(pct.tail(1).values[0])
    prob = 0.50 + np.clip(raw, -0.10, 0.10) * 2.0
    return float(np.clip(prob, 0.05, 0.95))

def simple_reasons(row, _):
    reasons = []
    reasons.append("📉 Prices are below normal levels" if row["trend_diff"] < 0 else "📈 Prices are holding above normal levels")
    reasons.append("↘️ Prices have been falling recently" if row["p13"] < 0 else "↗️ Prices have been moving upward")
    reasons.append("🏘️ More homes are coming onto the market" if row["vacancy_trend"] > 0 else "🏠 Fewer homes are available")
    return reasons

def price_direction(p13):
    return ("Rising", "Helps prices") if p13 > 0 else ("Falling", "Pushes prices down")

def rate_level(rate):
    if rate > 6.5: return "High", "Pushes prices down"
    elif rate < 5.5: return "Low", "Helps prices"
    return "Normal", "Neutral effect"

def supply_level(vacancy_trend):
    return ("Many", "Pushes prices down") if vacancy_trend > 0 else ("Few", "Helps prices")

def trend_position(trend_diff):
    return ("Above normal", "Helps prices") if trend_diff > 0 else ("Below normal", "Pushes prices down")

def meaning_and_action(situation):
    if situation == "Risky":
        return "Most conditions are working against prices", "Avoid rushing"
    elif situation == "Balanced":
        return "Conditions are neither clearly positive nor negative", "Start monitoring"
    return "Conditions are generally supporting prices", "Look for opportunities"

def early_market_signal(row, prev_row):
    if row["p13"] > prev_row["p13"]:
        return "🟡 Prices are still falling, but the decline is slowing."
    return "⚪ Prices are still falling at a similar or faster pace."

def detect_zillow_file_type(df):
    """
    Returns: 'weekly_price', 'monthly_zhvi', or None
    """

    # Must have RegionName
    if "RegionName" not in df.columns:
        return None

    # Try to extract date columns (Zillow dates start after col 5)
    date_cols = df.columns[5:]

    try:
        dates = pd.to_datetime(date_cols, errors="coerce")
        dates = dates[~pd.isna(dates)]
    except:
        return None

    if len(dates) < 5:
        return None

    # Measure spacing between dates
    deltas = dates.to_series().diff().dt.days.dropna()

    median_gap = deltas.median()

    if median_gap <= 10:
        return "weekly_price"

    if 25 <= median_gap <= 35:
        return "monthly_zhvi"

    return None

# =================================================
# FRED LOADER
# =================================================
def load_fred(series_id):
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": st.secrets["FRED_API_KEY"],
            "file_type": "json"
        },
        timeout=30
    )
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
    st.stop()

price_df = pd.read_csv(price_file)
value_df = pd.read_csv(value_file)

price_type = detect_zillow_file_type(price_df)
value_type = detect_zillow_file_type(value_df)

# Validate Weekly Median Sale Price file
if price_type != "weekly_price":
    if price_type == "monthly_zhvi":
        st.error("❌ Wrong file uploaded for **Weekly Median Sale Price**.\n\nYou uploaded a **Monthly ZHVI** file here. Please swap the files.")
    else:
        st.error("❌ Incorrect file uploaded for **Weekly Median Sale Price**.\n\nThis file does not match Zillow weekly price data.")
    st.stop()

# Validate Monthly ZHVI file
if value_type != "monthly_zhvi":
    if value_type == "weekly_price":
        st.error("❌ Wrong file uploaded for **Monthly ZHVI**.\n\nYou uploaded a **Weekly Median Sale Price** file here. Please swap the files.")
    else:
        st.error("❌ Incorrect file uploaded for **Monthly ZHVI**.\n\nThis file does not match Zillow monthly ZHVI data.")
    st.stop()

# =================================================
# TEXAS METROS
# =================================================
tx_metros = sorted([
    m for m in price_df["RegionName"].unique()
    if m.endswith(", TX") and m in value_df["RegionName"].values
])

selected_metro = st.selectbox("📍 Select Texas Metro", tx_metros)

if not st.button("✅ Run Forecast"):
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
# LOAD MARKET CONDITIONS DATA
# =================================================
interest = load_fred("MORTGAGE30US").rename(columns={"value": "interest"})
cpi = load_fred("CPIAUCSL").rename(columns={"value": "cpi"})
vacancy = load_fred("RRVRUSQ156N").rename(columns={"value": "vacancy"})

macro = pd.concat([interest, cpi, vacancy], axis=1).sort_index().ffill().dropna()
macro.index += timedelta(days=2)

data = macro.merge(zillow, left_index=True, right_index=True)

# =================================================
# FEATURES
# =================================================
data["adj_price"] = data["price"] / data["cpi"] * 100
data["p13"] = data["adj_price"].pct_change(13)
data["trend"] = data["adj_price"].rolling(52).mean()
data["trend_diff"] = data["adj_price"] - data["trend"]
data["vol"] = data["p13"].rolling(26).std()
data["vacancy_trend"] = data["vacancy"].diff(13)

data.dropna(inplace=True)

# =================================================
# MODEL
# =================================================
predictors = ["adj_price", "interest", "vacancy", "p13"]

temp = data.copy()
temp["future"] = temp["adj_price"].shift(-13)
temp["target"] = (temp["future"] > temp["adj_price"]).astype(int)
temp.dropna(inplace=True)

split = int(len(temp) * 0.7)
train, test = temp.iloc[:split], temp.iloc[split:]

rf = RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

acc = accuracy_score(test["target"], rf.predict(test[predictors]))
confidence_pct = int(round(acc * 100))

temp["prob_up"] = rf.predict_proba(temp[predictors])[:, 1]
temp["regime"] = temp["prob_up"].apply(regime_from_prob)

latest = temp.iloc[-1]
previous = temp.iloc[-2]
latest_prob = float(latest["prob_up"])
early_signal = early_market_signal(latest, previous)

# =================================================
# MARKET SNAPSHOT
# =================================================
st.markdown(f"## 📌 Market Snapshot — {selected_metro}")

st.write(f"**Current Market Condition:** {friendly_label(latest_prob)}")

st.write(f"**Model Reliability:** {confidence_badge(len(temp))}")

st.write(f"**Suggested Approach:** {suggested_action(latest_prob, None, None, None)}")

with st.expander("ℹ️ How reliable is this assessment?"):
    st.write(
        f"This outlook is based on historical patterns. "
        f"In past data, similar signals were correct about {confidence_pct}% of the time. "
        "This is meant to guide decisions, not predict exact prices."
    )

# =================================================
# RECENT VS LONGER-TERM CONTEXT
# =================================================
st.markdown("### 📉 Recent market movement (last few weeks)")
st.write(early_signal)

st.markdown("### 📈 Overall market direction (last few months)")
for r in simple_reasons(latest, latest_prob):
    st.write(f"- {r}")

# =================================================
# FUTURE MARKET OUTLOOK
# =================================================
st.markdown("---")
st.subheader("🗓️ Future Market Outlook")

# Helper for colored dots (place this once with other helpers if not already defined)
def situation_with_dot(situation):
    if situation == "Supportive":
        return "🟢 Supportive"
    elif situation == "Risky":
        return "🔴 Risky"
    return "🟡 Balanced"

horizons = [
    ("1 month", latest_prob - 0.10),
    ("2 months", latest_prob - 0.07),
    ("3 months", latest_prob - 0.05),
    ("6 months", latest_prob),
    ("1 year", latest_prob + 0.05),
]

rows = []
for label, score in horizons:
    score = min(max(score, 0), 1)

    situation = market_situation(score)
    situation_display = situation_with_dot(situation)

    effect = market_effect(score)
    meaning, action = meaning_and_action(situation)

    rows.append([
        label,
        situation_display,
        effect,
        meaning,
        action
    ])

st.dataframe(
    pd.DataFrame(
        rows,
        columns=[
            "Time Ahead",
            "Market Condition",
            "Market Effect",
            "What this means",
            "Suggested Approach"
        ]
    ),
    use_container_width=True
)

st.caption(
    "Market Condition shows overall pressure on prices. "
    "Shorter time frames are more uncertain; longer time frames reflect broader trends."
)
# =================================================
# WHY THE MARKET LOOKS THIS WAY
# =================================================
st.markdown("---")
st.subheader("🔍 Why the market looks this way")

price_seen, price_effect = price_direction(latest["p13"])
rate_seen, rate_effect = rate_level(latest["interest"])
supply_seen, supply_effect = supply_level(latest["vacancy_trend"])
trend_seen, trend_effect = trend_position(latest["trend_diff"])

st.dataframe(pd.DataFrame(
    [
        ["Recent prices movement", price_seen, price_effect],
        ["Interest rates", rate_seen, rate_effect],
        ["Homes for sale", supply_seen, supply_effect],
        ["Normal Price level", trend_seen, trend_effect],
    ],
    columns=["What the model looks at", "What it sees", "Effect on prices"]
), use_container_width=True)

# =================================================
# METRO COMPARISON — TOP 3
# =================================================
st.markdown("---")
st.subheader("🏙️ Texas Metro Comparison — Top 3")

rows = []
for m in tx_metros:
    pm = price_df[price_df["RegionName"] == m]
    if pm.empty: continue
    p = pd.DataFrame(pm.iloc[0, 5:])
    p.index = pd.to_datetime(p.index)
    p.columns = ["price"]
    prob = proxy_up_probability(p["price"])
    if prob is None: continue
    rows.append([m, f"{prob*100:.0f}%", friendly_label(prob), action_for_table(prob)])

if rows:
    st.dataframe(pd.DataFrame(
        rows,
        columns=["Metro", "Price Up Chance", "Market Condition", "Suggested Approach"]
    ).sort_values("Price Up Chance", ascending=False).head(3), use_container_width=True)

# =================================================
# ALL METROS RANKING
# =================================================
st.markdown("---")
st.subheader("🏙️ Texas Metro Rankings (All Metros)")

st.caption(
    "Market Condition reflects overall pressure on prices. "
    "Recent Price Movement shows what prices have been doing lately."
)

rank_rows = []
for m in tx_metros:
    pm = price_df[price_df["RegionName"] == m]
    if pm.empty:
        continue

    p = pd.DataFrame(pm.iloc[0, 5:])
    p.index = pd.to_datetime(p.index)
    p.columns = ["price"]

    prob = proxy_up_probability(p["price"])
    if prob is None:
        continue

    rank_rows.append([   # ✅ INSIDE LOOP
        m,
        friendly_label(prob),
        "Rising" if prob >= 0.55 else "Falling or Flat",
        confidence_badge(len(p.dropna()))
    ])

st.dataframe(pd.DataFrame(
    rank_rows,
    columns=["Metro", "Market Condition", "Recent Price Movement", "Confidence"]
), use_container_width=True)

# =================================================
# HISTORICAL PRICE & REGIMES
# =================================================
st.markdown("---")
st.subheader("📈 Historical Price Trend & Risk Regimes")

fig = plt.figure(figsize=(14,6))
plt.plot(temp.index, temp["adj_price"], color="black", linewidth=2)

for i in range(len(temp)-1):
    plt.axvspan(
        temp.index[i],
        temp.index[i+1],
        color="green" if temp["regime"].iloc[i] == "Supportive" else "red",
        alpha=0.15
    )

st.pyplot(fig)

# =================================================
# WEEKLY OUTLOOK
# =================================================
st.markdown("---")
st.subheader("📊 Weekly Outlook (Last 12 Weeks)")

recent = temp.tail(12)
fig2, ax = plt.subplots(figsize=(12,5))
ax.plot(recent.index, recent["prob_up"], marker="o", linewidth=2)
ax.axhline(0.65, linestyle="--", color="green", alpha=0.6)
ax.axhline(0.45, linestyle="--", color="red", alpha=0.6)
ax.set_ylim(0,1)

st.pyplot(fig2)
st.caption("Above 0.65 = market helping prices • Below 0.45 = market working against prices")


