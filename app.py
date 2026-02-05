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
from matplotlib.patches import Patch


# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="US Real Estate Price Outlook", layout="wide")
st.title("🏡 US Real Estate Price Outlook Dashboard see")
st.write("Zillow + FRED + ML → Client-ready housing market signals")
st.markdown("---")


# =================================================
# STATE MAP
# =================================================
STATE_MAP = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
    "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania",
    "RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee",
    "TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"
}


# =================================================
# HELPERS
# =================================================
def friendly_label(p):
    if p >= 0.65: return "🟢 Supportive"
    elif p <= 0.45: return "🔴 Risky"
    return "🟡 Unclear"

def regime_from_prob(p):
    if p >= 0.65: return "Supportive"
    elif p <= 0.45: return "Risky"
    return "Unclear"

def suggested_action(prob, trend_diff, vol, vacancy_trend):
    if prob >= 0.65:
        if trend_diff > 0:
            return "Market has strength. Buying can make sense if value is fair."
        else:
            return "Momentum is improving. Look for good deals."
    elif prob <= 0.45:
        if vacancy_trend > 0:
            return "Inventory is rising. Buyers can negotiate harder."
        else:
            return "Risk is elevated. Consider waiting for clarity."
    else:
        if vol > 0.04:
            return "Market is choppy. Avoid rushing decisions."
        else:
            return "Balanced market. Compare options carefully."

def action_for_table(prob):
    if prob >= 0.65: return "Favorable — consider buying"
    elif prob <= 0.45: return "Risky — be cautious"
    return "Mixed — take your time"

def proxy_up_probability(price_series):
    pct = price_series.pct_change(13).dropna()
    if pct.empty: return None
    raw = float(pct.tail(1).values[0])
    prob = 0.50 + np.clip(raw,-0.10,0.10)*2.0
    return float(np.clip(prob,0.05,0.95))


# --- UPGRADED METRO-SPECIFIC REASONS ---
def simple_reasons(latest_row, prob, trend_diff, vol, vacancy_trend):
    reasons=[]

    # Price vs trend
    if trend_diff < 0:
        reasons.append("📉 Prices sit below their yearly trend")
    else:
        reasons.append("📈 Prices are above their yearly trend")

    # Momentum
    if latest_row["p13"] < 0:
        reasons.append("↘️ Recent price momentum is slowing")
    else:
        reasons.append("↗️ Price momentum is improving")

    # Volatility
    if vol > 0.05:
        reasons.append("🎢 Prices have been volatile lately")
    else:
        reasons.append("📊 Prices have been relatively stable")

    # Vacancy trend
    if vacancy_trend > 0:
        reasons.append("🏘️ Inventory levels are rising")
    else:
        reasons.append("🏠 Inventory remains tight")

    # Final model signal
    if prob <= 0.45:
        reasons.append("⚠️ Model signals downside risk")
    elif prob >= 0.65:
        reasons.append("✅ Model signals supportive conditions")
    else:
        reasons.append("🤔 Model shows mixed signals")

    return reasons


# =================================================
# FRED LOADER
# =================================================
def load_fred(series_id):
    url="https://api.stlouisfed.org/fred/series/observations"
    params={"series_id":series_id,"api_key":st.secrets["FRED_API_KEY"],"file_type":"json"}
    r=requests.get(url,params=params,timeout=30)
    df=pd.DataFrame(r.json()["observations"])
    df["date"]=pd.to_datetime(df["date"])
    df["value"]=pd.to_numeric(df["value"],errors="coerce")
    return df.set_index("date")[["value"]]


# =================================================
# FILE UPLOAD
# =================================================
st.subheader("📤 Upload Zillow Files")
c1,c2=st.columns(2)

with c1:
    price_file=st.file_uploader("Weekly Median Sale Price CSV",type="csv")
with c2:
    value_file=st.file_uploader("Monthly ZHVI CSV",type="csv")

if not price_file or not value_file:
    st.info("Upload both Zillow files to continue.")
    st.stop()

price_df=pd.read_csv(price_file)
value_df=pd.read_csv(value_file)


# =================================================
# LOCATION SELECTION + SEARCH
# =================================================
st.subheader("🌍 Select Location")

metro_list=sorted(set(price_df["RegionName"]).intersection(set(value_df["RegionName"])))

records=[]
for m in metro_list:
    if "," not in m: continue
    city,abbr=m.rsplit(",",1)
    abbr=abbr.strip()
    if abbr not in STATE_MAP: continue
    records.append({
        "metro_raw":m,
        "metro_display":f"{city}, {STATE_MAP[abbr]}",
        "state_full":STATE_MAP[abbr]
    })

metro_df=pd.DataFrame(records)

search=st.text_input("🔍 Search metro (optional)","").strip()

auto_state=None
auto_metro=None
if search:
    matches=metro_df[metro_df["metro_display"].str.lower().str.contains(search.lower())]
    if not matches.empty:
        auto_state=matches.iloc[0]["state_full"]
        auto_metro=matches.iloc[0]["metro_raw"]

states=sorted(metro_df["state_full"].unique())
state_idx=states.index(auto_state) if auto_state in states else 0
selected_state=st.selectbox("Choose State",states,index=state_idx)

state_metros_df=metro_df[metro_df["state_full"]==selected_state]
metro_list_state=state_metros_df["metro_raw"].tolist()

metro_idx=metro_list_state.index(auto_metro) if auto_metro in metro_list_state else 0
selected_metro=st.selectbox("Choose Metro",metro_list_state,index=metro_idx)

if not st.button("✅ Run Forecast"):
    st.stop()


# =================================================
# PREP DATA
# =================================================
price=pd.DataFrame(price_df[price_df["RegionName"]==selected_metro].iloc[0,5:])
value=pd.DataFrame(value_df[value_df["RegionName"]==selected_metro].iloc[0,5:])

price.index=pd.to_datetime(price.index)
value.index=pd.to_datetime(value.index)

price.columns=["price"]
value.columns=["value"]

price["month"]=price.index.to_period("M")
value["month"]=value.index.to_period("M")

zillow=price.merge(value,on="month")
zillow.index=price.index
zillow.drop(columns="month",inplace=True)


# =================================================
# LOAD FRED
# =================================================
interest=load_fred("MORTGAGE30US").rename(columns={"value":"interest"})
cpi=load_fred("CPIAUCSL").rename(columns={"value":"cpi"})
vacancy=load_fred("RRVRUSQ156N").rename(columns={"value":"vacancy"})

macro=pd.concat([interest,cpi,vacancy],axis=1).sort_index().ffill().dropna()
macro.index+=timedelta(days=2)

data=macro.merge(zillow,left_index=True,right_index=True)


# =================================================
# FEATURES
# =================================================
data["adj_price"]=data["price"]/data["cpi"]*100
data["p13"]=data["adj_price"].pct_change(13)

data["trend"]=data["adj_price"].rolling(52).mean()
data["trend_diff"]=data["adj_price"]-data["trend"]

data["vol"]=data["p13"].rolling(26).std()

data["vacancy_trend"]=data["vacancy"].diff(13)

data.dropna(inplace=True)

predictors=["adj_price","interest","vacancy","p13"]

temp=data.copy()
temp["future"]=temp["adj_price"].shift(-13)
temp["target"]=(temp["future"]>temp["adj_price"]).astype(int)
temp.dropna(inplace=True)


# =================================================
# PROFESSIONAL BACKTEST
# =================================================
split=int(len(temp)*0.7)
train=temp.iloc[:split]
test=temp.iloc[split:]

rf=RandomForestClassifier(min_samples_split=10,random_state=1)
rf.fit(train[predictors],train["target"])

test_preds=rf.predict(test[predictors])
acc=accuracy_score(test["target"],test_preds)
confidence_pct=int(round(acc*100))

probs=rf.predict_proba(temp[predictors])[:,1]
temp["prob_up"]=probs
temp["regime"]=temp["prob_up"].apply(regime_from_prob)

latest_prob=float(temp["prob_up"].iloc[-1])
latest_row=temp.iloc[-1]

weekly_label=friendly_label(latest_prob)
clean_label=weekly_label.replace("🟢 ","").replace("🟡 ","").replace("🔴 ","")

monthly_regime=temp.resample("M")["regime"].agg(lambda x:x.value_counts().index[0]).iloc[-1]


# =================================================
# SNAPSHOT
# =================================================
st.markdown("---")
city,state_abbr=selected_metro.rsplit(",",1)

trend_diff=latest_row["trend_diff"]
vol=latest_row["vol"]
vacancy_trend=latest_row["vacancy_trend"]

st.markdown(f"## 📌 Market Snapshot — {city}, {state_abbr}")
st.write(f"**Market Outlook:** {clean_label}")
st.write(f"**Confidence:** ~{confidence_pct}% tested accuracy")
st.write(f"**Suggested Action:** {suggested_action(latest_prob,trend_diff,vol,vacancy_trend)}")

st.markdown("### Why this outlook:")
for r in simple_reasons(latest_row,latest_prob,trend_diff,vol,vacancy_trend):
    st.write(f"- {r}")


# =================================================
# WEEKLY + MONTHLY PREDICTIONS
# =================================================
st.markdown("---")
st.subheader("📌 Weekly Prediction")
st.info(f"Weekly Outlook: {weekly_label}")

st.markdown("---")
st.subheader("📌 Monthly Prediction")
st.info(f"Monthly Trend: {monthly_regime}")


# =================================================
# METRO COMPARISON
# =================================================
st.markdown("---")
st.subheader("🏙️ Metro Comparison (Same State) — Top 3")

rows=[]
for m in state_metros_df["metro_raw"]:
    pm=price_df[price_df["RegionName"]==m]
    if pm.empty: continue
    p=pd.DataFrame(pm.iloc[0,5:])
    p.index=pd.to_datetime(p.index)
    p.columns=["price"]

    prob=proxy_up_probability(p["price"])
    if prob is None: continue

    rows.append([m,f"{prob*100:.0f}%",friendly_label(prob),action_for_table(prob)])

if rows:
    comp_df=pd.DataFrame(rows,
        columns=["Metro","Price Up Chance","Outlook","What to Do"]
    ).sort_values("Price Up Chance",ascending=False).head(3)

    st.dataframe(comp_df,use_container_width=True)


# =================================================
# PRICE TREND
# =================================================
st.markdown("---")
st.subheader("📈 Price Trend + Risk Background")

fig=plt.figure(figsize=(14,6))
plt.plot(temp.index,temp["adj_price"],color="black",linewidth=2)

for i in range(len(temp)-1):
    color=("green" if temp["regime"].iloc[i]=="Supportive"
           else "gold" if temp["regime"].iloc[i]=="Unclear"
           else "red")
    plt.axvspan(temp.index[i],temp.index[i+1],color=color,alpha=0.15)

st.pyplot(fig)


# =================================================
# WEEKLY OUTLOOK CHART
# =================================================
st.markdown("---")
st.subheader("📊 Weekly Outlook (Last 12 Weeks)")

recent=temp.tail(12)

fig2,ax=plt.subplots(figsize=(12,5))
ax.plot(recent.index,recent["prob_up"],marker="o",linewidth=2,color="black")
ax.axhline(0.65,linestyle="--",color="green",alpha=0.6)
ax.axhline(0.45,linestyle="--",color="red",alpha=0.6)
ax.set_ylim(0,1)

st.pyplot(fig2)
st.caption("Above 0.65 = supportive • Below 0.45 = risky")
