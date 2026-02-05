# ----------------------------
# ✅ FIXED: Weekly + Monthly signals (3 month horizon)
# ----------------------------
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

# ✅ FIX: Handle empty walk-forward safely
if len(all_probs_3) == 0:
    # fallback: train once on full sample
    rf_fallback = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf_fallback.fit(temp3[predictors], temp3["target"])
    fallback_prob = rf_fallback.predict_proba(
        temp3[predictors].tail(1)
    )[:, 1]

    prob_data = temp3.tail(1).copy()
    prob_data["prob_up"] = fallback_prob
else:
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
