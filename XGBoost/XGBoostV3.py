import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_analysis import COLOR_ACTUAL, COLOR_PREDICTED

# ===========================================================================
# V3 — Rolling one-step-ahead forecast
#
# Realistic evaluation: when predicting day D, the model has access to
# actual sales through day D-1 (from truth.csv for the test period).
# This matches real-world use where yesterday's sales are always known.
#
# Key differences from V1 and V2:
#   V1: iterative test prediction — predicted values fed back as lags
#       (error compounding).
#   V2: direct prediction — lags mostly NaN during test, model falls back
#       to static features (low variance).
#   V3: rolling test prediction — actual truth values used as lags, just
#       like a real deployment where you observe sales each evening and
#       forecast the next day.  No error compounding, full lag signal.
# ===========================================================================

CAT_COLS = ["store", "is_state_holiday", "is_school_holiday", "is_special_day"]


def add_lag_features(frame):
    g = frame.groupby("store")["sales"]
    frame["sales_lag_1"]  = g.shift(1)
    frame["sales_lag_7"]  = g.shift(7)
    frame["sales_lag_14"] = g.shift(14)
    frame["sales_lag_28"] = g.shift(28)
    frame["rolling_mean_7"]  = g.shift(1).rolling(7).mean()
    frame["rolling_mean_14"] = g.shift(1).rolling(14).mean()
    frame["rolling_mean_28"] = g.shift(1).rolling(28).mean()
    return frame


def add_calendar_features(frame):
    frame["day_of_week"]  = frame["date"].dt.dayofweek
    frame["month"]        = frame["date"].dt.month
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["is_weekend"]   = (frame["day_of_week"] >= 5).astype(int)
    frame["store_time_index"] = frame.groupby("store").cumcount()
    return frame


# =====================  TRAINING  =====================

# -----------------------
# 1) Load + parse + sort
# -----------------------
df = pd.read_csv(PROJECT_ROOT / "train.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["store", "date"]).reset_index(drop=True)

# -----------------------
# 2) Lag + rolling features (all from actual sales)
# -----------------------
df = add_lag_features(df)

df = df.dropna(subset=[
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
]).copy()

# -----------------------
# 3) Calendar features
# -----------------------
df = add_calendar_features(df)

# -----------------------
# 4) One-hot encode
# -----------------------
df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)

# -----------------------
# 5) Build X / y
# -----------------------
dates = df["date"].copy()
df = df.drop(columns=["date", "unsold", "ordered"], errors="ignore")

X = df.drop(columns=["sales"])
y = df["sales"]

for bc in X.select_dtypes("bool").columns:
    X[bc] = X[bc].astype(int)

feature_names = list(X.columns)

# -----------------------
# 6) Per-store time split (80/20 — all stores in both sets)
# -----------------------
store_oh = [c for c in feature_names if c.startswith("store_store_")]

train_idx, val_idx = [], []
for sc in store_oh:
    sidx = X.loc[X[sc] == 1].index.tolist()
    n = int(len(sidx) * 0.8)
    train_idx.extend(sidx[:n])
    val_idx.extend(sidx[n:])

is_train = X.index.isin(train_idx)
is_val   = X.index.isin(val_idx)

X_train, y_train = X[is_train], y[is_train]
X_val,   y_val   = X[is_val],   y[is_val]

print(f"Train: {is_train.sum()} rows  |  Val: {is_val.sum()} rows  "
      f"(per-store 80/20 time split)")

# -----------------------
# 7) Train
# -----------------------
model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="rmse",
    early_stopping_rounds=200,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# -----------------------
# 8) Evaluate
# -----------------------
val_pred = model.predict(X_val)

val_mae  = mean_absolute_error(y_val, val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_r2   = r2_score(y_val, val_pred)

print(f"\nValidation (one-step-ahead) MAE:  {val_mae:.4f}")
print(f"Validation (one-step-ahead) RMSE: {val_rmse:.4f}")
print(f"Validation (one-step-ahead) R2:   {val_r2:.4f}")
print("Best iteration:", getattr(model, "best_iteration", None))

# -----------------------
# 9) Training-period plot: full 2021-2023 actual vs predicted
# -----------------------
all_pred = model.predict(X)
all_dates = dates.values
si = np.argsort(all_dates)

train_mae_full = mean_absolute_error(y, all_pred)
train_rmse_full = np.sqrt(mean_squared_error(y, all_pred))

fig_tr, ax_tr = plt.subplots(figsize=(14, 5), facecolor="white")
ax_tr.plot(all_dates[si], y.values[si],    color=COLOR_ACTUAL,    label="Actual",    linewidth=0.8, alpha=0.7)
ax_tr.plot(all_dates[si], all_pred[si],    color=COLOR_PREDICTED, label="Predicted", linewidth=0.8, alpha=0.7)
ax_tr.set_title("V3 Training Period: Actual vs Predicted (2021-2023)")
ax_tr.set_xlabel("Date")
ax_tr.set_ylabel("Sales")
ax_tr.legend(loc="upper right")
ax_tr.text(0.02, 0.95,
           f"RMSE: {train_rmse_full:.4f}\nMAE:  {train_mae_full:.4f}",
           transform=ax_tr.transAxes, fontsize=10, verticalalignment="top",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
ax_tr.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_tr.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
fig_tr.autofmt_xdate()
ax_tr.set_facecolor("white")
ax_tr.grid(False)
plt.tight_layout()
out_train = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV3_train_pred.png"
out_train.parent.mkdir(parents=True, exist_ok=True)
fig_tr.savefig(out_train, dpi=180)
plt.close(fig_tr)

# Training-period — single store
store_col = None
for c in store_oh:
    if (X[c] == 1).any():
        store_col = c
        break
if store_col is not None:
    sname = store_col.replace("store_", "", 1)
    ms = (X[store_col] == 1).values
    sd = all_dates[ms]
    ss = np.argsort(sd)

    fig_trs, ax_trs = plt.subplots(figsize=(14, 5), facecolor="white")
    ax_trs.plot(sd[ss], y.values[ms][ss],    color=COLOR_ACTUAL,    label="Actual",    linewidth=1, alpha=0.8)
    ax_trs.plot(sd[ss], all_pred[ms][ss],    color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
    ax_trs.set_title(f"V3 Training Period: Actual vs Predicted ({sname})")
    ax_trs.set_xlabel("Date")
    ax_trs.set_ylabel("Sales")
    ax_trs.legend(loc="upper right")
    ax_trs.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_trs.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig_trs.autofmt_xdate()
    ax_trs.set_facecolor("white")
    ax_trs.grid(False)
    plt.tight_layout()
    out_train_s = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV3_train_pred_store.png"
    fig_trs.savefig(out_train_s, dpi=180)
    plt.close(fig_trs)
    print("Saved:", out_train, "and", out_train_s)
else:
    print("Saved:", out_train)

# -----------------------
# 10) Feature importance
# -----------------------
importances = model.feature_importances_
fn = np.array(feature_names)
top = np.argsort(importances)[-15:]
fig_fi, ax_fi = plt.subplots()
ax_fi.barh(range(len(top)), importances[top])
ax_fi.set_yticks(range(len(top)))
clean = [s.replace("_", " ").title() for s in fn[top]]
ax_fi.set_yticklabels(clean)
ax_fi.set_xlabel("Importance (fraction of splits)")
ax_fi.set_title("V3 Feature Importance")
plt.tight_layout()
out_fi = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV3_feature_importance.png"
fig_fi.savefig(out_fi, dpi=180)
plt.close(fig_fi)

# =====================  TEST — ROLLING ONE-STEP-AHEAD  =====================

# -----------------------
# 11) Build test features using actual truth values as lags
# -----------------------
train_raw = pd.read_csv(PROJECT_ROOT / "train.csv")
train_raw["date"] = pd.to_datetime(train_raw["date"])

test_raw = pd.read_csv(PROJECT_ROOT / "test.csv")
test_raw["date"] = pd.to_datetime(test_raw["date"])

truth_df = pd.read_csv(PROJECT_ROOT / "truth.csv")

# Attach actual sales to test rows
test_truth = test_raw.merge(truth_df, on="row_id")

shared = ["date", "store", "is_state_holiday", "is_school_holiday",
          "is_special_day", "temperature_max", "temperature_min",
          "temperature_mean", "sunshine_sum", "precipitation_sum", "sales"]

combined = pd.concat([
    train_raw[shared],
    test_truth[shared],
], ignore_index=True).sort_values(["store", "date"]).reset_index(drop=True)

# Lag features — every lag references ACTUAL sales (train or truth)
combined = add_lag_features(combined)
combined = add_calendar_features(combined)

# Extract test rows
test_start = test_raw["date"].min()
test_feat = combined[combined["date"] >= test_start].copy()

# One-hot encode + align columns
test_feat = pd.get_dummies(test_feat, columns=CAT_COLS, drop_first=False)
test_feat = test_feat.drop(columns=["date", "unsold", "ordered", "sales"],
                           errors="ignore")
for col in feature_names:
    if col not in test_feat.columns:
        test_feat[col] = 0
X_test = test_feat[feature_names].copy()
for bc in X_test.select_dtypes("bool").columns:
    X_test[bc] = X_test[bc].astype(int)

test_pred = model.predict(X_test)

# Match predictions to row_ids (combined was sorted by store, date)
test_raw_sorted = test_raw.sort_values(["store", "date"]).reset_index(drop=True)
test_results = test_raw_sorted[["row_id", "date", "store"]].copy()
test_results["predicted"] = test_pred
test_results = test_results.merge(truth_df, on="row_id").rename(
    columns={"sales": "actual"})
test_results["date"] = pd.to_datetime(test_results["date"])

# -----------------------
# 12) Test metrics
# -----------------------
test_mae  = mean_absolute_error(test_results["actual"], test_results["predicted"])
test_rmse = np.sqrt(mean_squared_error(test_results["actual"], test_results["predicted"]))
test_r2   = r2_score(test_results["actual"], test_results["predicted"])

print(f"\nTest (rolling one-step-ahead) MAE:  {test_mae:.4f}")
print(f"Test (rolling one-step-ahead) RMSE: {test_rmse:.4f}")
print(f"Test (rolling one-step-ahead) R2:   {test_r2:.4f}")

# -----------------------
# 13) Test plot — all stores
# -----------------------
ts = test_results.sort_values("date")

fig_t, ax_t = plt.subplots(facecolor="white")
ax_t.plot(ts["date"].values, ts["actual"].values,
          color=COLOR_ACTUAL, label="Actual (truth)", linewidth=1, alpha=0.8)
ax_t.plot(ts["date"].values, ts["predicted"].values,
          color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
ax_t.set_title("V3 Test Set: Actual vs Predicted (rolling one-step-ahead)")
ax_t.set_xlabel("Date")
ax_t.set_ylabel("Sales")
ax_t.legend(loc="upper right")
ax_t.text(0.02, 0.95, f"RMSE: {test_rmse:.4f}\nMAE:  {test_mae:.4f}",
          transform=ax_t.transAxes, fontsize=10, verticalalignment="top",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_t.xaxis.set_major_locator(mdates.MonthLocator())
fig_t.autofmt_xdate()
ax_t.set_facecolor("white")
ax_t.grid(False)
plt.tight_layout()
out_test = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV3_test_pred.png"
fig_t.savefig(out_test, dpi=180)
plt.close(fig_t)

# -----------------------
# 14) Test plot — single store
# -----------------------
test_stores = test_results["store"].unique()
if len(test_stores) > 0:
    chosen = test_stores[0]
    st = test_results.loc[test_results["store"] == chosen].sort_values("date")

    st_mae  = mean_absolute_error(st["actual"], st["predicted"])
    st_rmse = np.sqrt(mean_squared_error(st["actual"], st["predicted"]))

    fig_ts, ax_ts = plt.subplots(facecolor="white")
    ax_ts.plot(st["date"].values, st["actual"].values,
               color=COLOR_ACTUAL, label="Actual (truth)", linewidth=1, alpha=0.8)
    ax_ts.plot(st["date"].values, st["predicted"].values,
               color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
    ax_ts.set_title(f"V3 Test Set: Actual vs Predicted ({chosen})")
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Sales")
    ax_ts.legend(loc="upper right")
    ax_ts.text(0.02, 0.95, f"RMSE: {st_rmse:.4f}\nMAE:  {st_mae:.4f}",
               transform=ax_ts.transAxes, fontsize=10, verticalalignment="top",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    fig_ts.autofmt_xdate()
    ax_ts.set_facecolor("white")
    ax_ts.grid(False)
    plt.tight_layout()
    out_test_st = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV3_test_pred_store.png"
    fig_ts.savefig(out_test_st, dpi=180)
    plt.close(fig_ts)
    print("Saved:", out_test, "and", out_test_st)
else:
    print("Saved:", out_test)
