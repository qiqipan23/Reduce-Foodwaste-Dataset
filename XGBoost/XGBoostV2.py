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
# V2 — Hybrid forecasting (no error compounding)
#
# Key differences from V1:
#   1. Per-store time-based split: every store has data in both train and val
#      so validation tests temporal generalisation per store.
#   2. Lag / rolling features use ACTUAL training sales only.  Where the
#      lookback window falls into the val / test period the lag is NaN;
#      XGBoost handles this natively.  No iterative prediction = no error
#      compounding.
#   3. Static historical aggregates (per-store, per-day-of-week, etc.) give
#      the model a baseline to fall back on when lags are unavailable.
#   4. Finer seasonal resolution (week-of-year).
# ===========================================================================

CAT_COLS = ["store", "is_state_holiday", "is_school_holiday", "is_special_day"]


def add_calendar_features(frame):
    frame["day_of_week"]  = frame["date"].dt.dayofweek
    frame["month"]        = frame["date"].dt.month
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["is_weekend"]   = (frame["day_of_week"] >= 5).astype(int)
    frame["day_of_month"] = frame["date"].dt.day
    frame["day_of_year"]  = frame["date"].dt.dayofyear
    return frame


def compute_historical_stats(source):
    return {
        "store_mean":       source.groupby("store")["sales"].mean().to_dict(),
        "store_std":        source.groupby("store")["sales"].std().to_dict(),
        "store_median":     source.groupby("store")["sales"].median().to_dict(),
        "store_dow_mean":   source.groupby(["store", "day_of_week"])["sales"].mean().to_dict(),
        "store_month_mean": source.groupby(["store", "month"])["sales"].mean().to_dict(),
        "store_woy_mean":   source.groupby(["store", "week_of_year"])["sales"].mean().to_dict(),
        "dow_mean":         source.groupby("day_of_week")["sales"].mean().to_dict(),
        "month_mean":       source.groupby("month")["sales"].mean().to_dict(),
        "woy_mean":         source.groupby("week_of_year")["sales"].mean().to_dict(),
    }


def apply_historical_features(target, stats):
    target["store_mean"]   = target["store"].map(stats["store_mean"])
    target["store_std"]    = target["store"].map(stats["store_std"])
    target["store_median"] = target["store"].map(stats["store_median"])

    keys = list(zip(target["store"], target["day_of_week"]))
    target["store_dow_mean"] = [stats["store_dow_mean"].get(k, np.nan) for k in keys]

    keys = list(zip(target["store"], target["month"]))
    target["store_month_mean"] = [stats["store_month_mean"].get(k, np.nan) for k in keys]

    keys = list(zip(target["store"], target["week_of_year"]))
    target["store_woy_mean"] = [stats["store_woy_mean"].get(k, np.nan) for k in keys]

    target["dow_mean"]   = target["day_of_week"].map(stats["dow_mean"])
    target["month_mean"] = target["month"].map(stats["month_mean"])
    target["woy_mean"]   = target["week_of_year"].map(stats["woy_mean"])
    return target


def add_lag_features(frame, sales_col="sales"):
    g = frame.groupby("store")[sales_col]
    frame["lag_7"]   = g.shift(7)
    frame["lag_14"]  = g.shift(14)
    frame["lag_28"]  = g.shift(28)
    frame["roll_7"]  = g.shift(1).rolling(7).mean()
    frame["roll_14"] = g.shift(1).rolling(14).mean()
    frame["roll_28"] = g.shift(1).rolling(28).mean()
    return frame


# =====================  TRAINING  =====================

# -----------------------
# 1) Load + parse + sort
# -----------------------
df = pd.read_csv(PROJECT_ROOT / "train.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["store", "date"]).reset_index(drop=True)
df = add_calendar_features(df)

# -----------------------
# 2) Per-store time-based split (80/20 per store → all stores in both sets)
# -----------------------
train_idx, val_idx = [], []
for store in df["store"].unique():
    s = df.loc[df["store"] == store].sort_values("date").index
    n = int(len(s) * 0.8)
    train_idx.extend(s[:n])
    val_idx.extend(s[n:])

is_train = df.index.isin(train_idx)
is_val   = df.index.isin(val_idx)

print(f"Train: {is_train.sum()} rows  |  Val: {is_val.sum()} rows  "
      f"(per-store 80/20 time split)")

# -----------------------
# 3) Historical stats from training portion only
# -----------------------
hist_stats = compute_historical_stats(df[is_train])
df = apply_historical_features(df, hist_stats)

# -----------------------
# 4) Lag features — val sales masked so lags only use actual training data
# -----------------------
df["_masked_sales"] = df["sales"].copy()
df.loc[is_val, "_masked_sales"] = np.nan
df = add_lag_features(df, sales_col="_masked_sales")
df.drop(columns=["_masked_sales"], inplace=True)

# Feature masking: randomly set ~80 % of training lag values to NaN so the
# model learns to handle the missing-lag regime it will face during val/test.
rng = np.random.RandomState(42)
lag_cols = ["lag_7", "lag_14", "lag_28", "roll_7", "roll_14", "roll_28"]
for col in lag_cols:
    avail = is_train & df[col].notna()
    n = avail.sum()
    drop_idx = rng.choice(df.index[avail], size=int(n * 0.8), replace=False)
    df.loc[drop_idx, col] = np.nan

# -----------------------
# 5) One-hot encode + build X / y
# -----------------------
dates_all = df["date"].copy()
df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)

drop_cols = ["date", "unsold", "ordered", "sales"]
X = df_enc.drop(columns=drop_cols, errors="ignore")
y = df_enc["sales"]

for bc in X.select_dtypes("bool").columns:
    X[bc] = X[bc].astype(int)

feature_names = list(X.columns)

X_train, y_train = X[is_train], y[is_train]
X_val,   y_val   = X[is_val],   y[is_val]

# -----------------------
# 6) Train
# -----------------------
model = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=5,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="rmse",
    early_stopping_rounds=500,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# -----------------------
# 7) Evaluate
# -----------------------
val_pred = model.predict(X_val)

val_mae  = mean_absolute_error(y_val, val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_r2   = r2_score(y_val, val_pred)

print(f"\nValidation MAE:  {val_mae:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R2:   {val_r2:.4f}")
print("Best iteration:", getattr(model, "best_iteration", None))

# -----------------------
# 8) Validation plots
# -----------------------
vd = dates_all.loc[y_val.index].values
si = np.argsort(vd)

fig, ax = plt.subplots(facecolor="white")
ax.plot(vd[si], y_val.values[si],  color=COLOR_ACTUAL,    label="Actual",    linewidth=1, alpha=0.8)
ax.plot(vd[si], val_pred[si],      color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
ax.set_title("V2 Validation: Actual vs Predicted")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend(loc="upper right")
ax.text(0.02, 0.95, f"RMSE: {val_rmse:.4f}\nMAE:  {val_mae:.4f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
fig.autofmt_xdate()
ax.set_facecolor("white")
ax.grid(False)
plt.tight_layout()
out_val = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV2_pred.png"
out_val.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_val, dpi=180)
plt.close(fig)

# Validation — single store
store_oh = [c for c in feature_names if c.startswith("store_store_")]
store_col = None
for c in store_oh:
    if (X_val[c] == 1).any():
        store_col = c
        break
if store_col is not None:
    sname = store_col.replace("store_", "", 1)
    ms = (X_val[store_col] == 1).values
    vd_s, si_s = vd[ms], np.argsort(vd[ms])

    s_mae  = mean_absolute_error(y_val.values[ms], val_pred[ms])
    s_rmse = np.sqrt(mean_squared_error(y_val.values[ms], val_pred[ms]))

    fig_s, ax_s = plt.subplots(facecolor="white")
    ax_s.plot(vd_s[si_s], y_val.values[ms][si_s], color=COLOR_ACTUAL,    label="Actual",    linewidth=1, alpha=0.8)
    ax_s.plot(vd_s[si_s], val_pred[ms][si_s],     color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
    ax_s.set_title(f"V2 Validation: Actual vs Predicted ({sname})")
    ax_s.set_xlabel("Date")
    ax_s.set_ylabel("Sales")
    ax_s.legend(loc="upper right")
    ax_s.text(0.02, 0.95, f"RMSE: {s_rmse:.4f}\nMAE:  {s_mae:.4f}",
              transform=ax_s.transAxes, fontsize=10, verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_s.xaxis.set_major_locator(mdates.MonthLocator())
    fig_s.autofmt_xdate()
    ax_s.set_facecolor("white")
    ax_s.grid(False)
    plt.tight_layout()
    out_val_s = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV2_pred_store.png"
    fig_s.savefig(out_val_s, dpi=180)
    plt.close(fig_s)
    print("Saved:", out_val, "and", out_val_s)
else:
    print("Saved:", out_val)

# -----------------------
# 9) Feature importance
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
ax_fi.set_title("V2 Feature Importance")
plt.tight_layout()
out_fi = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV2_feature_importance.png"
fig_fi.savefig(out_fi, dpi=180)
plt.close(fig_fi)

# =====================  TEST PREDICTION  =====================

# -----------------------
# 10) Build test features (no iterative prediction)
# -----------------------
test_raw = pd.read_csv(PROJECT_ROOT / "test.csv")
test_raw["date"] = pd.to_datetime(test_raw["date"])
truth_df = pd.read_csv(PROJECT_ROOT / "truth.csv")

# Reload clean training data
train_full = pd.read_csv(PROJECT_ROOT / "train.csv")
train_full["date"] = pd.to_datetime(train_full["date"])
train_full = train_full.sort_values(["store", "date"]).reset_index(drop=True)
train_full = add_calendar_features(train_full)

test_cal = add_calendar_features(test_raw.copy())

shared = ["date", "store", "is_state_holiday", "is_school_holiday",
          "is_special_day", "temperature_max", "temperature_min",
          "temperature_mean", "sunshine_sum", "precipitation_sum",
          "day_of_week", "month", "week_of_year", "is_weekend",
          "day_of_month", "day_of_year"]

# Combine train (actual sales) + test (NaN) for lag computation
combined = pd.concat([
    train_full[shared + ["sales"]],
    test_cal[shared].assign(sales=np.nan),
], ignore_index=True).sort_values(["store", "date"]).reset_index(drop=True)

combined = add_lag_features(combined, sales_col="sales")

# Historical stats from full training data
full_stats = compute_historical_stats(train_full)
combined = apply_historical_features(combined, full_stats)

# Extract test rows (sorted by store, date)
test_start = test_cal["date"].min()
test_feat = combined[combined["date"] >= test_start].copy()

test_feat = pd.get_dummies(test_feat, columns=CAT_COLS, drop_first=False)
for col in feature_names:
    if col not in test_feat.columns:
        test_feat[col] = 0
X_test = test_feat[feature_names].copy()
for bc in X_test.select_dtypes("bool").columns:
    X_test[bc] = X_test[bc].astype(int)

test_pred = model.predict(X_test)

# Match predictions to row_ids via consistent (store, date) ordering
test_raw_sorted = test_raw.sort_values(["store", "date"]).reset_index(drop=True)
test_results = test_raw_sorted[["row_id", "date", "store"]].copy()
test_results["predicted"] = test_pred
test_results = test_results.merge(truth_df, on="row_id").rename(columns={"sales": "actual"})
test_results["date"] = pd.to_datetime(test_results["date"])

# -----------------------
# 11) Test metrics
# -----------------------
test_mae  = mean_absolute_error(test_results["actual"], test_results["predicted"])
test_rmse = np.sqrt(mean_squared_error(test_results["actual"], test_results["predicted"]))
test_r2   = r2_score(test_results["actual"], test_results["predicted"])

print(f"\nTest MAE:  {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R2:   {test_r2:.4f}")

# -----------------------
# 12) Test-set plots
# -----------------------
ts = test_results.sort_values("date")

fig_t, ax_t = plt.subplots(facecolor="white")
ax_t.plot(ts["date"].values, ts["actual"].values,
          color=COLOR_ACTUAL, label="Actual (truth)", linewidth=1, alpha=0.8)
ax_t.plot(ts["date"].values, ts["predicted"].values,
          color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
ax_t.set_title("V2 Test Set: Actual vs Predicted")
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
out_test = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV2_test_pred.png"
fig_t.savefig(out_test, dpi=180)
plt.close(fig_t)

# Single-store test plot
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
    ax_ts.set_title(f"V2 Test Set: Actual vs Predicted ({chosen})")
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
    out_test_st = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoostV2_test_pred_store.png"
    fig_ts.savefig(out_test_st, dpi=180)
    plt.close(fig_ts)
    print("Saved:", out_test, "and", out_test_st)
else:
    print("Saved:", out_test)
