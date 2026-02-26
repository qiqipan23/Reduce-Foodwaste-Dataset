import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Project root (parent of this script's directory), so paths work from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Import plot colours to match data_analysis.py / data_preview.ipynb
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_analysis import COLOR_ACTUAL, COLOR_PREDICTED

# -----------------------
# 1) Load + parse
# -----------------------
df = pd.read_csv(PROJECT_ROOT / "train.csv")
df["date"] = pd.to_datetime(df["date"])

# -----------------------
# 2) Sort (required for lags)
# -----------------------
df = df.sort_values(["store", "date"]).copy()

# -----------------------
# 3) Lag + rolling features (per store)
# -----------------------
g = df.groupby("store")["sales"]

df["sales_lag_1"]  = g.shift(1)
df["sales_lag_7"]  = g.shift(7)
df["sales_lag_14"] = g.shift(14)
df["sales_lag_28"] = g.shift(28)

df["rolling_mean_7"]  = g.shift(1).rolling(7).mean()
df["rolling_mean_14"] = g.shift(1).rolling(14).mean()
df["rolling_mean_28"] = g.shift(1).rolling(28).mean()

# Drop rows where lag/rolling isn't available yet
df = df.dropna(subset=[
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"
]).copy()

# -----------------------
# 4) Calendar features
# -----------------------
df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon ... 6=Sun
df["month"] = df["date"].dt.month
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Optional: per-store time index (captures drift)
df["store_time_index"] = df.groupby("store").cumcount()

# -----------------------
# 5) One-hot encode categoricals
# -----------------------
cat_cols = ["store", "is_state_holiday", "is_school_holiday", "is_special_day"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# -----------------------
# 6) Drop columns not used as features
# -----------------------
dates = df["date"].copy()
df = df.drop(columns=["date", "unsold", "ordered"], errors="ignore")

# -----------------------
# 7) Build X and y
# -----------------------
X = df.drop(columns=["sales"])
y = df["sales"]

# Convert bool columns to 0/1 ints (tidy)
bool_cols = X.select_dtypes("bool").columns
X[bool_cols] = X[bool_cols].astype(int)

# -----------------------
# 8) Time-ordered train/val split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------
# 9) Train model (latest xgboost style)
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
    early_stopping_rounds=200
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# -----------------------
# 10) Evaluate
# -----------------------
val_pred = model.predict(X_val)

print("Validation MAE:", mean_absolute_error(y_val, val_pred))
print("Validation RMSE:", np.sqrt(mean_squared_error(y_val, val_pred)))
print("Validation R2:", r2_score(y_val, val_pred))
print("Best iteration:", getattr(model, "best_iteration", None))

# -----------------------
# 11) Strong baselines to sanity-check
# -----------------------
naive_1 = X_val["sales_lag_1"]
naive_7 = X_val["sales_lag_7"]
naive_28 = X_val["sales_lag_28"]

m1 = naive_1.notna()
m7 = naive_7.notna()
m28 = naive_28.notna()

print("Naive(lag1)  R2:", r2_score(y_val[m1], naive_1[m1]))
print("Naive(lag7)  R2:", r2_score(y_val[m7], naive_7[m7]))
print("Naive(lag28) R2:", r2_score(y_val[m28], naive_28[m28]))

# -----------------------
# 12) Out-of-fold: Actual vs Predicted (validation only) — all stores + store_0
# -----------------------
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Use explicit arrays for x so matplotlib doesn't use Series index (which can give wrong axis)
val_dates = dates.loc[y_val.index]
sort_idx = np.argsort(val_dates.values)
x_dates = val_dates.iloc[sort_idx].values
y_val_sorted = y_val.iloc[sort_idx].values
val_pred_sorted = val_pred[sort_idx]

fig, ax = plt.subplots(facecolor="white")
ax.plot(x_dates, y_val_sorted, color=COLOR_ACTUAL, label="Actual", linewidth=1, alpha=0.8)
ax.plot(x_dates, val_pred_sorted, color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
ax.set_title("Out-of-fold: Actual vs Predicted (validation only)")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
fig.autofmt_xdate()
ax.set_facecolor("white")
ax.grid(False)
plt.tight_layout()
out_path = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoost_pred.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=180)
plt.close(fig)

# Same plot for a single store (pick first store that has validation rows; data has store_1, store_4, store_5, store_7, store_8 — no store_0)
store_oh_cols = [c for c in X_val.columns if c.startswith("store_store_")]
store_col = None
for c in store_oh_cols:
    if (X_val[c] == 1).any():
        store_col = c
        break
if store_col is not None:
    store_name = store_col.replace("store_", "", 1)  # store_store_1 -> store_1
    mask_s0 = (X_val[store_col] == 1).values
    val_dates_s0 = dates.loc[y_val.index[mask_s0]]
    sort_idx_s0 = np.argsort(val_dates_s0.values)
    x_dates_s0 = val_dates_s0.iloc[sort_idx_s0].values
    y_val_s0_sorted = y_val.iloc[mask_s0].iloc[sort_idx_s0].values
    val_pred_s0_sorted = val_pred[mask_s0][sort_idx_s0]

    fig_s0, ax_s0 = plt.subplots(facecolor="white")
    ax_s0.plot(x_dates_s0, y_val_s0_sorted, color=COLOR_ACTUAL, label="Actual", linewidth=1, alpha=0.8)
    ax_s0.plot(x_dates_s0, val_pred_s0_sorted, color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
    ax_s0.set_title(f"Out-of-fold: Actual vs Predicted (validation only, {store_name})")
    ax_s0.set_xlabel("Date")
    ax_s0.set_ylabel("Sales")
    ax_s0.legend(loc="upper right")
    ax_s0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_s0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig_s0.autofmt_xdate()
    ax_s0.set_facecolor("white")
    ax_s0.grid(False)
    plt.tight_layout()
    out_path_s0 = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoost_pred_store.png"
    out_path_s0.parent.mkdir(parents=True, exist_ok=True)
    fig_s0.savefig(out_path_s0, dpi=180)
    plt.close(fig_s0)
    print("Saved:", out_path, "and", out_path_s0)
else:
    print("Saved:", out_path, "(no store subset: no store columns with validation data)")

# -----------------------
# 13) Feature importance (saved only, no window)
# -----------------------
importances = model.feature_importances_
feat_names = X.columns
sorted_idx = np.argsort(importances)[-15:]
fig_fi, ax_fi = plt.subplots()
ax_fi.barh(range(len(sorted_idx)), importances[sorted_idx])
ax_fi.set_yticks(range(len(sorted_idx)))
ax_fi.set_yticklabels(feat_names[sorted_idx])
out_path_fi = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoost_feature_importance.png"
out_path_fi.parent.mkdir(parents=True, exist_ok=True)
fig_fi.savefig(out_path_fi, dpi=180)
plt.close(fig_fi)



