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
clean = [s.replace("_", " ").title() for s in feat_names[sorted_idx]]
ax_fi.set_yticklabels(clean)
ax_fi.set_xlabel("Importance (fraction of splits)")
ax_fi.set_title("Feature Importance")
plt.tight_layout()
out_path_fi = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoost_feature_importance.png"
out_path_fi.parent.mkdir(parents=True, exist_ok=True)
fig_fi.savefig(out_path_fi, dpi=180)
plt.close(fig_fi)

# -----------------------
# 14) Predict on test.csv (iterative day-by-day forecasting)
# -----------------------
train_raw = pd.read_csv(PROJECT_ROOT / "train.csv")
train_raw["date"] = pd.to_datetime(train_raw["date"])

test_raw = pd.read_csv(PROJECT_ROOT / "test.csv")
test_raw["date"] = pd.to_datetime(test_raw["date"])

truth_df = pd.read_csv(PROJECT_ROOT / "truth.csv")

shared = ["date", "store", "is_state_holiday", "is_school_holiday",
          "is_special_day", "temperature_max", "temperature_min",
          "temperature_mean", "sunshine_sum", "precipitation_sum"]

comb = pd.concat([
    train_raw[shared + ["sales"]],
    test_raw[shared].assign(sales=np.nan),
], ignore_index=True).sort_values(["store", "date"]).reset_index(drop=True)

feature_names = list(X.columns)

_stores = comb["store"].unique()
_s_idx   = {s: comb.loc[comb["store"] == s].index.to_numpy() for s in _stores}
_s_dates = {s: comb.loc[_s_idx[s], "date"].values for s in _stores}

non_ohe = {"store_time_index"}
_ohe = {}
for cat in ["store", "is_state_holiday", "is_school_holiday", "is_special_day"]:
    prefix = cat + "_"
    plen = len(prefix)
    _ohe[cat] = [(c, c[plen:]) for c in feature_names
                 if c.startswith(prefix) and c not in non_ohe]

test_dates_sorted = sorted(test_raw["date"].unique())
preds_map = {}

for di, target_date in enumerate(test_dates_sorted):
    day_test = test_raw[test_raw["date"] == target_date]
    ts = pd.Timestamp(target_date)
    dow, mon, woy = ts.dayofweek, ts.month, ts.isocalendar()[1]
    wknd = int(dow >= 5)

    batch_feats, batch_rids, batch_cidxs = [], [], []

    for _, row in day_test.iterrows():
        store = row["store"]
        sidx = _s_idx[store]
        pos = np.searchsorted(_s_dates[store], np.datetime64(target_date))
        sales = comb.loc[sidx, "sales"].values

        feat = {
            "temperature_max": row["temperature_max"],
            "temperature_min": row["temperature_min"],
            "temperature_mean": row["temperature_mean"],
            "sunshine_sum": row["sunshine_sum"],
            "precipitation_sum": row["precipitation_sum"],
            "sales_lag_1":  sales[pos - 1]  if pos >= 1  else np.nan,
            "sales_lag_7":  sales[pos - 7]  if pos >= 7  else np.nan,
            "sales_lag_14": sales[pos - 14] if pos >= 14 else np.nan,
            "sales_lag_28": sales[pos - 28] if pos >= 28 else np.nan,
            "rolling_mean_7":  np.mean(sales[pos-7:pos])  if pos >= 7  else np.nan,
            "rolling_mean_14": np.mean(sales[pos-14:pos]) if pos >= 14 else np.nan,
            "rolling_mean_28": np.mean(sales[pos-28:pos]) if pos >= 28 else np.nan,
            "day_of_week": dow, "month": mon, "week_of_year": woy,
            "is_weekend": wknd, "store_time_index": pos,
        }

        for cat, pairs in _ohe.items():
            val = store if cat == "store" else row[cat]
            for col_name, suffix in pairs:
                feat[col_name] = int(suffix == val)

        batch_feats.append(feat)
        batch_rids.append(row["row_id"])
        batch_cidxs.append(sidx[pos])

    X_day = pd.DataFrame(batch_feats)[feature_names]
    for bc in X_day.select_dtypes("bool").columns:
        X_day[bc] = X_day[bc].astype(int)
    day_preds = model.predict(X_day)

    for i, (rid, cidx) in enumerate(zip(batch_rids, batch_cidxs)):
        preds_map[rid] = day_preds[i]
        comb.at[cidx, "sales"] = day_preds[i]

    if (di + 1) % 50 == 0 or di == len(test_dates_sorted) - 1:
        print(f"  Predicted {di+1}/{len(test_dates_sorted)} test dates")

# -----------------------
# 15) Evaluate against truth.csv
# -----------------------
test_results = (
    pd.DataFrame({"row_id": list(preds_map.keys()),
                   "predicted": list(preds_map.values())})
    .merge(truth_df, on="row_id")
    .rename(columns={"sales": "actual"})
    .merge(test_raw[["row_id", "date", "store"]], on="row_id")
)
test_results["date"] = pd.to_datetime(test_results["date"])

test_mae  = mean_absolute_error(test_results["actual"], test_results["predicted"])
test_rmse = np.sqrt(mean_squared_error(test_results["actual"], test_results["predicted"]))
test_r2   = r2_score(test_results["actual"], test_results["predicted"])

print(f"\nTest MAE:  {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R²:   {test_r2:.4f}")

# -----------------------
# 16) Test-set plots: Actual vs Predicted
# -----------------------
test_sorted = test_results.sort_values("date")
x_t  = test_sorted["date"].values
y_a  = test_sorted["actual"].values
y_p  = test_sorted["predicted"].values

fig_t, ax_t = plt.subplots(facecolor="white")
ax_t.plot(x_t, y_a, color=COLOR_ACTUAL,    label="Actual (truth)", linewidth=1, alpha=0.8)
ax_t.plot(x_t, y_p, color=COLOR_PREDICTED, label="Predicted",      linewidth=1, alpha=0.8)
ax_t.set_title("Test Set: Actual vs Predicted")
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
out_test = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoost_test_pred.png"
fig_t.savefig(out_test, dpi=180)
plt.close(fig_t)

# Single-store test plot
test_stores = test_results["store"].unique()
if len(test_stores) > 0:
    chosen = test_stores[0]
    mask_st = test_results["store"] == chosen
    st = test_results.loc[mask_st].sort_values("date")

    st_mae  = mean_absolute_error(st["actual"], st["predicted"])
    st_rmse = np.sqrt(mean_squared_error(st["actual"], st["predicted"]))

    fig_ts, ax_ts = plt.subplots(facecolor="white")
    ax_ts.plot(st["date"].values, st["actual"].values,
               color=COLOR_ACTUAL, label="Actual (truth)", linewidth=1, alpha=0.8)
    ax_ts.plot(st["date"].values, st["predicted"].values,
               color=COLOR_PREDICTED, label="Predicted", linewidth=1, alpha=0.8)
    ax_ts.set_title(f"Test Set: Actual vs Predicted ({chosen})")
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
    out_test_st = PROJECT_ROOT / "docs" / "assets" / "plots" / "XGBoost_test_pred_store.png"
    fig_ts.savefig(out_test_st, dpi=180)
    plt.close(fig_ts)
    print("Saved:", out_test, "and", out_test_st)
else:
    print("Saved:", out_test)

