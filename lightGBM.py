import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lightgbm import LGBMRegressor


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def normalize_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    In this dataset, holiday columns are strings like 'normal_day'.
    Convert them to binary: 1 if NOT 'normal_day', else 0.
    """
    df = df.copy()
    for c in ["is_state_holiday", "is_school_holiday", "is_special_day"]:
        if c in df.columns:
            # Keep robust to case/spacing
            s = df[c].astype(str).str.strip().str.lower()
            df[c] = (s != "normal_day").astype(int)
    return df


def add_sales_lag_features(df: pd.DataFrame, group_col="store") -> pd.DataFrame:
    """
    Adds lag + rolling stats per store. Assumes 'sales' exists but can contain NaN for test rows.
    Uses shift(1) before rolling to avoid peeking at the current day.
    IMPORTANT: This function sorts rows; so do NOT split train/test by iloc after calling it.
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"]).reset_index(drop=True)

    g = df.groupby(group_col, sort=False)["sales"]

    df["sales_lag_1"] = g.shift(1)
    df["sales_lag_7"] = g.shift(7)
    df["sales_lag_14"] = g.shift(14)

    df["sales_rollmean_7"] = g.shift(1).rolling(window=7, min_periods=1).mean()
    df["sales_rollmean_14"] = g.shift(1).rolling(window=14, min_periods=1).mean()
    df["sales_rollmean_28"] = g.shift(1).rolling(window=28, min_periods=1).mean()

    df["sales_rollstd_7"] = g.shift(1).rolling(window=7, min_periods=2).std()
    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    # ----------------------------
    # 1) Load
    # ----------------------------
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # ----------------------------
    # 2) Feature prep (date + holiday flags)
    # ----------------------------
    train = add_date_features(train)
    test = add_date_features(test)

    train = normalize_holiday_flags(train)
    test = normalize_holiday_flags(test)

    train["sales"] = pd.to_numeric(train["sales"], errors="coerce")

    # ----------------------------
    # 3) Combine train+test to build lags
    #    CRITICAL: keep a flag so we can split AFTER sorting
    # ----------------------------
    train_for_lags = train.copy()
    train_for_lags["__is_train"] = 1

    test_for_lags = test.copy()
    test_for_lags["sales"] = np.nan
    test_for_lags["__is_train"] = 0

    combined = pd.concat([train_for_lags, test_for_lags], ignore_index=True, sort=False)

    # Drop columns not reliable/available (KEEP row_id for test scoring)
    for c in ["unsold", "ordered"]:
        if c in combined.columns:
            combined = combined.drop(columns=[c])

    combined = add_sales_lag_features(combined, group_col="store")

    # Split using the flag (NOT by iloc)
    train_feat = combined[combined["__is_train"] == 1].copy()
    test_feat = combined[combined["__is_train"] == 0].copy()

    # ----------------------------
    # 4) Drop train rows with missing target
    # ----------------------------
    before = len(train_feat)
    train_feat = train_feat[train_feat["sales"].notna()].copy()
    after = len(train_feat)
    if after != before:
        print(f"Dropped {before - after} train rows with NaN sales.")

    # ----------------------------
    # 5) Build X/y
    # ----------------------------
    y = train_feat["sales"].astype(float)

    # row_id is only for test scoring; __is_train is internal; date removed from features
    drop_train = ["sales", "date", "__is_train"]
    drop_test = ["sales", "date", "__is_train"]

    if "row_id" in train_feat.columns:
        drop_train.append("row_id")
    if "row_id" in test_feat.columns:
        test_row_id = test_feat["row_id"].copy()
        drop_test.append("row_id")
    else:
        test_row_id = None

    X = train_feat.drop(columns=drop_train)
    X_test = test_feat.drop(columns=drop_test)

    # ----------------------------
    # 6) Preprocess: numeric + categorical
    # ----------------------------
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    def make_model():
        return LGBMRegressor(
            objective="regression",
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=10,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    # ----------------------------
    # 7) Time-series CV (walk-forward)
    # ----------------------------
    # NOTE: train_feat is already sorted by store/date because add_sales_lag_features sorted it,
    # but for TimeSeriesSplit we want global time order:
    order = train_feat.sort_values("date").index.to_numpy()
    X_ord = X.loc[order].reset_index(drop=True)
    y_ord = y.loc[order].reset_index(drop=True)
    dates_ord = train_feat.loc[order, "date"].reset_index(drop=True)
    store_ord = train_feat.loc[order, "store"].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []

    oof_parts = []
    all_train_rmse, all_valid_rmse = [], []

    print("\n=== TimeSeriesSplit Evaluation (LightGBM + lags) ===")
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_ord), start=1):
        X_tr, X_va = X_ord.iloc[tr_idx], X_ord.iloc[va_idx]
        y_tr, y_va = y_ord.iloc[tr_idx], y_ord.iloc[va_idx]

        X_tr_p = preprocess.fit_transform(X_tr, y_tr)
        X_va_p = preprocess.transform(X_va)

        model = make_model()
        model.fit(
            X_tr_p, y_tr,
            eval_set=[(X_tr_p, y_tr), (X_va_p, y_va)],
            eval_names=["train", "valid"],
            eval_metric="rmse",
        )

        pred = model.predict(X_va_p)

        oof_parts.append(pd.DataFrame({
            "date": dates_ord.iloc[va_idx].values,
            "store": store_ord.iloc[va_idx].values,
            "y_true": y_va.values,
            "y_pred": pred,
            "fold": fold,
        }))

        mae_val = mean_absolute_error(y_va, pred)
        rmse_val = rmse(y_va, pred)
        r2_val = r2_score(y_va, pred)

        maes.append(mae_val)
        rmses.append(rmse_val)
        r2s.append(r2_val)

        evals = model.evals_result_
        all_train_rmse.append(np.array(evals["train"]["rmse"], dtype=float))
        all_valid_rmse.append(np.array(evals["valid"]["rmse"], dtype=float))

        va_min = dates_ord.iloc[va_idx].min().date()
        va_max = dates_ord.iloc[va_idx].max().date()
        print(
            f"Fold {fold}: MAE={mae_val:.4f}  RMSE={rmse_val:.4f}  R2={r2_val:.4f}  "
            f"ValidRange={va_min}..{va_max}  (train={len(tr_idx)}, valid={len(va_idx)})"
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof["date"] = pd.to_datetime(oof["date"])
    oof = oof.sort_values(["store", "date"]).reset_index(drop=True)
    oof.to_csv("oof_predictions.csv", index=False)
    print("Wrote oof_predictions.csv")

    print("\n=== CV Summary ===")
    print(f"MAE  mean={float(np.mean(maes)):.4f}  std={float(np.std(maes)):.4f}")
    print(f"RMSE mean={float(np.mean(rmses)):.4f}  std={float(np.std(rmses)):.4f}")
    print(f"R2   mean={float(np.mean(r2s)):.4f}  std={float(np.std(r2s)):.4f}")

    # ----------------------------
    # 8) Train on full train and predict test
    # ----------------------------
    X_all_p = preprocess.fit_transform(X_ord, y_ord)
    X_test_p = preprocess.transform(X_test)

    final_model = make_model()
    final_model.fit(X_all_p, y_ord)

    test_pred = final_model.predict(X_test_p)

    # Write predictions in truth.csv format: row_id,sales
    if test_row_id is None:
        raise ValueError("test.csv does not contain row_id, cannot score against truth.csv.")

    rid = pd.to_numeric(test_row_id, errors="coerce").astype("Int64")
    if rid.isna().any():
        raise ValueError(f"row_id contains {int(rid.isna().sum())} missing values even after processing.")

    pred_out = pd.DataFrame({
        "row_id": rid.values,
        "sales": test_pred.astype(float),
    }).sort_values("row_id")

    pred_out.to_csv("predictions_lgbm_test.csv", index=False)
    print("\nWrote predictions_lgbm_test.csv (row_id, sales)")

    # ----------------------------
    # 9) Test-set evaluation using truth.csv
    # ----------------------------
    truth = pd.read_csv("truth.csv")
    merged = truth.merge(pred_out, on="row_id", how="inner", suffixes=("_true", "_pred"))

    if len(merged) != len(truth):
        print(f"Warning: matched {len(merged)} / {len(truth)} rows. Check row_id handling.")

    y_true_test = merged["sales_true"].astype(float)
    y_pred_test = merged["sales_pred"].astype(float)

    print("\n=== Test Set Evaluation (truth.csv) ===")
    print(f"MAE  = {mean_absolute_error(y_true_test, y_pred_test):.4f}")
    print(f"RMSE = {rmse(y_true_test, y_pred_test):.4f}")
    print(f"R2   = {r2_score(y_true_test, y_pred_test):.4f}")


    # ----------------------------
    # 10) Visualize TEST performance (truth.csv)
    # ----------------------------
    # Load raw test to get date/store for grouping
    test_raw = pd.read_csv("test.csv")
    test_raw["date"] = pd.to_datetime(test_raw["date"])

    # pred_out already exists in your script (row_id, sales)
    # If not, load it:
    # pred_out = pd.read_csv("predictions_lgbm_test.csv")

    truth = pd.read_csv("truth.csv")

    # Merge: row_id -> truth + preds + test metadata
    m = truth.merge(pred_out, on="row_id", how="inner", suffixes=("_true", "_pred"))
    m = m.merge(test_raw[["row_id", "date", "store"]], on="row_id", how="left")

    m = m.rename(columns={"sales_true": "y_true", "sales_pred": "y_pred"})
    m["residual"] = m["y_pred"] - m["y_true"]
    m["abs_error"] = (m["residual"]).abs()

    m.to_csv("test_predictions_with_truth.csv", index=False)
    print("Wrote test_predictions_with_truth.csv")

    # (A) Scatter: predicted vs true
    plt.figure(figsize=(7, 7))
    plt.scatter(m["y_true"], m["y_pred"], alpha=0.25)
    mn = float(min(m["y_true"].min(), m["y_pred"].min()))
    mx = float(max(m["y_true"].max(), m["y_pred"].max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True (truth.csv)")
    plt.ylabel("Predicted")
    plt.title("Test set: Predicted vs True")
    plt.tight_layout()
    plt.savefig("test_scatter_pred_vs_true.png")
    plt.close()
    print("Saved plot: test_scatter_pred_vs_true.png")

    # (B) Residuals histogram
    plt.figure(figsize=(10, 5))
    plt.hist(m["residual"], bins=50)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title("Test set: Residual distribution")
    plt.tight_layout()
    plt.savefig("test_residual_hist.png")
    plt.close()
    print("Saved plot: test_residual_hist.png")

    # (C) Residuals vs true (heteroscedasticity / extremes)
    plt.figure(figsize=(7, 5))
    plt.scatter(m["y_true"], m["residual"], alpha=0.25)
    plt.axhline(0)
    plt.xlabel("True (truth.csv)")
    plt.ylabel("Residual (pred - true)")
    plt.title("Test set: Residuals vs True")
    plt.tight_layout()
    plt.savefig("test_residuals_vs_true.png")
    plt.close()
    print("Saved plot: test_residuals_vs_true.png")

    # (D) Error by store (mean absolute error)
    store_mae = (
        m.groupby("store", as_index=False)["abs_error"]
        .mean()
        .sort_values("abs_error", ascending=False)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(store_mae["store"], store_mae["abs_error"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Store")
    plt.ylabel("Mean Absolute Error")
    plt.title("Test set: MAE by store")
    plt.tight_layout()
    plt.savefig("test_mae_by_store.png")
    plt.close()
    print("Saved plot: test_mae_by_store.png")

if __name__ == "__main__":
    main()