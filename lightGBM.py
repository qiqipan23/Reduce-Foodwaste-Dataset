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


def add_sales_lag_features(df: pd.DataFrame, group_col="store") -> pd.DataFrame:
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


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error in %.
    Ignores y_true values that are 0 (or extremely close to 0) to avoid exploding percentages.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def main():
    # ----------------------------
    # 1) Load
    # ----------------------------
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # ----------------------------
    # 2) Date features
    # ----------------------------
    train = add_date_features(train)
    test = add_date_features(test)

    train["sales"] = pd.to_numeric(train["sales"], errors="coerce")

    # Ensure holiday flags are numeric 0/1 (keeps them in numeric pipeline)
    for c in ["is_state_holiday", "is_school_holiday", "is_special_day"]:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce").fillna(0).astype(int)
        if c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0).astype(int)

    # ----------------------------
    # 3) Combine train+test to build lags
    # ----------------------------
    test_for_lags = test.copy()
    test_for_lags["sales"] = np.nan

    combined = pd.concat([train, test_for_lags], ignore_index=True, sort=False)

    # Drop columns not reliable/available
    for c in ["unsold", "ordered", "row_id"]:
        if c in combined.columns:
            combined = combined.drop(columns=[c])

    combined = add_sales_lag_features(combined, group_col="store")

    train_feat = combined.iloc[: len(train)].copy()
    test_feat = combined.iloc[len(train):].copy()

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
    X = train_feat.drop(columns=["sales", "date"])
    X_test = test_feat.drop(columns=["sales", "date"])

    # ----------------------------
    # 6) Auto-detect numeric vs categorical
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

    # ---- LightGBM model ----
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
    order = train_feat.sort_values("date").index.to_numpy()
    X_ord = X.loc[order].reset_index(drop=True)
    y_ord = y.loc[order].reset_index(drop=True)
    dates_ord = train_feat.loc[order, "date"].reset_index(drop=True)
    store_ord = train_feat.loc[order, "store"].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s, mapes = [], [], [], []

    # store learning curves for combined plot
    all_train_rmse = []
    all_valid_rmse = []

    # OOF storage for plotting
    oof_parts = []

    print("\n=== TimeSeriesSplit Evaluation (LightGBM + lags) ===")
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_ord), start=1):
        X_tr, X_va = X_ord.iloc[tr_idx], X_ord.iloc[va_idx]
        y_tr, y_va = y_ord.iloc[tr_idx], y_ord.iloc[va_idx]

        # Fit preprocessing on train fold only
        X_tr_p = preprocess.fit_transform(X_tr, y_tr)
        X_va_p = preprocess.transform(X_va)

        model = make_model()

        # Track RMSE per boosting round on train + valid
        model.fit(
            X_tr_p, y_tr,
            eval_set=[(X_tr_p, y_tr), (X_va_p, y_va)],
            eval_names=["train", "valid"],
            eval_metric="rmse",
        )

        pred = model.predict(X_va_p)

        # OOF dataframe for this fold (for plots)
        fold_df = pd.DataFrame({
            "date": dates_ord.iloc[va_idx].values,
            "store": store_ord.iloc[va_idx].values,
            "y_true": y_va.values,
            "y_pred": pred,
            "fold": fold,
        })
        oof_parts.append(fold_df)

        mae_val = mean_absolute_error(y_va, pred)
        rmse_val = rmse(y_va, pred)
        r2_val = r2_score(y_va, pred)
        mape_val = mape(y_va, pred)

        maes.append(mae_val)
        rmses.append(rmse_val)
        r2s.append(r2_val)
        mapes.append(mape_val)

        # collect learning curves
        evals = model.evals_result_
        train_rmse_curve = np.array(evals["train"]["rmse"], dtype=float)
        valid_rmse_curve = np.array(evals["valid"]["rmse"], dtype=float)
        all_train_rmse.append(train_rmse_curve)
        all_valid_rmse.append(valid_rmse_curve)

        va_min = dates_ord.iloc[va_idx].min().date()
        va_max = dates_ord.iloc[va_idx].max().date()
        print(
            f"Fold {fold}: MAE={mae_val:.4f}  RMSE={rmse_val:.4f}  MAPE={mape_val:.2f}%  R2={r2_val:.4f}  "
            f"ValidRange={va_min}..{va_max}  (train={len(tr_idx)}, valid={len(va_idx)})"
        )

    # Build OOF dataframe
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["date"] = pd.to_datetime(oof["date"])
    oof = oof.sort_values(["store", "date"]).reset_index(drop=True)

    oof.to_csv("oof_predictions.csv", index=False)
    print("Wrote oof_predictions.csv")

    print("\n=== Summary ===")
    print(f"MAE  mean={float(np.mean(maes)):.4f}  std={float(np.std(maes)):.4f}")
    print(f"RMSE mean={float(np.mean(rmses)):.4f}  std={float(np.std(rmses)):.4f}")
    print(f"R2   mean={float(np.mean(r2s)):.4f}  std={float(np.std(r2s)):.4f}")
    print(f"MAPE mean={float(np.mean(mapes)):.2f}%  std={float(np.std(mapes)):.2f}%")

    # ----------------------------
    # 8) Combined learning curve plot (all folds in ONE figure)
    # ----------------------------
    max_len = max(len(v) for v in all_valid_rmse)

    def pad_to(arr, n):
        out = np.full(n, np.nan, dtype=float)
        out[: len(arr)] = arr
        return out

    train_mat = np.vstack([pad_to(v, max_len) for v in all_train_rmse])
    valid_mat = np.vstack([pad_to(v, max_len) for v in all_valid_rmse])

    mean_train = np.nanmean(train_mat, axis=0)
    mean_valid = np.nanmean(valid_mat, axis=0)

    iters = np.arange(1, max_len + 1)

    plt.figure(figsize=(10, 6))

    # overlay each fold's validation curve
    for i, v in enumerate(all_valid_rmse, start=1):
        plt.plot(np.arange(1, len(v) + 1), v, alpha=0.35, label=f"valid fold {i}")

    # mean curves (bold)
    plt.plot(iters, mean_train, linewidth=2.5, label="train mean")
    plt.plot(iters, mean_valid, linewidth=2.5, label="valid mean")

    plt.xlabel("Boosting round (tree)")
    plt.ylabel("RMSE")
    plt.title("LightGBM learning curves (all folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lgbm_learning_curves_all_folds.png")
    plt.close()

    print("Saved combined plot: lgbm_learning_curves_all_folds.png")

    # ----------------------------
    # 8b) OOF plots: actual vs predicted
    # ----------------------------
    # (A) time-series across ALL stores (can be noisy)
    oof_time = oof.sort_values("date").reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    plt.plot(oof_time["date"], oof_time["y_true"], label="Actual", alpha=0.8)
    plt.plot(oof_time["date"], oof_time["y_pred"], label="Predicted", alpha=0.8)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Out-of-fold: Actual vs Predicted (validation only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("oof_actual_vs_pred_time.png")
    plt.close()
    print("Saved plot: oof_actual_vs_pred_time.png")

    # (B) one-store time-series (cleaner)
    store_id = oof["store"].iloc[0]
    one = oof[oof["store"] == store_id].sort_values("date")
    plt.figure(figsize=(12, 6))
    plt.plot(one["date"], one["y_true"], label="Actual")
    plt.plot(one["date"], one["y_pred"], label="Predicted")
    plt.title(f"OOF Actual vs Predicted — Store {store_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("oof_actual_vs_pred_time_store.png")
    plt.close()
    print("Saved plot: oof_actual_vs_pred_time_store.png")

    # (C) scatter plot predicted vs actual
    plt.figure(figsize=(7, 7))
    plt.scatter(oof["y_true"], oof["y_pred"], alpha=0.25)
    mn = float(min(oof["y_true"].min(), oof["y_pred"].min()))
    mx = float(max(oof["y_true"].max(), oof["y_pred"].max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual sales")
    plt.ylabel("Predicted sales")
    plt.title("Out-of-fold: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig("oof_actual_vs_pred_scatter.png")
    plt.close()
    print("Saved plot: oof_actual_vs_pred_scatter.png")

    # ----------------------------
    # 9) Train on full train and predict test
    # ----------------------------
    X_all_p = preprocess.fit_transform(X_ord, y_ord)
    X_test_p = preprocess.transform(X_test)

    final_model = make_model()
    final_model.fit(X_all_p, y_ord)

    test_pred = final_model.predict(X_test_p)
    test_pred = np.maximum(0, test_pred)

    out = test.copy()
    out["sales"] = test_pred
    out[["date", "store", "sales"]].to_csv("predictions_lgbm.csv", index=False)
    print("\nWrote predictions_lgbm.csv")


if __name__ == "__main__":
    main()