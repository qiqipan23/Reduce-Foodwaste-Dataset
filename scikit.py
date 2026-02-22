import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor


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

    df["sales_rollmean_7"] = g.shift(1).rolling(window=7, min_periods=1).mean()
    df["sales_rollmean_14"] = g.shift(1).rolling(window=14, min_periods=1).mean()
    df["sales_rollstd_7"] = g.shift(1).rolling(window=7, min_periods=2).std()

    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    # 1) Load
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # 2) Date features
    train = add_date_features(train)
    test = add_date_features(test)

    # Ensure numeric sales
    train["sales"] = pd.to_numeric(train["sales"], errors="coerce")

    # 3) Create combined to build lags for both train+test
    test_for_lags = test.copy()
    test_for_lags["sales"] = np.nan

    combined = pd.concat([train, test_for_lags], ignore_index=True, sort=False)

    # Drop columns we don't want as features
    for c in ["unsold", "ordered", "row_id"]:
        if c in combined.columns:
            combined = combined.drop(columns=[c])

    # Add lag features
    combined = add_sales_lag_features(combined, group_col="store")

    # Split back
    train_feat = combined.iloc[: len(train)].copy()
    test_feat = combined.iloc[len(train):].copy()

    # 4) Drop training rows where target is NaN (prevents scoring crash)
    before = len(train_feat)
    train_feat = train_feat[train_feat["sales"].notna()].copy()
    after = len(train_feat)
    if after != before:
        print(f"Dropped {before-after} train rows with NaN sales.")

    # 5) Build X/y
    y = train_feat["sales"].astype(float)
    X = train_feat.drop(columns=["sales", "date"])
    X_test = test_feat.drop(columns=["sales", "date"])

    # 6) Auto-detect numeric vs categorical
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

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=800,
        random_state=42,
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

    # 7) Time-series CV on chronological order of TRAIN rows
    order = train_feat.sort_values("date").index.to_numpy()
    X_ord = X.loc[order].reset_index(drop=True)
    y_ord = y.loc[order].reset_index(drop=True)
    dates_ord = train_feat.loc[order, "date"].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []

    print("\n=== TimeSeriesSplit Evaluation (HistGBR + lags) ===")
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_ord), start=1):
        X_tr, X_va = X_ord.iloc[tr_idx], X_ord.iloc[va_idx]
        y_tr, y_va = y_ord.iloc[tr_idx], y_ord.iloc[va_idx]

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)

        # Safety net: remove any NaNs in y_va before scoring
        mask = y_va.notna()
        y_va2 = y_va[mask]
        pred2 = pred[mask.to_numpy()]

        mae_val = mean_absolute_error(y_va2, pred2)
        rmse_val = rmse(y_va2, pred2)
        r2_val = r2_score(y_va2, pred2)

        maes.append(mae_val)
        rmses.append(rmse_val)
        r2s.append(r2_val)

        va_min = dates_ord.iloc[va_idx].min().date()
        va_max = dates_ord.iloc[va_idx].max().date()
        print(
            f"Fold {fold}: MAE={mae_val:.4f}  RMSE={rmse_val:.4f}  R2={r2_val:.4f}  "
            f"ValidRange={va_min}..{va_max}  (train={len(tr_idx)}, valid={len(va_idx)})"
        )

    print("\n=== Summary ===")
    print(f"MAE  mean={float(np.mean(maes)):.4f}  std={float(np.std(maes)):.4f}")
    print(f"RMSE mean={float(np.mean(rmses)):.4f}  std={float(np.std(rmses)):.4f}")
    print(f"R2   mean={float(np.mean(r2s)):.4f}  std={float(np.std(r2s)):.4f}")

    # 8) Train on full train and predict test
    pipe.fit(X_ord, y_ord)
    test_pred = pipe.predict(X_test)

    out = test.copy()
    out["sales"] = test_pred
    out[["date", "store", "sales"]].to_csv("predictions_lags.csv", index=False)
    print("\nWrote predictions_lags.csv")


if __name__ == "__main__":
    main()
