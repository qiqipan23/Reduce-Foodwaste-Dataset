"""
Generate dataset visualization PNGs for docs/assets/plots/

Input:  train.csv  (in project root by default)
Output: docs/assets/plots/*.png

Creates:
- dist_sales.png
- dist_unsold.png
- dist_weather.png
- timeseries_sales_total.png
- sales_by_weekday.png
- sales_holiday_effect.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Config ----------
CSV_PATH = Path("train.csv")
OUT_DIR = Path("docs/assets/plots")

DATE_COL = "date"
STORE_COL = "store"

HOLIDAY_COLS = ["is_state_holiday", "is_school_holiday", "is_special_day"]
NUM_COLS = [
    "temperature_max",
    "temperature_min",
    "temperature_mean",
    "sunshine_sum",
    "precipitation_sum",
    "sales",
    "unsold",
    "ordered",
]


# ---------- Helpers ----------
def ensure_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {name}: {missing}")


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def hist(series: pd.Series, title: str, xlabel: str, outpath: Path, bins: int = 50) -> None:
    s = series.dropna()
    plt.figure(figsize=(8, 5))
    if s.empty:
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.title(title)
        plt.axis("off")
        savefig(outpath)
        return

    plt.hist(s.values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    savefig(outpath)


# ---------- Main ----------
def main() -> int:
    if not CSV_PATH.exists():
        print(f"❌ Cannot find {CSV_PATH.resolve()}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(CSV_PATH)
    ensure_columns(df, [DATE_COL, STORE_COL], "basic fields")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = safe_numeric(df, NUM_COLS)

    # Holiday cols may be missing in some variants; handle gracefully
    for c in HOLIDAY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # ---------- 1) dist_sales.png ----------
    if "sales" in df.columns:
        hist(
            df["sales"],
            title="Sales Distribution",
            xlabel="sales",
            outpath=OUT_DIR / "dist_sales.png",
            bins=60,
        )
    else:
        print("⚠️ 'sales' column not found; skipping dist_sales.png")

    # ---------- 2) dist_unsold.png ----------
    if "unsold" in df.columns:
        hist(
            df["unsold"],
            title="Unsold Distribution",
            xlabel="unsold",
            outpath=OUT_DIR / "dist_unsold.png",
            bins=60,
        )
    else:
        print("⚠️ 'unsold' column not found; skipping dist_unsold.png")

    # ---------- 3) dist_weather.png ----------
    weather_cols = [c for c in ["temperature_mean", "sunshine_sum", "precipitation_sum"] if c in df.columns]
    plt.figure(figsize=(10, 7))
    if not weather_cols:
        plt.text(0.5, 0.5, "No weather columns found", ha="center", va="center")
        plt.title("Weather Feature Distributions")
        plt.axis("off")
        savefig(OUT_DIR / "dist_weather.png")
    else:
        # One figure with stacked histograms (separate axes)
        fig, axes = plt.subplots(len(weather_cols), 1, figsize=(10, 3.2 * len(weather_cols)))
        if len(weather_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, weather_cols):
            s = df[col].dropna()
            if s.empty:
                ax.text(0.5, 0.5, f"No data: {col}", ha="center", va="center")
                ax.set_axis_off()
                continue
            ax.hist(s.values, bins=50)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        fig.suptitle("Weather Feature Distributions", y=0.98)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "dist_weather.png", dpi=180)
        plt.close(fig)

    # ---------- 4) timeseries_sales_total.png ----------
    if "sales" in df.columns and df[DATE_COL].notna().any():
        ts = (
            df.dropna(subset=[DATE_COL])
            .groupby(DATE_COL, as_index=True)["sales"]
            .sum()
            .sort_index()
        )
        plt.figure(figsize=(10, 5))
        if ts.empty:
            plt.text(0.5, 0.5, "No time series data available", ha="center", va="center")
            plt.title("Total Sales Over Time")
            plt.axis("off")
        else:
            plt.plot(ts.index, ts.values)
            plt.title("Total Sales Over Time (All Stores)")
            plt.xlabel("date")
            plt.ylabel("total sales")
        savefig(OUT_DIR / "timeseries_sales_total.png")
    else:
        print("⚠️ Missing 'sales' or valid 'date'; skipping timeseries_sales_total.png")

    # ---------- 5) sales_by_weekday.png ----------
    if "sales" in df.columns and df[DATE_COL].notna().any():
        tmp = df.dropna(subset=[DATE_COL]).copy()
        tmp["weekday"] = tmp[DATE_COL].dt.day_name()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        w = tmp.groupby("weekday")["sales"].mean().reindex(order)

        plt.figure(figsize=(9, 5))
        if w.dropna().empty:
            plt.text(0.5, 0.5, "No weekday aggregation available", ha="center", va="center")
            plt.title("Average Sales by Weekday")
            plt.axis("off")
        else:
            plt.bar(w.index, w.values)
            plt.title("Average Sales by Weekday")
            plt.xlabel("weekday")
            plt.ylabel("avg sales")
            plt.xticks(rotation=20, ha="right")
        savefig(OUT_DIR / "sales_by_weekday.png")
    else:
        print("⚠️ Missing 'sales' or valid 'date'; skipping sales_by_weekday.png")

    # ---------- 6) sales_holiday_effect.png ----------
    # Combine holiday flags into a single "any_holiday" if columns exist
    holiday_present = [c for c in HOLIDAY_COLS if c in df.columns]
    if "sales" in df.columns and holiday_present:
        tmp = df.copy()
        tmp["any_holiday"] = (tmp[holiday_present].sum(axis=1) > 0).astype(int)

        sales0 = tmp.loc[tmp["any_holiday"] == 0, "sales"].dropna()
        sales1 = tmp.loc[tmp["any_holiday"] == 1, "sales"].dropna()

        plt.figure(figsize=(9, 5))
        if sales0.empty and sales1.empty:
            plt.text(0.5, 0.5, "No sales data available", ha="center", va="center")
            plt.title("Sales: Holiday vs Non-holiday")
            plt.axis("off")
        else:
            plt.hist([sales0.values, sales1.values], bins=50, label=["Non-holiday", "Holiday"], stacked=False)
            plt.title("Sales Distribution: Holiday vs Non-holiday")
            plt.xlabel("sales")
            plt.ylabel("Count")
            plt.legend()
        savefig(OUT_DIR / "sales_holiday_effect.png")
    else:
        print("⚠️ Holiday columns not found (or missing 'sales'); skipping sales_holiday_effect.png")

    print(f"✅ Done. PNGs saved to: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())