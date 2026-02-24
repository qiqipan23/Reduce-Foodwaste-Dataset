from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

CSV_PATH = Path("train.csv")

DATE_COL = "date"
HOLIDAY_COLS = ["is_state_holiday", "is_school_holiday", "is_special_day"]

NUM_SUMMARY_ROWS = ["sales", "unsold", "ordered", "temperature_mean"]
MISSING_ROWS = ["temperature_mean", "sunshine_sum", "precipitation_sum"]


def pct(x: float) -> str:
    return f"{x*100:.2f}%"


def fmt(x) -> str:
    if pd.isna(x):
        return "—"
    # 你可以按需要调整小数位
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        # 对 sales/unsold/ordered 通常用 2 位小数即可
        return f"{float(x):.4f}".rstrip("0").rstrip(".")
    return str(x)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Cannot find {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)

    # Parse date if exists (not required for these tables)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Ensure numeric conversions where appropriate
    for c in set(NUM_SUMMARY_ROWS + MISSING_ROWS):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in HOLIDAY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)

    # -------- Table 1: Numeric summary --------
    num_rows = []
    for col in NUM_SUMMARY_ROWS:
        if col not in df.columns:
            num_rows.append((col, "—", "—", "—", "—"))
            continue
        s = df[col].dropna()
        if s.empty:
            num_rows.append((col, "—", "—", "—", "—"))
            continue
        num_rows.append(
            (
                col,
                fmt(s.mean()),
                fmt(s.std(ddof=1)),
                fmt(s.min()),
                fmt(s.max()),
            )
        )

    # -------- Table 2: Holiday proportions --------
    hol_rows = []
    for col in HOLIDAY_COLS:
        if col not in df.columns:
            hol_rows.append((col, "—%", "—%"))
            continue
        s = df[col].fillna(0)
        # treat any non-zero as 1
        s01 = (s.astype(float) != 0).astype(int)
        p1 = float(s01.mean()) if n else float("nan")
        p0 = 1.0 - p1 if n else float("nan")
        hol_rows.append((col, pct(p0), pct(p1)))

    # -------- Table 3: Missing overview --------
    miss_rows = []
    for col in MISSING_ROWS:
        if col not in df.columns:
            miss_rows.append((col, "—", "—%"))
            continue
        miss = int(df[col].isna().sum())
        miss_pct = (miss / n) if n else float("nan")
        miss_rows.append((col, str(miss), pct(miss_pct)))

    # -------- Print results as HTML rows --------
    print("\n=== Paste into HTML: Numeric summary table rows ===")
    for col, mean, std, mn, mx in num_rows:
        print(f"<tr><td>{col}</td><td>{mean}</td><td>{std}</td><td>{mn}</td><td>{mx}</td></tr>")

    print("\n=== Paste into HTML: Holiday proportions table rows ===")
    for col, p0, p1 in hol_rows:
        print(f"<tr><td>{col}</td><td>{p0}</td><td>{p1}</td></tr>")

    print("\n=== Paste into HTML: Missing values overview table rows ===")
    for col, mc, mp in miss_rows:
        print(f"<tr><td>{col}</td><td>{mc}</td><td>{mp}</td></tr>")


if __name__ == "__main__":
    main()