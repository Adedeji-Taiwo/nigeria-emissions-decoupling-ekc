# 1. Augmented Dickey-Fuller (ADF) Unit Root Tests
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from nigeria_agric_data import df, UNIT_ROOT_VARS, N
import os

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt(x, d=3):
    return f"{x:.{d}f}"


def adf(series, regression):
    res = adfuller(series, regression=regression, autolag="BIC")
    stat, pval, usedlag = res[0], res[1], res[2]
    return stat, pval, usedlag


print(
    f"ADF unit root tests — Nigeria Agricultural Sector, 1990–2021 (n = {N})")
print("Lag selection: BIC (preferred for small samples)")
print("ln(AF²) inherits integration order of ln(AF) and is not tested separately.\n")

records = []

for var in UNIT_ROOT_VARS:
    s = df[var].dropna().values

    # Level (constant + trend)
    l_stat, l_p, l_lag = adf(s, regression="ct")
    # First difference (constant only)
    d = np.diff(s)
    d_stat, d_p, d_lag = adf(d, regression="c")

    if l_p < 0.05:
        order = "I(0)"
    elif d_p < 0.05:
        order = "I(1)"
    elif d_p < 0.10:
        order = "I(1)*"
    else:
        order = "Review"

    records.append({
        "Variable": var,
        "Level": f"{fmt(l_stat)} ({fmt(l_p)}){stars(l_p)}",
        "Diff":  f"{fmt(d_stat)} ({fmt(d_p)}){stars(d_p)}",
        "Lags":  f"{l_lag}/{d_lag}",
        "Order": order,
        "Level_stat": l_stat,
        "Level_p": l_p,
        "Diff_stat": d_stat,
        "Diff_p": d_p,
    })

# Save as CSV
pd.DataFrame(records).to_csv(os.path.join(
    output_dir, "adf_unit_root.csv"), index=False)

# Print table
print(f"  {'Variable':<14} {'Level (C,T)':>22} {'1st Diff (C)':>22} {'Lags':>6}  Order")
print("  " + "-" * 80)
for r in records:
    print(
        f"  {r['Variable']:<14} {r['Level']:>22} {r['Diff']:>22} {r['Lags']:>6}  {r['Order']}")
print("  " + "-" * 80)
print("  *** p<0.01  ** p<0.05  * p<0.10  |  Lags: level/diff")
print("\nSaved → EKC/results/adf_unit_root.csv")
print("ADF testing complete.\n")
