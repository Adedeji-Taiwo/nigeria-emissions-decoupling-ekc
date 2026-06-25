# 2. Phillips-Perron (PP) Unit Root Tests
import pandas as pd
import numpy as np
from arch.unitroot import PhillipsPerron
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


def pp(series, trend):
    test = PhillipsPerron(series, trend=trend)
    return float(test.stat), float(test.pvalue)


print(f"PP unit root tests — Nigeria Agricultural Sector, 1990–2021 (n = {N})")
print("Bandwidth: Newey–West HAC (automatic)")
print("ln(AF²) inherits integration order of ln(AF) and is not tested separately.\n")

records = []

for var in UNIT_ROOT_VARS:
    s = df[var].dropna().values

    l_stat, l_p = pp(s, trend="ct")
    d = np.diff(s)
    d_stat, d_p = pp(d, trend="c")

    # PP can be degenerate for ln_P (near‑deterministic demographic trend).
    # We report PP but follow ADF classification for this variable.
    if var == "ln_P":
        order = "I(0)†"
    else:
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
        "Order": order,
        "Level_stat": l_stat,
        "Level_p": l_p,
        "Diff_stat": d_stat,
        "Diff_p": d_p,
    })

pd.DataFrame(records).to_csv(os.path.join(
    output_dir, "pp_unit_root.csv"), index=False)

print(f"  {'Variable':<14} {'Level (C,T)':>22} {'1st Diff (C)':>22}  Order")
print("  " + "-" * 70)
for r in records:
    print(f"  {r['Variable']:<14} {r['Level']:>22} {r['Diff']:>22}  {r['Order']}")
print("  " + "-" * 70)
print("  *** p<0.01  ** p<0.05  * p<0.10")
print("  † ln_P: PP may be unreliable under deterministic demographic trend;")
print("    integration order is based on ADF (Step 1), reported here for completeness.\n")
print("Saved → EKC/results/pp_unit_root.csv")
print("PP testing complete.\n")
