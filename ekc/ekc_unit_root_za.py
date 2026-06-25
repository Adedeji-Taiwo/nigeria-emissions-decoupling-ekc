# 3. Zivot-Andrews (ZA) Structural Break Unit Root Tests
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import zivot_andrews
from nigeria_agric_data import df, UNIT_ROOT_VARS, N
import os

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)


def fmt(x, d=3):
    return f"{x:.{d}f}"


print(
    f"ZA structural break tests — Nigeria Agricultural Sector, 1990–2021 (n = {N})")
print("Model: intercept + trend break (ct)")
print("H0: unit root with no structural break")
print("Max lags = 2 (small‑sample protection)\n")

records = []

for var in UNIT_ROOT_VARS:
    s = df[var].dropna()

    stat, pval, crit, used_lag, bp = zivot_andrews(
        s.values,
        regression="ct",
        maxlag=2,
        autolag="BIC"
    )

    break_year = int(s.index[int(bp)]) if bp is not None else None

    cv1 = crit["1%"]
    cv5 = crit["5%"]
    cv10 = crit["10%"]

    decision = "I(0) w/ break" if stat < cv5 else "I(1)"

    records.append({
        "Variable": var,
        "ZA_stat": stat,
        "p_value": pval,
        "1%_CV": cv1,
        "5%_CV": cv5,
        "10%_CV": cv10,
        "Lags": used_lag,
        "Break_year": break_year,
        "Decision": decision
    })

pd.DataFrame(records).to_csv(os.path.join(
    output_dir, "za_unit_root.csv"), index=False)

print(f"  {'Variable':<14} {'ZA Stat':>9} {'p-val':>8} {'1% CV':>9} {'5% CV':>9} {'10% CV':>9} {'Lag':>5} {'Break':>6}  Decision")
print("  " + "-" * 100)
for r in records:
    brk = str(r["Break_year"]) if r["Break_year"] is not None else "—"
    print(f"  {r['Variable']:<14} {fmt(r['ZA_stat']):>9} {fmt(r['p_value']):>8} "
          f"{fmt(r['1%_CV']):>9} {fmt(r['5%_CV']):>9} {fmt(r['10%_CV']):>9} "
          f"{str(r['Lags']):>5} {brk:>6}  {r['Decision']}")

print("  " + "-" * 100)
print("  I(0) w/ break: ZA rejects unit root at level (stationary around a break).")
print("  I(1): ZA fails to reject unit root (consistent with I(1)).")
print("  ln(AF²) inherits integration order of ln(AF) and is not tested separately.")
print("\nSaved → EKC/results/za_unit_root.csv")
print("ZA testing complete.\n")
