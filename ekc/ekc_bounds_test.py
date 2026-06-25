# ARDL Bounds Test (Pesaran, Shin & Smith 2001, Case III)
import warnings
import pandas as pd
import os
from statsmodels.tsa.ardl import ardl_select_order, ARDL, UECM
from nigeria_agric_data import df, Y_COL, N

warnings.filterwarnings("ignore")

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)

# Data preparation
cols = [Y_COL, "ln_AF", "ln_AF2", "ln_SI", "ln_ST"]
data = df[cols].dropna().reset_index(drop=True)

y = data[Y_COL]
X_dyn = data[["ln_AF", "ln_AF2", "ln_SI"]]
X_fix = data[["ln_ST"]]

# Lag selection for dynamic regressors
sel = ardl_select_order(
    endog=y, exog=X_dyn,
    maxlag=2, maxorder=2,
    ic="aic", trend="c"
)

dyn_mod = sel.model
p = dyn_mod.ardl_order[0]
q_tuple = dyn_mod.ardl_order[1:]
order_dict = {"ln_AF": q_tuple[0], "ln_AF2": q_tuple[1], "ln_SI": q_tuple[2]}

# Build final ARDL model (same as lag selection step)
final_mod = ARDL(
    endog=y, lags=p,
    exog=X_dyn, order=order_dict,
    fixed=X_fix, trend="c"
)
final_res = final_mod.fit()

# UECM and bounds test
ecm = UECM.from_ardl(final_mod)
ecm_fit = ecm.fit()
bt = ecm_fit.bounds_test(case=3)
F = float(bt.stat)
cv = bt.crit_vals

# Extract 5% critical values


def detect_cols(cv):
    cols = list(cv.columns)
    lower_col = next((c for c in cols if "lower" in str(
        c).lower() or "i(0" in str(c).lower()), cols[0])
    upper_col = next((c for c in cols if "upper" in str(
        c).lower() or "i(1" in str(c).lower()), cols[-1])
    return lower_col, upper_col


def get_cv5(cv, lower_col, upper_col):
    for key in ["5%", 95.0, 0.05, 5]:
        try:
            return float(cv.loc[key, lower_col]), float(cv.loc[key, upper_col])
        except Exception:
            continue
    row = cv.iloc[1]
    return float(row[lower_col]), float(row[upper_col])


lower_col, upper_col = detect_cols(cv)
lb_5, ub_5 = get_cv5(cv, lower_col, upper_col)

# Verdict
if F > ub_5:
    verdict = "Cointegration exists"
elif F < lb_5:
    verdict = "No cointegration"
else:
    verdict = "Inconclusive"

print(f"Bounds test: F = {F:.4f}")
print(f"5% critical values: I(0) = {lb_5:.4f}, I(1) = {ub_5:.4f}")
print(f"Verdict: {verdict}")
if verdict == "Inconclusive":
    print("Use ECT sign & significance as tie-breaker.")

# Save result
res = pd.DataFrame({
    "F_stat": [round(F, 4)],
    "I0_5pct": [round(lb_5, 4)],
    "I1_5pct": [round(ub_5, 4)],
    "Verdict": [verdict]
})
res.to_csv(os.path.join(output_dir, "bounds_test_result.csv"), index=False)
print(f"\nSaved → EKC/results/bounds_test_result.csv")
