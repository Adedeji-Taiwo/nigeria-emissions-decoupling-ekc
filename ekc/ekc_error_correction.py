# Error-Correction Term (ECT) extraction — same ARDL model as bounds test
import warnings
import pandas as pd
import os
from statsmodels.tsa.ardl import ardl_select_order, ARDL, UECM
from nigeria_agric_data import df, Y_COL, N

warnings.filterwarnings("ignore")

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)

# Data
cols = [Y_COL, "ln_AF", "ln_AF2", "ln_SI", "ln_ST"]
data = df[cols].dropna().reset_index(drop=True)

y = data[Y_COL]
X_dyn = data[["ln_AF", "ln_AF2", "ln_SI"]]
X_fix = data[["ln_ST"]]

# Lag selection (same as previous steps)
sel = ardl_select_order(
    endog=y, exog=X_dyn,
    maxlag=2, maxorder=2,
    ic="aic", trend="c"
)

dyn_mod = sel.model
p = dyn_mod.ardl_order[0]
q_tuple = dyn_mod.ardl_order[1:]
order_dict = {"ln_AF": q_tuple[0], "ln_AF2": q_tuple[1], "ln_SI": q_tuple[2]}

# Final ARDL model
final_mod = ARDL(
    endog=y, lags=p,
    exog=X_dyn, order=order_dict,
    fixed=X_fix, trend="c"
)

ecm_fit = UECM.from_ardl(final_mod).fit()

# Extract ECT
ect_name = f"{Y_COL}.L1"
ect_coef = float(ecm_fit.params[ect_name])
ect_pval = float(ecm_fit.pvalues[ect_name])

# Interpretation
if ect_coef < 0 and ect_pval < 0.01:
    strength = "Strong (1% level)"
elif ect_coef < 0 and ect_pval < 0.05:
    strength = "Moderate (5% level)"
elif ect_coef < 0 and ect_pval < 0.10:
    strength = "Weak (10% level)"
elif ect_coef < 0:
    strength = "Not significant"
else:
    strength = "No adjustment (ECT ≥ 0)"

print(f"ECT coefficient: {ect_coef:.4f}  (p = {ect_pval:.4f})")
print(f"Speed of adjustment: {abs(ect_coef)*100:.1f}% per year")
print(f"Evidence of long‑run equilibrium: {strength}")

# Save
res = pd.DataFrame({
    "ECT_coefficient": [round(ect_coef, 4)],
    "p_value": [round(ect_pval, 4)],
    "adjustment_speed_pct": [round(abs(ect_coef)*100, 1)],
    "verdict": [strength]
})
res.to_csv(os.path.join(output_dir, "ect_result.csv"), index=False)
print(f"\nSaved → EKC/results/ect_result.csv")
