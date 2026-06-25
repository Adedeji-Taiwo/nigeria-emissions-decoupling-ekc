# Long-run EKC coefficients — from UECM cointegrating vector
import warnings
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.ardl import ardl_select_order, ARDL, UECM
from nigeria_agric_data import df, Y_COL, N

warnings.filterwarnings("ignore")

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)

cols = [Y_COL, "ln_AF", "ln_AF2", "ln_SI", "ln_ST"]
data = df[cols].dropna().reset_index(drop=True)
y = data[Y_COL]
X_dyn = data[["ln_AF", "ln_AF2", "ln_SI"]]
X_fix = data[["ln_ST"]]

sel = ardl_select_order(
    endog=y, exog=X_dyn,
    maxlag=2, maxorder=2,
    ic="aic", trend="c"
)
dyn_mod = sel.model
p = dyn_mod.ardl_order[0]
q_af, q_af2, q_si = dyn_mod.ardl_order[1:]
q_st = 0

order_dict = {"ln_AF": q_af, "ln_AF2": q_af2, "ln_SI": q_si}
print(f"ARDL order: ({p}, {q_af}, {q_af2}, {q_si}, {q_st})")

final_mod = ARDL(
    endog=y, lags=p,
    exog=X_dyn, order=order_dict,
    fixed=X_fix, trend="c"
)
ecm_fit = UECM.from_ardl(final_mod).fit()

# Full cointegrating vector (coefficient, std error, t, p-value)
ci_params = ecm_fit.ci_params
ci_bse = ecm_fit.ci_bse
ci_tvalues = ecm_fit.ci_tvalues
ci_pvalues = ecm_fit.ci_pvalues

lr_table = pd.DataFrame({
    "variable": ci_params.index,
    "coefficient": ci_params.values,
    "std_error": ci_bse.values,
    "t_stat": ci_tvalues.values,
    "p_value": ci_pvalues.values
})
lr_table.to_csv(os.path.join(
    output_dir, "longrun_coefficients.csv"), index=False)

# Extract betas for EKC shape (β = -γ_x / γ_y)
gamma_y = float(ci_params["ln_AGHGpc"])
beta_af = -float(ci_params["ln_AF"]) / gamma_y
beta_af2 = -float(ci_params["ln_AF2"]) / gamma_y
beta_si = -float(ci_params["ln_SI"]) / gamma_y

print(f"β1 (ln_AF)  = {beta_af:.4f}")
print(f"β2 (ln_AF²) = {beta_af2:.4f}")
print(f"β3 (ln_SI)  = {beta_si:.4f}")

if beta_af > 0 and beta_af2 < 0:
    shape = "Inverted-U"
elif beta_af < 0 and beta_af2 > 0:
    shape = "U-shaped"
else:
    shape = "Indeterminate"

turning_point = None
if beta_af2 != 0 and shape != "Indeterminate":
    turning_point = np.exp(-beta_af / (2 * beta_af2))

print(f"EKC shape: {shape}")
if turning_point:
    print(
        f"Turning point: {turning_point:.2f} thousand USD per capita ({turning_point*1000:.0f} USD)")

summary = pd.DataFrame({
    "EKC_shape": [shape],
    "Turning_point_kUSD": [round(turning_point, 4) if turning_point else None]
})
summary.to_csv(os.path.join(output_dir, "ekc_shape_summary.csv"), index=False)

print(f"\nFull long-run table saved → EKC/results/longrun_coefficients.csv")
print(f"EKC shape summary saved → EKC/results/ekc_shape_summary.csv")
