# ARDL lag selection (AIC)
import pandas as pd
from statsmodels.tsa.ardl import ardl_select_order, ARDL
from nigeria_agric_data import df, Y_COL, N

cols = [Y_COL, "ln_AF", "ln_AF2", "ln_SI", "ln_ST"]
data = df[cols].dropna().reset_index(drop=True)

y = data[Y_COL]
X_dyn = data[["ln_AF", "ln_AF2", "ln_SI"]]
X_fix = data[["ln_ST"]]

# Select lags only for dynamic regressors
sel = ardl_select_order(
    endog=y, exog=X_dyn,
    maxlag=2, maxorder=2,
    ic="aic", trend="c"
)

ardl_dyn_mod = sel.model
p = ardl_dyn_mod.ardl_order[0]
q_tuple = ardl_dyn_mod.ardl_order[1:]
q_af, q_af2, q_si = q_tuple[0], q_tuple[1], q_tuple[2]
q_st = 0   # fixed at lag 0

# Build final model with fixed ln_ST
order_dict = {"ln_AF": q_af, "ln_AF2": q_af2, "ln_SI": q_si}
final_mod = ARDL(
    endog=y, lags=p,
    exog=X_dyn, order=order_dict,
    fixed=X_fix, trend="c"
)
final_res = final_mod.fit()

print(f"ARDL lag structure: ARDL({p}, {q_af}, {q_af2}, {q_si}, {q_st})")
print(f"  where q_ST = {q_st} (ln_ST included contemporaneously only)")
print(f"AIC = {final_res.aic:.4f},  BIC = {final_res.bic:.4f}")

if any("ln_ST" in name for name in final_res.params.index):
    print("ln_ST confirmed in the model (fixed L0).")
else:
    print("Warning: ln_ST is missing — review specification immediately.")

print("ARDL lag selection complete.")
