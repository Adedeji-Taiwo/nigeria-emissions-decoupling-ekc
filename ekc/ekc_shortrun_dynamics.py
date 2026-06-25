# Short-run dynamics (ECM) — same ARDL model as bounds test and long-run coefficients
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

# Dynamic lag selection (AIC) — identical to previous steps
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

print(f"ARDL lag structure: ARDL({p}, {q_af}, {q_af2}, {q_si}, {q_st})")

# Final ARDL + UECM
final_mod = ARDL(
    endog=y, lags=p,
    exog=X_dyn, order=order_dict,
    fixed=X_fix, trend="c"
)
ecm_fit = UECM.from_ardl(final_mod).fit()

params = ecm_fit.params
bse = ecm_fit.bse
tvals = ecm_fit.tvalues
pvals = ecm_fit.pvalues


def sigstars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# Collect short-run terms (Δ variables)
rows = []
for term in params.index:
    if term.startswith("D."):
        c = float(params[term])
        se = float(bse[term])
        t = float(tvals[term])
        p = float(pvals[term])
        rows.append({
            "term": term,
            "coefficient": round(c, 4),
            "std_error": round(se, 4),
            "t_stat": round(t, 3),
            "p_value": round(p, 4),
            "significance": sigstars(p)
        })

# Fixed contemporaneous regressor ln_ST
if "ln_ST" in params.index:
    c = float(params["ln_ST"])
    se = float(bse["ln_ST"])
    t = float(tvals["ln_ST"])
    p = float(pvals["ln_ST"])
    rows.append({
        "term": "ln_ST",
        "coefficient": round(c, 4),
        "std_error": round(se, 4),
        "t_stat": round(t, 3),
        "p_value": round(p, 4),
        "significance": sigstars(p)
    })

# Error correction term
ect_name = f"{Y_COL}.L1"
if ect_name in params.index:
    c = float(params[ect_name])
    se = float(bse[ect_name])
    t = float(tvals[ect_name])
    p = float(pvals[ect_name])
    rows.append({
        "term": ect_name,
        "coefficient": round(c, 4),
        "std_error": round(se, 4),
        "t_stat": round(t, 3),
        "p_value": round(p, 4),
        "significance": sigstars(p)
    })
    adj_speed = abs(c) * 100
    print(
        f"ECT coefficient: {c:.4f} (p = {p:.4f}) — adjustment speed: {adj_speed:.1f}% per year")
else:
    print("ECT term not found — review model specification.")

# Save and print
shortrun_df = pd.DataFrame(rows)
shortrun_df.to_csv(os.path.join(
    output_dir, "shortrun_dynamics.csv"), index=False)

print("\nShort‑run dynamics:")
print(shortrun_df[["term", "coefficient", "std_error",
      "t_stat", "p_value", "significance"]].to_string(index=False))
print(f"\nSaved → EKC/results/shortrun_dynamics.csv")
