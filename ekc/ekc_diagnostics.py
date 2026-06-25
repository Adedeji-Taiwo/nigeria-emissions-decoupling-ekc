# Residual diagnostics for the official ARDL(2,2,1,1,0) model
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, het_arch
from statsmodels.stats.stattools import jarque_bera
from nigeria_agric_data import df, Y_COL

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)

data = df[[Y_COL, "ln_AF", "ln_AF2", "ln_SI", "ln_ST"]
          ].dropna().reset_index(drop=True)
y = data[Y_COL]
X_dyn = data[["ln_AF", "ln_AF2", "ln_SI"]]
X_fix = data[["ln_ST"]]

final_mod = ARDL(
    endog=y, lags=2,
    exog=X_dyn,
    order={"ln_AF": 2, "ln_AF2": 1, "ln_SI": 1},
    fixed=X_fix,
    trend="c"
)
res = final_mod.fit()

hb = res.model.hold_back
fitted = res.predict(start=hb, end=len(y) - 1)
y_aln = y.iloc[hb:].values
r = (y_aln - fitted.values).astype(float)

X_exog = res.model.exog[-len(r):]
X_diag = X_exog if np.any(np.all(X_exog == 1, axis=0)
                          ) else sm.add_constant(X_exog)
ols_diag = sm.OLS(y_aln, X_diag).fit()

bg = acorr_breusch_godfrey(ols_diag, nlags=2)   # (LM, LM p, F, F p)
bp = het_breuschpagan(r, X_diag)                 # (LM, LM p, F, F p)
jb = jarque_bera(r)                              # (JB, JB p, skew, kurt)
arch = het_arch(r, nlags=2)                        # (LM, LM p, F, F p)

print("Diagnostics for ARDL(2,2,1,1,0) with fixed ln_ST:")
print(f"  Breusch–Godfrey (serial corr) p = {bg[1]:.4f}")
print(f"  Breusch–Pagan (heterosked.)   p = {bp[1]:.4f}")
print(f"  Jarque–Bera (normality)       p = {jb[1]:.4f}")
print(f"  ARCH(2) (cond. heterosked.)   p = {arch[1]:.4f}")

diag_df = pd.DataFrame({
    "Test": ["Breusch-Godfrey", "Breusch-Pagan", "Jarque-Bera", "ARCH(2)"],
    "Statistic": [bg[0], bp[0], jb[0], arch[0]],
    "p_value": [bg[1], bp[1], jb[1], arch[1]]
})
diag_df.to_csv(os.path.join(output_dir, "diagnostics.csv"), index=False)
print("\nSaved → EKC/results/diagnostics.csv")
