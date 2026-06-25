# Cubic robustness test (quadratic vs cubic EKC)
import warnings
import os
import pandas as pd
import statsmodels.api as sm
from nigeria_agric_data import df

warnings.filterwarnings("ignore")

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)

# Ensure cubic term exists
if "ln_AF3" not in df.columns:
    df["ln_AF3"] = df["ln_AF"] ** 3

data = df[["ln_AGHGpc", "ln_AF", "ln_AF2", "ln_AF3"]].dropna()
y = data["ln_AGHGpc"]

# Quadratic
Xq = sm.add_constant(data[["ln_AF", "ln_AF2"]])
res_q = sm.OLS(y, Xq).fit()

# Cubic
Xc = sm.add_constant(data[["ln_AF", "ln_AF2", "ln_AF3"]])
res_c = sm.OLS(y, Xc).fit()

b3 = res_c.params["ln_AF3"]
p3 = res_c.pvalues["ln_AF3"]

# Model comparison table
comp = pd.DataFrame({
    "Model": ["Quadratic", "Cubic"],
    "AIC": [round(res_q.aic, 3), round(res_c.aic, 3)],
    "BIC": [round(res_q.bic, 3), round(res_c.bic, 3)]
})
comp.to_csv(os.path.join(output_dir, "cubic_robustness.csv"), index=False)

print(f"Cubic term coefficient: {b3:.4f} (p = {p3:.4f})")
if p3 > 0.10:
    print("Cubic term not significant – quadratic specification preferred.")
else:
    print("Cubic term significant – evidence of N‑shape; discuss as robustness check.")
print(f"AIC comparison saved → EKC/results/cubic_robustness.csv")
