import numpy as np
import pandas as pd
import os
from nigeria_agric_data import df, N

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)

# 1. Descriptive Statistics
print(
    f"Descriptive statistics – Nigeria Agricultural Sector, 1990–2021 (n = {N})")

desc_vars = {
    'Total Agric GHG (MtCO₂e)':              'GHG_agric',
    'Agric GHG Per Capita (Mt/Mn People)':   'AGHGpc',
    'Affluence (AF - kUSD 2015)':            'AF',
    'Population (P - Million)':              'P',
    'Emission Intensity (EI - Mt/TWh)':      'EI',
    'Energy Intensity (SI - TWh/Bn USD)':    'SI',
    'Sectoral Composition (ST)':             'ST',
}

rows = []
for label, col in desc_vars.items():
    if col in df.columns:
        s = df[col]
        rows.append({
            'Variable':  label,
            'Obs':       int(s.count()),
            'Mean':      round(s.mean(), 4),
            'Std_Dev':   round(s.std(), 4),
            'Min':       round(s.min(), 4),
            'Max':       round(s.max(), 4),
        })

desc_df = pd.DataFrame(rows).set_index('Variable')
desc_df.to_csv(os.path.join(output_dir, "descriptive_stats.csv"))
print(desc_df.to_string())
print("Saved → EKC/results/descriptive_stats.csv\n")

# 2. Pairwise Correlation Matrix
print("Pairwise correlation matrix – level variables")
print("Flagging |r| > 0.85\n")

corr_vars = ['GHG_agric', 'AGHGpc', 'AF', 'P', 'EI', 'SI', 'ST']
corr_labels = ['GHG', 'AGHGpc', 'AF', 'P', 'EI', 'SI', 'ST']

cm = df[corr_vars].corr().round(3)
cm.index = cm.columns = corr_labels
cm.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
print(cm.to_string())
print("Saved → EKC/results/correlation_matrix.csv\n")

print("High-correlation pairs (|r| > 0.85):")
flagged = False
for i in range(len(corr_labels)):
    for j in range(i + 1, len(corr_labels)):
        r = cm.iloc[i, j]
        if abs(r) > 0.85:
            print(f"  {corr_labels[i]} & {corr_labels[j]}: r = {r:.3f}")
            flagged = True
if not flagged:
    print("  None above threshold.")

print("\nPreliminary analysis complete.")
