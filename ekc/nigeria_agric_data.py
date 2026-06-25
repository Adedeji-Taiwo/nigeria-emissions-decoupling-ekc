import pandas as pd
import numpy as np

RAW = {
    'Year': list(range(1990, 2022)),
    'GHG_agric': [39.44, 41.26, 41.79, 42.88, 44.19, 45.11, 47.20, 49.34, 52.19, 53.05,
                  53.79, 53.12, 53.77, 55.50, 56.57, 60.12, 60.47, 59.49, 59.73, 57.61,
                  62.09, 66.31, 69.45, 70.99, 72.37, 74.44, 78.15, 78.76, 80.32, 81.98,
                  84.64, 85.16],
    'EC_agric':  [0.04806, 0.04806, 0.06000, 0.06000, 0.06000, 0.06000, 0.06000, 0.06000,
                  0.06000, 0.07222, 0.07222, 0.07222, 0.10833, 0.10833, 0.12028, 0.12028,
                  0.13222, 0.06000, 0.12056, 0.07222, 0.12056, 0.04806, 0.04806, 0.04806,
                  0.04806, 0.04806, 0.04806, 0.04972, 0.05056, 0.05194, 0.05306, 0.05417],
    'GDP_agric': [22.09, 22.90, 23.43, 23.87, 24.48, 25.36, 26.36, 27.46, 28.54, 29.99,
                  30.87, 32.04, 49.85, 53.34, 56.68, 60.69, 65.18, 69.88, 74.26, 78.63,
                  83.21, 85.63, 91.37, 94.06, 98.07, 101.72, 105.90, 109.55, 111.87, 114.51,
                  117.56, 122.00],
    'GDP_total': [153.18, 153.73, 160.85, 157.57, 154.71, 154.60, 161.09, 165.82, 170.10, 171.09,
                  179.68, 190.31, 219.48, 235.61, 257.40, 273.97, 290.58, 309.73, 330.68, 357.26,
                  385.86, 406.34, 423.53, 451.78, 480.29, 493.03, 485.06, 488.96, 498.37, 509.37,
                  476.93, 482.22],
    'Population': [97.12, 99.72, 102.37, 105.12, 107.94, 110.82, 113.75, 116.75, 119.85, 123.05,
                   126.38, 129.86, 133.47, 137.20, 141.06, 145.02, 149.08, 153.27, 157.60, 162.05,
                   166.64, 171.38, 176.20, 181.05, 185.90, 190.67, 195.44, 200.25, 204.94, 209.49,
                   214.00, 218.53],
}

df = pd.DataFrame(RAW).set_index('Year')

# Derived variables for decomposition and regression
# Emission intensity (MtCO2e / TWh)
df['EI'] = df['GHG_agric'] / df['EC_agric']
# Emissions per capita (MtCO2e / million)
df['AGHGpc'] = df['GHG_agric'] / df['Population']
# Affluence (GDP per capita, thousand USD)
df['AF'] = df['GDP_total'] / df['Population']
# Energy intensity (TWh / Bn USD)
df['SI'] = df['EC_agric'] / df['GDP_agric']
# Sectoral composition (share of agriculture)
df['ST'] = df['GDP_agric'] / df['GDP_total']
# Population (million)
df['P'] = df['Population']

# Logarithmic transformations for the EKC model
df['ln_AGHGpc'] = np.log(df['AGHGpc'])
df['ln_AF'] = np.log(df['AF'])
df['ln_AF2'] = df['ln_AF'] ** 2
df['ln_SI'] = np.log(df['SI'])
df['ln_ST'] = np.log(df['ST'])
df['ln_P'] = np.log(df['P'])

# Convenience constants for downstream scripts
# Dependent variable
Y_COL = 'ln_AGHGpc'
# Regressors (population already in Y_COL)
EXOG_COLS = ['ln_AF', 'ln_AF2', 'ln_SI', 'ln_ST']
UNIT_ROOT_VARS = ['ln_AGHGpc', 'ln_AF', 'ln_AF2', 'ln_SI',
                  'ln_ST', 'ln_P']  # Variables for unit root testing

N = len(df)

if __name__ == '__main__':
    print(f"Data loaded: {N} observations ({df.index[0]}–{df.index[-1]})")
    print(f"Target: {Y_COL}  |  Regressors: {EXOG_COLS}")
