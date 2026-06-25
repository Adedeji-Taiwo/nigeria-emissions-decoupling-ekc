# ARDL‑implied long‑run EKC curve
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from nigeria_agric_data import df

# Output directory
fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)

# Global style
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
    "axes.linewidth": 1.1,
})

# Observed data (levels)
x = df["AF"].values                     # thousand USD per capita
y = df["AGHGpc"].values                 # MtCO2e per million people
years = df.index.values
lnSI_bar = float(np.mean(df["ln_SI"]))  # hold ln_SI at mean

# Long‑run coefficients (from Step 7)
beta1 = -0.8874
beta2 = 0.6099
beta3 = 0.0162

# Calibrate intercept so curve sits on observed scale
alpha = float(np.mean(df["ln_AGHGpc"]) - (
    beta1 * np.mean(df["ln_AF"]) +
    beta2 * np.mean(df["ln_AF"]**2) +
    beta3 * lnSI_bar
))

# Grid for the curve
xf = np.linspace(x.min() * 0.97, x.max() * 1.03, 400)
ln_yhat = alpha + beta1 * np.log(xf) + beta2 * \
    (np.log(xf)**2) + beta3 * lnSI_bar
yhat = np.exp(ln_yhat)

# Plot
fig, ax = plt.subplots(figsize=(12.5, 8.5))

# Scatter by period
for i, yr in enumerate(years):
    if yr <= 2010:
        ax.scatter(x[i], y[i], color="#1F4E79", marker="o", s=115,
                   edgecolors="white", linewidth=0.9, alpha=0.95, zorder=5,
                   label="1990–2010" if yr == years[0] else None)
    else:
        ax.scatter(x[i], y[i], color="#C00000", marker="s", s=115,
                   edgecolors="white", linewidth=0.9, alpha=0.95, zorder=5,
                   label="2011–2021" if yr == 2011 else None)

# ARDL‑implied long‑run curve
ax.plot(xf, yhat, color="#2E75B6", linewidth=3.2, zorder=4,
        label="Long‑run EKC curve")

# Labels and axes
ax.set_xlabel("GDP per Capita (thousand constant 2015 USD)", labelpad=10)
ax.set_ylabel(
    "Agricultural GHG per Capita\n(MtCO₂e per million people)", labelpad=10)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.grid(axis="y", alpha=0.25, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
leg = ax.legend(loc="upper right", framealpha=0.95,
                fancybox=True, borderpad=0.8)
leg.get_frame().set_edgecolor("#686767")

fig.tight_layout()
plt.savefig(os.path.join(fig_dir, "ekc_ardl_curve.png"),
            dpi=600, bbox_inches="tight")
plt.close()
print("Saved → EKC/figures/ekc_ardl_curve.png (600 dpi)")
