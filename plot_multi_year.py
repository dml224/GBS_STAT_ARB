import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
import os


# Plot settings #

plt.rcParams.update({
    'font.family': 'Courier New',
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20,
    "font.weight": "bold",
    "axes.labelweight": "bold",
})

# Palette
palette = {
    "Benchmark": "#7f7f7f",  
    "Spectral": "#00cbf7",
    "SPONGE": "#055ae2",
    "GBS Boost": "#b390a5", 
    "GBS Roots": "#ff9eb0", 
    "QIC-GBS": "#5A5A5A"    
}


# Data #

data_rows = []

# 2008
vix_2008 = 46.2 
data_rows.extend([
    {'Algorithm': 'Benchmark', 'Sharpe_Ratio': -1.1133, 'Density': np.nan, 'Year': 2008, 'VIX': vix_2008},
    {'Algorithm': 'GBS Boost', 'Sharpe_Ratio': 1.4070,  'Density': 0.2093, 'Year': 2008, 'VIX': vix_2008},
    {'Algorithm': 'QIC-GBS',   'Sharpe_Ratio': 2.2439,  'Density': 0.2099, 'Year': 2008, 'VIX': vix_2008},
    {'Algorithm': 'GBS Roots', 'Sharpe_Ratio': 2.0140,  'Density': 0.2647, 'Year': 2008, 'VIX': vix_2008},
    {'Algorithm': 'SPONGE',    'Sharpe_Ratio': 2.1641,  'Density': 0.2262, 'Year': 2008, 'VIX': vix_2008},
    {'Algorithm': 'Spectral',  'Sharpe_Ratio': 1.0435,  'Density': 0.1934, 'Year': 2008, 'VIX': vix_2008},
])

# 2017
vix_2017 = 12.8
data_rows.extend([
    {'Algorithm': 'Benchmark', 'Sharpe_Ratio': 2.2088, 'Density': np.nan, 'Year': 2017, 'VIX': vix_2017},
    {'Algorithm': 'GBS Boost', 'Sharpe_Ratio': 0.1380, 'Density': 0.2090, 'Year': 2017, 'VIX': vix_2017},
    {'Algorithm': 'QIC-GBS',   'Sharpe_Ratio': 0.3925, 'Density': 0.2105, 'Year': 2017, 'VIX': vix_2017},
    {'Algorithm': 'GBS Roots', 'Sharpe_Ratio': 0.6227, 'Density': 0.2347, 'Year': 2017, 'VIX': vix_2017},
    {'Algorithm': 'SPONGE',    'Sharpe_Ratio': 1.4856, 'Density': 0.4136, 'Year': 2017, 'VIX': vix_2017},
    {'Algorithm': 'Spectral',  'Sharpe_Ratio': 1.1647, 'Density': 0.4935, 'Year': 2017, 'VIX': vix_2017},
])

# 2020
vix_2020 = 29.2 
data_rows.extend([
    {'Algorithm': 'Benchmark', 'Sharpe_Ratio': 1.2188, 'Density': np.nan, 'Year': 2020, 'VIX': vix_2020},
    {'Algorithm': 'GBS Boost', 'Sharpe_Ratio': 2.3748, 'Density': 0.1983, 'Year': 2020, 'VIX': vix_2020},
    {'Algorithm': 'QIC-GBS',   'Sharpe_Ratio': 1.0230, 'Density': 0.1992, 'Year': 2020, 'VIX': vix_2020},
    {'Algorithm': 'GBS Roots', 'Sharpe_Ratio': 2.2648, 'Density': 0.2244, 'Year': 2020, 'VIX': vix_2020},
    {'Algorithm': 'SPONGE',    'Sharpe_Ratio': 1.90, 'Density': 0.230,  'Year': 2020, 'VIX': vix_2020}, 
    {'Algorithm': 'Spectral',  'Sharpe_Ratio': 1.10, 'Density': 0.200,  'Year': 2020, 'VIX': vix_2020},
])

# 2022
vix_2022 = 25.6 
data_rows.extend([
    {'Algorithm': 'Benchmark', 'Sharpe_Ratio': 0.1029, 'Density': np.nan, 'Year': 2022, 'VIX': vix_2022},
    {'Algorithm': 'GBS Boost', 'Sharpe_Ratio': 0.5974, 'Density': 0.2293, 'Year': 2022, 'VIX': vix_2022},
    {'Algorithm': 'QIC-GBS',   'Sharpe_Ratio': 0.9628, 'Density': 0.2264, 'Year': 2022, 'VIX': vix_2022},
    {'Algorithm': 'GBS Roots', 'Sharpe_Ratio': 1.3260, 'Density': 0.2848, 'Year': 2022, 'VIX': vix_2022},
    {'Algorithm': 'SPONGE',    'Sharpe_Ratio': 1.2928, 'Density': 0.4905, 'Year': 2022, 'VIX': vix_2022},
    {'Algorithm': 'Spectral',  'Sharpe_Ratio': 2.1515, 'Density': 0.5419, 'Year': 2022, 'VIX': vix_2022},
])

df = pd.DataFrame(data_rows)


# Plotting #

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.8], hspace=0.35)


# Top panel #

ax1 = fig.add_subplot(gs[0])

sns.barplot(
    data=df,
    x='Year',
    y='Sharpe_Ratio',
    hue='Algorithm',
    palette=palette,
    linewidth=0, 
    ax=ax1
)

ax1.axhline(0, color='black', linewidth=2)
ax1.set_xlabel("")
ax1.set_ylabel("SHARPE RATIO")
ax1.grid(False) 
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=False)

# Bottom panel #

ax2 = fig.add_subplot(gs[1])

df_density = df[df['Algorithm'] != 'Benchmark']

sns.lineplot(
    data=df_density,
    x=df_density['Year'].astype(str), 
    y='Density',
    hue='Algorithm',
    palette=palette,
    marker='o',
    markersize=10,
    linewidth=3,
    ax=ax2,
    legend=False 
)

# Plot VIX on secondary axis
ax3 = ax2.twinx()
vix_data = df.groupby('Year')['VIX'].mean()
ax3.plot(vix_data.index.astype(str), vix_data.values, color='red', linestyle='-', linewidth=2, marker='x', markersize=12, label='Market VIX')

ax2.set_ylabel("WEIGHTED DENSITY") 
ax2.set_xlabel("MARKET REGIME (YEAR)")
ax2.grid(False)

ax3.set_ylabel("MARKET VIX", color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax3.spines['right'].set_color('red')
ax3.grid(False)

vix_line = mlines.Line2D([], [], color='red', marker='x', linestyle='-', markersize=10, label='Market VIX')
ax2.legend(handles=[vix_line], loc='upper center', frameon=True)


# Save figures #

output_dir = "output_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "performance_regime_analysis.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "performance_regime_analysis.svg"), bbox_inches='tight')
plt.show()