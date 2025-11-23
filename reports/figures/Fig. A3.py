import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Conversion factor: 1 foot = 0.3048 meters
FT_TO_M = 0.3048

# Read the Excel data from the Plot sheet
df = pd.read_excel('/mnt/user-data/uploads/18TCs34GagesUncertaintyQuantification.xlsx', 
                   sheet_name='Plot')

# Convert from feet to meters
df['ModeledPeaks_m'] = df['ModeledPeaks'] * FT_TO_M
df['ObsPeaks_m'] = df['ObsPeaks'] * FT_TO_M

# Define colors and markers for each TC with a better color palette
tc_styles = {
    'Barry': {'color': '#1f77b4', 'marker': 'o', 'facecolors': 'none'},  # Blue
    'Bonnie': {'color': '#2ca02c', 'marker': 'o', 'facecolors': 'none'},  # Green
    'Cindy': {'color': '#d62728', 'marker': 'o', 'facecolors': 'none'},  # Red
    'Cristobal': {'color': '#9467bd', 'marker': 'o', 'facecolors': 'none'},  # Purple
    'Delta': {'color': '#ff7f0e', 'marker': 'o', 'facecolors': '#ff7f0e'},  # Orange filled
    'Gustav': {'color': '#ffd700', 'marker': 'o', 'facecolors': '#ffd700'},  # Gold filled
    'Harvey': {'color': '#8b4513', 'marker': 'o', 'facecolors': '#8b4513'},  # Saddle brown filled
    'Ida': {'color': '#e377c2', 'marker': 'o', 'facecolors': 'none'},  # Pink
    'Imelda': {'color': '#ff4500', 'marker': '^', 'facecolors': 'none'},  # Orange-red triangle
    'Isaac': {'color': '#800080', 'marker': 's', 'facecolors': 'none'},  # Purple square
    'Laura': {'color': '#1e90ff', 'marker': '^', 'facecolors': 'none'},  # Dodger blue triangle
    'Lee': {'color': '#87ceeb', 'marker': '^', 'facecolors': 'none'},  # Sky blue triangle
    'Nate': {'color': '#dc143c', 'marker': 's', 'facecolors': 'none'},  # Crimson square
    'Olga': {'color': '#ff69b4', 'marker': 's', 'facecolors': 'none'},  # Hot pink square
    'Sally&Beta': {'color': '#4169e1', 'marker': 's', 'facecolors': 'none'},  # Royal blue square
    'Zeta': {'color': '#696969', 'marker': 's', 'facecolors': 'none'}  # Dim gray square
}

# Calculate linear regression for all data
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df['ModeledPeaks_m'], df['ObsPeaks_m'])
r_squared = r_value**2

# Calculate RMSE
rmse = np.sqrt(np.mean((df['ObsPeaks_m'] - df['ModeledPeaks_m'])**2))

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

# Plot each TC with its specific style
for tc, style in tc_styles.items():
    tc_data = df[df['TC'] == tc]
    if len(tc_data) > 0:
        if style['facecolors'] == 'none':
            # Unfilled markers
            ax.scatter(tc_data['ModeledPeaks_m'], tc_data['ObsPeaks_m'],
                      facecolors='none', edgecolors=style['color'],
                      marker=style['marker'], s=60, linewidths=1.5, 
                      alpha=0.7, label=tc)
        else:
            # Filled markers
            ax.scatter(tc_data['ModeledPeaks_m'], tc_data['ObsPeaks_m'],
                      c=style['color'], marker=style['marker'], 
                      edgecolors=style['color'], s=60, linewidths=1.5, 
                      alpha=0.7, label=tc)

# Plot 1:1 line
max_val = max(df['ModeledPeaks_m'].max(), df['ObsPeaks_m'].max())
ax.plot([0, max_val], [0, max_val], 'k-', linewidth=2, label='1to1')

# Plot regression line
x_line = np.array([0, max_val])
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'b--', linewidth=2, label='Linear (all)')

# Plot +/- 1 RMSE bounds
ax.plot(x_line, x_line + rmse, 'k--', linewidth=1.5, label='+/- 1 RMSE')
ax.plot(x_line, x_line - rmse, 'k--', linewidth=1.5)

# Add regression equation and R² to plot
equation_text = f'y = {slope:.4f}x - {abs(intercept):.4f}\nR² = {r_squared:.4f}'
ax.text(0.05, 0.95, equation_text, transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Formatting
ax.set_xlabel('Modeled water surface elevation, m NAVD88', fontsize=14, fontweight='bold')
ax.set_ylabel('Observed water surface elevation, m NAVD88', fontsize=14, fontweight='bold')
ax.set_title('TCs numerical model uncertainty for Gage data and High-\nwater marks', 
             fontsize=14, fontweight='bold')

# Set equal aspect ratio and limits
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, max_val * 1.05)
ax.set_ylim(0, max_val * 1.05)

# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Legend - create in multiple columns to fit
ax.legend(loc='lower right', fontsize=9, frameon=True, 
          shadow=True, ncol=3, columnspacing=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/mnt/user-data/outputs/tc_uncertainty_meters.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/tc_uncertainty_meters.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')

print("Plot created successfully!")
print(f"Number of data points: {len(df)}")
print(f"Conversion: feet to meters (×{FT_TO_M})")
print(f"Max observed: {df['ObsPeaks_m'].max():.2f} m (was {df['ObsPeaks'].max():.2f} ft)")
print(f"Max modeled: {df['ModeledPeaks_m'].max():.2f} m (was {df['ModeledPeaks'].max():.2f} ft)")
print(f"RMSE: {rmse:.3f} m")
print(f"R²: {r_squared:.4f}")
