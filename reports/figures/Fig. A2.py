import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Conversion factor: 1 foot = 0.3048 meters
FT_TO_M = 0.3048

# Read the data - the file has a single column header but is actually tab-delimited with 3 columns
# Skip the header and read manually
with open('/mnt/user-data/uploads/FigureA1_Data.txt', 'r') as f:
    lines = f.readlines()[1:]  # Skip header
    
dates = []
modeled = []
observed = []

for line in lines:
    parts = line.strip().split('\t')
    if len(parts) == 3:
        dates.append(parts[0])
        # Convert from feet to meters
        modeled.append(float(parts[1]) * FT_TO_M)
        observed.append(float(parts[2]) * FT_TO_M)

# Create dataframe
data = pd.DataFrame({
    'DateTime': pd.to_datetime(dates, format='%m/%d/%Y %H:%M'),
    'Modeled': modeled,
    'Observed': observed
})

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Plot the data
ax.plot(data['DateTime'], data['Observed'], 
        linewidth=2, color='#2E86AB', label='Observed', linestyle='-')
ax.plot(data['DateTime'], data['Modeled'], 
        linewidth=2, color='#A23B72', label='Modeled', linestyle='--')

# Formatting
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Flow (m$^3$/s)', fontsize=14, fontweight='bold')
ax.set_title('Katrina — Amite River', fontsize=16, fontweight='bold', pad=20)

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45, ha='right', fontsize=12)

# Increase y-axis tick label font size
plt.yticks(fontsize=12)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper left', fontsize=12, frameon=True, 
          shadow=True, fancybox=True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('/mnt/user-data/outputs/stage_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/stage_comparison.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')

print("Plot created successfully!")
print(f"Data range: {data['DateTime'].min()} to {data['DateTime'].max()}")
print(f"Peak observed: {data['Observed'].max():.2f} m³/s")
print(f"Peak modeled: {data['Modeled'].max():.2f} m³/s")
print(f"(Original peak observed: {data['Observed'].max()/FT_TO_M:.2f} ft)")
print(f"(Original peak modeled: {data['Modeled'].max()/FT_TO_M:.2f} ft)")
