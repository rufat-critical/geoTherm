import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# Read the CSV file
df = pd.read_csv('nsds.csv', header=None)

# Create a logarithmic grid for interpolation
ns_grid = np.logspace(np.log10(0.1), np.log10(1000), 200)
ds_grid = np.logspace(np.log10(0.1), np.log10(100), 200)
NS, DS = np.meshgrid(ns_grid, ds_grid)

# Collect all points and their efficiencies
all_points_ns = []
all_points_ds = []
all_efficiencies = []

# Define efficiencies
efficiencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for eff in efficiencies:
    col_name = f'Full_{eff}'
    col_idx = df.iloc[0].str.contains(col_name, na=False)
    if any(col_idx):
        ns_col = df.columns[col_idx][0]
        ds_col = ns_col + 1
        
        # Extract and clean data
        ns = pd.to_numeric(df[ns_col].iloc[1:], errors='coerce')
        ds = pd.to_numeric(df[ds_col].iloc[1:], errors='coerce')
        
        # Remove NaN values
        mask = ~(ns.isna() | ds.isna())
        ns = ns[mask].values
        ds = ds[mask].values
        
        # Store points and efficiencies
        all_points_ns.extend(ns)
        all_points_ds.extend(ds)
        all_efficiencies.extend([eff] * len(ns))

# Convert to numpy arrays
all_points_ns = np.array(all_points_ns)
all_points_ds = np.array(all_points_ds)
all_efficiencies = np.array(all_efficiencies)

# Create RBF interpolator
# Using logarithmic coordinates for better interpolation
log_ns = np.log10(all_points_ns)
log_ds = np.log10(all_points_ds)
rbf = Rbf(log_ns, log_ds, all_efficiencies, function='multiquadric', smooth=0.1)

# Evaluate on grid
log_NS = np.log10(NS)
log_DS = np.log10(DS)
efficiency_grid = rbf(log_NS, log_DS)

# Mask values below 0.1
masked_efficiency = np.ma.masked_where(efficiency_grid < 0.1, efficiency_grid)

# Create the contour plot
plt.figure(figsize=(10, 8))

# Add filled contours for continuous color mapping
contour_filled = plt.contourf(NS, DS, masked_efficiency, 
                            levels=np.linspace(0.1, 0.8, 71),  # 71 levels for smooth transition
                            cmap='viridis', extend='neither')  # Changed to 'neither' to not show values outside range

# Add distinct contour lines for specific efficiencies
contour_lines = plt.contour(NS, DS, efficiency_grid,
                           levels=efficiencies,
                           colors='black', linewidths=0.5)

# Add contour labels
plt.clabel(contour_lines, inline=True, fmt='%.1f', fontsize=8)

# Add colorbar back
plt.colorbar(contour_filled, label='Efficiency')

# Set background color to white for masked regions
plt.gca().patch.set_color('white')

# Customize the plot
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Specific Speed (Ns)')
plt.ylabel('Specific Diameter (Ds)')
plt.title('Turbine Efficiency Map (RBF Interpolation)')
plt.grid(True, which='both', ls='-', alpha=0.2)

# Set axis limits
plt.xlim(0.1, 1000)
plt.ylim(0.1, 100)

# Plot original data points
plt.scatter(all_points_ns, all_points_ds, c='black', s=1, alpha=0.2)

plt.tight_layout()
plt.show()