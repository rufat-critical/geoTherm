import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import h5py

def create_pump_map(csv_file, output_file):
    """Create pump map HDF5 file from CSV data.
    
    Args:
        csv_file (str): Input CSV file path
        output_file (str): Output HDF5 file path
    """
    # Load and clean data
    df = pd.read_csv(csv_file)
    df_cleaned = df.dropna()
    
    # Extract data points
    X_250 = df_cleaned["X_250"]  # RPM
    Y_250 = df_cleaned["Y_250"]  # GPM at 250 PSI
    X_1200 = df_cleaned["X_1200"]  # RPM
    Y_1200 = df_cleaned["Y_1200"]  # GPM at 1200 PSI
    
    # Perform linear regression
    slope_250, intercept_250, r_value_250, _, _ = linregress(X_250, Y_250)
    slope_1200, intercept_1200, r_value_1200, _, _ = linregress(X_1200, Y_1200)
    
    # Generate interpolated points
    rpm_range = np.arange(100, 1201, 5)  # RPM points
    flow_250 = slope_250 * rpm_range + intercept_250  # GPM
    flow_1200 = slope_1200 * rpm_range + intercept_1200  # GPM
    
    # Convert units
    GPM_TO_M3S = 0.0000630902
    BAR_PER_PSI = 0.0689476
    
    # Create HDF5 file
    pressures = {
        f"{17}": {  # 250 PSI converted to bar
            "RPM": rpm_range,
            "Q": flow_250 * GPM_TO_M3S
        },
        f"{83}": {  # 1200 PSI converted to bar
            "RPM": rpm_range,
            "Q": flow_1200 * GPM_TO_M3S
        }
    }
    
    with h5py.File(output_file, "w") as f:
        for pressure, data in pressures.items():
            group = f.create_group(pressure)
            group.create_dataset("RPM", data=data["RPM"])
            group.create_dataset("Q", data=data["Q"])
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw data plot
    ax1.scatter(X_250, Y_250, label="250 PSI Data", color="blue", alpha=0.6)
    ax1.scatter(X_1200, Y_1200, label="1200 PSI Data", color="red", alpha=0.6)
    ax1.plot(rpm_range, flow_250, '--', color="blue", 
             label=f"250 PSI Fit (R²={r_value_250**2:.3f})")
    ax1.plot(rpm_range, flow_1200, '--', color="red", 
             label=f"1200 PSI Fit (R²={r_value_1200**2:.3f})")
    ax1.set_xlabel("RPM")
    ax1.set_ylabel("Flow Rate [GPM]")
    ax1.set_title("Raw Pump Data and Fits")
    ax1.grid(True)
    ax1.legend()
    
    # Converted units plot
    ax2.plot(rpm_range, flow_250 * GPM_TO_M3S, '--', color="blue", 
             label="17 bar")
    ax2.plot(rpm_range, flow_1200 * GPM_TO_M3S, '--', color="red", 
             label="83 bar")
    ax2.set_xlabel("RPM")
    ax2.set_ylabel("Flow Rate [m³/s]")
    ax2.set_title("Converted Pump Map")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nData saved to {output_file}")
    print(f"RPM range: {rpm_range[0]} to {rpm_range[-1]}")
    print(f"Flow range: {min(flow_250 * GPM_TO_M3S):.6f} to {max(flow_1200 * GPM_TO_M3S):.6f} m³/s")
    print(f"Pressure range: 17 to 83 bar")

if __name__ == "__main__":
    create_pump_map("wpd_datasets.csv", "D35e.h5")
