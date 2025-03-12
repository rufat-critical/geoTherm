import numpy as np
import matplotlib.pyplot as plt
from interp import PumpMapGrid
import h5py
import seaborn as sns
import geoTherm as gt

def examine_h5_file(filename):
    """Print contents and structure of HDF5 file."""
    with h5py.File(filename, 'r') as f:
        print("\nHDF5 File Structure:")
        print("-------------------")
        for pressure in f.keys():
            rpm = np.array(f[pressure]['RPM'])
            flow = np.array(f[pressure]['Q'])
            print(f"\nPressure: {pressure} bar")
            print(f"RPM range: {rpm.min():.0f} to {rpm.max():.0f}")
            print(f"Flow range: {flow.min():.6f} to {flow.max():.6f} m³/s")

def analyze_pump_map(filename, test_rpms=[600, 800, 1000, 1150], T=55, 
                    inlet_pressures=[2.8, 3.8]):
    """Analyze and visualize pump map data with acetone properties.
    
    Args:
        filename (str): Path to pump map HDF5 file
        test_rpms (list): RPM values for Q-P sweeps
        T (float): Acetone temperature in °C
        inlet_pressures (list): Inlet pressures to analyze [bar]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create a different line style for each inlet pressure
    styles = ['-', '--']
    
    for inlet_p, style in zip(inlet_pressures, styles):
        # Initialize acetone properties
        acetone = gt.thermo()
        acetone.TPY = (T, 'degC'), (inlet_p, 'bar'), 'acetone'
        rho = acetone.density
        
        # Load pump map
        pump_map = PumpMapGrid(filename)
        
        # Get raw data
        pressures = []
        rpms = []
        flows = []
        
        # Plot 1: Raw data - Flow vs RPM at different pressures
        with h5py.File(filename, 'r') as f:
            for pressure in f.keys():
                pressures.append(float(pressure))
                rpm = np.array(f[pressure]['RPM'])
                flow = np.array(f[pressure]['Q'])
                # Convert volumetric flow to mass flow
                mass_flow = flow * rho
                rpms.append(rpm)
                flows.append(mass_flow)
                # Plot raw data lines
                ax1.plot(rpm, mass_flow, style, 
                        label=f'ΔP={pressure} bar, Pin={inlet_p} bar')
        
        # Get flow bounds for plotting
        mdot_bounds = np.array(pump_map.flow_bounds) * rho
        
        # Plot 2: Q-P sweeps at specified RPMs
        colors = sns.color_palette('husl', len(test_rpms))
        for rpm, color in zip(test_rpms, colors):
            # Get Q-P curve using only points within bounds
            mdot_sweep = np.linspace(mdot_bounds[0], mdot_bounds[1], 100)
            Q_sweep = mdot_sweep / rho
            P_sweep = [pump_map.get_outlet_pressure(rpm, q) + inlet_p for q in Q_sweep]
            
            # Plot on mdot-P plot
            ax2.plot(mdot_sweep, P_sweep, style, color=color,
                    label=f'{rpm} RPM, Pin={inlet_p} bar')
            
            # Get and plot operating points on RPM-mdot plot
            Q_points = []
            for p in pressures:
                Q_points.append(pump_map.interpolator((p, rpm)) * rho)
            ax1.scatter([rpm]*len(pressures), Q_points, color=color)
    
    ax1.set_xlabel('RPM')
    ax1.set_ylabel('Mass Flow Rate [kg/s]')
    ax1.set_title(f'Pump Characteristic Curves (Acetone at {T}°C)')
    ax1.grid(True)
    ax1.legend()
    
    # Add flow markers every 0.1 kg/s
    marker_flows = np.arange(np.ceil(mdot_bounds[0]*10)/10, 
                           np.floor(mdot_bounds[1]*10)/10 + 0.1, 0.1)
    for mdot in marker_flows:
        ax2.axvline(mdot, color='gray', linestyle=':', alpha=0.3)
    
    ax2.set_xlabel('Mass Flow Rate [kg/s]')
    ax2.set_ylabel('Absolute Pressure [bar]')
    ax2.set_title('Total Pressure vs Mass Flow at Various RPMs')
    ax2.grid(True)
    ax2.set_xlim(mdot_bounds)
    
    # Add operating bounds
    ax2.axvline(mdot_bounds[0], color='r', linestyle='--', 
                label='Flow Bounds')
    ax2.axvline(mdot_bounds[1], color='r', linestyle='--')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print data ranges
    print("\nPump Map Boundaries:")
    print(f"RPM range: {pump_map.rpm_bounds[0]:.0f} to {pump_map.rpm_bounds[1]:.0f}")
    print(f"Mass flow range: {mdot_bounds[0]:.2f} to {mdot_bounds[1]:.2f} kg/s")
    print(f"Pressure range: {pump_map.pressure_bounds[0]:.1f} to {pump_map.pressure_bounds[1]:.1f} bar")

if __name__ == "__main__":
    filename = "D35e.h5"
    examine_h5_file(filename)
    analyze_pump_map(filename)
    
    # Test a specific operating point
    acetone = gt.thermo()
    acetone.TPY = (65, 'degC'), (2.8, 'bar'), 'acetone'
    
    mdot = 1.7  # kg/s
    Q = mdot/acetone.density
    
    rpm = 600
    pump_map = PumpMapGrid(filename)
    is_extrapolating = pump_map.check_extrapolation(rpm, Q)
    P = pump_map.get_outlet_pressure(rpm, Q)
    print(f"\nTest Point:")
    print(f"RPM = {rpm}")
    print(f"Mass flow = {mdot:.2f} kg/s")
    print(f"Volume flow = {Q:.6f} m³/s")
    print(f"P = {P:.2f} bar")