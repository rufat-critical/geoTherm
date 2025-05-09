import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import os
import re
from geoTherm.units import toSI
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from geoTherm.logger import logger


class PumpMap:
    def __init__(self, csv_file_path, fallback_method='extrapolate'):
        """
        Initialize the pump data interpolator with a CSV file path.
        
        Args:
            csv_file_path (str): Path to the CSV file containing pump data
            fallback_method (str): Method to use for points outside convex hull.
                                 Options: 'nearest' or 'extrapolate'. Defaults to 'nearest'.
        """
        self.csv_file_path = csv_file_path
        self.info = {}
        self.interpolator = None
        
        # Validate and store the fallback method
        valid_methods = ['nearest', 'extrapolate']
        if fallback_method not in valid_methods:
            raise ValueError(f"fallback_method must be one of {valid_methods}")
        self.fallback_method = fallback_method
        
        # Load the data
        self.load_data()
        
        # Create the interpolator
        self.create_interpolator()
        
    def load_data(self):
        """Load and parse the pump data from the CSV file."""
        # Read the CSV file as raw text
        with open(self.csv_file_path, 'r') as f:
            lines = f.readlines()
        
        # Extract metadata
        self.info['Type'] = lines[0].strip().split(',')[1]
        self.info['Desc'] = lines[1].strip().split(',')[1]
        
        # Extract column names and units
        value_names = [val for val in lines[2].strip().split(',')[1:4] if val]
        value_units = [val for val in lines[3].strip().split(',')[1:4] if val]
        
        # Find where the data starts (look for 'DATA' marker)
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('DATA'):
                data_start_idx = i + 2  # Skip the DATA row and header row
                break
        
        # Extract data
        rpm_data = []
        q_data = []
        pressure_data = []

        # Use regular expressions to parse data rows
        # Pattern to match groups of three numeric values separated by commas
        pattern = re.compile(r'([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)')
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            
            # Find all matches of three consecutive numeric values
            matches = pattern.findall(line)
            
            # Process all matches in the line
            for match in matches:
                try:
                    rpm, q, pressure = match
                    rpm_data.append(float(rpm))
                    q_data.append(float(q))
                    pressure_data.append(float(pressure))
                except (ValueError, IndexError):
                    # Skip invalid data
                    continue
        
        # Convert to SI units
        self.rpm_values = np.array([toSI((rpm_data[i], value_units[0]), 'RPM') for i, _ in enumerate(rpm_data)])
        self.q_values = np.array([toSI((q_data[i], value_units[1]), 'VOLUMETRICFLOW') for i, _ in enumerate(q_data)])
        self.pressure_data = np.array([toSI((pressure_data[i], value_units[2]), 'PRESSURE') for i, _ in enumerate(pressure_data)])


        # Store the units
        self.units = {
            'rpm': value_units[0] if value_units else 'RPM',
            'q': value_units[1] if len(value_units) > 1 else 'm³/s',
            'pressure': value_units[2] if len(value_units) > 2 else 'bar'
        }
        
    def create_interpolator(self):
        """Create the interpolator for pressure based on RPM and Q."""
        # Combine RPM and Q as the points for interpolation
        points = np.column_stack((self.rpm_values, self.q_values))
        
        # Create the primary interpolator
        self.interpolator = LinearNDInterpolator(points, self.pressure_data)
        
        if self.fallback_method == 'nearest':
            # Create a nearest neighbor interpolator for fallback
            self.fallback_interpolator = NearestNDInterpolator(points, self.pressure_data)
        else:  # 'extrapolate'
            # Create a linear extrapolation function
            def extrapolate(rpm, q):
                # Find the k nearest points to use for extrapolation
                k = 3  # Use 3 nearest points to establish the linear trend
                distances = np.sqrt((self.rpm_values - rpm)**2 + (self.q_values - q)**2)
                nearest_indices = np.argsort(distances)[:k]
                
                # Get the coordinates and values for these points
                nearest_points = np.column_stack((self.rpm_values[nearest_indices], 
                                                self.q_values[nearest_indices]))
                nearest_pressures = self.pressure_data[nearest_indices]
                
                # Fit a linear plane through these points
                # Add a column of ones for the intercept term
                A = np.column_stack((nearest_points, np.ones(k)))
                # Solve for coefficients [a, b, c] in z = ax + by + c
                coeffs = np.linalg.lstsq(A, nearest_pressures, rcond=None)[0]
                
                # Return the extrapolated value
                return coeffs[0] * rpm + coeffs[1] * q + coeffs[2]
            
            self.fallback_interpolator = extrapolate
    

    def get_dP(self, rpm, q):
        """
        Get the interpolated dP for a given RPM and flow rate.
        
        Args:
            rpm (float): The RPM value
            q (float): The flow rate value in m³/s
            
        Returns:
            tuple: (float, bool) The interpolated pressure value in bar and whether extrapolation was used
        """
        # Check if we're extrapolating
        data_range = self.get_data_range()
        is_extrapolating = False
        
        if rpm < data_range['rpm_min'] or rpm > data_range['rpm_max']:
            is_extrapolating = True
            logger.warn(f"Warning: Extrapolating RPM value ({rpm}) outside data range "
                  f"[{data_range['rpm_min']}, {data_range['rpm_max']}]")
        
        if q < data_range['q_min'] or q > data_range['q_max']:
            is_extrapolating = True
            logger.warn(f"Warning: Extrapolating flow rate value ({q}) outside data range "
                  f"[{data_range['q_min']}, {data_range['q_max']}]")
        
        result = self.interpolator(rpm, q)
        
        # If the result is NaN (outside the convex hull), use the fallback interpolator
        if np.isnan(result):
            if not is_extrapolating:
                method_name = "nearest neighbor" if self.fallback_method == 'nearest' else "linear extrapolation"
                logger.warn(f"Warning: Using {method_name} for point outside convex hull")
            
            if self.fallback_method == 'nearest':
                result = self.fallback_interpolator(rpm, q)
            else:
                result = self.fallback_interpolator(rpm, q)
        
        return float(result), is_extrapolating
    
    def get_data_range(self):
        """
        Get the range of RPM and Q values in the data.
        
        Returns:
            dict: A dictionary containing min and max values for RPM and Q
        """
        return {
            'rpm_min': np.min(self.rpm_values),
            'rpm_max': np.max(self.rpm_values),
            'q_min': np.min(self.q_values),
            'q_max': np.max(self.q_values)
        }
    
    def __str__(self):
        """
        Return a string representation of the pump data interpolator.
        
        Returns:
            str: A string containing pump name, description, and data ranges
        """
        data_range = self.get_data_range()
        return (
            f"Pump: {self.info.get('Type', 'Unknown')}\n"
            f"Description: {self.info.get('Desc', 'Unknown')}\n"
            f"RPM Range: {data_range['rpm_min']} to {data_range['rpm_max']} {self.units['rpm']}\n"
            f"Flow Rate Range: {data_range['q_min']} to {data_range['q_max']} {self.units['q']}"
        )
    
    def plot(self, operating_points=None, rpm_values=None, figsize=(16, 8), show_op_labels=True):
        """
        Plot the pump map as combined subplots showing both pressure vs flow and flow vs RPM views.
        
        Args:
            operating_points (list or tuple, optional): 
                Either a single tuple (rpm, q, pressure) or a list of such tuples to mark on the plots.
            rpm_values (list, optional): List of RPM values to plot.
            figsize (tuple, optional): Figure size as (width, height) in inches
            show_op_labels (bool, optional): Whether to show labels and legend for operating points. Defaults to True.
        """
        # Get data range
        data_range = self.get_data_range()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # --- First subplot: Pressure vs Q for different RPM values ---
        
        # If rpm_values not provided, create a range of values
        if rpm_values is None:
            rpm_min = data_range['rpm_min']
            rpm_max = data_range['rpm_max']
            rpm_values = np.linspace(rpm_min, rpm_max, 5)  # 5 RPM curves
        
        # Create a colormap for the RPM curves
        cmap = get_cmap('viridis')
        norm = colors.Normalize(vmin=min(rpm_values), vmax=max(rpm_values))
        
        # Create q values for plotting
        q_min = data_range['q_min']
        q_max = data_range['q_max']
        q_values = np.linspace(q_min, q_max, 100)  # 100 points for smooth curves
        
        # Plot each RPM curve on the first subplot
        for i, rpm in enumerate(rpm_values):
            pressures = [self.get_pressure(rpm, q) for q in q_values]
            color = cmap(norm(rpm))
            ax1.plot(q_values, pressures, color=color, label=f"{rpm:.0f} RPM")
        
        # --- Second subplot: Q vs RPM with isopressure lines ---
        # Clear previous content for second subplot
        ax2.clear()
        
        # Get unique pressure values (rounded to avoid floating point issues)
        unique_pressures = np.unique(np.round(self.pressure_data, decimals=4))
        
        # Create a colormap for the pressure lines
        pressure_cmap = get_cmap('viridis')
        pressure_norm = colors.Normalize(vmin=min(unique_pressures), vmax=max(unique_pressures))
        
        # Plot each pressure line
        for pressure in unique_pressures:
            # Find indices where pressure is approximately equal to the current value
            indices = np.where(np.isclose(self.pressure_data, pressure, rtol=1e-4))[0]
            
            if len(indices) > 1:  # Need at least 2 points to form a line
                # Get corresponding RPM and Q values
                rpm_points = self.rpm_values[indices]
                q_points = self.q_values[indices]
                
                # Sort points by RPM to connect them properly
                sort_idx = np.argsort(rpm_points)
                rpm_sorted = rpm_points[sort_idx]
                q_sorted = q_points[sort_idx]
                
                # Plot as a continuous line with color based on pressure
                color = pressure_cmap(pressure_norm(pressure))
                ax2.plot(rpm_sorted, q_sorted, color=color, 
                        linewidth=2, label=f'{pressure/1e5:.1f} bar')
        
        # Add legend to the second subplot with fewer entries to avoid overcrowding
        handles, labels = ax2.get_legend_handles_labels()
        # Select every nth label to reduce legend size
        n = max(1, len(labels) // 5)  # Show about 5 pressure values in legend
        ax2.legend(handles[::n], labels[::n], loc='best', title='Pressure Values')
        
        # --- Plot operating points on both subplots if provided ---
        if operating_points is not None:
            # Convert single point to list for uniform processing
            if not isinstance(operating_points, list):
                operating_points = [operating_points]
            
            # Plot each operating point on both subplots
            for i, point in enumerate(operating_points):
                # Unpack the point
                if len(point) == 2:
                    rpm, q = point
                    pressure = self.get_pressure(rpm, q)
                else:
                    rpm, q, pressure = point
                
                # Use different marker styles for multiple points
                marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
                marker = marker_styles[i % len(marker_styles)]
                
                # Plot on first subplot (Pressure vs Q) with smaller marker
                label = f'Point {i+1}: ({rpm:.0f} RPM, {q:.4f} m³/s, {pressure:.2f} bar)' if show_op_labels else None
                ax1.scatter(q, pressure, color='red', s=50, zorder=5, marker=marker, label=label)
                
                # Add text label on first subplot if enabled
                if show_op_labels:
                    ax1.annotate(f'({rpm:.0f}, {q:.4f}, {pressure:.2f})',
                                (q, pressure), xytext=(10, 5), textcoords='offset points',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                # Plot on second subplot (Q vs RPM) with smaller marker
                ax2.scatter(rpm, q, color='red', s=50, zorder=5, marker=marker)
                
                # Add text label on second subplot if enabled
                if show_op_labels:
                    ax2.annotate(f'({rpm:.0f}, {q:.4f}, {pressure:.2f})',
                                (rpm, q), xytext=(10, 5), textcoords='offset points',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # --- Add labels, titles, and formatting to both subplots ---
        
        # First subplot
        ax1.set_xlabel(f'Flow ({self.units["q"]})')
        ax1.set_ylabel(f'Pressure ({self.units["pressure"]})')
        ax1.set_title(f'Pressure vs Flow at Different RPM')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best')
        
        # Second subplot
        ax2.set_xlabel(f'RPM ({self.units["rpm"]})')
        ax2.set_ylabel(f'Flow ({self.units["q"]})')
        ax2.set_title(f'Flow vs RPM with Pressure Contours')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # No need for legend in second subplot since we're using direct labels
        
        # Add main title
        fig.suptitle(f'Pump Map: {self.info.get("Type", "Unknown")}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make room for the suptitle
        
        # Show the figure
        fig.show()

    def get_rpm(self, pressure, q, tolerance=1e-6, max_iterations=100):
        """
        Find the RPM value that produces the target pressure at the given flow rate.
        
        Args:
            pressure (float): Target pressure value in bar
            q (float): Flow rate value in m³/s
            tolerance (float, optional): Acceptable error in pressure. Defaults to 1e-6.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
            
        Returns:
            tuple: (rpm, bool) where rpm is the found RPM value and bool indicates if solution converged
        """
        # Get data range
        data_range = self.get_data_range()
        rpm_min = data_range['rpm_min']
        rpm_max = data_range['rpm_max']
        
        # Binary search for RPM value
        left = rpm_min
        right = rpm_max
        iterations = 0
        
        while iterations < max_iterations:
            rpm = (left + right) / 2
            current_pressure, _ = self.get_pressure(rpm, q)
            
            # Check if we're close enough
            if abs(current_pressure - pressure) < tolerance:
                return rpm, True
            
            # Adjust search range
            if current_pressure < pressure:
                left = rpm
            else:
                right = rpm
            
            iterations += 1
        
        # If we exit the loop, we didn't converge
        # Return best approximation found
        return (left + right) / 2, False


# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Create the full path to the CSV file
csv_path = os.path.join(current_dir, 'D35e.csv')

# Create and export the PumpMap instance
PumpMap = PumpMap(csv_path)

# Example usage
if __name__ == "__main__":
    # Use the already created instance
    pump_map = PumpMap
    
    # Get metadata
    print(pump_map)

    # Example interpolation
    test_rpm = 500
    test_q = 0.001
    pressure = pump_map.get_pressure(test_rpm, test_q)

    print(f"Interpolated Pressure at RPM={test_rpm}, Q={test_q} m³/s: {pressure} bar")
    
    # Example of plotting with multiple operating points
    # Point 1: Specify RPM and Q, let the method calculate pressure
    point1 = (500, 0.001)
    # Point 2: Specify RPM, Q, and pressure
    point2 = (700, 0.002, pump_map.get_pressure(700, 0.002))
    
    # Create the combined plot view
    pump_map.plot(operating_points=[point1, point2])