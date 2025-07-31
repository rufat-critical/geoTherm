import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import re
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
            fallback_method (str): Method to use for points outside data range.
                                 Options: 'nearest' or 'extrapolate'. Defaults to 'extrapolate'.
        """
        self.csv_file_path = csv_file_path
        self.info = {}
        self.head_interpolator = None
        self.q_interpolator = None
        
        # Validate and store the fallback method
        valid_methods = ['nearest', 'extrapolate']
        if fallback_method not in valid_methods:
            raise ValueError(f"fallback_method must be one of {valid_methods}")
        self.fallback_method = fallback_method
        
        # Load the data
        self.load_data()
        
        # Create the interpolators
        self.create_interpolators()
        
    def load_data(self):
        """Load and parse the pump data from the CSV file."""
        # Read the CSV file as raw text
        with open(self.csv_file_path, 'r') as f:
            lines = f.readlines()
        
        # Extract metadata
        self.info['Type'] = lines[0].strip().split(',')[1]
        self.info['Desc'] = lines[1].strip().split(',')[1]
        
        # Extract column names and units
        value_names = [val for val in lines[2].strip().split(',')[1:3] if val]
        value_units = [val for val in lines[3].strip().split(',')[1:3] if val]
        
        # Find where the data starts (look for 'DATA' marker)
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('DATA'):
                data_start_idx = i + 2  # Skip the DATA row and header row
                break
        
        # Extract data
        q_data = []
        head_data = []

        # Use regular expressions to parse data rows
        # Pattern to match groups of two numeric values separated by commas
        pattern = re.compile(r'([0-9.]+)\s*,\s*([0-9.]+)')
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            
            # Find all matches of two consecutive numeric values
            matches = pattern.findall(line)
            
            # Process all matches in the line
            for match in matches:
                try:
                    q, head = match
                    # Convert flow rate from m³/h to m³/s
                    q_m3s = float(q) / 3600.0  # Convert m³/h to m³/s
                    q_data.append(q_m3s)
                    head_data.append(float(head))
                except (ValueError, IndexError):
                    # Skip invalid data
                    continue
        
        # Store converted values
        self.q_values = np.array(q_data)
        self.head_values = np.array(head_data)

        # Store the units - Q is now in m³/s
        self.units = {
            'q': 'm³/s',  # Always store in m³/s
            'head': value_units[1] if len(value_units) > 1 else 'm'
        }
        
        # Debug: Print the loaded data
        print(f"Loaded {len(self.q_values)} data points")
        print(f"Q range: {np.min(self.q_values):.6f} to {np.max(self.q_values):.6f} {self.units['q']}")
        print(f"Head range: {np.min(self.head_values):.3f} to {np.max(self.head_values):.3f} {self.units['head']}")
        
    def create_interpolators(self):
        """Create interpolators for head from Q and Q from head."""
        # Sort data by Q values for proper interpolation
        sort_idx = np.argsort(self.q_values)
        q_sorted = self.q_values[sort_idx]
        head_sorted = self.head_values[sort_idx]
        
        # Create interpolator for head from Q
        self.head_interpolator = interp1d(q_sorted, head_sorted, 
                                        kind='linear', 
                                        bounds_error=False, 
                                        fill_value='extrapolate')
        
        # Create interpolator for Q from head (inverse relationship)
        # Sort data by head values for proper interpolation
        sort_idx_head = np.argsort(head_sorted)
        head_sorted_for_q = head_sorted[sort_idx_head]
        q_sorted_for_head = q_sorted[sort_idx_head]
        
        self.q_interpolator = interp1d(head_sorted_for_q, q_sorted_for_head, 
                                     kind='linear', 
                                     bounds_error=False, 
                                     fill_value='extrapolate')
    
    def get_head(self, q):
        """
        Get the interpolated head for a given flow rate.
        
        Args:
            q (float): The flow rate value in m³/s
            
        Returns:
            float: The interpolated head value in m
        """
        # Check if we're extrapolating
        data_range = self.get_data_range()
        is_extrapolating = False
        
        if q < data_range['q_min'] or q > data_range['q_max']:
            is_extrapolating = True
            logger.warn(f"Warning: Extrapolating flow rate value ({q}) outside data range "
                  f"[{data_range['q_min']}, {data_range['q_max']}]")
        
        result = self.head_interpolator(q)
        
        # Ensure the result is positive
        result = max(0.0, float(result))
        
        return result
    
    def get_q(self, head):
        """
        Get the interpolated flow rate for a given head.
        
        Args:
            head (float): The head value in m
            
        Returns:
            float: The interpolated flow rate value in m³/s
        """
        # Check if we're extrapolating
        data_range = self.get_data_range()
        is_extrapolating = False
        
        if head < data_range['head_min'] or head > data_range['head_max']:
            is_extrapolating = True
            logger.warn(f"Warning: Extrapolating head value ({head}) outside data range "
                  f"[{data_range['head_min']}, {data_range['head_max']}]")
        
        result = self.q_interpolator(head)
        
        # Ensure the result is positive
        result = max(0.0, float(result))
        
        return result
    
    def get_data_range(self):
        """
        Get the range of Q and Head values in the data.
        
        Returns:
            dict: A dictionary containing min and max values for Q and Head
        """
        return {
            'q_min': np.min(self.q_values),
            'q_max': np.max(self.q_values),
            'head_min': np.min(self.head_values),
            'head_max': np.max(self.head_values)
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
            f"Flow Rate Range: {data_range['q_min']:.3f} to {data_range['q_max']:.3f} {self.units['q']}\n"
            f"Head Range: {data_range['head_min']:.3f} to {data_range['head_max']:.3f} {self.units['head']}"
        )
    
    def plot(self, operating_points=None, figsize=(12, 8), show_op_labels=True):
        """
        Plot the pump map showing head vs flow rate.
        
        Args:
            operating_points (list or tuple, optional): 
                Either a single tuple (q, head) or a list of such tuples to mark on the plot.
            figsize (tuple, optional): Figure size as (width, height) in inches
            show_op_labels (bool, optional): Whether to show labels and legend for operating points. Defaults to True.
        """
        # Get data range
        data_range = self.get_data_range()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot original data points first
        ax.scatter(self.q_values, self.head_values, color='red', s=30, 
                  zorder=5, label='Data Points')
        
        # Create smooth curve for plotting - only within the data range
        q_min = data_range['q_min']
        q_max = data_range['q_max']
        q_plot = np.linspace(q_min, q_max, 200)  # 200 points for smooth curve
        head_plot = [self.get_head(q) for q in q_plot]
        
        # Plot the pump curve
        ax.plot(q_plot, head_plot, 'b-', linewidth=2, label='Pump Curve')
        
        # --- Plot operating points if provided ---
        if operating_points is not None:
            # Convert single point to list for uniform processing
            if not isinstance(operating_points, list):
                operating_points = [operating_points]
            
            # Plot each operating point
            for i, point in enumerate(operating_points):
                # Unpack the point
                if len(point) == 1:
                    q = point[0]
                    head = self.get_head(q)
                else:
                    q, head = point
                
                # Use different marker styles for multiple points
                marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
                marker = marker_styles[i % len(marker_styles)]
                
                # Plot operating point
                label = f'Point {i+1}: ({q:.3f} {self.units["q"]}, {head:.3f} {self.units["head"]})' if show_op_labels else None
                ax.scatter(q, head, color='green', s=100, zorder=10, marker=marker, label=label)
                
                # Add text label if enabled
                if show_op_labels:
                    ax.annotate(f'({q:.3f}, {head:.3f})',
                                (q, head), xytext=(10, 5), textcoords='offset points',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # --- Add labels, titles, and formatting ---
        ax.set_xlabel(f'Flow Rate ({self.units["q"]})')
        ax.set_ylabel(f'Head ({self.units["head"]})')
        ax.set_title(f'Pump Performance Curve: {self.info.get("Type", "Unknown")}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Set axis limits to data range with some padding
        ax.set_xlim(q_min * 0.95, q_max * 1.05)
        ax.set_ylim(data_range['head_min'] * 0.95, data_range['head_max'] * 1.05)
        
        # Add main title
        fig.suptitle(f'Pump Map: {self.info.get("Type", "Unknown")}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make room for the suptitle
        
        # Show the figure and keep it open
        plt.show()


# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Create the full path to the CSV file
csv_path = os.path.join(current_dir, 'FlowServePump.csv')

# Create and export the PumpMap instance
PumpMap = PumpMap(csv_path)

# Example usage
if __name__ == "__main__":
    # Use the already created instance
    pump_map = PumpMap
    
    # Get metadata
    print(pump_map)

    # Example interpolation - get head from flow rate
    test_q = 50.0 / 3600.0  # Convert 50 m³/h to m³/s
    head = pump_map.get_head(test_q)
    print(f"Interpolated Head at Q={test_q:.6f} {pump_map.units['q']}: {head:.3f} {pump_map.units['head']}")
    
    # Example interpolation - get flow rate from head
    test_head = 20.0  # m
    q = pump_map.get_q(test_head)
    print(f"Interpolated Flow Rate at Head={test_head} {pump_map.units['head']}: {q:.6f} {pump_map.units['q']}")
    
    # Create the plot without any operating points
    pump_map.plot()
